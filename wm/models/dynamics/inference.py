"""Inference utilities for dynamics model."""
import torch
import torch.nn as nn


def denoise_frame(
    model: nn.Module,
    z_context_noisy: torch.Tensor,
    z_target_shape: tuple,
    action_t: torch.Tensor,
    K: int,
    context_noise: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> list[dict]:
    """
    Run K-step denoising for a single frame, returning all intermediates.

    Key: d = 1/K (step size is inversely proportional to number of steps)
    - K=4:  d=0.25,   tau progresses as [0, 0.25, 0.5, 0.75]
    - K=8:  d=0.125,  tau progresses as [0, 0.125, 0.25, ..., 0.875]
    - K=30: d=0.0333, tau progresses as [0, 0.0333, 0.0667, ..., 0.9667]

    Args:
        model: DynamicsModel
        z_context_noisy: (B, T_ctx, N, D) context with noise applied
        z_target_shape: Shape of target frame (B, 1, N, D)
        action_t: (B, 1) or (B, 1, action_dim) action for current frame
        K: Number of denoising steps
        context_noise: Noise level for context (tau_ctx)
        device: Device
        generator: Optional random generator for reproducibility

    Returns:
        List of dicts with step, tau, d, and z_pred for each denoising step
    """
    B, T_ctx, N, D = z_context_noisy.shape

    # Initialize from pure noise
    if generator is not None:
        z_t = torch.randn(z_target_shape, device=device, generator=generator)
    else:
        z_t = torch.randn(z_target_shape, device=device)

    # Step size depends on K
    d = 1.0 / K

    # Create context actions matching action_t shape
    # action_t can be (B, 1) or (B, 1, action_dim) for one-hot
    if action_t.dim() == 3:
        # One-hot actions: (B, 1, action_dim) -> context (B, T_ctx, action_dim)
        action_dim = action_t.shape[-1]
        actions_ctx = torch.zeros(B, T_ctx, action_dim, dtype=action_t.dtype, device=device)
    else:
        # Discrete actions: (B, 1) -> context (B, T_ctx)
        actions_ctx = torch.zeros(B, T_ctx, dtype=action_t.dtype, device=device)

    intermediates = []
    for k in range(K):
        tau_k = k / K  # Signal level increases from 0 to (K-1)/K

        # Build full sequence: context + current prediction
        z_full = torch.cat([z_context_noisy, z_t], dim=1)  # (B, T_ctx+1, N, D)

        # Build tau sequence: context at context_noise level, current at tau_k
        tau_ctx = torch.full((B, T_ctx), context_noise, device=device)
        tau_curr = torch.full((B, 1), tau_k, device=device)
        tau_full = torch.cat([tau_ctx, tau_curr], dim=1)  # (B, T_ctx+1)

        # Build d sequence: constant d for all
        d_full = torch.full((B, T_ctx + 1), d, device=device)

        # Build action sequence
        action_full = torch.cat([actions_ctx, action_t], dim=1)  # (B, T_ctx+1) or (B, T_ctx+1, action_dim)

        # Forward pass - model predicts clean representation
        z_pred = model(z_full, action_full, tau_full, d_full)

        # Extract prediction for current frame
        z_t = z_pred[:, -1:, :, :]  # (B, 1, N, D)

        intermediates.append({
            "step": k,
            "tau": tau_k,
            "d": d,
            "z_pred": z_t.clone().cpu(),
        })

    return intermediates
