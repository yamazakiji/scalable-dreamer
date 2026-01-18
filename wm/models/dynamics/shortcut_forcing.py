"""Shortcut forcing and flow matching objectives for dynamics training."""
import math
import torch
from torch import nn, Tensor


class ShortcutForcing(nn.Module):
    """
    Shortcut forcing training objective from Dreamer 4 (arXiv:2509.24527).

    Combines diffusion forcing with shortcut models for fast inference.
    The network predicts clean representations (x-prediction) to prevent
    error accumulation over long rollouts.

    Key features:
    - For finest step size d_min: standard flow matching loss
    - For larger step sizes: bootstrap loss distilling two smaller steps
    - Ramp loss weight: w(τ) = 0.9τ + 0.1
    - X-space formulation for stable long-horizon generation
    """

    def __init__(
        self,
        max_steps: int = 64,
        ramp_weight: bool = True,
    ):
        """
        Args:
            max_steps: Maximum number of sampling steps K_max (defines d_min = 1/K_max)
            ramp_weight: Whether to use ramp loss weight w(τ) = 0.9τ + 0.1
        """
        super().__init__()

        self.max_steps = max_steps
        self.d_min = 1.0 / max_steps
        self.ramp_weight = ramp_weight

        # Precompute valid step sizes: 1, 2, 4, 8, ..., K_max
        self.step_sizes = [2 ** i for i in range(int(math.log2(max_steps)) + 1)]
        self.num_step_sizes = len(self.step_sizes)

    def sample_schedule(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """
        Sample (τ, d) pairs according to paper Eq. 4.

        d ~ 1/U({1, 2, 4, 8, ..., K_max})
        τ ~ U({0, 1/d, ..., 1 - 1/d})

        Following diffusion forcing, each timestep gets independent noise levels.

        Args:
            batch_size: Batch size B
            seq_len: Sequence length T
            device: Device to create tensors on

        Returns:
            tau: (B, T) signal levels in [0, 1)
            d: (B, T) step sizes
        """
        # Sample step size index uniformly: d ~ 1/U({1, 2, 4, ..., K_max})
        step_idx = torch.randint(0, self.num_step_sizes, (batch_size, seq_len), device=device)

        # Convert index to actual step size d = 1/K where K is power of 2
        K = torch.tensor(self.step_sizes, device=device)[step_idx]  # (B, T)
        d = 1.0 / K  # (B, T)

        # Sample τ uniformly from grid determined by step size
        # τ ~ U({0, d, 2d, ..., 1 - d}) = U({0, 1/K, 2/K, ..., (K-1)/K})
        num_grid_points = K  # K grid points: 0, d, 2d, ..., (K-1)d
        grid_idx = torch.randint(0, self.max_steps, (batch_size, seq_len), device=device)
        grid_idx = grid_idx % num_grid_points  # Clamp to valid range
        tau = grid_idx.float() * d  # (B, T)

        return tau, d

    def corrupt(
        self,
        z_clean: Tensor,
        tau: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Create corrupted representation z̃ = (1-τ)z_0 + τz_1.

        Args:
            z_clean: (B, T, N, D) clean latent representations z_1
            tau: (B, T) signal levels

        Returns:
            z_corrupted: (B, T, N, D) corrupted representations z̃
            z_noise: (B, T, N, D) noise samples z_0
        """
        # Sample noise z_0 ~ N(0, I)
        z_noise = torch.randn_like(z_clean)

        # Expand tau for broadcasting: (B, T) -> (B, T, 1, 1)
        tau_expanded = tau.unsqueeze(-1).unsqueeze(-1)

        # Linear interpolation: z̃ = (1-τ)z_0 + τz_1
        z_corrupted = (1 - tau_expanded) * z_noise + tau_expanded * z_clean

        return z_corrupted, z_noise

    def compute_loss(
        self,
        model: nn.Module,
        z_clean: Tensor,
        actions: Tensor,
        tau: Tensor | None = None,
        d: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        Compute shortcut forcing loss.

        For d = d_min: L = ||ẑ_1 - z_1||²  (flow matching in x-space)
        For d > d_min: bootstrap loss with two smaller steps

        Args:
            model: Dynamics model that predicts clean z given corrupted z
            z_clean: (B, T, N, D) clean latent representations
            actions: (B, T) discrete action indices
            tau: (B, T) signal levels (sampled if None)
            d: (B, T) step sizes (sampled if None)

        Returns:
            Dictionary with 'total' loss and optional 'flow' and 'bootstrap' components
        """
        B, T, N, D = z_clean.shape
        device = z_clean.device

        # Sample schedule if not provided
        if tau is None or d is None:
            tau, d = self.sample_schedule(B, T, device)

        # Corrupt clean representations
        z_corrupted, z_noise = self.corrupt(z_clean, tau)

        # Identify which samples use flow matching vs bootstrap
        is_finest = (d <= self.d_min + 1e-6)  # (B, T)

        # Forward pass: predict clean representations
        z_pred = model(z_corrupted, actions, tau, d)

        # Compute ramp weight: w(τ) = 0.9τ + 0.1
        if self.ramp_weight:
            weight = 0.9 * tau + 0.1  # (B, T)
            weight = weight.unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
        else:
            weight = 1.0

        # Flow matching loss for finest step size: L = ||ẑ_1 - z_1||²
        flow_loss = (z_pred - z_clean).pow(2)  # (B, T, N, D)
        flow_loss = (flow_loss * weight).mean(dim=(-1, -2))  # (B, T)

        # For non-finest steps, compute bootstrap loss
        # This requires two forward passes with half step size
        if not is_finest.all():
            bootstrap_loss = self._compute_bootstrap_loss(
                model, z_clean, z_corrupted, z_noise, actions, tau, d, weight
            )
        else:
            bootstrap_loss = torch.zeros_like(flow_loss)

        # Combine losses: flow for finest, bootstrap for others
        is_finest_expanded = is_finest.float()
        total_loss = is_finest_expanded * flow_loss + (1 - is_finest_expanded) * bootstrap_loss

        return {
            "total": total_loss.mean(),
            "flow": (flow_loss * is_finest_expanded).sum() / (is_finest_expanded.sum() + 1e-8),
            "bootstrap": (bootstrap_loss * (1 - is_finest_expanded)).sum() / ((1 - is_finest_expanded).sum() + 1e-8),
        }

    def _compute_bootstrap_loss(
        self,
        model: nn.Module,
        z_clean: Tensor,
        z_corrupted: Tensor,
        z_noise: Tensor,
        actions: Tensor,
        tau: Tensor,
        d: Tensor,
        weight: Tensor | float,
    ) -> Tensor:
        """
        Compute bootstrap loss for larger step sizes.

        The bootstrap target is computed by taking two smaller steps:
        b' = f(z̃, τ, d/2)
        z' = z̃ + b' * d/2  (in v-space)
        b'' = f(z', τ + d/2, d/2)
        v_target = (b' + b'') / 2  (stopped gradient)

        Then loss is computed in x-space with (1-τ)² scaling.
        """
        B, T, N, D = z_clean.shape
        device = z_clean.device

        # Half step size
        d_half = d / 2  # (B, T)
        tau_expanded = tau.unsqueeze(-1).unsqueeze(-1)
        d_expanded = d.unsqueeze(-1).unsqueeze(-1)
        d_half_expanded = d_half.unsqueeze(-1).unsqueeze(-1)

        # First half step: predict clean from current corrupted
        with torch.no_grad():
            z_pred_1 = model(z_corrupted, actions, tau, d_half)
            # Convert x-prediction to v-prediction: v = (x̂_1 - x_τ) / (1 - τ)
            v_pred_1 = (z_pred_1 - z_corrupted) / (1 - tau_expanded + 1e-8)

            # Step forward in corrupted space: z' = z̃ + v * d/2
            z_mid = z_corrupted + v_pred_1 * d_half_expanded

            # Second half step from z_mid
            tau_mid = tau + d_half
            z_pred_2 = model(z_mid, actions, tau_mid, d_half)
            # Convert to v-prediction
            tau_mid_expanded = tau_mid.unsqueeze(-1).unsqueeze(-1)
            v_pred_2 = (z_pred_2 - z_mid) / (1 - tau_mid_expanded + 1e-8)

            # Target velocity is average of two steps
            v_target = (v_pred_1 + v_pred_2) / 2

        # Full step prediction
        z_pred_full = model(z_corrupted, actions, tau, d)

        # Convert prediction to v-space for loss
        v_pred_full = (z_pred_full - z_corrupted) / (1 - tau_expanded + 1e-8)

        # Bootstrap loss in v-space, scaled back to x-space
        # L = (1-τ)² ||v̂ - v_target||²
        scale = (1 - tau_expanded).pow(2)
        bootstrap_loss = scale * (v_pred_full - v_target).pow(2)

        if isinstance(weight, Tensor):
            bootstrap_loss = bootstrap_loss * weight

        return bootstrap_loss.mean(dim=(-1, -2))  # (B, T)


class FlowMatching(nn.Module):
    """
    Simple flow matching objective (no shortcut, for comparison).

    Standard flow matching predicts velocity v = z_1 - z_0 from corrupted z_τ.
    This implementation uses x-prediction for consistency with shortcut forcing.
    """

    def __init__(
        self,
        ramp_weight: bool = True,
        tau_distribution: str = "uniform",
    ):
        """
        Args:
            ramp_weight: Whether to use ramp loss weight w(τ) = 0.9τ + 0.1
            tau_distribution: Distribution for sampling τ ("uniform" or "logit_normal")
        """
        super().__init__()

        self.ramp_weight = ramp_weight
        self.tau_distribution = tau_distribution

    def sample_tau(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Tensor:
        """
        Sample signal levels τ.

        Args:
            batch_size: Batch size B
            seq_len: Sequence length T
            device: Device to create tensors on

        Returns:
            tau: (B, T) signal levels in [0, 1)
        """
        if self.tau_distribution == "uniform":
            tau = torch.rand(batch_size, seq_len, device=device)
        elif self.tau_distribution == "logit_normal":
            # Logit-normal distribution (concentrated in middle)
            logits = torch.randn(batch_size, seq_len, device=device)
            tau = torch.sigmoid(logits)
        else:
            raise ValueError(f"Unknown tau distribution: {self.tau_distribution}")

        return tau

    def corrupt(
        self,
        z_clean: Tensor,
        tau: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Create corrupted representation z̃ = (1-τ)z_0 + τz_1.

        Args:
            z_clean: (B, T, N, D) clean latent representations z_1
            tau: (B, T) signal levels

        Returns:
            z_corrupted: (B, T, N, D) corrupted representations z̃
            z_noise: (B, T, N, D) noise samples z_0
        """
        z_noise = torch.randn_like(z_clean)
        tau_expanded = tau.unsqueeze(-1).unsqueeze(-1)
        z_corrupted = (1 - tau_expanded) * z_noise + tau_expanded * z_clean
        return z_corrupted, z_noise

    def compute_loss(
        self,
        model: nn.Module,
        z_clean: Tensor,
        actions: Tensor,
        tau: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        Compute flow matching loss: L = ||ẑ_1 - z_1||² (x-prediction).

        Args:
            model: Dynamics model that predicts clean z given corrupted z
            z_clean: (B, T, N, D) clean latent representations
            actions: (B, T) discrete action indices
            tau: (B, T) signal levels (sampled if None)

        Returns:
            Dictionary with 'total' loss
        """
        B, T, N, D = z_clean.shape
        device = z_clean.device

        # Sample tau if not provided
        if tau is None:
            tau = self.sample_tau(B, T, device)

        # Corrupt clean representations
        z_corrupted, z_noise = self.corrupt(z_clean, tau)

        # Forward pass: predict clean representations
        # For simple flow matching, d is not used (or set to d_min)
        d = torch.full((B, T), 1.0 / 64, device=device)  # dummy step size
        z_pred = model(z_corrupted, actions, tau, d)

        # Compute ramp weight
        if self.ramp_weight:
            weight = 0.9 * tau + 0.1
            weight = weight.unsqueeze(-1).unsqueeze(-1)
        else:
            weight = 1.0

        # Flow matching loss in x-space
        loss = (z_pred - z_clean).pow(2)
        loss = (loss * weight).mean()

        return {"total": loss}
