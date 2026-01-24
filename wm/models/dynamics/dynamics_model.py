"""Interactive Dynamics model from Dreamer 4."""
import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from einops import rearrange

from wm.models.transformer.block import TransformerBlock
from wm.models.dynamics.action_encoder import ActionEncoder


class DynamicsModel(nn.Module):
    """
    Interactive Dynamics model from Dreamer 4 (arXiv:2509.24527).

    Predicts next latent representations given actions and past context.
    Uses shortcut forcing for fast inference with K=4 sampling steps.

    The model operates on interleaved sequences of:
    - Actions: encoded via discrete embeddings
    - Signal tokens: (τ, d) encoded via discrete embeddings
    - Corrupted representations: z̃ projected to embed_dim

    Architecture:
    - 2D transformer with space + time attention
    - Temporal attention only every 4 layers (efficiency)
    - Block-causal masking
    - Register tokens for temporal consistency
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 24,
        temporal_layer_freq: int = 4,
        num_spatial_tokens: int = 256,
        latent_dim: int = 32,
        num_register_tokens: int = 8,
        num_actions: int = 18,
        num_action_tokens: int = 1,
        max_sampling_steps: int = 64,
    ):
        """
        Args:
            embed_dim: Transformer embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            temporal_layer_freq: Apply temporal attention every N layers
            num_spatial_tokens: Number of spatial tokens per timestep (S_z)
            latent_dim: Dimension of tokenizer latent space
            num_register_tokens: Number of register tokens per timestep (S_r)
            num_actions: Size of discrete action vocabulary
            num_action_tokens: Number of tokens per action (S_a)
            max_sampling_steps: Maximum sampling steps K_max (for τ, d embeddings)
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.temporal_layer_freq = temporal_layer_freq
        self.num_spatial_tokens = num_spatial_tokens
        self.latent_dim = latent_dim
        self.num_register_tokens = num_register_tokens
        self.num_action_tokens = num_action_tokens
        self.max_sampling_steps = max_sampling_steps

        # Action encoder
        self.action_encoder = ActionEncoder(
            num_actions=num_actions,
            embed_dim=embed_dim,
            num_tokens=num_action_tokens,
        )

        # Signal level (τ) and step size (d) embeddings
        # τ is discretized to max_sampling_steps + 1 values (including τ=1)
        # d is log2(K) values where K is power of 2
        num_tau_values = max_sampling_steps + 1
        num_d_values = int(math.log2(max_sampling_steps)) + 1

        self.tau_embed = nn.Embedding(num_tau_values, embed_dim // 2)
        self.d_embed = nn.Embedding(num_d_values, embed_dim // 2)

        # Linear projection for concatenated (τ, d) embeddings
        self.signal_proj = nn.Linear(embed_dim, embed_dim)

        # Representation projection: latent_dim -> embed_dim
        self.z_proj = nn.Linear(latent_dim, embed_dim)

        # Register tokens (learned per-timestep, shared across batch)
        self.register_tokens = nn.Parameter(
            torch.randn(1, 1, num_register_tokens, embed_dim) * 0.02
        )

        # Build transformer blocks
        # Space attention at every layer, time attention every temporal_layer_freq layers
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            # Space attention layer
            self.blocks.append(TransformerBlock(embed_dim, num_heads, "space"))

            # Time attention layer every N space layers
            if (i + 1) % temporal_layer_freq == 0:
                self.blocks.append(TransformerBlock(embed_dim, num_heads, "time"))

        # Output projection: embed_dim -> latent_dim (predict clean z)
        self.output_norm = nn.RMSNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, latent_dim)

    def _discretize_tau(self, tau: Tensor) -> Tensor:
        """Discretize continuous τ to embedding indices."""
        # τ ∈ [0, 1] -> index ∈ [0, max_sampling_steps]
        indices = (tau * self.max_sampling_steps).round().long()
        return indices.clamp(0, self.max_sampling_steps)

    def _discretize_d(self, d: Tensor) -> Tensor:
        """Discretize step size d to embedding indices."""
        # d = 1/K where K is power of 2
        # index = log2(1/d) = log2(K)
        K = (1.0 / d).round()
        indices = torch.log2(K).round().long()
        num_d_values = int(math.log2(self.max_sampling_steps)) + 1
        return indices.clamp(0, num_d_values - 1)

    def _encode_signal(self, tau: Tensor, d: Tensor) -> Tensor:
        """
        Encode (τ, d) into signal token embeddings.

        Args:
            tau: (B, T) signal levels ∈ [0, 1]
            d: (B, T) step sizes

        Returns:
            signal_tokens: (B, T, 1, embed_dim) signal token embeddings
        """
        B, T = tau.shape

        # Discretize to indices
        tau_idx = self._discretize_tau(tau)  # (B, T)
        d_idx = self._discretize_d(d)  # (B, T)

        # Embed
        tau_emb = self.tau_embed(tau_idx)  # (B, T, embed_dim//2)
        d_emb = self.d_embed(d_idx)  # (B, T, embed_dim//2)

        # Concatenate and project
        signal_emb = torch.cat([tau_emb, d_emb], dim=-1)  # (B, T, embed_dim)
        signal_emb = self.signal_proj(signal_emb)  # (B, T, embed_dim)

        # Add token dimension
        return signal_emb.unsqueeze(2)  # (B, T, 1, embed_dim)

    def forward(
        self,
        z_corrupted: Tensor,
        actions: Tensor,
        tau: Tensor,
        d: Tensor,
    ) -> Tensor:
        """
        Predict clean representations from corrupted inputs.

        Args:
            z_corrupted: (B, T, N_z, latent_dim) corrupted latent representations
            actions: (B, T) discrete action indices
            tau: (B, T) signal levels ∈ [0, 1]
            d: (B, T) step sizes

        Returns:
            z_pred: (B, T, N_z, latent_dim) predicted clean representations
        """
        B, T, N_z, D = z_corrupted.shape

        # Project corrupted z to embed_dim: (B, T, N_z, embed_dim)
        z_emb = self.z_proj(z_corrupted)

        # Encode actions: (B, T, num_action_tokens, embed_dim)
        action_emb = self.action_encoder(actions)

        # Encode signal (τ, d): (B, T, 1, embed_dim)
        signal_emb = self._encode_signal(tau, d)

        # Expand register tokens: (B, T, num_register_tokens, embed_dim)
        register_emb = self.register_tokens.expand(B, T, -1, -1)

        # Interleave tokens per timestep: [action, signal, z, registers]
        # Total tokens per timestep: num_action_tokens + 1 + N_z + num_register_tokens
        x = torch.cat([action_emb, signal_emb, z_emb, register_emb], dim=2)
        # x: (B, T, N_total, embed_dim)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Extract z tokens for output prediction
        # z tokens are at positions [num_action_tokens + 1 : num_action_tokens + 1 + N_z]
        z_start = self.num_action_tokens + 1
        z_end = z_start + N_z
        z_out = x[:, :, z_start:z_end, :]  # (B, T, N_z, embed_dim)

        # Project to latent space
        z_out = self.output_norm(z_out)
        z_pred = self.output_proj(z_out)  # (B, T, N_z, latent_dim)

        return z_pred

    @torch.no_grad()
    def sample(
        self,
        z_context: Tensor,
        actions: Tensor,
        num_steps: int = 4,
        context_noise: float = 0.1,
    ) -> Tensor:
        """
        Autoregressively sample future representations.

        Args:
            z_context: (B, T_ctx, N_z, latent_dim) context representations (clean)
            actions: (B, T_gen) future action indices to condition on
            num_steps: Number of denoising steps K per frame
            context_noise: Noise level for context (τ_ctx = 0.1 from paper)

        Returns:
            z_generated: (B, T_gen, N_z, latent_dim) generated representations
        """
        B, T_ctx, N_z, D = z_context.shape
        T_gen = actions.shape[1]
        device = z_context.device

        # Step size for sampling
        step_size = 1.0 / num_steps

        # Add slight noise to context for robustness
        if context_noise > 0:
            noise = torch.randn_like(z_context)
            z_context = (1 - context_noise) * z_context + context_noise * noise

        # Generate frame by frame
        z_generated = []
        z_history = z_context  # Start with context

        for t in range(T_gen):
            # Current action
            action_t = actions[:, t:t+1]  # (B, 1)

            # Initialize from pure noise
            z_t = torch.randn(B, 1, N_z, D, device=device)

            # Denoise with K steps
            for k in range(num_steps):
                tau_k = k * step_size  # Signal level at this step
                tau = torch.full((B, 1), tau_k, device=device)
                d = torch.full((B, 1), step_size, device=device)

                # Prepare context (add slight noise to past generated frames)
                if z_history.shape[1] > T_ctx:
                    # Only use recent context
                    z_ctx_use = z_history[:, -T_ctx:, :, :]
                else:
                    z_ctx_use = z_history

                # Set context noise level
                T_ctx_use = z_ctx_use.shape[1]
                tau_ctx = torch.full((B, T_ctx_use), context_noise, device=device)
                d_ctx = torch.full((B, T_ctx_use), step_size, device=device)

                # Combine context and current frame
                z_full = torch.cat([z_ctx_use, z_t], dim=1)  # (B, T_ctx+1, N_z, D)
                tau_full = torch.cat([tau_ctx, tau], dim=1)  # (B, T_ctx+1)
                d_full = torch.cat([d_ctx, d], dim=1)  # (B, T_ctx+1)

                # Get context actions (use dummy value or repeat)
                action_ctx = torch.zeros(B, T_ctx_use, dtype=torch.long, device=device)
                action_full = torch.cat([action_ctx, action_t], dim=1)  # (B, T_ctx+1)

                # Predict clean z
                z_pred = self(z_full, action_full, tau_full, d_full)

                # Extract prediction for current frame
                z_t = z_pred[:, -1:, :, :]  # (B, 1, N_z, D)

                # Update z_t for next denoising step
                # For shortcut model, prediction is already clean so we use it directly
                # For iterative refinement, we would interpolate

            # Append generated frame
            z_generated.append(z_t)

            # Update history
            z_history = torch.cat([z_history, z_t], dim=1)

        # Concatenate all generated frames
        z_generated = torch.cat(z_generated, dim=1)  # (B, T_gen, N_z, D)

        return z_generated

    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())
