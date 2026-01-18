import torch

from torch import nn
from torch.utils.checkpoint import checkpoint
from einops.layers.torch import Rearrange

from wm.models.transformer.block import TransformerBlock, SpaceAttention


class CausalTokenizer(nn.Module):
    """
    Causal Tokenizer from Dreamer 4 (arXiv:2509.24527).

    Compresses video frames into continuous latent representations using masked
    autoencoding. The encoder processes patch tokens + latent tokens, and the
    latent representations are squeezed through a low-dimensional bottleneck
    with tanh activation. The decoder reconstructs patches from the latents.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=512,
        num_heads=8,
        num_latents=128,
        latent_dim=128,
        gradient_checkpointing=False,
    ):
        super().__init__()

        self.gradient_checkpointing = gradient_checkpointing
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.num_latents = num_latents
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        # Patch embedding: (B, T, C, H, W) -> (B, T, N, E)
        self.patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(self.patch_dim, embed_dim)
        )

        # Mask token for masked patches in encoder
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, embed_dim))

        # Learnable latent tokens prepended to encoder input
        self.latent_tokens = nn.Parameter(torch.randn(1, 1, num_latents, embed_dim))

        # Decoder tokens for reading out patch reconstructions
        self.decoder_tokens = nn.Parameter(torch.randn(1, 1, self.num_patches, embed_dim))

        # Encoder: process patches + latent tokens
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, "space") for _ in range(3)
        ])
        self.encoder_blocks.append(TransformerBlock(embed_dim, num_heads, "time"))

        # Projection to bottleneck latent representation with tanh
        self.to_latent = nn.Sequential(
            nn.Linear(embed_dim, latent_dim),
            nn.Tanh()
        )

        # Projection from bottleneck back to model dimension
        self.from_latent = nn.Linear(latent_dim, embed_dim)

        # Decoder: reconstruct patches from latents + decoder tokens
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, "space") for _ in range(3)
        ])
        self.decoder_blocks.append(TransformerBlock(embed_dim, num_heads, "time"))

        # Output projection for patch reconstruction
        self.to_pixels = nn.Sequential(
            nn.Linear(embed_dim, self.patch_dim),
            Rearrange("b t (h w) (p1 p2 c) -> b t c (h p1) (w p2)", h=int(img_size / patch_size), p1=patch_size, p2=patch_size)
        )

    def _run_block(self, block, x, attn_mask=None):
        """Run a transformer block with optional gradient checkpointing."""
        if self.gradient_checkpointing and self.training:
            return checkpoint(
                block,
                x,
                attn_mask,
                # use_reentrant=False,
            )
        else:
            return block(x, attn_mask=attn_mask)

    def random_masking(self, patches):
        """
        Perform random masking by replacing patches with mask token.
        Mask ratio is sampled from U(0, 0.9) per image as per Dreamer 4 paper.

        Args:
            patches: [B, T, N, E] patch embeddings
        Returns:
            patches_masked: patches with masked positions replaced by mask_token
            mask: binary mask where 1 = visible, 0 = masked
        """
        B, T, N, E = patches.shape
        device = patches.device

        # Sample mask ratio from U(0, 0.9) for each sample in the batch
        # This ensures the tokenizer sometimes trains on p=0 case used during inference
        mask_ratios = torch.rand(B, device=device) * 0.9  # [B]

        # Generate random noise for shuffling
        # Paper states U(0, 0.9) is randomized across images
        noise = torch.rand(B, T, N, device=device)

        # Sort noise to get shuffle indices
        ids_shuffle = torch.argsort(noise, dim=2)
        ids_unshuffle = torch.argsort(ids_shuffle, dim=2)

        # Create binary mask: 1 is keep, 0 is remove
        # Each sample has different len_keep based on its mask_ratio
        len_keep = (N * (1 - mask_ratios)).int()  # [B]

        # Create mask per sample
        indices = torch.arange(N, device=device).expand(B, T, N)
        mask = (indices < len_keep.view(B, 1, 1)).float()

        # Unshuffle to get original order
        mask = torch.gather(mask, dim=2, index=ids_unshuffle)

        # Replace masked patches with mask token
        mask_tokens = self.mask_token.expand(B, T, N, E)
        patches_masked = patches * mask.unsqueeze(-1) + mask_tokens * (1 - mask.unsqueeze(-1))

        return patches_masked, mask

    def _create_latent_encoder_mask(self, device):
        """
        Create latent encoder attention mask where:
        - Latent tokens (first num_latents) can attend to latent tokens and other modalities
        - Pixel tokens (remaining num_patches) can attend within itself

        Returns:
            mask: Boolean tensor of shape (num_latents + num_patches, num_latents + num_patches)
                  True = can attend, False = cannot attend
        """
        total_tokens = self.num_latents + self.num_patches

        # Start with all True (all can attend to all)
        mask = torch.ones(total_tokens, total_tokens, dtype=torch.bool, device=device)

        # Latent tokens cannot attend to pixel tokens
        # mask[i, j] = False means position i cannot attend to position j
        mask[self.num_latents:, :self.num_latents] = False

        return mask
    
    def _create_latent_decoder_mask(self, device):
        """
        Create latent decoder attention mask where:
        - Latents attend only within themselves
        - Pixel tokens can attend to any modalities

        Returns:
            mask: Boolean tensor of shape (num_latents + num_patches, num_latents + num_patches)
                  True = can attend, False = cannot attend
        """
        total_tokens = self.num_latents + self.num_patches

        mask = torch.ones(total_tokens, total_tokens, dtype=torch.bool, device=device)

        mask[:self.num_latents, self.num_latents:] = False

        return mask

    def encode(self, x, apply_masking=True):
        """
        Encode video frames to latent representation.

        Args:
            x: Input video tensor of shape (B, T, C, H, W)
            apply_masking: Whether to apply random masking (True for training, False for inference)
        Returns:
            z: Latent representation of shape (B, T, num_latents, latent_dim)
            mask: Binary mask indicating visible patches (or None if no masking)
        """
        B, T = x.shape[0], x.shape[1]

        # Patch embedding: (B, T, C, H, W) -> (B, T, N, E)
        patches = self.patch_embed(x)

        # Random masking during training
        if apply_masking:
            patches_masked, mask = self.random_masking(patches)
        else:
            patches_masked = patches
            mask = None

        # Prepend learnable latent tokens: (B, T, num_latents, E)
        latent_tokens = self.latent_tokens.expand(B, T, -1, -1)

        # Concatenate latent tokens with patches: (B, T, num_latents + N, E)
        encoder_input = torch.cat([latent_tokens, patches_masked], dim=2)

        # Create attention mask for space attention blocks
        attn_mask = self._create_latent_encoder_mask(x.device)

        # Encode all tokens through encoder blocks
        encoded = encoder_input
        for block in self.encoder_blocks:
            if isinstance(block.attention, SpaceAttention):
                encoded = self._run_block(block, encoded, attn_mask)
            else:
                encoded = self._run_block(block, encoded)

        # Extract latent tokens only (first num_latents tokens)
        latent_encoded = encoded[:, :, :self.num_latents, :]

        # Project to bottleneck with tanh: (B, T, num_latents, latent_dim)
        z = self.to_latent(latent_encoded)

        return z, mask

    def decode(self, z):
        """
        Decode latent representation to reconstructed patches.

        Args:
            z: Latent representation of shape (B, T, num_latents, latent_dim)
        Returns:
            recon: Reconstructed patches of shape (B, T, N, patch_dim)
        """
        B, T = z.shape[0], z.shape[1]

        # Project latents back to model dimension: (B, T, num_latents, E)
        latents = self.from_latent(z)

        # Expand decoder tokens: (B, T, N, E)
        decoder_tokens = self.decoder_tokens.expand(B, T, -1, -1)

        # Concatenate latents with decoder tokens: (B, T, num_latents + N, E)
        decoder_input = torch.cat([latents, decoder_tokens], dim=2)

        # Create attention mask for space attention blocks
        attn_mask = self._create_latent_decoder_mask(z.device)

        # Decode through decoder blocks
        decoded = decoder_input
        for block in self.decoder_blocks:
            if isinstance(block.attention, SpaceAttention):
                decoded = self._run_block(block, decoded, attn_mask)
            else:
                decoded = self._run_block(block, decoded)

        # Extract patch reconstructions (last N tokens)
        patch_decoded = decoded[:, :, self.num_latents:, :]

        # Project to pixel space
        recon = self.to_pixels(patch_decoded)

        return recon

    def forward(self, x, apply_masking=True):
        """
        Forward pass for MAE-based tokenizer (training).

        Args:
            x: Input video tensor of shape (B, T, C, H, W)
            apply_masking: Whether to apply random masking (True for training)
        Returns:
            z: Latent representation of shape (B, T, num_latents, latent_dim)
            recon: Reconstructed patches of shape (B, T, N, patch_dim)
            mask: Binary mask indicating visible patches
            patches: Original patch embeddings for computing loss
        """
        # Get original patches for loss computation
        patches = self.patch_embed(x)

        # Encode with masking
        z, mask = self.encode(x, apply_masking=apply_masking)

        # Decode to reconstruct patches
        recon = self.decode(z)

        return z, recon, mask, patches

    def tokenize(self, x):
        """
        Tokenize video frames for downstream use (inference mode, no masking).

        Args:
            x: Input video tensor of shape (B, T, C, H, W)
        Returns:
            z: Latent representation of shape (B, T, num_latents, latent_dim)
        """
        z, _ = self.encode(x, apply_masking=False)
        return z
