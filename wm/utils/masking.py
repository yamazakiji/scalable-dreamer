"""Masking strategies for masked autoencoder training."""
import torch
import torch.nn.functional as F


def random_mask(frames: torch.Tensor, mask_ratio: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply random masking to frames.

    Args:
        frames: [B, T, C, H, W] input frames
        mask_ratio: Ratio of pixels to mask (0.0 to 1.0)

    Returns:
        masked_frames: Frames with random pixels masked (set to 0.5)
        mask: Binary mask [B, T, 1, H, W] where 1 = visible, 0 = masked
    """
    B, T, C, H, W = frames.shape

    # Generate random mask
    mask = torch.rand(B, T, 1, H, W, device=frames.device) > mask_ratio

    # Apply mask (set masked pixels to gray 0.5)
    masked_frames = frames * mask + 0.5 * (~mask)

    return masked_frames, mask.float()


def block_mask(frames: torch.Tensor, mask_ratio: float = 0.5, block_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply block/patch-based masking to frames.

    Args:
        frames: [B, T, C, H, W] input frames
        mask_ratio: Ratio of blocks to mask (0.0 to 1.0)
        block_size: Size of square blocks to mask

    Returns:
        masked_frames: Frames with random blocks masked (set to 0.5)
        mask: Binary mask [B, T, 1, H, W] where 1 = visible, 0 = masked
    """
    B, T, C, H, W = frames.shape

    # Calculate number of blocks
    num_blocks_h = H // block_size
    num_blocks_w = W // block_size

    # Generate random block mask
    block_mask = torch.rand(B, T, 1, num_blocks_h, num_blocks_w, device=frames.device) > mask_ratio

    # Upsample to full resolution using nearest neighbor
    mask = F.interpolate(
        block_mask.view(B * T, 1, num_blocks_h, num_blocks_w),
        size=(H, W),
        mode='nearest'
    )
    mask = mask.view(B, T, 1, H, W)

    # Apply mask
    masked_frames = frames * mask + 0.5 * (~mask)

    return masked_frames, mask.float()


def grid_mask(frames: torch.Tensor, mask_ratio: float = 0.5, grid_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply checkerboard/grid masking to frames.

    Args:
        frames: [B, T, C, H, W] input frames
        mask_ratio: Approximate ratio of pixels to mask (0.0 to 1.0)
        grid_size: Size of grid cells

    Returns:
        masked_frames: Frames with grid pattern masked (set to 0.5)
        mask: Binary mask [B, T, 1, H, W] where 1 = visible, 0 = masked
    """
    B, T, C, H, W = frames.shape

    # Create checkerboard pattern
    mask = torch.zeros(1, 1, 1, H, W, device=frames.device)

    # Determine which grid cells to keep based on mask_ratio
    # For 50% masking, use standard checkerboard
    # For 75% masking, mask more aggressively
    if mask_ratio <= 0.5:
        # Checkerboard pattern - keep every other block
        for i in range(0, H, grid_size):
            for j in range(0, W, grid_size):
                if (i // grid_size + j // grid_size) % 2 == 0:
                    mask[:, :, :, i:i+grid_size, j:j+grid_size] = 1
    else:
        # For higher mask ratios, keep fewer blocks
        keep_ratio = 1 - mask_ratio
        num_blocks_h = H // grid_size
        num_blocks_w = W // grid_size
        total_blocks = num_blocks_h * num_blocks_w
        num_keep_blocks = int(total_blocks * keep_ratio)

        # Randomly select which blocks to keep
        all_blocks = [(i, j) for i in range(num_blocks_h) for j in range(num_blocks_w)]
        keep_blocks = torch.randperm(total_blocks)[:num_keep_blocks]

        for idx in keep_blocks:
            i, j = all_blocks[idx]
            mask[:, :, :, i*grid_size:(i+1)*grid_size, j*grid_size:(j+1)*grid_size] = 1

    # Broadcast to batch size
    mask = mask.expand(B, T, -1, -1, -1)

    # Apply mask
    masked_frames = frames * mask + 0.5 * (~mask)

    return masked_frames, mask.float()


def apply_masking_strategy(
    frames: torch.Tensor,
    strategy: str,
    mask_ratio: float = 0.5,
    block_size: int = 16,
    grid_size: int = 16
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply specified masking strategy to frames.

    Args:
        frames: [B, T, C, H, W] input frames
        strategy: One of 'none', 'random', 'block', 'grid'
        mask_ratio: Ratio of content to mask
        block_size: Size for block masking
        grid_size: Size for grid masking

    Returns:
        masked_frames: Masked frames
        mask: Binary mask
    """
    if strategy == 'none':
        mask = torch.ones_like(frames[:, :, :1])
        return frames, mask
    elif strategy == 'random':
        return random_mask(frames, mask_ratio)
    elif strategy == 'block':
        return block_mask(frames, mask_ratio, block_size)
    elif strategy == 'grid':
        return grid_mask(frames, mask_ratio, grid_size)
    else:
        raise ValueError(f"Unknown masking strategy: {strategy}")
