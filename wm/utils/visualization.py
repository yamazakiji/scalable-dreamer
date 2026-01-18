"""Visualization utilities for reconstruction tasks."""
import torch
import wandb
import numpy as np


def create_reconstruction_grid(
    original: torch.Tensor,
    masked: torch.Tensor,
    reconstructed: torch.Tensor,
    num_samples: int = 4
) -> np.ndarray:
    """
    Create a grid showing original, masked, and reconstructed images.

    Args:
        original: [B, T, C, H, W] original frames
        masked: [B, T, C, H, W] masked frames
        reconstructed: [B, T, C, H, W] reconstructed frames
        num_samples: Number of samples to include in grid

    Returns:
        Grid image as numpy array [H, W, C]
    """
    B, T, C, H, W = original.shape

    # Select first frame from sequence and first num_samples from batch
    num_samples = min(num_samples, B)
    original = original[:num_samples, 0]  # [N, C, H, W]
    masked = masked[:num_samples, 0]
    reconstructed = reconstructed[:num_samples, 0]

    # Stack horizontally: original | masked | reconstructed
    images = torch.cat([original, masked, reconstructed], dim=0)  # [N*3, C, H, W]

    # Clamp to valid range
    images = torch.clamp(images, 0, 1)

    # Convert to numpy and rearrange
    images = images.cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))  # [N*3, H, W, C]

    # Create grid: rows are samples, columns are [original, masked, reconstructed]
    rows = []
    for i in range(num_samples):
        row = np.concatenate([
            images[i],                      # original
            images[i + num_samples],        # masked
            images[i + 2 * num_samples]     # reconstructed
        ], axis=1)
        rows.append(row)

    grid = np.concatenate(rows, axis=0)

    return grid


def log_reconstruction_comparison(
    logger,
    original: torch.Tensor,
    masked_dict: dict,
    reconstructed_dict: dict,
    step: int,
    num_samples: int = 4
):
    """
    Log reconstruction comparisons for multiple masking strategies to wandb.

    Args:
        logger: WandbLogger instance
        original: [B, T, C, H, W] original frames
        masked_dict: Dictionary mapping strategy name to masked frames
        reconstructed_dict: Dictionary mapping strategy name to reconstructed frames
        step: Current training step
        num_samples: Number of samples per strategy
    """
    if not logger or not logger.enabled:
        return

    images_to_log = {}

    for strategy in masked_dict.keys():
        grid = create_reconstruction_grid(
            original,
            masked_dict[strategy],
            reconstructed_dict[strategy],
            num_samples=num_samples
        )

        # Convert to wandb Image
        images_to_log[f'reconstruction/{strategy}'] = wandb.Image(
            grid,
            caption=f'{strategy} reconstruction'
        )

    # Log all images at once
    logger.log(images_to_log, step=step)


def create_comparison_figure(
    frames_dict: dict,
    titles: list[str],
    num_samples: int = 4
) -> np.ndarray:
    """
    Create a comparison figure with multiple columns.

    Args:
        frames_dict: Dictionary mapping column names to frame tensors [B, T, C, H, W]
        titles: List of titles for each column
        num_samples: Number of rows in the figure

    Returns:
        Figure as numpy array [H, W, C]
    """
    # Get first batch
    first_key = list(frames_dict.keys())[0]
    B = frames_dict[first_key].shape[0]
    num_samples = min(num_samples, B)

    columns = []
    for key in frames_dict.keys():
        frames = frames_dict[key][:num_samples, 0]  # [N, C, H, W]
        frames = torch.clamp(frames, 0, 1)
        frames = frames.cpu().numpy()
        frames = np.transpose(frames, (0, 2, 3, 1))  # [N, H, W, C]
        columns.append(frames)

    # Stack columns horizontally for each row
    rows = []
    for i in range(num_samples):
        row_images = [col[i] for col in columns]
        row = np.concatenate(row_images, axis=1)
        rows.append(row)

    # Stack rows vertically
    figure = np.concatenate(rows, axis=0)

    return figure
