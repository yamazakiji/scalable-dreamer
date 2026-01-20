"""Test script for CausalTokenizer with FSDP inference."""

import argparse
import os
from collections import OrderedDict
from pathlib import Path

import torch
import torch.distributed as dist
from torchvision.utils import save_image

from wm.models.tokenizer.causal_tokenizer import CausalTokenizer
from wm.data.video_dataset import VideoDataset
from wm.training.distributed import setup_distributed, cleanup_distributed
from wm.training.fsdp import apply_fsdp, get_memory_stats


def load_checkpoint(checkpoint_path: str, device: torch.device) -> dict:
    """Load checkpoint and strip DDP 'module.' prefix if present."""
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)["model_state_dict"]

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v

    return new_state_dict


def save_frames(
    frames: torch.Tensor,
    output_dir: Path,
    prefix: str,
    num_frames: int | None = None,
    step: int = 1,
):
    """
    Save video frames as individual images.

    Args:
        frames: Tensor of shape [T, C, H, W] with values in [0, 1]
        output_dir: Directory to save frames
        prefix: Filename prefix (e.g., 'original' or 'reconstructed')
        num_frames: Maximum number of frames to save (None = all)
        step: Save every nth frame
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_to_save = frames[::step]
    if num_frames is not None:
        frames_to_save = frames_to_save[:num_frames]

    for i, frame in enumerate(frames_to_save):
        frame_idx = i * step
        save_image(frame, output_dir / f"{prefix}_frame_{frame_idx:04d}.png")

    print(f"Saved {len(frames_to_save)} {prefix} frames to {output_dir}")


def save_comparison_grid(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    output_path: Path,
    num_frames: int = 8,
    step: int | None = None,
):
    """
    Save a side-by-side comparison grid of original and reconstructed frames.

    Args:
        original: Tensor of shape [T, C, H, W]
        reconstructed: Tensor of shape [T, C, H, W]
        output_path: Path to save the grid image
        num_frames: Number of frames to include
        step: Frame step (auto-calculated if None)
    """
    T = original.shape[0]
    if step is None:
        step = max(1, T // num_frames)

    indices = list(range(0, T, step))[:num_frames]

    orig_frames = original[indices]
    recon_frames = reconstructed[indices]

    # Interleave original and reconstructed for comparison
    comparison = torch.stack([orig_frames, recon_frames], dim=1).reshape(-1, *orig_frames.shape[1:])

    save_image(comparison, output_path, nrow=2, padding=2, pad_value=1.0)
    print(f"Saved comparison grid to {output_path}")


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    video: torch.Tensor,
    device: torch.device,
    apply_masking: bool = False,
) -> torch.Tensor:
    """Run model inference on video tensor."""
    video = video.unsqueeze(0).to(device)  # Add batch dimension

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, reconstruction, _, _ = model(video, apply_masking=apply_masking)

    return reconstruction[0].float().cpu()  # Remove batch dim and move to CPU


def main():
    parser = argparse.ArgumentParser(description="Test CausalTokenizer with FSDP")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./outputs/dreamer4_tokenizer/checkpoint_step_30000.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Path to video data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_outputs",
        help="Directory to save output frames",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="Index of video sample to use from dataset",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames to save",
    )
    parser.add_argument(
        "--fsdp",
        action="store_true",
        help="Use FSDP sharding for inference",
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Save all frames (not just sampled)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup distributed if using FSDP
    if args.fsdp:
        rank, world_size = setup_distributed()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        is_main = rank == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True
        rank = 0

    if is_main:
        print(f"Device: {device}")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"FSDP: {args.fsdp}")

    # Create model
    model = CausalTokenizer().to(device)

    if is_main:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    # Load checkpoint (before FSDP wrapping)
    if is_main:
        print("Loading checkpoint...")
    state_dict = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(state_dict)

    # Apply FSDP if requested
    if args.fsdp:
        if is_main:
            print("Applying FSDP sharding...")
        model = apply_fsdp(model, mixed_precision="bf16")
        if is_main:
            mem_stats = get_memory_stats(local_rank)
            print(f"GPU memory after FSDP wrap: {mem_stats['memory_allocated_gb']:.2f} GB")

    model.eval()

    # Load dataset and get sample
    if is_main:
        print(f"Loading video sample {args.sample_idx} from {args.data_dir}...")
    dataset = VideoDataset(video_dir=args.data_dir)
    video, _ = dataset[args.sample_idx]

    if is_main:
        print(f"Video shape: {video.shape}")  # [T, C, H, W]

    # Run inference
    if is_main:
        print("Running inference...")

    torch.cuda.reset_peak_memory_stats()
    reconstruction = run_inference(model, video, device, apply_masking=False)

    if is_main:
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak GPU memory during inference: {peak_mem:.2f} GB")
        print(f"Reconstruction shape: {reconstruction.shape}")

    # Save outputs (only on main rank)
    if is_main:
        # Clamp values to valid range
        video_clamped = video.clamp(0, 1)
        reconstruction_clamped = reconstruction.clamp(0, 1)

        # Calculate step for frame sampling
        T = video.shape[0]
        step = max(1, T // args.num_frames) if not args.save_all else 1
        num_to_save = T if args.save_all else args.num_frames

        # Save individual frames
        save_frames(
            video_clamped,
            output_dir / "original",
            "original",
            num_frames=num_to_save,
            step=step if not args.save_all else 1,
        )
        save_frames(
            reconstruction_clamped,
            output_dir / "reconstructed",
            "reconstructed",
            num_frames=num_to_save,
            step=step if not args.save_all else 1,
        )

        # Save comparison grid
        save_comparison_grid(
            video_clamped,
            reconstruction_clamped,
            output_dir / "comparison_grid.png",
            num_frames=min(8, args.num_frames),
        )

        print(f"\nAll outputs saved to {output_dir}")

    # Cleanup
    if args.fsdp:
        cleanup_distributed()


if __name__ == "__main__":
    main()
