"""Visualize latent representations from dynamics model evaluation.

Decodes latent tensors back to images using CausalTokenizer and generates
detailed visualizations of denoising progression.

Usage:
    python visualize_latents.py \
        --eval-folder outputs/dreamer4_dynamics/eval/eval_step_1000/batch_0 \
        --tokenizer-checkpoint outputs/dreamer4_tokenizer/checkpoint_step_30000.pt \
        --fsdp
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from wm.models.tokenizer.causal_tokenizer import CausalTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize latent representations")
    parser.add_argument(
        "--eval-folder",
        type=str,
        required=True,
        help="Path to eval batch folder containing frame_* directories",
    )
    parser.add_argument(
        "--tokenizer-checkpoint",
        type=str,
        required=True,
        help="Path to tokenizer .pt checkpoint file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: {eval_folder}/visualizations)",
    )
    parser.add_argument(
        "--fsdp",
        action="store_true",
        help="Enable FSDP for memory-efficient model loading",
    )
    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Skip GIF generation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    return parser.parse_args()


def load_tokenizer(checkpoint_path: str, device: str, use_fsdp: bool = False) -> CausalTokenizer:
    """Load CausalTokenizer from checkpoint."""
    model = CausalTokenizer().to(device)

    # Load checkpoint and strip DDP/FSDP prefix if present
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    if use_fsdp:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://", world_size=1, rank=0)
        from wm.training.fsdp import apply_fsdp
        model = apply_fsdp(model, mixed_precision="bf16")

    model.eval()
    return model


@torch.no_grad()
def decode_latent(tokenizer: CausalTokenizer, z: torch.Tensor, device: str) -> np.ndarray:
    """Decode a latent tensor to an image array.

    Args:
        tokenizer: CausalTokenizer model
        z: Latent tensor of shape (1, 1, num_latents, latent_dim) or (1, 1, 128, 128)
        device: Device to use

    Returns:
        Image as numpy array of shape (H, W, 3) with uint8 dtype
    """
    z = z.to(device=device, dtype=torch.float32)

    with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16):
        recon = tokenizer.decode(z)  # (1, 1, 3, H, W)

    recon = recon.squeeze()  # (3, H, W)
    recon = torch.clamp(recon, 0, 1)
    return (recon.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()


def discover_frames(eval_folder: Path) -> list[Path]:
    """Find all frame_* directories sorted by frame number."""
    frame_dirs = sorted(
        eval_folder.glob("frame_*"),
        key=lambda p: int(re.search(r"frame_(\d+)", p.name).group(1)),
    )
    return frame_dirs


def discover_k_values(frame_dir: Path) -> list[int]:
    """Find all K* subdirectories and return K values sorted."""
    k_dirs = list(frame_dir.glob("K*"))
    k_values = sorted(int(d.name[1:]) for d in k_dirs if d.is_dir())
    return k_values


def discover_steps(k_dir: Path) -> list[int]:
    """Find all z_pred_step_*.pt files and return step numbers sorted."""
    step_files = list(k_dir.glob("z_pred_step_*.pt"))
    steps = sorted(int(re.search(r"z_pred_step_(\d+)", f.name).group(1)) for f in step_files)
    return steps


def create_side_by_side(
    target_img: np.ndarray,
    predicted_imgs: dict[int, np.ndarray],
    k_values: list[int],
) -> Image.Image:
    """Create horizontal strip: Target | K4 | K8 | K30.

    Args:
        target_img: Target image array (H, W, 3)
        predicted_imgs: Dict mapping K value to final predicted image
        k_values: List of K values to include

    Returns:
        PIL Image with side-by-side comparison
    """
    h, w = target_img.shape[:2]
    n_images = 1 + len(k_values)
    padding = 10
    label_height = 30

    total_width = n_images * w + (n_images - 1) * padding
    total_height = h + label_height

    canvas = Image.new("RGB", (total_width, total_height), (255, 255, 255))

    # Paste target
    x_offset = 0
    canvas.paste(Image.fromarray(target_img), (x_offset, label_height))

    # Add label
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    draw.text((x_offset + w // 2 - 20, 5), "Target", fill=(0, 0, 0))

    # Paste predictions
    for k in k_values:
        x_offset += w + padding
        if k in predicted_imgs:
            canvas.paste(Image.fromarray(predicted_imgs[k]), (x_offset, label_height))
            draw.text((x_offset + w // 2 - 10, 5), f"K{k}", fill=(0, 0, 0))

    return canvas


def create_denoising_grid(
    step_images: list[np.ndarray],
    k_value: int,
    cols: int = 4,
) -> Image.Image:
    """Create a grid showing denoising progression.

    Args:
        step_images: List of images for each denoising step
        k_value: K value for labeling
        cols: Number of columns in grid

    Returns:
        PIL Image with grid layout
    """
    if not step_images:
        return Image.new("RGB", (100, 100), (255, 255, 255))

    h, w = step_images[0].shape[:2]
    n_steps = len(step_images)
    rows = (n_steps + cols - 1) // cols
    padding = 5
    label_height = 20

    total_width = cols * w + (cols - 1) * padding
    total_height = rows * (h + label_height) + (rows - 1) * padding

    canvas = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for i, img in enumerate(step_images):
        row = i // cols
        col = i % cols
        x = col * (w + padding)
        y = row * (h + label_height + padding)

        canvas.paste(Image.fromarray(img), (x, y + label_height))
        draw.text((x + 5, y + 2), f"Step {i}", fill=(0, 0, 0))

    return canvas


def create_denoising_gif(
    step_images: list[np.ndarray],
    output_path: Path,
    duration: int = 200,
) -> None:
    """Create animated GIF of denoising progression.

    Args:
        step_images: List of images for each denoising step
        output_path: Path to save GIF
        duration: Duration per frame in milliseconds
    """
    if not step_images:
        return

    frames = [Image.fromarray(img) for img in step_images]
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )


def create_summary_grid(
    all_frame_data: dict[int, dict],
    k_values: list[int],
) -> Image.Image:
    """Create summary grid: rows=frames, cols=Target+K values.

    Args:
        all_frame_data: Dict mapping frame_idx -> {target, predictions}
        k_values: List of K values

    Returns:
        PIL Image with summary grid
    """
    if not all_frame_data:
        return Image.new("RGB", (100, 100), (255, 255, 255))

    first_frame = next(iter(all_frame_data.values()))
    h, w = first_frame["target"].shape[:2]
    n_frames = len(all_frame_data)
    n_cols = 1 + len(k_values)  # Target + K values

    padding = 5
    label_width = 60
    label_height = 25

    total_width = label_width + n_cols * w + (n_cols - 1) * padding
    total_height = label_height + n_frames * h + (n_frames - 1) * padding

    canvas = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Column headers
    x = label_width
    draw.text((x + w // 2 - 20, 5), "Target", fill=(0, 0, 0))
    for k in k_values:
        x += w + padding
        draw.text((x + w // 2 - 10, 5), f"K{k}", fill=(0, 0, 0))

    # Rows
    for i, (frame_idx, data) in enumerate(sorted(all_frame_data.items())):
        y = label_height + i * (h + padding)

        # Row label
        draw.text((5, y + h // 2 - 5), f"Frame {frame_idx}", fill=(0, 0, 0))

        # Target
        x = label_width
        canvas.paste(Image.fromarray(data["target"]), (x, y))

        # Predictions
        for k in k_values:
            x += w + padding
            if k in data["predictions"]:
                canvas.paste(Image.fromarray(data["predictions"][k]), (x, y))

    return canvas


# Import ImageDraw at module level for label functions
from PIL import ImageDraw


def main():
    args = parse_args()

    eval_folder = Path(args.eval_folder)
    if not eval_folder.exists():
        raise FileNotFoundError(f"Eval folder not found: {eval_folder}")

    output_dir = Path(args.output_dir) if args.output_dir else eval_folder / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from {args.tokenizer_checkpoint}")
    tokenizer = load_tokenizer(args.tokenizer_checkpoint, args.device, args.fsdp)

    # Discover frames
    frame_dirs = discover_frames(eval_folder)
    if not frame_dirs:
        raise ValueError(f"No frame_* directories found in {eval_folder}")

    print(f"Found {len(frame_dirs)} frames")

    # Get K values from first frame
    k_values = discover_k_values(frame_dirs[0])
    print(f"K values: {k_values}")

    # Store data for summary
    all_frame_data = {}

    for frame_dir in frame_dirs:
        frame_idx = int(re.search(r"frame_(\d+)", frame_dir.name).group(1))
        print(f"\nProcessing frame {frame_idx}...")

        frame_output_dir = output_dir / frame_dir.name
        frame_output_dir.mkdir(parents=True, exist_ok=True)

        # Load and decode target
        target_path = frame_dir / "z_target.pt"
        if not target_path.exists():
            print(f"  Warning: z_target.pt not found, skipping frame")
            continue

        z_target = torch.load(target_path, map_location="cpu", weights_only=True)
        target_img = decode_latent(tokenizer, z_target, args.device)

        frame_data = {"target": target_img, "predictions": {}}

        for k in k_values:
            k_dir = frame_dir / f"K{k}"
            if not k_dir.exists():
                print(f"  Warning: K{k} directory not found")
                continue

            steps = discover_steps(k_dir)
            if not steps:
                print(f"  Warning: No step files found in K{k}")
                continue

            print(f"  Processing K{k} with {len(steps)} steps")

            # Decode all steps
            step_images = []
            for step in steps:
                z_path = k_dir / f"z_pred_step_{step}.pt"
                z_pred = torch.load(z_path, map_location="cpu", weights_only=True)
                step_img = decode_latent(tokenizer, z_pred, args.device)
                step_images.append(step_img)

            # Store final prediction
            if step_images:
                frame_data["predictions"][k] = step_images[-1]

            # Create denoising grid
            grid = create_denoising_grid(step_images, k)
            grid.save(frame_output_dir / f"denoising_K{k}_grid.png")

            # Create GIF if requested
            if not args.no_gif and step_images:
                create_denoising_gif(
                    step_images,
                    frame_output_dir / f"denoising_K{k}.gif",
                )

        # Create side-by-side comparison
        if frame_data["predictions"]:
            comparison = create_side_by_side(target_img, frame_data["predictions"], k_values)
            comparison.save(frame_output_dir / "target_vs_predicted.png")

        all_frame_data[frame_idx] = frame_data

    # Create summary grid
    if all_frame_data:
        summary_dir = output_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)

        summary = create_summary_grid(all_frame_data, k_values)
        summary.save(summary_dir / "all_frames_comparison.png")

    print(f"\nVisualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
