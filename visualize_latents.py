"""Visualize world models pipeline: encode -> dynamics -> decode.

Demonstrates the full pipeline comparing teacher forcing vs autoregressive
generation modes, producing GIF visualizations.

Usage:
    python visualize_latents.py \
        --video-source 0 \
        --data-dir ./data \
        --tokenizer-checkpoint outputs/dreamer4_tokenizer/checkpoint_step_30000.pt \
        --dynamics-checkpoint outputs/dynamics_default/checkpoint_step_40000.pt \
        --output-dir outputs/visualizations/test \
        --num-frames 100 \
        --midpoint 50
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from wm.models.tokenizer.causal_tokenizer import CausalTokenizer
from wm.models.dynamics.dynamics_model import DynamicsModel
from wm.models.dynamics.inference import denoise_frame
from wm.data.video_dataset import VideoDataset


# Fixed parameters from plan
K = 4  # Denoising steps
CONTEXT_WINDOW = 128  # Model's training sequence length
CONTEXT_NOISE = 0.1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize world models encode->dynamics->decode pipeline"
    )
    parser.add_argument(
        "--video-source",
        type=int,
        default=0,
        help="Episode index (default: 0)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Video data directory (default: ./data)",
    )
    parser.add_argument(
        "--tokenizer-checkpoint",
        type=str,
        required=True,
        help="Path to tokenizer .pt file",
    )
    parser.add_argument(
        "--dynamics-checkpoint",
        type=str,
        required=True,
        help="Path to dynamics .pt file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/visualizations",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=100,
        help="Total frames to process (default: 100)",
    )
    parser.add_argument(
        "--midpoint",
        type=int,
        default=50,
        help="Frame to switch to autoregressive (default: 50)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="GIF frame rate (default: 10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: cuda/cpu/auto (default: auto)",
    )
    return parser.parse_args()


def get_device(device_arg: str) -> str:
    """Resolve device argument to actual device."""
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def load_tokenizer(checkpoint_path: str, device: str) -> CausalTokenizer:
    """Load CausalTokenizer from checkpoint."""
    model = CausalTokenizer(img_size=128, latent_dim=64).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    model.eval()
    return model


def load_dynamics_model(checkpoint_path: str, device: str) -> DynamicsModel:
    """Load DynamicsModel from checkpoint with hardcoded config."""
    model = DynamicsModel(
        embed_dim=512,
        num_heads=8,
        num_layers=24,
        temporal_layer_freq=4,
        num_spatial_tokens=128,  # Matches tokenizer
        latent_dim=64,  # dont forget to change back
        num_register_tokens=8,
        num_actions=12,
        num_action_tokens=1,
        max_sampling_steps=64,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    model.eval()
    return model


@torch.no_grad()
def encode_video(
    tokenizer: CausalTokenizer,
    frames: torch.Tensor,
    device: str,
    batch_size: int = 16,
) -> torch.Tensor:
    """Encode video frames to latents.

    Args:
        tokenizer: CausalTokenizer model
        frames: (T, C, H, W) video frames
        device: Device to use
        batch_size: Batch size for memory efficiency

    Returns:
        z: (T, N, D) latent representations
    """
    T = frames.shape[0]
    latents = []

    for i in range(0, T, batch_size):
        batch = frames[i : i + batch_size]  # (B, C, H, W)
        batch = batch.unsqueeze(1).to(device)  # (B, 1, C, H, W) - single frame per "video"

        with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16):
            z = tokenizer.tokenize(batch)  # (B, 1, N, D)

        z = z.squeeze(1).cpu()  # (B, N, D)
        latents.append(z)

    return torch.cat(latents, dim=0)  # (T, N, D)


@torch.no_grad()
def decode_latents(
    tokenizer: CausalTokenizer,
    latents: torch.Tensor,
    device: str,
    batch_size: int = 16,
) -> list[np.ndarray]:
    """Decode latents to images.

    Args:
        tokenizer: CausalTokenizer model
        latents: (T, N, D) latent representations
        device: Device to use
        batch_size: Batch size for memory efficiency

    Returns:
        List of (H, W, 3) uint8 numpy arrays
    """
    T = latents.shape[0]
    images = []

    for i in range(0, T, batch_size):
        batch = latents[i : i + batch_size]  # (B, N, D)
        batch = batch.unsqueeze(1).to(device=device, dtype=torch.float32)  # (B, 1, N, D)

        with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16):
            recon = tokenizer.decode(batch)  # (B, 1, C, H, W)

        recon = recon.squeeze(1)  # (B, C, H, W)
        recon = torch.clamp(recon, 0, 1)
        recon = (recon.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()

        for j in range(recon.shape[0]):
            images.append(recon[j])

    return images


def add_noise(z: torch.Tensor, noise_level: float) -> torch.Tensor:
    """Add noise to latent representations."""
    noise = torch.randn_like(z)
    return (1 - noise_level) * z + noise_level * noise


@torch.no_grad()
def run_teacher_forcing(
    dynamics: DynamicsModel,
    z_gt: torch.Tensor,
    actions: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Run teacher forcing prediction.

    For each frame, use ground truth history as context.

    Args:
        dynamics: DynamicsModel
        z_gt: (T, N, D) ground truth latents
        actions: (T, 12) boolean actions
        device: Device

    Returns:
        z_pred_tf: (T, N, D) teacher forcing predictions
    """
    T, N, D = z_gt.shape
    z_pred_tf = torch.zeros_like(z_gt)

    for t in tqdm(range(T), desc="Teacher forcing"):
        # Use as much ground truth history as available (up to CONTEXT_WINDOW frames)
        ctx_start = max(0, t - CONTEXT_WINDOW)
        z_context = z_gt[ctx_start:t]  # Ground truth context

        if z_context.shape[0] == 0:
            # No context available (first frame) - use zeros
            z_context = torch.zeros(1, N, D)

        # Add noise to context
        z_context_noisy = add_noise(z_context, CONTEXT_NOISE)

        # Prepare action for current frame
        action_t = actions[t : t + 1].unsqueeze(0).to(device)  # (1, 1, 12)

        intermediates = denoise_frame(
            model=dynamics,
            z_context_noisy=z_context_noisy.unsqueeze(0).to(device),
            z_target_shape=(1, 1, N, D),
            action_t=action_t,
            K=K,
            context_noise=CONTEXT_NOISE,
            device=device,
        )

        z_pred_tf[t] = intermediates[-1]["z_pred"].squeeze(0).squeeze(0)  # (N, D)

    return z_pred_tf


@torch.no_grad()
def run_autoregressive(
    dynamics: DynamicsModel,
    z_gt: torch.Tensor,
    actions: torch.Tensor,
    midpoint: int,
    device: str,
) -> torch.Tensor:
    """Run autoregressive prediction from midpoint.

    Uses ground truth for frames [0:midpoint], then autoregressively
    generates frames [midpoint:T].

    Args:
        dynamics: DynamicsModel
        z_gt: (T, N, D) ground truth latents
        actions: (T, 12) boolean actions
        midpoint: Frame index to start autoregressive generation
        device: Device

    Returns:
        z_pred_ar: (T, N, D) predictions (GT for [0:midpoint], AR for rest)
    """
    T, N, D = z_gt.shape
    z_pred_ar = z_gt.clone()  # Start with ground truth

    # Build history: ground truth prefix
    z_history = z_gt[:midpoint].clone()

    for t in tqdm(range(midpoint, T), desc="Autoregressive"):
        # Use as much history as available (up to CONTEXT_WINDOW frames)
        ctx_start = max(0, len(z_history) - CONTEXT_WINDOW)
        z_context = z_history[ctx_start:]  # Mix of GT prefix + predictions

        # Add noise to context
        z_context_noisy = add_noise(z_context, CONTEXT_NOISE)

        # Prepare action for current frame (use GT actions throughout)
        action_t = actions[t : t + 1].unsqueeze(0).to(device)  # (1, 1, 12)

        intermediates = denoise_frame(
            model=dynamics,
            z_context_noisy=z_context_noisy.unsqueeze(0).to(device),
            z_target_shape=(1, 1, N, D),
            action_t=action_t,
            K=K,
            context_noise=CONTEXT_NOISE,
            device=device,
        )

        z_pred_t = intermediates[-1]["z_pred"].squeeze(0)  # (1, N, D)

        # Store prediction
        z_pred_ar[t] = z_pred_t.squeeze(0)  # (N, D)

        # Append prediction to history for next iteration
        z_history = torch.cat([z_history, z_pred_t.cpu()], dim=0)

    return z_pred_ar


def create_gif(
    images: list[np.ndarray],
    output_path: Path,
    fps: int = 10,
) -> None:
    """Create GIF from list of images.

    Args:
        images: List of (H, W, 3) uint8 numpy arrays
        output_path: Path to save GIF
        fps: Frames per second
    """
    if not images:
        return

    frames = [Image.fromarray(img) for img in images]
    duration = int(1000 / fps)  # milliseconds per frame

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )


def create_comparison_gif(
    original: list[np.ndarray],
    reconstruction: list[np.ndarray],
    teacher_forcing: list[np.ndarray],
    autoregressive: list[np.ndarray],
    output_path: Path,
    fps: int = 10,
    midpoint: int = 50,
) -> None:
    """Create side-by-side comparison GIF.

    Args:
        original: Original video frames
        reconstruction: Tokenizer encode->decode frames
        teacher_forcing: Teacher forcing predictions
        autoregressive: Autoregressive predictions
        output_path: Path to save GIF
        fps: Frames per second
        midpoint: Frame where AR starts (for visual indicator)
    """
    if not original:
        return

    h, w = original[0].shape[:2]
    padding = 10
    label_height = 30
    labels = ["Original", "Recon", "TF", "AR"]
    n_cols = 4

    total_width = n_cols * w + (n_cols - 1) * padding
    total_height = h + label_height

    frames = []
    for i, (orig, recon, tf, ar) in enumerate(
        zip(original, reconstruction, teacher_forcing, autoregressive)
    ):
        # Create canvas
        canvas = Image.new("RGB", (total_width, total_height), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Paste images
        images = [orig, recon, tf, ar]
        for j, (img, label) in enumerate(zip(images, labels)):
            x = j * (w + padding)

            # Add label with indicator for AR midpoint
            if label == "AR" and i >= midpoint:
                label_text = f"{label}*"  # Indicate AR is now generating
            else:
                label_text = label

            draw.text((x + w // 2 - 20, 5), label_text, fill=(0, 0, 0))
            canvas.paste(Image.fromarray(img), (x, label_height))

        frames.append(canvas)

    duration = int(1000 / fps)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )


def main():
    args = parse_args()
    device = get_device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Output directory: {output_dir}")

    # Load models
    print(f"Loading tokenizer from {args.tokenizer_checkpoint}")
    tokenizer = load_tokenizer(args.tokenizer_checkpoint, device)

    print(f"Loading dynamics model from {args.dynamics_checkpoint}")
    dynamics = load_dynamics_model(args.dynamics_checkpoint, device)

    # Load video data
    print(f"Loading video from {args.data_dir}, episode {args.video_source}")
    dataset = VideoDataset(
        video_dir=args.data_dir,
        sequence_length=args.num_frames,
        frame_size=(128, 128),
    )
    frames, actions = dataset[args.video_source]
    # frames: (T, C, H, W), actions: (T, 12)

    print(f"Loaded {frames.shape[0]} frames with shape {frames.shape[1:]}")
    print(f"Actions shape: {actions.shape}")

    # Step 1: Encode video to latents
    print("Encoding video to latents...")
    z_gt = encode_video(tokenizer, frames, device)  # (T, N, D)
    print(f"Latent shape: {z_gt.shape}")

    # Step 2: Decode latents back to images (reconstruction baseline)
    print("Decoding latents (reconstruction baseline)...")
    reconstruction_images = decode_latents(tokenizer, z_gt, device)

    # Step 3: Run teacher forcing
    print("Running teacher forcing predictions...")
    z_pred_tf = run_teacher_forcing(dynamics, z_gt, actions.float(), device)

    # Step 4: Run autoregressive from midpoint
    print(f"Running autoregressive predictions from frame {args.midpoint}...")
    z_pred_ar = run_autoregressive(
        dynamics, z_gt, actions.float(), args.midpoint, device
    )

    # Step 5: Decode predictions
    print("Decoding teacher forcing predictions...")
    tf_images = decode_latents(tokenizer, z_pred_tf, device)

    print("Decoding autoregressive predictions...")
    ar_images = decode_latents(tokenizer, z_pred_ar, device)

    # Step 6: Convert original frames for GIF
    print("Preparing original frames...")
    original_images = []
    for i in range(frames.shape[0]):
        img = frames[i].permute(1, 2, 0).numpy()  # (H, W, C)
        img = (img * 255).astype(np.uint8)
        original_images.append(img)

    # Step 7: Create GIFs
    print("Creating GIFs...")

    create_gif(original_images, output_dir / "original.gif", args.fps)
    print(f"  Saved: {output_dir / 'original.gif'}")

    create_gif(reconstruction_images, output_dir / "reconstruction.gif", args.fps)
    print(f"  Saved: {output_dir / 'reconstruction.gif'}")

    create_gif(tf_images, output_dir / "teacher_forcing.gif", args.fps)
    print(f"  Saved: {output_dir / 'teacher_forcing.gif'}")

    create_gif(ar_images, output_dir / "autoregressive.gif", args.fps)
    print(f"  Saved: {output_dir / 'autoregressive.gif'}")

    create_comparison_gif(
        original_images,
        reconstruction_images,
        tf_images,
        ar_images,
        output_dir / "comparison.gif",
        args.fps,
        args.midpoint,
    )
    print(f"  Saved: {output_dir / 'comparison.gif'}")

    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
