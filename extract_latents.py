"""Extract latent representations from video dataset using CausalTokenizer.

This script processes videos through a trained CausalTokenizer and saves
(z, a, z_next) transition tuples for dynamics model training.

Usage:
    # Single GPU
    python extract_latents.py --checkpoint outputs/tokenizer/checkpoint.pt

    # Multi-GPU with FSDP
    torchrun --nproc_per_node=4 extract_latents.py --checkpoint outputs/tokenizer/checkpoint.pt --fsdp
"""

import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from wm.models.tokenizer.causal_tokenizer import CausalTokenizer
from wm.data.video_dataset import VideoDataset
from wm.training.distributed import setup_distributed, cleanup_distributed

# Optional FSDP imports
try:
    from wm.training.fsdp import apply_fsdp, get_memory_stats
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False


def load_checkpoint(checkpoint_path: str, device: torch.device) -> dict:
    """Load checkpoint and strip DDP 'module.' prefix if present."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v

    return new_state_dict


def create_transitions(
    latents: torch.Tensor,
    actions: torch.Tensor,
) -> dict:
    """
    Create (z_t, a_t, z_{t+1}) transition tuples from sequential latents.

    The last frame has no z_next, so we produce T-1 transitions from T frames.

    Args:
        latents: (T, num_latents, latent_dim) latent sequence
        actions: (T, action_dim) or (T,) action sequence

    Returns:
        dict with z, a, z_next tensors and metadata
    """
    T = latents.shape[0]

    return {
        "z": latents[:-1],           # (T-1, num_latents, latent_dim)
        "a": actions[:-1],           # (T-1, action_dim) or (T-1,)
        "z_next": latents[1:],       # (T-1, num_latents, latent_dim)
        "num_transitions": T - 1,
        "num_frames": T,
    }


def save_transitions(
    output_dir: Path,
    episode_id: int,
    transitions: dict,
):
    """Save transitions for an episode."""
    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    # Add episode_id to transitions dict
    transitions["episode_id"] = episode_id

    save_path = episodes_dir / f"{episode_id:04d}.pt"
    torch.save(transitions, save_path)


def process_dataset(
    model: torch.nn.Module,
    dataloader: DataLoader,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    is_main: bool,
) -> tuple[int, int]:
    """
    Process entire dataset and extract latents.

    Returns:
        (num_episodes, total_transitions)
    """
    num_episodes = 0
    total_transitions = 0

    iterator = tqdm(dataloader, desc="Extracting latents", disable=not is_main)

    for idx, sample in enumerate(iterator):
        # Handle both tuple (frames, actions) and single tensor returns
        if isinstance(sample, list):  # default collate_fn returns list
            frames, actions = sample
        else:
            frames = sample
            # Placeholder actions if not provided
            T = frames.shape[0]
            actions = torch.zeros(T, dtype=torch.long)

        frames = frames.to(device)

        # Extract latents
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                latents, _, _, _ = model(frames, apply_masking=False)  # Note: FSDP does not work with methods other than forward

        # Create and save transitions
        transitions = create_transitions(latents.squeeze(0).cpu().to(torch.float32), actions.squeeze(0))  # note: needed as later these latents will be used for training
        save_transitions(output_dir, idx, transitions)

        num_episodes += 1
        total_transitions += transitions["num_transitions"]

        if is_main:
            iterator.set_postfix(
                episodes=num_episodes,
                transitions=total_transitions,
            )

    return num_episodes, total_transitions


def main():
    parser = argparse.ArgumentParser(description="Extract latents from video dataset")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to tokenizer checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Input video data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/latents",
        help="Output latents directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Frames to process at once (for memory efficiency)",
    )
    parser.add_argument(
        "--fsdp",
        action="store_true",
        help="Use FSDP sharding for inference",
    )
    args = parser.parse_args()

    # Setup distributed if using FSDP
    if args.fsdp:
        if not FSDP_AVAILABLE:
            raise RuntimeError("FSDP requested but wm.training.fsdp module not available")
        rank, world_size = setup_distributed()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        is_main = rank == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True
        rank = 0
        world_size = 1

    if is_main:
        print(f"Device: {device}")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"FSDP: {args.fsdp}")
        print(f"Data directory: {args.data_dir}")
        print(f"Output directory: {args.output_dir}")

    # Create output directory
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create model and load checkpoint
    model = CausalTokenizer().to(device)

    if is_main:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")

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

    # Create dataset
    dataset = VideoDataset(video_dir=args.data_dir)

    if is_main:
        print(f"Dataset size: {len(dataset)} episodes")

    # Use DistributedSampler for multi-GPU
    if args.fsdp and world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=False)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one episode at a time
        sampler=sampler,
        num_workers=0,  # Videos need sequential access
    )

    # Process dataset
    if is_main:
        print("Starting latent extraction...")

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    num_episodes, total_transitions = process_dataset(
        model=model,
        dataloader=dataloader,
        output_dir=output_dir,
        device=device,
        batch_size=args.batch_size,
        is_main=is_main,
    )

    if torch.cuda.is_available() and is_main:
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak GPU memory: {peak_mem:.2f} GB")

    # Synchronize and save global metadata on rank 0
    if args.fsdp and world_size > 1:
        dist.barrier()

    if is_main:
        # Get latent shape from model
        latent_shape = [model.num_latents, model.latent_dim]

        metadata = {
            "tokenizer_checkpoint": str(args.checkpoint),
            "latent_shape": latent_shape,
            "latent_range": [-1, 1],
            "num_episodes": num_episodes,
            "total_transitions": total_transitions,
            "extraction_timestamp": datetime.now().isoformat(),
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nExtraction complete!")
        print(f"Total episodes: {num_episodes}")
        print(f"Total transitions: {total_transitions}")
        print(f"Output directory: {output_dir}")

    if args.fsdp:
        cleanup_distributed()


if __name__ == "__main__":
    main()
