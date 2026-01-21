"""Train dynamics model with shortcut forcing objective.

Usage:
    # Single GPU
    python train_dynamics.py --latent-dir ./data/latents

    # Multi-GPU with FSDP
    torchrun --nproc_per_node=4 train_dynamics.py --latent-dir ./data/latents --fsdp
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path

from wm.configs.base import ExperimentConfig
from wm.models.dynamics.dynamics_model import DynamicsModel
from wm.models.dynamics.shortcut_forcing import ShortcutForcing
from wm.data.latent_dataset import LatentDataset
from wm.training.distributed import setup_distributed, cleanup_distributed, wrap_model_ddp
from wm.training.optimizer import get_optimizer, get_warmup_scheduler
from wm.utils.logging import WandbLogger

# FSDP imports (optional, only used when --fsdp is enabled)
try:
    from wm.training.fsdp import (
        apply_fsdp,
        save_checkpoint,
        load_checkpoint,
        get_memory_stats,
    )
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False


def train_step(
    model: nn.Module,
    objective: ShortcutForcing,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    grad_clip: float = 1.0,
) -> dict:
    """Single training step with shortcut forcing.

    Args:
        model: DynamicsModel
        objective: ShortcutForcing loss function
        batch: Dict with "z", "a", "z_next" from LatentSequenceDataset
        optimizer: Optimizer
        grad_clip: Gradient clipping value

    Returns:
        Dictionary of metrics
    """
    optimizer.zero_grad()

    # batch from LatentSequenceDataset: {"z": (B, T, N, D), "a": (B, T), "z_next": (B, T, N, D)}
    z_clean = batch["z_next"]  # Use z_next as prediction targets
    actions = batch["a"]

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # ShortcutForcing.compute_loss handles tau/d sampling and corruption internally
        loss_dict = objective.compute_loss(model, z_clean, actions)

    loss_dict["total"].backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

    return {
        "loss": loss_dict["total"].item(),
        "flow_loss": loss_dict["flow"].item(),
        "bootstrap_loss": loss_dict["bootstrap"].item(),
        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    eval_dataloader: DataLoader,
    logger,
    step: int,
    device: torch.device,
    rank: int = 0,
    num_inference_steps: int = 4,
    use_fsdp: bool = False,
):
    """Evaluation phase for DynamicsModel with autoregressive sampling."""
    model.eval()

    batch = next(iter(eval_dataloader))
    batch = {k: v.to(device) for k, v in batch.items()}

    B, T, N, D = batch["z"].shape
    context_len = min(4, T // 2)  # Use first few frames as context
    gen_len = T - context_len

    z_context = batch["z"][:, :context_len]  # (B, ctx_len, N, D)
    actions = batch["a"][:, context_len:]    # (B, gen_len) actions for generation
    z_target = batch["z_next"][:, context_len:]  # (B, gen_len, N, D) targets

    all_metrics = {}

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # Autoregressive sampling
        z_generated = model.sample(
            z_context,
            actions,
            num_steps=num_inference_steps,
            context_noise=0.1,
        )

        # Compute MSE between generated and target (up to generated length)
        gen_frames = min(z_generated.shape[1], z_target.shape[1])
        mse = nn.functional.mse_loss(
            z_generated[:, :gen_frames],
            z_target[:, :gen_frames]
        )
        all_metrics["eval/mse"] = mse.item()

        # Per-step MSE for first few frames
        for t in range(min(4, gen_frames)):
            step_mse = nn.functional.mse_loss(z_generated[:, t], z_target[:, t])
            all_metrics[f"eval/mse_step_{t+1}"] = step_mse.item()

    if rank == 0:
        if logger:
            if use_fsdp and FSDP_AVAILABLE:
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                mem_stats = get_memory_stats(local_rank)
                all_metrics.update({f"memory/{k}": v for k, v in mem_stats.items()})
            logger.log(all_metrics, step=step)

        print(f"\nEvaluation at step {step}:")
        print(f"  MSE (avg): {all_metrics.get('eval/mse', 0):.6f}")
        for t in range(min(4, gen_frames)):
            print(f"  MSE step {t+1}: {all_metrics.get(f'eval/mse_step_{t+1}', 0):.6f}")

    model.train()


def train(config: ExperimentConfig, latent_dir: str, sequence_length: int, use_fsdp: bool = False):
    """Main training loop with optional FSDP support."""
    if use_fsdp and not FSDP_AVAILABLE:
        raise RuntimeError("FSDP requested but wm.training.fsdp module not available")

    rank, world_size = setup_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    if is_main:
        print(f"Starting dynamics training: {config.name}")
        print(f"World size: {world_size}")
        print(f"Device: {device}")
        print(f"Mode: {'FSDP' if use_fsdp else 'DDP'}")
        print(f"Latent directory: {latent_dir}")
        print(f"Sequence length: {sequence_length}")

    torch.manual_seed(config.seed + rank)

    # Create dynamics model
    model = DynamicsModel(
        embed_dim=config.dynamics.embed_dim,
        num_heads=config.dynamics.num_heads,
        num_layers=config.dynamics.num_layers,
        temporal_layer_freq=config.dynamics.temporal_layer_freq,
        num_spatial_tokens=config.dynamics.num_spatial_tokens,
        latent_dim=config.dynamics.latent_dim,
        num_register_tokens=config.dynamics.num_register_tokens,
        num_actions=config.dynamics.num_actions,
        num_action_tokens=config.dynamics.num_action_tokens,
        max_sampling_steps=config.dynamics.max_sampling_steps,
    ).to(device)

    # Create shortcut forcing objective
    objective = ShortcutForcing(
        max_steps=config.dynamics.max_sampling_steps,
        ramp_weight=config.dynamics.ramp_weight,
    )

    if is_main:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")
        print(f"Gradient checkpointing: {config.dynamics.gradient_checkpointing}")

    if use_fsdp:
        model = apply_fsdp(
            model,
            mixed_precision=config.training.fsdp_mixed_precision,
        )
        optimizer = get_optimizer(
            model, config.training.learning_rate, config.training.weight_decay
        )
        if is_main:
            print("Model wrapped with FSDP")
            mem_stats = get_memory_stats(local_rank)
            print(f"GPU memory after FSDP wrap: {mem_stats['memory_allocated_gb']:.2f} GB")
    else:
        if world_size > 1:
            model = wrap_model_ddp(model, rank)
        optimizer = get_optimizer(
            model, config.training.learning_rate, config.training.weight_decay
        )

    scheduler = get_warmup_scheduler(
        optimizer, config.training.warmup_steps, config.training.max_steps
    )

    # Create dataset
    dataset = LatentDataset(
        latent_dir=latent_dir,
        sequence_length=sequence_length,
    )
    sampler = DistributedSampler(dataset, shuffle=True) if (world_size > 1 or use_fsdp) else None

    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    eval_dataset = LatentDataset(
        latent_dir=latent_dir,
        sequence_length=sequence_length,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    if is_main:
        print(f"Dataset size: {len(dataset)} sequences")
        print(f"Latent shape: {dataset.latent_shape}")

    logger = None
    if is_main and config.use_wandb:
        logger = WandbLogger(
            project=config.wandb_project,
            name=config.name,
            config=config.to_dict(),
            enabled=config.use_wandb,
        )
        if not use_fsdp:
            logger.watch(model, log_freq=config.training.log_every)

    start_step = 0
    start_epoch = 0

    if config.training.resume_from:
        if is_main:
            print(f"Resuming from checkpoint: {config.training.resume_from}")
        if use_fsdp:
            start_step, start_epoch = load_checkpoint(
                model, optimizer, scheduler, config.training.resume_from, rank
            )
        else:
            checkpoint = torch.load(config.training.resume_from, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_step = checkpoint["step"]
            start_epoch = checkpoint["epoch"]

    model.train()
    step = start_step
    epoch = start_epoch

    if is_main:
        print(f"Starting training from step {step}...")

    while step < config.training.max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)

        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            metrics = train_step(
                model, objective, batch, optimizer, config.training.grad_clip
            )
            scheduler.step()

            if is_main and step % config.training.log_every == 0:
                metrics["lr"] = scheduler.get_last_lr()[0]
                metrics["step"] = step
                metrics["epoch"] = epoch

                print(
                    f"Step {step}/{config.training.max_steps} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"Flow: {metrics['flow_loss']:.4f} | "
                    f"Bootstrap: {metrics['bootstrap_loss']:.4f} | "
                    f"LR: {metrics['lr']:.2e}"
                )

                if logger:
                    logger.log(metrics, step=step)

            if step > 0 and step % config.training.eval_every == 0:
                if is_main:
                    print(f"\nRunning evaluation at step {step}...")
                evaluate(
                    model, eval_dataloader, logger, step, device, rank,
                    num_inference_steps=config.dynamics.num_inference_steps,
                    use_fsdp=use_fsdp,
                )

            if step > 0 and step % config.training.checkpoint_every == 0:
                save_dir = Path(config.training.save_dir) / config.name
                if is_main:
                    save_dir.mkdir(parents=True, exist_ok=True)

                if use_fsdp:
                    if dist.is_initialized():
                        dist.barrier()
                    save_checkpoint(
                        model, optimizer, scheduler, step, epoch,
                        str(save_dir / f"checkpoint_step_{step}.pt"),
                        rank, config.to_dict(),
                    )
                elif is_main:
                    checkpoint = {
                        "step": step,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "config": config.to_dict(),
                    }
                    torch.save(checkpoint, save_dir / f"checkpoint_step_{step}.pt")
                    print(f"Saved checkpoint at step {step}")

            step += 1
            if step >= config.training.max_steps:
                break

        epoch += 1

    if logger:
        logger.finish()
    cleanup_distributed()

    if is_main:
        print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train dynamics model")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--name", type=str, default="dreamer4_dynamics", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--fsdp", action="store_true", help="Use FSDP instead of DDP")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")

    # Data arguments
    parser.add_argument("--latent-dir", type=str, default="./data/latents",
                        help="Directory containing extracted latents")
    parser.add_argument("--sequence-length", type=int, default=16,
                        help="Sequence length for training")

    # Training overrides
    parser.add_argument("--max-steps", type=int, help="Override max training steps")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")

    args = parser.parse_args()

    if args.config:
        config = ExperimentConfig.from_yaml(Path(args.config))
    else:
        config = ExperimentConfig(
            name=args.name,
            seed=args.seed,
            use_wandb=not args.no_wandb,
        )

    # Apply overrides
    if args.resume:
        config.training.resume_from = args.resume
    if args.max_steps:
        config.training.max_steps = args.max_steps
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr

    train(config, args.latent_dir, args.sequence_length, use_fsdp=args.fsdp)


if __name__ == "__main__":
    main()
