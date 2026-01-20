import argparse
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path

from wm.configs.base import ExperimentConfig
from wm.models.tokenizer.causal_tokenizer import CausalTokenizer
from wm.data.video_dataset import VideoDataset
from wm.training.distributed import setup_distributed, cleanup_distributed, wrap_model_ddp
from wm.training.optimizer import get_optimizer, get_warmup_scheduler
from wm.utils.logging import WandbLogger
from wm.utils.metrics import LPIPSMetric, compute_reconstruction_metrics
from wm.utils.visualization import log_reconstruction_comparison

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
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    lpips_model: LPIPSMetric,
    grad_clip: float = 1.0,
) -> dict:
    """Single training step."""
    optimizer.zero_grad()

    B, T, C, H, W = batch.shape

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, recon, _, _ = model(batch, apply_masking=True)

        pred_flat = recon.view(B * T, C, H, W)
        target_flat = batch.view(B * T, C, H, W)

        mse_loss = nn.functional.mse_loss(recon, batch)
        lpips_loss = lpips_model(pred_flat, target_flat)
        loss = mse_loss + 0.2 * lpips_loss

    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

    return {
        "loss": loss.item(),
        "mse_loss": mse_loss.item(),
        "lpips_loss": lpips_loss.item(),
        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    eval_dataloader: DataLoader,
    lpips_model: LPIPSMetric,
    logger,
    step: int,
    device: torch.device,
    rank: int = 0,
    use_fsdp: bool = False,
):
    """Evaluation phase for CausalTokenizer."""
    model.eval()

    batch, _ = next(iter(eval_dataloader))
    batch = batch.to(device)

    all_metrics = {}
    reconstructed_dict = {}

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, recon_masked, _, _ = model(batch, apply_masking=True)
        metrics_masked = compute_reconstruction_metrics(
            recon_masked, batch, lpips_model=lpips_model
        )
        for k, v in metrics_masked.items():
            all_metrics[f"eval/masked_{k}"] = v
        reconstructed_dict["masked"] = recon_masked

        _, recon_clean, _, _ = model(batch, apply_masking=False)
        metrics_clean = compute_reconstruction_metrics(
            recon_clean, batch, lpips_model=lpips_model
        )
        for k, v in metrics_clean.items():
            all_metrics[f"eval/inference_{k}"] = v
        reconstructed_dict["inference"] = recon_clean

    if rank == 0:
        if logger:
            if use_fsdp and FSDP_AVAILABLE:
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                mem_stats = get_memory_stats(local_rank)
                all_metrics.update({f"memory/{k}": v for k, v in mem_stats.items()})
            logger.log(all_metrics, step=step)

        num_samples = min(4, batch.shape[0])
        original_dict = {
            "masked": batch[:num_samples],
            "inference": batch[:num_samples],
        }
        log_reconstruction_comparison(
            logger,
            batch[:num_samples],
            original_dict,
            {k: v[:num_samples] for k, v in reconstructed_dict.items()},
            step=step,
            num_samples=num_samples,
        )

        print(f"\nEvaluation at step {step}:")
        print(
            f"  masked: LPIPS={all_metrics.get('eval/masked_lpips', 0):.4f}, "
            f"MSE={all_metrics.get('eval/masked_mse', 0):.4f}"
        )
        print(
            f"  inference: LPIPS={all_metrics.get('eval/inference_lpips', 0):.4f}, "
            f"MSE={all_metrics.get('eval/inference_mse', 0):.4f}"
        )

    model.train()


def train(config: ExperimentConfig, use_fsdp: bool = False):
    """Main training loop with optional FSDP support."""
    if use_fsdp and not FSDP_AVAILABLE:
        raise RuntimeError("FSDP requested but wm.training.fsdp module not available")

    rank, world_size = setup_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    if is_main:
        print(f"Starting training: {config.name}")
        print(f"World size: {world_size}")
        print(f"Device: {device}")
        print(f"Mode: {'FSDP' if use_fsdp else 'DDP'}")

    torch.manual_seed(config.seed + rank)

    model = CausalTokenizer(
        gradient_checkpointing=config.tokenizer.gradient_checkpointing,
    ).to(device)

    if is_main:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")
        print(f"Gradient checkpointing: {config.tokenizer.gradient_checkpointing}")

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

    dataset = VideoDataset(video_dir="./data")
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

    eval_dataset = VideoDataset(video_dir="./data")
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    lpips_model = LPIPSMetric(net="alex", device=device)

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

        for batch, _ in dataloader:
            batch = batch.to(device)

            metrics = train_step(
                model, batch, optimizer, lpips_model, config.training.grad_clip
            )
            scheduler.step()

            if is_main and step % config.training.log_every == 0:
                metrics["lr"] = scheduler.get_last_lr()[0]
                metrics["step"] = step
                metrics["epoch"] = epoch

                print(
                    f"Step {step}/{config.training.max_steps} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"LR: {metrics['lr']:.2e}"
                )

                if logger:
                    logger.log(metrics, step=step)

            if step > 0 and step % config.training.eval_every == 0:
                if is_main:
                    print(f"\nRunning evaluation at step {step}...")
                evaluate(
                    model, eval_dataloader, lpips_model, logger, step, device, rank, use_fsdp
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
    parser = argparse.ArgumentParser(description="Train causal tokenizer")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--name", type=str, default="dreamer4_tokenizer", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--fsdp", action="store_true", help="Use FSDP instead of DDP")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    if args.config:
        config = ExperimentConfig.from_yaml(Path(args.config))
    else:
        config = ExperimentConfig(
            name=args.name,
            seed=args.seed,
            use_wandb=not args.no_wandb,
        )

    if args.resume:
        config.training.resume_from = args.resume

    train(config, use_fsdp=args.fsdp)


if __name__ == "__main__":
    main()
