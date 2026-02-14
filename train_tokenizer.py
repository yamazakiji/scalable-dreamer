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


class RunningRMSNormalizer:
    """Normalizes losses by their running RMS estimate (EMA of squared values).

    Each loss component is divided by its RMS estimate so that differently-scaled
    losses contribute roughly equally to the combined gradient.
    """

    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.ema_sq: dict[str, float] = {}
        self.steps: dict[str, int] = {}

    def __call__(self, loss: torch.Tensor, name: str) -> torch.Tensor:
        val = loss.detach().float().item()

        if name not in self.ema_sq:
            # First call — seed the estimate, return raw loss
            self.ema_sq[name] = val ** 2
            self.steps[name] = 1
            return loss

        self.steps[name] += 1
        self.ema_sq[name] = self.decay * self.ema_sq[name] + (1 - self.decay) * val ** 2

        # Bias-corrected RMS estimate
        bias_correction = 1 - self.decay ** self.steps[name]
        rms = (self.ema_sq[name] / bias_correction) ** 0.5

        return loss / max(rms, 1e-8)

    def get_rms_estimates(self) -> dict[str, float]:
        """Return current bias-corrected RMS estimates for logging."""
        out = {}
        for name in self.ema_sq:
            bias_correction = 1 - self.decay ** self.steps[name]
            out[name] = (self.ema_sq[name] / bias_correction) ** 0.5
        return out


def train_step(
    model: nn.Module,
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    lpips_model: LPIPSMetric,
    rms_normalizer: RunningRMSNormalizer,
    grad_clip: float = 1.0,
    compute_diagnostics: bool = False,
) -> dict:
    """Single training step."""
    optimizer.zero_grad()

    B, T, C, H, W = batch.shape

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        z, recon, _, _ = model(batch, apply_masking=True)

        pred_flat = recon.view(B * T, C, H, W)
        target_flat = batch.view(B * T, C, H, W)

        mse_loss = nn.functional.mse_loss(recon, batch)
        lpips_loss = lpips_model(pred_flat, target_flat)
        loss = rms_normalizer(mse_loss, "mse") + 0.2 * rms_normalizer(lpips_loss, "lpips")

    metrics = {
        "loss": loss.item(),
        "mse_loss": mse_loss.item(),
        "lpips_loss": lpips_loss.item(),
    }

    if compute_diagnostics:
        # Latent stats
        z_abs = z.detach().float().abs()
        metrics["latent/mean_abs_z"] = z_abs.mean().item()
        metrics["latent/frac_saturated"] = (z_abs > 0.99).float().mean().item()
        metrics["latent/std_z"] = z.detach().float().std().item()
        # Loss RMS estimates
        for name, rms in rms_normalizer.get_rms_estimates().items():
            metrics[f"loss_rms/{name}"] = rms
        loss.backward()
    else:
        loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    metrics["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

    optimizer.step()

    return metrics


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
    """Evaluation phase for CausalTokenizer.

    Iterates over the full eval dataloader to compute mean metrics.
    Uses the first batch for visualization.
    """
    model.eval()

    metric_sums: dict[str, float] = {}
    num_batches = 0
    first_batch = None
    first_recon_dict: dict[str, torch.Tensor] = {}

    for batch, _ in eval_dataloader:
        batch = batch.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, recon_masked, _, _ = model(batch, apply_masking=True)
            metrics_masked = compute_reconstruction_metrics(
                recon_masked, batch, lpips_model=lpips_model
            )

            _, recon_clean, _, _ = model(batch, apply_masking=False)
            metrics_clean = compute_reconstruction_metrics(
                recon_clean, batch, lpips_model=lpips_model
            )

        for k, v in metrics_masked.items():
            key = f"eval/masked_{k}"
            metric_sums[key] = metric_sums.get(key, 0.0) + v
        for k, v in metrics_clean.items():
            key = f"eval/inference_{k}"
            metric_sums[key] = metric_sums.get(key, 0.0) + v

        if first_batch is None:
            first_batch = batch
            first_recon_dict["masked"] = recon_masked
            first_recon_dict["inference"] = recon_clean

        num_batches += 1

    # Compute mean metrics
    all_metrics = {k: v / num_batches for k, v in metric_sums.items()}

    if rank == 0:
        if logger:
            if use_fsdp and FSDP_AVAILABLE:
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                mem_stats = get_memory_stats(local_rank)
                all_metrics.update({f"memory/{k}": v for k, v in mem_stats.items()})
            logger.log(all_metrics, step=step)

        # Visualization from first batch
        if first_batch is not None:
            num_samples = min(1, first_batch.shape[0])
            original_dict = {
                "masked": first_batch[:num_samples],
                "inference": first_batch[:num_samples],
            }
            log_reconstruction_comparison(
                logger,
                first_batch[:num_samples],
                original_dict,
                {k: v[:num_samples] for k, v in first_recon_dict.items()},
                step=step,
                num_samples=num_samples,
                num_frames=8,
            )

        print(f"\nEvaluation at step {step} ({num_batches} batches):")
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
        img_size=config.tokenizer.image_size,
        patch_size=config.tokenizer.patch_size,
        in_channels=config.tokenizer.in_channels,
        embed_dim=config.tokenizer.embed_dim,
        num_heads=config.tokenizer.num_heads,
        num_latents=config.tokenizer.num_latents,
        latent_dim=config.tokenizer.latent_dim,
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

    dataset = VideoDataset(video_dir="./data/rollouts")
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

    eval_dataset = VideoDataset(video_dir="./data/rollouts_val")
    # eval_dataset = VideoDataset(video_dir="./data/rollouts_one_batch")
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

    rms_normalizer = RunningRMSNormalizer()

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

            is_log_step = is_main and step % config.training.log_every == 0
            metrics = train_step(
                model, batch, optimizer, lpips_model, rms_normalizer,
                config.training.grad_clip, compute_diagnostics=is_log_step,
            )
            scheduler.step()

            if is_log_step:
                metrics["lr"] = scheduler.get_last_lr()[0]
                metrics["step"] = step
                metrics["epoch"] = epoch

                sat_info = ""
                if "latent/frac_saturated" in metrics:
                    sat_info = (
                        f" | Sat: {metrics['latent/frac_saturated']:.3f}"
                        f" | |z|: {metrics['latent/mean_abs_z']:.3f}"
                    )

                print(
                    f"Step {step}/{config.training.max_steps} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"LR: {metrics['lr']:.2e}"
                    f"{sat_info}"
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
    parser.add_argument(
        "--set", "-s",
        action="append",
        dest="overrides",
        metavar="KEY=VALUE",
        help="Override config values (e.g., --set training.learning_rate=0.001)"
    )
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
        config = config.update(**{"training.resume_from": args.resume})

    # Apply generic overrides
    if args.overrides:
        for override in args.overrides:
            key, value = override.split("=", 1)
            # Parse value type
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # Keep as string
            config = config.update(**{key: value})

    train(config, use_fsdp=args.fsdp)


if __name__ == "__main__":
    main()
