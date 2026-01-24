"""Evaluation helper functions for dynamics model."""
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# FSDP memory stats (optional)
try:
    from wm.training.fsdp import get_memory_stats
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False


def prepare_eval_batch(
    dataloader: DataLoader,
    device: torch.device,
    context_len: int = 4,
    max_gen_len: int = 4,
) -> dict:
    """
    Prepare batch for evaluation.

    Args:
        dataloader: Evaluation dataloader
        device: Device to move tensors to
        context_len: Number of context frames (default 4)
        max_gen_len: Maximum frames to generate (default 4)

    Returns:
        Dictionary with keys: z_context, z_targets, actions, context_len, gen_len
    """
    batch = next(iter(dataloader))
    batch = {k: v.to(device) for k, v in batch.items()}

    B, T, N, D = batch["z"].shape
    actual_context_len = min(context_len, T // 2)
    gen_len = min(max_gen_len, T - actual_context_len)

    z_context = batch["z"][:, :actual_context_len]  # (B, ctx_len, N, D)
    actions = batch["a"]  # (B, T)
    z_targets = batch["z_next"][:, actual_context_len:actual_context_len + gen_len]  # (B, gen_len, N, D)

    return {
        "z_context": z_context,
        "z_targets": z_targets,
        "actions": actions,
        "context_len": actual_context_len,
        "gen_len": gen_len,
        "batch_shape": (B, T, N, D),
    }


def aggregate_metrics(
    all_metrics: dict,
    K_values: list[int],
    gen_len: int,
) -> dict:
    """
    Compute aggregate metrics (mean MSE/cosine_sim) across frames.

    Args:
        all_metrics: Dictionary of per-frame metrics
        K_values: List of K values evaluated
        gen_len: Number of generated frames

    Returns:
        Updated metrics dictionary with aggregate values
    """
    for K in K_values:
        frame_mses = [all_metrics[f"eval/K{K}_frame{t}_mse"] for t in range(gen_len)]
        frame_cos_sims = [all_metrics[f"eval/K{K}_frame{t}_cosine_sim"] for t in range(gen_len)]

        all_metrics[f"eval/K{K}_final_mse"] = sum(frame_mses) / len(frame_mses)
        all_metrics[f"eval/K{K}_final_cosine_sim"] = sum(frame_cos_sims) / len(frame_cos_sims)

    return all_metrics


def log_evaluation(
    logger,
    metrics: dict,
    step: int,
    K_values: list[int],
    output_dir: Path | None,
    use_fsdp: bool,
) -> None:
    """
    Handle console printing and wandb logging.

    Args:
        logger: WandbLogger or None
        metrics: Dictionary of metrics
        step: Current training step
        K_values: List of K values evaluated
        output_dir: Output directory (for printing location)
        use_fsdp: Whether FSDP is being used
    """
    if logger:
        if use_fsdp and FSDP_AVAILABLE:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            mem_stats = get_memory_stats(local_rank)
            metrics.update({f"memory/{k}": v for k, v in mem_stats.items()})
        logger.log(metrics, step=step)

    print(f"\nEvaluation at step {step}:")
    for K in K_values:
        print(f"  K={K}:")
        print(f"    Final MSE: {metrics[f'eval/K{K}_final_mse']:.6f}")
        print(f"    Cosine Sim: {metrics[f'eval/K{K}_final_cosine_sim']:.4f}")
        if f"eval/K{K}_step0_mse" in metrics:
            print(f"    Step 0 MSE: {metrics[f'eval/K{K}_step0_mse']:.6f}")
            print(f"    Step {K-1} MSE: {metrics.get(f'eval/K{K}_step{K-1}_mse', 'N/A')}")

    if output_dir:
        print(f"  Outputs saved to: {output_dir}")
