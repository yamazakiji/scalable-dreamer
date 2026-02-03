"""Evaluation helper functions for dynamics model."""
import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

from wm.models.dynamics.inference import denoise_frame
from wm.utils.metrics import compute_latent_metrics

# FSDP memory stats (optional)
try:
    from wm.training.fsdp import get_memory_stats
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False


@torch.no_grad()
def evaluate_full(
    model: nn.Module,
    eval_dataloader: DataLoader,
    device: torch.device,
    context_len: int = 4,
    num_gen_frames: int = 8,
    K: int = 4,
    context_noise: float = 0.1,
) -> dict:
    """
    Full evaluation over the entire eval dataloader.

    For each batch, generates frames autoregressively using K-step denoising
    (matching the paper's inference procedure) and compares against ground truth.

    Args:
        model: DynamicsModel
        eval_dataloader: DataLoader for evaluation data
        device: Device to run on
        context_len: Number of GT context frames
        num_gen_frames: Number of frames to generate autoregressively
        K: Number of denoising steps (paper uses 4)
        context_noise: Noise level for context frames (tau_ctx)

    Returns:
        Dictionary of aggregated metrics (overall and per-frame)
    """
    model.eval()

    # Per-frame accumulators
    frame_mse: list[float] = [0.0] * num_gen_frames
    frame_cosine_sim: list[float] = [0.0] * num_gen_frames
    frame_relative_error: list[float] = [0.0] * num_gen_frames
    frame_counts: list[int] = [0] * num_gen_frames
    total_sequences = 0

    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        z = batch["z"]        # (B, T, N, D)
        a = batch["a"]        # (B, T)
        z_next = batch["z_next"]  # (B, T, N, D)

        B, T, N, D = z.shape
        ctx_len = min(context_len, T - 1)
        gen_len = min(num_gen_frames, T - ctx_len)

        if gen_len <= 0:
            continue

        # Context from GT with noise
        z_ctx = z[:, :ctx_len]  # (B, ctx_len, N, D)
        noise = torch.randn_like(z_ctx)
        z_ctx_noisy = (1 - context_noise) * z_ctx + context_noise * noise

        z_history = z_ctx_noisy.clone()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for t in range(gen_len):
                frame_idx = ctx_len + t
                action_t = a[:, frame_idx:frame_idx + 1]  # (B, 1)
                z_target = z_next[:, frame_idx:frame_idx + 1]  # (B, 1, N, D)

                # Sliding window context
                ctx_window = z_history[:, -context_len:]

                intermediates = denoise_frame(
                    model=model,
                    z_context_noisy=ctx_window,
                    z_target_shape=(B, 1, N, D),
                    action_t=action_t,
                    K=K,
                    context_noise=context_noise,
                    device=device,
                )

                z_pred = intermediates[-1]["z_pred"].to(device)

                # Compute metrics vs GT
                metrics = compute_latent_metrics(z_pred, z_target)
                frame_mse[t] += metrics["mse"] * B
                frame_cosine_sim[t] += metrics["cosine_sim"] * B
                frame_relative_error[t] += metrics["relative_error"] * B
                frame_counts[t] += B

                # AR: append prediction to history (with context noise)
                pred_noise = torch.randn_like(z_pred)
                z_pred_noisy = (1 - context_noise) * z_pred + context_noise * pred_noise
                z_history = torch.cat([z_history, z_pred_noisy], dim=1)

        total_sequences += B

    # Convert to tensors for potential distributed reduction
    counts_tensor = torch.tensor(frame_counts, dtype=torch.float64, device=device)
    mse_tensor = torch.tensor(frame_mse, dtype=torch.float64, device=device)
    cosine_tensor = torch.tensor(frame_cosine_sim, dtype=torch.float64, device=device)
    relerr_tensor = torch.tensor(frame_relative_error, dtype=torch.float64, device=device)
    total_seq_tensor = torch.tensor([total_sequences], dtype=torch.float64, device=device)

    # Distributed aggregation
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(mse_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(cosine_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(relerr_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_seq_tensor, op=dist.ReduceOp.SUM)

    # Compute averages
    result = {}
    overall_mse = 0.0
    overall_cosine = 0.0
    overall_relerr = 0.0
    num_frames_with_data = 0

    for t in range(num_gen_frames):
        count = counts_tensor[t].item()
        if count > 0:
            result[f"eval/frame_{t}_mse"] = mse_tensor[t].item() / count
            result[f"eval/frame_{t}_cosine_sim"] = cosine_tensor[t].item() / count
            result[f"eval/frame_{t}_relative_error"] = relerr_tensor[t].item() / count
            overall_mse += result[f"eval/frame_{t}_mse"]
            overall_cosine += result[f"eval/frame_{t}_cosine_sim"]
            overall_relerr += result[f"eval/frame_{t}_relative_error"]
            num_frames_with_data += 1

    if num_frames_with_data > 0:
        result["eval/mse"] = overall_mse / num_frames_with_data
        result["eval/cosine_sim"] = overall_cosine / num_frames_with_data
        result["eval/relative_error"] = overall_relerr / num_frames_with_data
    else:
        result["eval/mse"] = 0.0
        result["eval/cosine_sim"] = 0.0
        result["eval/relative_error"] = 0.0

    result["eval/num_sequences"] = int(total_seq_tensor[0].item())

    model.train()
    return result


def log_evaluation(
    logger,
    metrics: dict,
    step: int,
    use_fsdp: bool = False,
) -> None:
    """
    Log evaluation metrics to console and wandb.

    Args:
        logger: WandbLogger or None
        metrics: Dictionary of evaluation metrics from evaluate_full
        step: Current training step
        use_fsdp: Whether FSDP is being used (for memory stats)
    """
    if logger:
        log_dict = dict(metrics)
        if use_fsdp and FSDP_AVAILABLE:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            mem_stats = get_memory_stats(local_rank)
            log_dict.update({f"memory/{k}": v for k, v in mem_stats.items()})
        logger.log(log_dict, step=step)

    print(f"\nEvaluation at step {step}:")
    print(f"  Overall MSE: {metrics.get('eval/mse', 0):.6f}")
    print(f"  Overall Cosine Sim: {metrics.get('eval/cosine_sim', 0):.4f}")
    print(f"  Overall Relative Error: {metrics.get('eval/relative_error', 0):.4f}")
    print(f"  Sequences evaluated: {metrics.get('eval/num_sequences', 0)}")

    # Per-frame breakdown
    t = 0
    while f"eval/frame_{t}_mse" in metrics:
        print(
            f"  Frame {t}: MSE={metrics[f'eval/frame_{t}_mse']:.6f} "
            f"CosSim={metrics[f'eval/frame_{t}_cosine_sim']:.4f} "
            f"RelErr={metrics[f'eval/frame_{t}_relative_error']:.4f}"
        )
        t += 1
