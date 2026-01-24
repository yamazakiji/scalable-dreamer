"""I/O utilities for evaluation outputs."""
import json
from pathlib import Path

import torch


def save_frame_outputs(
    batch_dir: Path,
    t: int,
    z_target: torch.Tensor,
    frame_results: dict,
    K_values: list[int],
) -> None:
    """
    Save outputs for a single frame (target + predictions for each K).

    Args:
        batch_dir: Base directory for batch outputs (e.g., eval_step_100/batch_0)
        t: Frame index
        z_target: (B, 1, N, D) target latent
        frame_results: Dict mapping K -> {"intermediates", "final_metrics", "step_metrics"}
        K_values: List of K values evaluated
    """
    frame_dir = batch_dir / f"frame_{t}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Save target
    torch.save(z_target.cpu(), frame_dir / "z_target.pt")

    # Save results for each K
    for K in K_values:
        k_dir = frame_dir / f"K{K}"
        k_dir.mkdir(exist_ok=True)

        res = frame_results[K]

        # Save intermediates
        for inter in res["intermediates"]:
            torch.save(
                inter["z_pred"],
                k_dir / f"z_pred_step_{inter['step']}.pt"
            )

        # Save metrics
        metrics_to_save = {
            "final": res["final_metrics"],
            "per_step": res["step_metrics"],
        }
        with open(k_dir / "metrics.json", "w") as f:
            json.dump(metrics_to_save, f, indent=2)


def save_evaluation_summary(
    output_dir: Path,
    step: int,
    context_len: int,
    gen_len: int,
    K_values: list[int],
    metrics: dict,
) -> None:
    """
    Save summary metrics JSON.

    Args:
        output_dir: Directory to save summary (e.g., outputs/exp/eval/eval_step_100)
        step: Current training step
        context_len: Number of context frames used
        gen_len: Number of frames generated
        K_values: List of K values evaluated
        metrics: Dictionary of all metrics
    """
    summary = {
        "step": step,
        "context_len": context_len,
        "gen_len": gen_len,
        "K_values": K_values,
        "metrics": {k: v for k, v in metrics.items()},
    }
    with open(output_dir / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
