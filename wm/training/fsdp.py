"""FSDP2 training utilities using PyTorch's fully_shard API."""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)


def get_mixed_precision_policy(dtype: str = "bf16") -> MixedPrecisionPolicy | None:
    """Create mixed precision policy for FSDP2."""
    if dtype == "bf16":
        return MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    elif dtype == "fp16":
        return MixedPrecisionPolicy(param_dtype=torch.float16, reduce_dtype=torch.float32)
    return None


def apply_fsdp(
    model: nn.Module,
    mixed_precision: str = "bf16",
) -> nn.Module:
    """
    Apply FSDP2 sharding to model.

    Shards transformer blocks first, then root model (required by FSDP2 API).
    """
    from wm.models.transformer.block import TransformerBlock

    mp_policy = get_mixed_precision_policy(mixed_precision)
    fsdp_kwargs = {"mp_policy": mp_policy} if mp_policy else {}

    # Shard transformer blocks first (must be done before root)
    for module in model.modules():
        if isinstance(module, TransformerBlock):
            fully_shard(module, **fsdp_kwargs)

    # Shard root model
    fully_shard(model, **fsdp_kwargs)

    return model


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    epoch: int,
    save_path: str,
    rank: int,
    config: dict | None = None,
):
    """Save checkpoint using DCP API (full state dict gathered to rank 0)."""
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    model_sd = get_model_state_dict(model, options=options)
    optim_sd = get_optimizer_state_dict(model, optimizer, options=options)

    if rank == 0:
        ckpt = {
            "step": step,
            "epoch": epoch,
            "model_state_dict": model_sd,
            "optimizer_state_dict": optim_sd,
            "scheduler_state_dict": scheduler.state_dict(),
        }
        if config:
            ckpt["config"] = config
        torch.save(ckpt, save_path)
        print(f"Saved checkpoint to {save_path}")

    if dist.is_initialized():
        dist.barrier()


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    checkpoint_path: str,
    rank: int,
) -> tuple[int, int]:
    """Load checkpoint using DCP API."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False) if rank == 0 else None

    if dist.is_initialized():
        ckpt_list = [ckpt]
        dist.broadcast_object_list(ckpt_list, src=0)
        ckpt = ckpt_list[0]

    options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
    set_model_state_dict(model, ckpt["model_state_dict"], options=options)
    set_optimizer_state_dict(model, optimizer, ckpt["optimizer_state_dict"], options=options)
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return ckpt["step"], ckpt["epoch"]


def get_memory_stats(device: int) -> dict:
    """Get GPU memory statistics in GB."""
    if torch.cuda.is_available():
        return {
            "memory_allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
            "memory_reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
            "max_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        }
    return {}
