"""Distributed training utilities."""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed() -> tuple[int, int]:
    """
    Initialize distributed training.

    Returns:
        rank: Process rank
        world_size: Total number of processes
    """
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )

    return rank, world_size


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_ddp(
    model: torch.nn.Module,
    rank: int,
    find_unused_parameters: bool = False
) -> DDP:
    """Wrap model with DistributedDataParallel."""
    return DDP(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=True
    )
