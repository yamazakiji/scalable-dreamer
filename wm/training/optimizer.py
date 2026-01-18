"""Optimizer and scheduler utilities."""
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def get_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.999)
) -> AdamW:
    """Create AdamW optimizer."""
    return AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=betas
    )


def get_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int
) -> LambdaLR:
    """
    Create learning rate scheduler with linear warmup and cosine decay.

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        max_steps: Total training steps

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())

    return LambdaLR(optimizer, lr_lambda)
