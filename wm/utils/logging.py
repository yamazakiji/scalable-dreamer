"""Logging utilities with wandb integration."""
import wandb
from typing import Dict, Any, Optional
from pathlib import Path


class WandbLogger:
    """Wrapper for wandb logging."""

    def __init__(
        self,
        project: str,
        name: str,
        config: Dict[str, Any],
        entity: Optional[str] = None,
        enabled: bool = True,
        **kwargs
    ):
        self.enabled = enabled
        if self.enabled:
            self.run = wandb.init(
                project=project,
                name=name,
                config=config,
                entity=entity,
                **kwargs
            )
        else:
            self.run = None

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to wandb."""
        if self.enabled:
            wandb.log(metrics, step=step)

    def finish(self):
        """Finish wandb run."""
        if self.enabled and self.run is not None:
            wandb.finish()

    def watch(self, model, log_freq: int = 1000):
        """Watch model gradients."""
        if self.enabled:
            wandb.watch(model, log_freq=log_freq)
