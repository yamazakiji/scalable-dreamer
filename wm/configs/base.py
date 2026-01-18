"""Configuration classes with CLI + YAML override support."""
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import TypeVar
import yaml

T = TypeVar('T', bound='BaseConfig')


@dataclass
class BaseConfig:
    """Base configuration with serialization support."""

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    def to_yaml(self, path: Path):
        """Save config to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls: type[T], path: Path) -> T:
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls: type[T], data: dict) -> T:
        """Create config from dictionary, properly handling nested dataclasses."""
        kwargs = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            value = data[f.name]
            # Handle nested dataclass configs
            if hasattr(f.type, 'from_dict') and isinstance(value, dict):
                kwargs[f.name] = f.type.from_dict(value)
            else:
                kwargs[f.name] = value
        return cls(**kwargs)

    def update(self: T, **kwargs) -> T:
        """Return new config with updated values (for CLI overrides)."""
        data = self.to_dict()
        for key, value in kwargs.items():
            if value is not None:
                # Handle nested updates like "training.learning_rate"
                if '.' in key:
                    parts = key.split('.')
                    target = data
                    for part in parts[:-1]:
                        target = target[part]
                    target[parts[-1]] = value
                else:
                    data[key] = value
        return self.from_dict(data)


# =============================================================================
# Model Configs (one per model type)
# =============================================================================

@dataclass
class TokenizerConfig(BaseConfig):
    """Causal tokenizer model configuration."""
    # Input
    image_height: int = 224
    image_width: int = 320
    patch_size: int = 16

    # Architecture
    dim: int = 512
    dim_latent: int = 32
    num_latent_tokens: int = 64
    num_register_tokens: int = 8
    encoder_layers: int = 3
    decoder_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.0

    # Masking (training behavior)
    mask_ratio_min: float = 0.0
    mask_ratio_max: float = 0.9


# =============================================================================
# Training Config (reusable across models)
# =============================================================================

@dataclass
class TrainingConfig(BaseConfig):
    """Training hyperparameters."""
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100_000
    grad_clip: float = 1.0

    # Batching
    batch_size: int = 1
    num_workers: int = 0

    checkpoint_every: int = 10_000
    eval_every: int = 5_000
    log_every: int = 100
    save_dir: str = "outputs"

    # FSDP settings
    fsdp_mixed_precision: str = "bf16"
    resume_from: str | None = None


# =============================================================================
# Experiment Config (ties model + training together)
# =============================================================================

@dataclass
class ExperimentConfig(BaseConfig):
    """Complete experiment configuration."""
    name: str = "experiment"
    seed: int = 42

    # Sub-configs
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Logging
    use_wandb: bool = True
    wandb_project: str = "world_models"
