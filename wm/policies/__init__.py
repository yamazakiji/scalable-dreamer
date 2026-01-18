"""Policy implementations for rollout collection."""
from .base import Policy
from .random_policy import RandomPolicy

__all__ = ['Policy', 'RandomPolicy']
