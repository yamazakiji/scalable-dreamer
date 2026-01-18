"""Abstract base class for policies."""
from abc import ABC, abstractmethod
import numpy as np


class Policy(ABC):
    """Abstract policy interface for action selection."""

    @abstractmethod
    def reset(self, obs: np.ndarray) -> None:
        """Reset policy state at episode start.

        Args:
            obs: Initial observation from environment
        """
        pass

    @abstractmethod
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Get action for current observation.

        Args:
            obs: Current observation from environment

        Returns:
            Action to take in environment
        """
        pass
