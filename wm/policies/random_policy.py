import numpy as np
from gymnasium.spaces import Space

from wm.policies.base import Policy


class RandomPolicy(Policy):
    """Policy that samples random actions from the action space."""

    def __init__(self, action_space: Space, seed: int = None):
        """Initialize random policy.

        Args:
            action_space: Gymnasium action space to sample from
            seed: Random seed for reproducibility
        """
        self.action_space = action_space
        self.rng = np.random.RandomState(seed)
        # Seed the action space as well for consistent sampling
        if seed is not None:
            self.action_space.seed(seed)

    def reset(self, obs: np.ndarray) -> None:
        """Reset policy state (no-op for stateless random policy).

        Args:
            obs: Initial observation (unused)
        """
        pass

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Sample random action from action space.

        Args:
            obs: Current observation (unused)

        Returns:
            Random action sampled from action space
        """
        return self.action_space.sample()
