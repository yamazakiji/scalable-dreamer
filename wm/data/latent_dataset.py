"""Dataset for loading pre-computed latent sequences."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class LatentDataset(Dataset):
    """
    Dataset for loading latent sequences from pre-computed episodes.

    Each __getitem__ loads one episode and samples a random sequence from it.
    Memory efficient: only one episode in memory per worker at a time.

    Example:
        dataset = LatentDataset("./data/latents", sequence_length=16)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        for batch in dataloader:
            z = batch["z"]        # (B, seq_len, num_latents, latent_dim)
            a = batch["a"]        # (B, seq_len, action_dim)
            z_next = batch["z_next"]  # (B, seq_len, num_latents, latent_dim)
    """

    def __init__(
        self,
        latent_dir: str,
        sequence_length: int = 128,
    ):
        """
        Args:
            latent_dir: Directory containing extracted latents with episodes/ subdir
            sequence_length: Number of consecutive transitions per sample
        """
        self.latent_dir = Path(latent_dir)
        self.sequence_length = sequence_length

        # Load metadata
        metadata_path = self.latent_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        # Find episode files
        episodes_dir = self.latent_dir / "episodes"
        self.episode_files = sorted(episodes_dir.glob("*.pt"))

        if not self.episode_files:
            raise ValueError(f"No episode files found in {episodes_dir}")

        # Get episode lengths from metadata or by loading
        self._init_episode_info()

    def _init_episode_info(self):
        """Initialize episode length info from metadata or by loading episodes."""
        if "episode_lengths" in self.metadata:
            # Use metadata - no episode loading needed
            lengths = self.metadata["episode_lengths"]
            self.episode_lengths = [
                lengths.get(f.name, lengths.get(str(i), 0))
                for i, f in enumerate(self.episode_files)
            ]
        else:
            # Fallback: load each episode once to get lengths
            self.episode_lengths = []
            for ep_file in self.episode_files:
                ep = torch.load(ep_file, weights_only=True)
                self.episode_lengths.append(ep["num_transitions"])

        # Filter to episodes with enough transitions for sequence_length
        self.valid_episodes = [
            i for i, length in enumerate(self.episode_lengths)
            if length >= self.sequence_length
        ]

        if not self.valid_episodes:
            raise ValueError(
                f"No episodes have >= {self.sequence_length} transitions"
            )

    def __len__(self) -> int:
        return len(self.valid_episodes)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Load episode and return a random sequence.

        Returns:
            dict with:
                z: (seq_len, num_latents, latent_dim) current latents
                a: (seq_len, action_dim) or (seq_len,) actions
                z_next: (seq_len, num_latents, latent_dim) next latents
        """
        ep_idx = self.valid_episodes[idx]
        episode = torch.load(self.episode_files[ep_idx], weights_only=True)

        num_trans = episode["num_transitions"]
        max_start = num_trans - self.sequence_length

        # Random start position within episode
        start = torch.randint(0, max_start + 1, (1,)).item()
        end = start + self.sequence_length

        return {
            "z": episode["z"][start:end],
            "a": episode["a"][start:end].float(),
            "z_next": episode["z_next"][start:end],
        }

    @property
    def latent_shape(self) -> tuple[int, int]:
        """Return (num_latents, latent_dim)."""
        if "latent_shape" in self.metadata:
            return tuple(self.metadata["latent_shape"])
        sample = self[0]
        return tuple(sample["z"].shape[1:])

    @property
    def action_dim(self) -> int | None:
        """Return action dimension, or None if scalar."""
        if "action_dim" in self.metadata:
            return self.metadata["action_dim"]
        sample = self[0]
        return sample["a"].shape[-1] if sample["a"].dim() > 1 else None
