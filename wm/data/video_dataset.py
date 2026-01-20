import json

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import cv2


class DummyVideoDataset(Dataset):
    """
    Dummy dataset that generates random video sequences. for testing purposes only
    """

    def __init__(
        self,
        num_samples: int = 1000,
        sequence_length: int = 64,
        frame_size: tuple[int, int] = (224, 320),
        channels: int = 3
    ):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.channels = channels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            frames: [T, C, H, W] video sequence (float32, range [0, 1])
            actions: [T, 12] random action tensor
        """
        frames = torch.randn(
            self.sequence_length,
            self.channels,
            self.frame_size[0],
            self.frame_size[1]
        )
        actions = torch.randint(0, 2, (self.sequence_length, 12), dtype=torch.int32)
        return frames, actions


class VideoDataset(Dataset):
    """
    Video dataset for loading real video files.
    maybe i need to revisit it sometime
    """
    def __init__(
        self,
        video_dir: str,
        sequence_length: int = 384,
        frame_size: tuple[int, int] = (224, 224),
    ):
        self.video_dir = Path(video_dir)
        self.sequence_length = sequence_length
        self.frame_size = frame_size

        # Discover episode folders: {video_dir}/rollouts/episodes/{NNNN}/
        episodes_dir = self.video_dir / "rollouts" / "episodes"
        if not episodes_dir.exists():
            raise ValueError(f"Episodes directory not found: {episodes_dir}")

        # Find all numbered folders containing video.mp4 and metadata.json
        self.episode_paths: list[Path] = []
        for folder in sorted(episodes_dir.iterdir()):
            if folder.is_dir():
                video_path = folder / "video.mp4"
                metadata_path = folder / "metadata.json"
                if video_path.exists() and metadata_path.exists():
                    self.episode_paths.append(folder)

        if not self.episode_paths:
            raise ValueError(f"No valid episodes found in {episodes_dir}")

    def __len__(self) -> int:
        return len(self.episode_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and return video sequence with corresponding actions.

        Returns:
            frames: (T, C, H, W) float32 tensor normalized to [0, 1]
            actions: (T, 12) int32 tensor of boolean actions
        """
        episode_path = self.episode_paths[idx]
        video_path = episode_path / "video.mp4"
        metadata_path = episode_path / "metadata.json"

        # Load metadata for actions
        with open(metadata_path) as f:
            metadata = json.load(f)

        actions_list = metadata["actions"]
        num_video_frames = len(actions_list)

        # Load video frames
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        actions = []
        frame_idx = 0

        for _ in range(self.sequence_length):
            ret, frame = cap.read()
            if not ret:
                # Loop video back to beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                frame_idx = 0

            # Resize frame (frame_size is (H, W), cv2 expects (W, H))
            frame = cv2.resize(frame, self.frame_size[::-1])
            frames.append(frame)

            # Get corresponding action (loop if video loops)
            action_idx = frame_idx % num_video_frames
            actions.append(actions_list[action_idx])
            frame_idx += 1

        cap.release()

        # Convert frames to tensor [T, C, H, W]
        frames = np.stack(frames, axis=0)
        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)

        # Convert actions to tensor [T, 12]
        actions = torch.tensor(actions, dtype=torch.int32)

        return frames, actions
