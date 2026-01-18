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

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns:
            frames: [T, C, H, W] video sequence (float32, range [0, 1])
        """
        frames = torch.randn(
            self.sequence_length,
            self.channels,
            self.frame_size[0],
            self.frame_size[1]
        )
        return frames


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

        # Find video files (supports nested directories)
        self.video_paths = list(self.video_dir.glob("**/*.mp4"))
        self.video_paths.extend(self.video_dir.glob("**/*.avi"))

        if not self.video_paths:
            raise ValueError(f"No videos found in {video_dir}")

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load and return video sequence."""
        video_path = self.video_paths[idx]

        # Load video
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        for _ in range(self.sequence_length):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()

            # Resize and convert to RGB
            frame = cv2.resize(frame, self.frame_size[::-1])
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        # Convert to tensor [T, C, H, W]
        frames = np.stack(frames, axis=0)
        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)

        return frames
