import argparse
import json
import cv2
import numpy as np
import stable_retro as retro
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from wm.policies import RandomPolicy


def collect_episode(
    env,
    policy,
    episode_idx: int,
    output_dir: Path,
    fps: int = 30,
    max_steps: int = 1000,
    frame_height: int = 224,
    frame_width: int = 320
):
    """Collect a single episode and save video with metadata.

    Args:
        env: Gymnasium environment
        policy: Policy for action selection
        episode_idx: Episode number for naming
        output_dir: Base directory for saving data
        fps: Video frames per second
        max_steps: Maximum steps per episode
        frame_height: Target frame height
        frame_width: Target frame width

    Returns:
        dict: Episode metadata
    """
    # Create episode directory
    episode_dir = output_dir / "episodes" / f"{episode_idx:04d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    # Initialize video writer
    video_path = episode_dir / "video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        str(video_path),
        fourcc,
        fps,
        (frame_width, frame_height)
    )

    # Reset environment and policy
    obs, _ = env.reset()
    policy.reset(obs)

    # Tracking
    actions = []
    rewards = []
    terminated = False
    truncated = False
    step_count = 0
    total_reward = 0.0

    # Episode loop
    for _ in range(max_steps):
        # Get action from policy
        action = policy.get_action(obs)

        # Step environment
        obs, reward, terminated, truncated, _ = env.step(action)

        # Resize and write frame
        frame = cv2.resize(obs, (frame_width, frame_height))
        video_writer.write(frame)

        # Track metadata
        actions.append(action.tolist() if isinstance(action, np.ndarray) else action)
        rewards.append(float(reward))
        total_reward += reward
        step_count += 1

        # Check episode end
        if terminated or truncated:
            break

    # Release video writer
    video_writer.release()

    # Save metadata
    metadata = {
        "episode_id": episode_idx,
        "environment": "MortalKombatII-Genesis-v0",
        "num_steps": step_count,
        "total_reward": total_reward,
        "fps": fps,
        "frame_size": [frame_height, frame_width],
        "actions": actions,
        "rewards": rewards,
        "terminated": terminated,
        "truncated": truncated,
        "timestamp": datetime.now().isoformat(),
        "policy_type": "random",
    }

    metadata_path = episode_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Collect rollouts from MK2 environment"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of episodes to collect (default: 100)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/rollouts",
        help="Output directory for collected data (default: data/rollouts)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=19,
        help="Video frames per second (default: 60)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps per episode (default: 1000)"
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=224,
        help="Video frame height (default: 224)"
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=320,
        help="Video frame width (default: 320)"
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Mortal Kombat II Rollout Collection")
    print("=" * 60)
    print(f"Number of episodes: {args.num_episodes}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {args.seed}")
    print(f"Video settings: {args.frame_width}x{args.frame_height} @ {args.fps} fps")
    print(f"Max steps per episode: {args.max_steps}")
    print()

    # Create environment
    print("Creating environment...")
    env = retro.make(game='MortalKombatII-Genesis-v0')
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print()

    # Create policy
    print("Initializing random policy...")
    policy = RandomPolicy(env.action_space, seed=args.seed)
    print()

    # Collect episodes
    print("Collecting episodes...")
    all_metadata = []

    for episode_idx in tqdm(range(1, args.num_episodes + 1), desc="Episodes"):
        metadata = collect_episode(
            env=env,
            policy=policy,
            episode_idx=episode_idx,
            output_dir=output_dir,
            fps=args.fps,
            max_steps=args.max_steps,
            frame_height=args.frame_height,
            frame_width=args.frame_width
        )
        all_metadata.append(metadata)

    # Close environment
    env.close()

    # Save summary
    total_frames = sum(m["num_steps"] for m in all_metadata)
    total_reward = sum(m["total_reward"] for m in all_metadata)
    avg_episode_length = total_frames / len(all_metadata)
    avg_reward = total_reward / len(all_metadata)

    summary = {
        "num_episodes": args.num_episodes,
        "total_frames": total_frames,
        "total_reward": total_reward,
        "avg_episode_length": avg_episode_length,
        "avg_reward": avg_reward,
        "collection_timestamp": datetime.now().isoformat(),
        "config": {
            "seed": args.seed,
            "fps": args.fps,
            "max_steps": args.max_steps,
            "frame_size": [args.frame_height, args.frame_width],
            "environment": "MortalKombatII-Genesis-v0",
            "policy_type": "random"
        }
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print("Collection Summary")
    print("=" * 60)
    print(f"Total episodes: {args.num_episodes}")
    print(f"Total frames: {total_frames:,}")
    print(f"Average episode length: {avg_episode_length:.1f} steps")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Output directory: {output_dir}")
    print(f"Summary saved to: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
