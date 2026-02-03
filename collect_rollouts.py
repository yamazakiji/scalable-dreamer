import argparse
import json
import signal
import sys
import cv2
import numpy as np
import stable_retro as retro
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from wm.policies import RandomPolicy


def worker_collect_episode(args: tuple) -> dict:
    """Worker function for parallel episode collection.

    Each worker creates its own env and policy instances since retro
    environments are not thread-safe.

    Args:
        args: Tuple of (episode_idx, base_seed, output_dir, fps, max_steps,
                       frame_height, frame_width, game_name)

    Returns:
        dict: Episode metadata
    """
    (episode_idx, base_seed, output_dir, fps, max_steps,
     frame_height, frame_width, game_name) = args

    episode_seed = base_seed + episode_idx
    env = retro.make(game=game_name)
    policy = RandomPolicy(env.action_space, seed=episode_seed)

    try:
        return collect_episode(env, policy, episode_idx, Path(output_dir),
                               fps, max_steps, frame_height, frame_width)
    finally:
        env.close()


def collect_episodes_parallel(
    worker_args: list,
    num_workers: int,
    num_episodes: int
) -> list:
    """Collect episodes in parallel with progress tracking and graceful shutdown.

    Args:
        worker_args: List of argument tuples for worker_collect_episode
        num_workers: Number of worker processes
        num_episodes: Total number of episodes (for progress bar)

    Returns:
        list: Sorted list of episode metadata dictionaries
    """
    results = []
    shutdown_requested = False
    executor = None

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        if not shutdown_requested:
            shutdown_requested = True
            print("\nShutdown requested, waiting for current episodes to complete...")
        else:
            print("\nForce shutdown...")
            sys.exit(1)

    original_sigint = signal.signal(signal.SIGINT, signal_handler)
    original_sigterm = signal.signal(signal.SIGTERM, signal_handler)

    try:
        executor = ProcessPoolExecutor(max_workers=num_workers)
        futures = {executor.submit(worker_collect_episode, args): args[0]
                   for args in worker_args}

        with tqdm(total=num_episodes, desc="Episodes") as pbar:
            for future in as_completed(futures):
                if shutdown_requested:
                    break
                try:
                    metadata = future.result()
                    results.append(metadata)
                    pbar.update(1)
                except Exception as e:
                    episode_idx = futures[future]
                    print(f"\nError in episode {episode_idx}: {e}")

        if shutdown_requested:
            executor.shutdown(wait=False, cancel_futures=True)
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
        if executor is not None:
            executor.shutdown(wait=True)

    # Sort by episode_id for consistent ordering
    results.sort(key=lambda x: x["episode_id"])
    return results


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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)"
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    game_name = 'MortalKombatII-Genesis-v0'

    print("=" * 60)
    print("Mortal Kombat II Rollout Collection")
    print("=" * 60)
    print(f"Number of episodes: {args.num_episodes}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {args.seed}")
    print(f"Video settings: {args.frame_width}x{args.frame_height} @ {args.fps} fps")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Number of workers: {args.num_workers}")
    print()

    # Create environment to display info
    print("Creating environment...")
    env = retro.make(game=game_name)
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    env.close()
    print()

    # Collect episodes
    print("Collecting episodes...")

    if args.num_workers > 1:
        # Parallel collection
        worker_args = [
            (episode_idx, args.seed, str(output_dir), args.fps, args.max_steps,
             args.frame_height, args.frame_width, game_name)
            for episode_idx in range(1, args.num_episodes + 1)
        ]
        all_metadata = collect_episodes_parallel(
            worker_args, args.num_workers, args.num_episodes
        )
    else:
        # Sequential collection (original behavior)
        env = retro.make(game=game_name)
        policy = RandomPolicy(env.action_space, seed=args.seed)
        all_metadata = []

        for episode_idx in tqdm(range(1, args.num_episodes + 1), desc="Episodes"):
            # Use deterministic seeding per episode for consistency with parallel mode
            episode_seed = args.seed + episode_idx
            policy = RandomPolicy(env.action_space, seed=episode_seed)

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
            "environment": game_name,
            "policy_type": "random",
            "num_workers": args.num_workers
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
