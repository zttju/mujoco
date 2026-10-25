import argparse
from pathlib import Path
import time

import gymnasium as gym
import numpy as np

import mujoco_rl  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the MuJoCo CartPole")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs") / "cem_cartpole" / "policy.npz",
        help="Path to policy.npz created by scripts/train.py",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Run random actions instead of a trained policy",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed: 1.0 is real time, 0.5 is half speed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.random and not args.model.exists():
        raise FileNotFoundError(
            f"policy not found: {args.model}; train first or pass --random",
        )

    env = gym.make("MujocoCartPole-v0", render_mode="human")
    policy = None if args.random else np.load(args.model)["policy"]
    if args.speed <= 0:
        raise ValueError("--speed must be greater than zero")

    frame_interval = 1.0 / (float(env.metadata["render_fps"]) * args.speed)

    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            episode_reward = 0.0
            terminated = truncated = False
            next_frame = time.perf_counter()

            while not (terminated or truncated):
                if policy is None:
                    action = env.action_space.sample()
                else:
                    action = np.array(
                        [observation @ policy[:-1] + policy[-1]],
                        dtype=np.float32,
                    )
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_reward += float(reward)

                # launch_passive.sync() does not enforce real-time playback.
                next_frame += frame_interval
                sleep_time = next_frame - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_frame = time.perf_counter()

            print(f"episode {episode + 1}: reward={episode_reward:.2f}")
    except KeyboardInterrupt:
        print("\nvisualization stopped")
    finally:
        env.close()


if __name__ == "__main__":
    main()
