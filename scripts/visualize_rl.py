import argparse
from pathlib import Path
import time

import gymnasium as gym
import numpy as np

import mujoco_rl  # noqa: F401


ALGORITHMS = ("ppo", "a2c", "sac", "td3", "ddpg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a trained SB3 policy")
    parser.add_argument(
        "--algorithm",
        choices=ALGORITHMS,
        default="ppo",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs") / "rl_benchmark" / "ppo" / "model.zip",
    )
    parser.add_argument("--env-id", default="MujocoReacher2D-v0")
    parser.add_argument(
        "--random",
        action="store_true",
        help="Run random actions instead of loading an SB3 model",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.speed <= 0:
        raise ValueError("--speed must be positive")
    if not args.random and not args.model.exists():
        raise FileNotFoundError(
            f"model not found: {args.model}; train first or pass --random"
        )

    env = gym.make(args.env_id, render_mode="human")
    model = None
    if not args.random:
        from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

        algorithm_types = {
            "ppo": PPO,
            "a2c": A2C,
            "sac": SAC,
            "td3": TD3,
            "ddpg": DDPG,
        }
        model = algorithm_types[args.algorithm].load(args.model)

    frame_interval = 1.0 / (float(env.metadata["render_fps"]) * args.speed)
    try:
        for episode in range(args.episodes):
            observation, info = env.reset(seed=args.seed + episode)
            episode_return = 0.0
            terminated = truncated = False
            next_frame = time.perf_counter()

            while not (terminated or truncated):
                if model is None:
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(
                    np.asarray(action),
                )
                episode_return += float(reward)

                next_frame += frame_interval
                sleep_time = next_frame - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_frame = time.perf_counter()

            print(
                f"episode {episode + 1}: return={episode_return:.2f} "
                f"distance={info['distance']:.3f} success={info['success']}"
            )
    except KeyboardInterrupt:
        print("\nvisualization stopped")
    finally:
        env.close()


if __name__ == "__main__":
    main()
