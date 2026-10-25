import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

import mujoco_rl  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a linear policy with Cross-Entropy Method (CEM)",
    )
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population-size", type=int, default=48)
    parser.add_argument("--elite-fraction", type=float, default=0.2)
    parser.add_argument("--episodes-per-policy", type=int, default=2)
    parser.add_argument("--episode-length", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs") / "cem_cartpole",
    )
    return parser.parse_args()


def rollout(policy: np.ndarray, env: gym.Env, seed: int, episode_length: int) -> float:
    weights = policy[:-1]
    bias = policy[-1]
    observation, _ = env.reset(seed=seed)
    total_reward = 0.0

    for _ in range(episode_length):
        action = float(observation @ weights + bias)
        observation, reward, terminated, truncated, _ = env.step(
            np.array([action], dtype=np.float32),
        )
        total_reward += float(reward)
        if terminated or truncated:
            break

    return total_reward


def evaluate(
    policy: np.ndarray,
    env: gym.Env,
    seed: int,
    episodes: int,
    episode_length: int,
) -> float:
    returns = [
        rollout(policy, env, seed + episode, episode_length)
        for episode in range(episodes)
    ]
    return float(np.mean(returns))


def main() -> None:
    args = parse_args()
    if args.population_size < 4 or not 0.0 < args.elite_fraction < 1.0:
        raise ValueError("use population_size >= 4 and 0 < elite_fraction < 1")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make("MujocoCartPole-v0")
    rng = np.random.default_rng(args.seed)

    # The policy is [w_cart, w_velocity, w_sin(angle), w_cos(angle), bias].
    mean = np.zeros(5, dtype=np.float64)
    std = np.ones(5, dtype=np.float64)
    elite_count = max(2, int(args.population_size * args.elite_fraction))
    best_return = -np.inf
    best_policy = mean.copy()
    history = []

    try:
        for generation in range(args.generations):
            candidates = rng.normal(mean, std, size=(args.population_size, 5))
            returns = np.array(
                [
                    evaluate(
                        candidate,
                        env,
                        seed=args.seed + 10_000 * generation + candidate_index,
                        episodes=args.episodes_per_policy,
                        episode_length=args.episode_length,
                    )
                    for candidate_index, candidate in enumerate(candidates)
                ],
            )
            elite_indices = np.argsort(returns)[-elite_count:]
            elites = candidates[elite_indices]
            mean = elites.mean(axis=0)
            std = np.maximum(elites.std(axis=0), 0.05)

            current_return = float(returns[elite_indices[-1]])
            history.append(
                {
                    "generation": generation + 1,
                    "mean_return": float(returns.mean()),
                    "best_return": current_return,
                }
            )
            print(
                f"generation {generation + 1:02d}/({args.generations}) "
                f"mean={returns.mean():7.2f} best={current_return:7.2f}"
            )

            if current_return > best_return:
                best_return = current_return
                best_policy = candidates[elite_indices[-1]].copy()
                np.savez(args.run_dir / "policy.npz", policy=best_policy)
    finally:
        env.close()

    metadata = {
        "best_return": best_return,
        "generations": args.generations,
        "population_size": args.population_size,
        "seed": args.seed,
        "history": history,
    }
    (args.run_dir / "training.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"best return: {best_return:.2f}")
    print(f"policy: {args.run_dir / 'policy.npz'}")


if __name__ == "__main__":
    main()
