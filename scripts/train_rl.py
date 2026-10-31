import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback

import mujoco_rl  # noqa: F401


ALGORITHMS = ("ppo", "a2c", "sac", "td3", "ddpg")
ON_POLICY = {"ppo", "a2c"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare continuous-control RL algorithms",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=ALGORITHMS,
        default=["ppo", "sac", "td3"],
    )
    parser.add_argument("--env-id", default="MujocoReacher2D-v0")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
        help="number of environment steps between evaluations",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="disable TensorBoard logs (terminal CSV logs are still written)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs") / "rl_benchmark",
    )
    return parser.parse_args()


def algorithm_kwargs(algorithm: str, seed: int) -> dict:
    common = {"seed": seed, "verbose": 1}
    if algorithm == "ppo":
        return common | {
            "learning_rate": 3e-4,
            "n_steps": 256,
            "batch_size": 256,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
        }
    if algorithm == "a2c":
        return common | {
            "learning_rate": 7e-4,
            "n_steps": 64,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.01,
        }

    # Deterministic off-policy methods are generally more stable with one
    # environment and a replay buffer.
    return common | {
        "learning_rate": 1e-3,
        "buffer_size": 300_000,
        "batch_size": 256,
        "learning_starts": 1_000,
        "gamma": 0.99,
        "tau": 0.005,
    }


class ReacherEvalCallback(EvalCallback):
    """Evaluate and log the env's Gymnasium-style ``success`` field."""

    def _log_success_callback(self, locals_, globals_) -> None:
        info = locals_["info"]
        success = info.get("is_success", info.get("success"))
        if locals_["done"] and success is not None:
            self._is_success_buffer.append(bool(success))


def train_one(
    algorithm: str,
    env_id: str,
    total_timesteps: int,
    requested_n_envs: int,
    eval_episodes: int,
    eval_freq: int,
    tensorboard: bool,
    seed: int,
    run_dir: Path,
) -> dict:
    from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.env_util import make_vec_env

    algorithm_types = {
        "ppo": PPO,
        "a2c": A2C,
        "sac": SAC,
        "td3": TD3,
        "ddpg": DDPG,
    }

    algorithm_dir = run_dir / algorithm
    algorithm_dir.mkdir(parents=True, exist_ok=True)
    n_envs = requested_n_envs if algorithm in ON_POLICY else 1
    monitor_dir = algorithm_dir / "monitor"
    env = make_vec_env(env_id, n_envs=n_envs, seed=seed, monitor_dir=str(monitor_dir))
    eval_env = make_vec_env(
        env_id,
        n_envs=1,
        seed=seed + 5_000,
        monitor_dir=str(algorithm_dir / "eval_monitor"),
    )
    model_kwargs = algorithm_kwargs(algorithm, seed)
    if tensorboard:
        model_kwargs["tensorboard_log"] = str(algorithm_dir)

    started = time.perf_counter()
    try:
        callback = ReacherEvalCallback(
            eval_env,
            log_path=str(algorithm_dir),
            best_model_save_path=str(algorithm_dir),
            n_eval_episodes=eval_episodes,
            eval_freq=max(1, eval_freq // n_envs),
            deterministic=True,
            verbose=1,
        )
        model = algorithm_types[algorithm](
            "MlpPolicy",
            env,
            **model_kwargs,
        )
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False,
        )
        model.save(algorithm_dir / "model")
    finally:
        env.close()
        eval_env.close()

    eval_env = gym.make(env_id)
    episode_returns = []
    episode_lengths = []
    successes = []
    try:
        for episode in range(eval_episodes):
            observation, _ = eval_env.reset(seed=seed + 10_000 + episode)
            episode_return = 0.0
            episode_length = 0
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = eval_env.step(
                    np.asarray(action),
                )
                episode_return += float(reward)
                episode_length += 1
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            successes.append(bool(info["success"]))
    finally:
        eval_env.close()

    result = {
        "algorithm": algorithm,
        "env_id": env_id,
        "total_timesteps": total_timesteps,
        "train_seconds": time.perf_counter() - started,
        "eval_episodes": eval_episodes,
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "mean_length": float(np.mean(episode_lengths)),
        "success_rate": float(np.mean(successes)),
        "model_path": str(algorithm_dir / "model.zip"),
        "tensorboard_log": str(algorithm_dir) if tensorboard else None,
        "evaluations": str(algorithm_dir / "evaluations.npz"),
    }
    (algorithm_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    args = parse_args()
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive")
    if args.eval_episodes <= 0:
        raise ValueError("--eval-episodes must be positive")
    if args.eval_freq <= 0:
        raise ValueError("--eval-freq must be positive")
    if args.n_envs <= 0:
        raise ValueError("--n-envs must be positive")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    tensorboard = not args.no_tensorboard
    results = []
    for algorithm in args.algorithms:
        print(f"\n=== training {algorithm.upper()} ===")
        results.append(
            train_one(
                algorithm=algorithm,
                env_id=args.env_id,
                total_timesteps=args.total_timesteps,
                requested_n_envs=args.n_envs,
                eval_episodes=args.eval_episodes,
                eval_freq=args.eval_freq,
                tensorboard=tensorboard,
                seed=args.seed,
                run_dir=args.run_dir,
            )
        )

    (args.run_dir / "benchmark.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n=== benchmark summary ===")
    for result in results:
        print(
            f"{result['algorithm']:>4}: return={result['mean_return']:8.2f} "
            f"success={result['success_rate']:.0%} "
            f"time={result['train_seconds']:.1f}s"
        )
    print(f"results: {args.run_dir / 'benchmark.json'}")


if __name__ == "__main__":
    main()
