import gymnasium as gym

import mujoco_rl  # noqa: F401


def main() -> None:
    env = gym.make("MujocoCartPole-v0")
    observation, info = env.reset(seed=42)

    for _ in range(100):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()

    env.close()
    print(f"final observation: {observation}")


if __name__ == "__main__":
    main()
