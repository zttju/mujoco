import numpy as np

import mujoco_rl  # noqa: F401
from mujoco_rl.envs import CartPoleEnv


def test_reset_and_step_shapes() -> None:
    env = CartPoleEnv()
    observation, _ = env.reset(seed=7)

    assert observation.shape == env.observation_space.shape
    assert env.action_space.contains(np.array([0.0], dtype=np.float32))

    observation, reward, terminated, truncated, _ = env.step(np.array([0.0]))
    assert observation.shape == (4,)
    assert reward >= 0.0
    assert not terminated
    assert not truncated

    env.close()
