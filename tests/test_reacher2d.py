import numpy as np

from mujoco_rl.envs import Reacher2DEnv


def test_reset_step_shapes() -> None:
    env = Reacher2DEnv()
    observation, info = env.reset(seed=3)

    assert observation.shape == env.observation_space.shape
    assert env.action_space.shape == (2,)
    assert 0.0 <= info["distance"] <= 0.5

    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    assert observation.shape == (10,)
    assert isinstance(reward, float)
    assert isinstance(info["success"], bool)

    env.close()


def test_observation_space_matches_reset() -> None:
    env = Reacher2DEnv()
    observation, _ = env.reset(seed=4)

    assert observation.shape == env.observation_space.shape
    assert env.observation_space.contains(observation)
    env.close()


def test_fixed_target_option() -> None:
    env = Reacher2DEnv()
    _, first = env.reset(seed=1, options={"target_xy": [0.15, 0.0]})
    _, second = env.reset(seed=2, options={"target_xy": [0.15, 0.0]})

    assert np.allclose(first["target_xy"], second["target_xy"])
    env.close()
