from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces


ASSET_PATH = Path(__file__).resolve().parent / "assets" / "cartpole.xml"


class CartPoleEnv(gym.Env):
    """A small balance task used as a template for custom MuJoCo RL models.

    Actions are motor signals in ``[-1, 1]``. The actuator converts them into
    force on the cart. Episodes end when the cart leaves the track or the pole
    falls too far from vertical.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 250,
    ) -> None:
        if render_mode not in [*self.metadata["render_modes"], None]:
            raise ValueError(f"unsupported render_mode: {render_mode}")

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.model = mujoco.MjModel.from_xml_path(str(ASSET_PATH))
        self.data = mujoco.MjData(self.model)

        obs_high = np.full(4, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        self._step_count = 0
        self._viewer = None
        self._renderer = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = np.array([0.0, 0.05])
        self.data.qvel[:] = self.np_random.uniform(-0.05, 0.05, size=2)
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (1,):
            raise ValueError(f"expected action with shape (1,), got {action.shape}")
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        cart_pos = float(self.data.qpos[0])
        pole_angle = float(self.data.qpos[1])
        action_value = float(action[0])
        terminated = bool(abs(cart_pos) > 2.0 or abs(pole_angle) > 0.35)
        truncated = bool(self._step_count >= self.max_episode_steps)
        reward = (
            1.0
            - 0.01 * cart_pos**2
            - 0.001 * pole_angle**2
            - 0.0001 * action_value**2
        )

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), terminated, truncated, {}

    def _get_obs(self) -> np.ndarray:
        pole_angle = float(self.data.qpos[1])
        return np.array(
            [
                self.data.qpos[0],
                self.data.qvel[0],
                np.sin(pole_angle),
                np.cos(pole_angle),
            ],
            dtype=np.float32,
        )

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None

        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()

        return None

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
