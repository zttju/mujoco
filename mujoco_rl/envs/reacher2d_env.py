from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


ASSET_PATH = Path(__file__).resolve().parent / "assets" / "reacher2d.xml"
SUCCESS_DISTANCE = 0.07


class Reacher2DEnv(gym.Env):
    """A two-joint reaching task used as a small robotic-arm template.

    The policy commands normalized joint torques. Each episode places a target
    within the reachable workspace, and reward decreases with the fingertip-to-
    target distance and torque use.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 100,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 200,
        torque_scale: float = 1.0,
    ) -> None:
        if render_mode not in [*self.metadata["render_modes"], None]:
            raise ValueError(f"unsupported render_mode: {render_mode}")
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if torque_scale <= 0:
            raise ValueError("torque_scale must be positive")

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.torque_scale = torque_scale
        self.model = mujoco.MjModel.from_xml_path(str(ASSET_PATH))
        self.data = mujoco.MjData(self.model)

        ctrl_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        ctrl_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.action_space = spaces.Box(ctrl_low, ctrl_high, dtype=np.float32)

        obs_dim = 2 * self.model.nq + self.model.nv + 4
        obs_high = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self._step_count = 0
        self._viewer = None
        self._renderer = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:] = self.np_random.uniform(-np.pi, np.pi, size=self.model.nq)
        self.data.qvel[:] = self.np_random.uniform(-0.2, 0.2, size=self.model.nv)

        requested_target = None
        if options is not None and "target_xy" in options:
            requested_target = np.asarray(options["target_xy"], dtype=np.float64)
            if requested_target.shape != (2,):
                raise ValueError("options['target_xy'] must have shape (2,)")
        else:
            radius = self.np_random.uniform(0.10, 0.30)
            angle = self.np_random.uniform(-np.pi, np.pi)
            requested_target = np.array(
                [radius * np.cos(angle), radius * np.sin(angle)]
            )

        if np.linalg.norm(requested_target) > 0.33:
            raise ValueError("target_xy must lie inside the reachable workspace")
        self.data.mocap_pos[0, :2] = requested_target
        self.data.mocap_pos[0, 2] = 0.08

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        return self._get_obs(), self._task_info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (self.model.nu,):
            raise ValueError(
                f"expected action with shape ({self.model.nu},), got {action.shape}"
            )
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.data.ctrl[:] = action * self.torque_scale
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        distance = self._distance()
        terminated = bool(distance < SUCCESS_DISTANCE)
        truncated = bool(self._step_count >= self.max_episode_steps)
        reward = -distance - 0.001 * float(np.dot(action, action))
        if terminated:
            reward += 1.0

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), terminated, truncated, self._task_info()

    def _distance(self) -> float:
        fingertip = self.data.site("fingertip").xpos
        target = self.data.site("target_site").xpos
        return float(np.linalg.norm(fingertip[:2] - target[:2]))

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos
        target = self.data.site("target_site").xpos
        fingertip = self.data.site("fingertip").xpos
        return np.concatenate(
            [
                np.sin(qpos),
                np.cos(qpos),
                self.data.qvel,
                target[:2],
                fingertip[:2] - target[:2],
            ],
            dtype=np.float32,
        )

    def _task_info(self) -> dict:
        distance = self._distance()
        target = self.data.site("target_site").xpos
        return {
            "distance": distance,
            "success": bool(distance < SUCCESS_DISTANCE),
            "target_xy": np.array(target[:2], dtype=np.float32),
        }

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
