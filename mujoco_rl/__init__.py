from gymnasium.envs.registration import register

register(
    id="MujocoCartPole-v0",
    entry_point="mujoco_rl.envs:CartPoleEnv",
    max_episode_steps=250,
)
