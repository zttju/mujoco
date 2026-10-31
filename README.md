# MuJoCo RL Environment

这个目录包含 MuJoCo/Gymnasium 强化学习模板环境。当前有两个任务：

- `MujocoCartPole-v0`：小车倒立摆，适合验证环境流程。
- `MujocoReacher2D-v0`：平面二连杆机械臂 reacher 任务，适合作为真实机械臂控制模板。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

如果只做开发测试，也可以使用：

```bash
pip install -r requirements.txt
```

## 运行

无界面随机 rollout，用于确认环境可用：

```bash
python examples/random_rollout.py
```

带 MuJoCo 窗口渲染：

```bash
python -c "import gymnasium as gym, mujoco_rl; \
env = gym.make('MujocoCartPole-v0', render_mode='human'); \
env.reset(seed=0); \
env.step(env.action_space.sample()); \
env.close()"
```

## 训练

内置训练器使用 Cross-Entropy Method（CEM），只需要 NumPy，不需要 PyTorch。开始训练：

```bash
python scripts/train.py
```

快速冒烟训练：

```bash
python scripts/train.py --generations 8 --population-size 24
```

模型和训练记录会写入 `runs/cem_cartpole/`：

```text
runs/cem_cartpole/policy.npz
runs/cem_cartpole/training.json
```

增加 `--generations`、`--population-size` 或 `--episodes-per-policy` 通常能提升策略稳定性。

## 可视化

查看训练得到的模型：

```bash
python scripts/visualize.py
```

指定其他策略：

```bash
python scripts/visualize.py --model runs/cem_cartpole/policy.npz
```

不加载模型，先看随机动作：

```bash
python scripts/visualize.py --random
```

调整播放速度，例如半速慢放：

```bash
python scripts/visualize.py --speed 0.5
```

关闭窗口即可停止可视化。Linux 桌面环境需要有 OpenGL 支持；如果在 SSH 会话中运行，请先开启 X11 或 Wayland 转发。

## 机械臂环境

`MujocoReacher2D-v0` 控制两个旋转关节，把末端 fingertip 移动到随机目标点。动作是两关节的归一化力矩，观测包含关节角、角速度、目标坐标和末端误差。奖励由末端到目标的距离、动作能耗和成功奖励组成。

随机 rollout：

```bash
python -c "import gymnasium as gym, mujoco_rl; \
env = gym.make('MujocoReacher2D-v0', render_mode='human'); \
obs, info = env.reset(seed=0); \
env.step(env.action_space.sample()); \
env.close()"
```

环境单元测试：

```bash
pytest tests/test_reacher2d.py
```

## 常用强化学习算法对比

安装训练依赖。`stable-baselines3` 会安装 PyTorch，下载体积较大：

```bash
pip install -e ".[train]"
```

如果默认 PyPI 下载的 CUDA 版 PyTorch 太大，或网络下载失败，先安装官方 CPU 版 PyTorch：

```bash
pip install -r requirements-cpu-train.txt
pip install -e ".[train]"
```

训练并对比 PPO、SAC 和 TD3：

```bash
python scripts/train_rl.py
```

常用算法还支持 A2C 和 DDPG：

```bash
python scripts/train_rl.py --algorithms ppo a2c sac td3 ddpg
```

短训练用于检查链路：

```bash
python scripts/train_rl.py \
  --algorithms ppo sac \
  --total-timesteps 10000 \
  --eval-episodes 3
```

每个算法的结果保存在 `runs/rl_benchmark/<algorithm>/`，汇总结果在 `runs/rl_benchmark/benchmark.json`。指标包括 return、成功率、episode 长度和训练时间。

训练过程默认写入 TensorBoard 和周期评估文件：

```text
runs/rl_benchmark/ppo/events.out.tfevents.*   # reward/loss/熵/学习率曲线
runs/rl_benchmark/ppo/monitor/*.monitor.csv   # 训练 episode return 和长度
runs/rl_benchmark/ppo/evaluations.npz         # 定期评估的 reward/length/success
runs/rl_benchmark/ppo/best_model.zip          # 周期评估中的最好模型
```

启动训练曲线界面：

```bash
python scripts/train_rl.py --algorithms ppo sac td3 --total-timesteps 100000
.venv/bin/tensorboard --logdir runs/rl_benchmark --host 127.0.0.1 --port 6006
```

然后浏览器打开：

```text
http://127.0.0.1:6006
```

重点查看 `eval/mean_reward`、`eval/success_rate`、`rollout/ep_rew_mean`、`train/loss` 和 `time/fps`。如果 6006 端口被占用，改成 `--port 6007`。

查看训练后的机械臂策略：

```bash
python scripts/visualize_rl.py --algorithm ppo
python scripts/visualize_rl.py --algorithm sac --model runs/rl_benchmark/sac/model.zip
```

不加载策略，先查看随机动作：

```bash
python scripts/visualize_rl.py --random
```

## 修改成自己的环境

1. 替换 [cartpole.xml](mujoco_rl/envs/assets/cartpole.xml) 为你的 MuJoCo 模型。
2. 修改 `CartPoleEnv` 中的 `observation_space`、`action_space`、reward、termination 和 truncation 逻辑。
3. 在 `mujoco_rl/__init__.py` 中注册新的环境 ID。

标准调用方式：

```python
import gymnasium as gym
import mujoco_rl

env = gym.make("MujocoCartPole-v0")
obs, info = env.reset(seed=0)
```

如果要迁移到自己的机械臂，推荐以 `Reacher2DEnv` 为模板：

1. 把 [reacher2d.xml](mujoco_rl/envs/assets/reacher2d.xml) 替换成你的 MJCF 模型。
2. 根据 `model.nq`、`model.nv`、`model.nu` 自动生成 observation/action 维度。
3. 根据实际任务重写距离、抓取、避障或轨迹跟踪 reward。
4. 在 `mujoco_rl/__init__.py` 注册新的 `env_id`。
5. 用 `python scripts/train_rl.py --env-id YourRobot-v0 --algorithms ppo sac td3` 训练对比。
