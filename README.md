# MuJoCo RL Environment

这个目录包含一个可直接用于强化学习的 Gymnasium/MuJoCo 模板环境。任务是控制小车让上方摆杆保持直立。

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
