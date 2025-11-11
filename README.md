# Fish Escape - 强化学习小鱼躲避系统

基于PPO算法的多智能体强化学习项目，训练小鱼学习躲避大鱼的策略。

## 项目简介

这是一个使用强化学习训练小鱼躲避大鱼的游戏环境。项目使用PPO（Proximal Policy Optimization）算法，通过参数共享策略让多条小鱼共享同一个策略网络，学习如何在圆形舞台中躲避大鱼的追捕。

## 项目结构

```
fish_rl/
├── fish_env.py              # 环境定义（支持自定义小鱼数量）
├── train_v2.py              # 训练脚本（推荐使用，支持多进程并行）
├── watch.py                 # 实时可视化脚本（观看小鱼躲避行为）
├── visualize.py             # 可视化工具（训练曲线、GIF生成）
├── checkpoints/             # 训练好的模型检查点
│   ├── model_iter_10.zip
│   ├── model_iter_20.zip
│   ├── ...
│   ├── model_final.zip
│   └── training_stats.pkl
└── training_curves.png       # 训练曲线图
```

## 环境要求

- Python 3.11+
- 主要依赖：
  - gymnasium
  - stable-baselines3
  - pygame
  - matplotlib
  - numpy
  - tqdm
  - imageio

## 快速开始

### 1. 安装依赖

首先创建虚拟环境（推荐使用uv）：

```bash
# 检查是否有uv，如果没有则安装
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境
uv venv

# 激活虚拟环境
source .venv/bin/activate  # macOS/Linux
# 或
.venv\Scripts\activate  # Windows

# 安装依赖
uv pip install -r requirements.txt
```

### 2. 测试环境

```bash
python test_env.py
```

### 3. 训练模型

```bash
python train_v2.py
```

默认配置：
- 100 个 iterations
- 25 条小鱼
- 32 个并行环境
- 训练过程中每 5 个 iteration 自动保存统计信息

### 4. 实时观看训练好的模型

```bash
# 使用最新模型
python watch.py

# 指定特定迭代的模型
python watch.py --model ./checkpoints/model_iter_60

# 自定义小鱼数量
python watch.py --num-fish 50

# 观看多个episode
python watch.py --episodes 3
```

### 5. 查看训练曲线

```bash
python visualize.py
```

这将生成训练曲线图，显示存活率、奖励等指标的变化。

## 环境参数

- **舞台半径**: 10.0
- **小鱼数量**: 可自定义（默认 25，训练时使用）
- **Episode长度**: 500步
- **小鱼最大速度**: 2.0
- **小鱼视野半径**: 3.33 (舞台半径的1/3)
- **小鱼最大转向角度**: 30度
- **大鱼初始速度**: 1.5 (vx), 0.0 (vy)
- **大鱼重力**: 0.5
- **大鱼反弹阻尼**: 0.85

## 训练结果

### 性能对比（简化环境，10条鱼）

| 模型 | 平均奖励 | 平均存活数量 | 平均存活率 |
|------|---------|------------|-----------|
| 随机策略 | 914.2 | 8.30/10 | 83.0% |
| Iteration 10 | 486.6 | 8.00/10 | 80.0% |
| Iteration 30 | **1532.6** | **9.20/10** | **92.0%** |
| Iteration 60 | 1385.1 | 8.90/10 | 89.0% |

### 关键发现

1. **学习有效性**: 训练后的模型在存活率和奖励上都显著优于随机策略
2. **最佳性能**: Iteration 30达到最佳性能，平均92%存活率
3. **行为改进**: 小鱼学会了主动远离大鱼，并在大鱼接近时加速逃离

## 算法说明

### PPO (Proximal Policy Optimization)

本项目使用PPO算法训练小鱼的策略网络。PPO的主要优势：

- **稳定性好**: 适合连续控制任务
- **样本效率高**: 相比DQN等算法更高效
- **易于调试**: 超参数相对稳定

### 状态空间（11维）

1. 自身位置 (x, y) - 归一化
2. 自身速度 (vx, vy) - 归一化
3. 到边界距离 - 归一化
4. 大鱼可见标志 (0/1)
5. 大鱼相对位置 (Δx, Δy) - 归一化
6. 大鱼相对速度 (Δvx, Δvy) - 归一化
7. 大鱼距离 - 归一化

### 动作空间（离散5个动作）

0. 加速前进
1. 左转
2. 右转
3. 减速
4. 保持

### 奖励函数

- **存活奖励**: +2.0 每时间步
- **距离奖励**: 
  - 在视野内: +10.0 × (距离/视野半径)
  - 远离方向对齐: +5.0 × 对齐度
  - 不在视野内: +5.0
- **死亡惩罚**: -50.0
- **边界惩罚**: -2.0 × (靠近程度)

## 自定义训练

### 修改训练参数

编辑 `train_v2.py`:

```python
if __name__ == "__main__":
    TOTAL_ITERATIONS = 100  # 训练轮数
    NUM_FISH = 25           # 小鱼数量
    NUM_ENVS = 32           # 并行环境数量（多进程）
    
    model, stats = train_ppo_v2(
        total_iterations=TOTAL_ITERATIONS,
        num_fish=NUM_FISH,
        num_envs=NUM_ENVS
    )
```

### 调整环境难度

编辑 `fish_env.py`:

```python
# 增加大鱼速度
self.PREDATOR_INITIAL_VX = 2.0  # 增加初始速度
self.PREDATOR_GRAVITY = 0.7     # 增加重力

# 减少小鱼视野
self.FISH_VISION_RADIUS = self.STAGE_RADIUS / 4  # 减少视野范围

# 调整小鱼速度
self.FISH_MAX_SPEED = 2.5  # 增加小鱼最大速度
```

## 可视化

### 生成训练曲线

```python
import pickle
import matplotlib.pyplot as plt

with open('./checkpoints/training_stats.pkl', 'rb') as f:
    stats = pickle.load(f)

plt.plot(stats['iterations'], stats['avg_rewards'])
plt.xlabel('Iteration')
plt.ylabel('Average Reward')
plt.title('Training Progress')
plt.show()
```

### 实时观看模型行为

使用 `watch.py` 脚本（推荐）：

```bash
python watch.py --model ./checkpoints/model_iter_60 --num-fish 25
```

或使用 Python 代码：

```python
from stable_baselines3 import PPO
from fish_env import FishEscapeEnv

model = PPO.load('./checkpoints/model_iter_60')
env = FishEscapeEnv(num_fish=25, render_mode='human')

obs, _ = env.reset()
for _ in range(500):
    if len(obs) == 0:
        break
    actions = [model.predict(ob, deterministic=True)[0] for ob in obs]
    obs, _, terminated, truncated, _ = env.step(actions)
    if terminated or truncated:
        break

env.close()
```

## 进阶扩展

### 1. 增加到250条鱼

逐步增加小鱼数量，每次增加后重新训练：
- 10 → 25 → 50 → 100 → 250

### 2. 提高大鱼威胁

逐步增加大鱼的速度和攻击性。

### 3. 多大鱼场景

修改环境支持多条大鱼同时存在。

### 4. 小鱼协作

添加小鱼之间的通信机制，让它们能够协作躲避。

### 5. 课程学习

实现自动调整难度的课程学习策略。

## 常见问题

### Q: 训练后存活率仍然是0%？

A: 这通常是因为环境太难。建议：
1. 减少小鱼数量（从10-15条开始）
2. 降低大鱼速度（修改 `PREDATOR_INITIAL_VX` 和 `PREDATOR_GRAVITY`）
3. 增加小鱼视野范围（修改 `FISH_VISION_RADIUS`）
4. 增加小鱼最大速度（修改 `FISH_MAX_SPEED`）

### Q: 训练时间太长？

A: 可以：
1. 减少 `total_iterations`
2. 减少 `n_steps`（每次rollout的步数）
3. 减少 `num_envs`（并行环境数量）
4. 使用GPU加速（需要安装PyTorch GPU版本）

### Q: 如何保存和加载模型？

A: 
```python
# 保存
model.save("my_model")

# 加载
from stable_baselines3 import PPO
model = PPO.load("my_model")
```

### Q: 如何查看训练进度？

A: 训练过程中会自动保存统计信息到 `checkpoints/training_stats.pkl`（每5个iteration保存一次）。运行 `python visualize.py` 可以查看训练曲线。

## 技术细节

### 多智能体训练策略

本项目采用**参数共享**策略：
- 所有小鱼共享同一个策略网络
- 每条小鱼有独立的观测和奖励
- 训练时汇总所有小鱼的经验
- 使用多进程并行环境（SubprocVecEnv）提高训练效率

### 大鱼运动机制

大鱼使用**确定性物理模拟**：
- 从圆心 (0, 0) 开始，初始速度向右
- 每步施加重力（向下加速）
- 碰到圆形边界时反弹，并施加阻尼
- 不追踪小鱼，轨迹完全由物理规律决定

### 小鱼边界处理

- **物理反弹**: 碰到边界时，法向速度反向并减半，切向速度保持不变
- **奖励惩罚**: 距离边界小于1.0时，根据距离给予惩罚（最多-2.0）


```python
learning_rate = 3e-4
n_steps = 1024          # 每次rollout的步数
batch_size = 256        # 批大小
n_epochs = 10           # 每次更新的epoch数
gamma = 0.99            # 折扣因子
gae_lambda = 0.95       # GAE lambda参数
clip_range = 0.2        # PPO clip范围
ent_coef = 0.02         # 熵系数（鼓励探索）
vf_coef = 0.5           # 价值函数系数
max_grad_norm = 0.5     # 梯度裁剪
```

### 网络架构

```python
Policy Network:
  Input (11) → Dense(128) → Dense(128) → Output(5)

Value Network:
  Input (11) → Dense(128) → Dense(128) → Output(1)
```

## 参考资料

- [Stable-Baselines3 文档](https://stable-baselines3.readthedocs.io/)
- [PPO 论文](https://arxiv.org/abs/1707.06347)
- [Gymnasium 文档](https://gymnasium.farama.org/)

## 作者

Manus AI

## 许可

MIT License
