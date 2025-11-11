# dev_v2

## 2025-11-11 更新
- `VectorizedFishEnv` 区分了实时存活数 (`num_alive`)、episode 平均 (`avg_num_alive`) 与终局 (`final_num_alive`)，并通过 `VecMonitor`/`CheckpointCallback` 全部写入日志与 TensorBoard（`experiments/v2/artifacts/tb_logs/`）。
- `CheckpointCallback` 在 verbose 日志里输出平均/终局存活数，并把三类指标连同自定义 logger 事件一起持久化至 `training_stats.pkl`。
- `FishEscapeEnv` 增加 `REWARD_SCALE=0.1` 统一缩放奖励，防止 PPO 优势值被过大的绝对回报压缩。
- `PPO` 现在带 `tensorboard_log`，配合 `tensorboard --logdir experiments/v2/artifacts/tb_logs` 可实时查看 loss、熵以及新增自定义指标。

### 全量运行 (Iter=100, num_envs=32)
- 命令：`python experiments/v2/train.py --total_iterations 100 --num_envs 32`（env：`venv`）。日志路径 `experiments/v2/artifacts/logs/train_v2_iter100.log`，快照+曲线归档于 `experiments/v2/artifacts/checkpoints/v2_iter100/`。
- 指标：平均存活率长期在 69%~74%，终局存活 17±1 条鱼（25 条上限），Rewards 约 250±7，所有 episode 仍跑满 500 步。
- 最终输出：`Final survival rate=69.12%`、`Final avg alive=17.3`、`Final avg reward=251.01`。`avg_num_alive` 与 `final_num_alive` 曲线基本平行，说明死亡主要集中在收尾阶段且策略未形成协同逃跑。
- 训练曲线：见 `experiments/v2/artifacts/checkpoints/v2_iter100/training_curve.png`；可观察到最高点在 Iter40 左右后即往下回落，推测策略在更高熵阶段短暂受益，但随后再次陷入同质化动作。

### 分析
- **Reward 缩放** 带来了更稳定的 ADV，但不足以让策略提高终局存活，因为动作共享的 credit assignment 仍是瓶颈。
- **Logging**：`info` 中的 `num_alive/avg_num_alive/final_num_alive` 均可见；TensorBoard 尚未实际打开确认曲线，需下轮执行 Checklist 第 2 项。
- **瓶颈推测**：单一动作广播导致即使部分观测提示“逃离”也会被其它方向拉稀，训练后 survival 在 70% 左右徘徊；要冲击高难度配置，必须让策略能针对个体条件做决策。

### 下一步计划（提案）
1. **实测 TensorBoard**：运行 `tensorboard --logdir experiments/v2/artifacts/tb_logs --port <port>`，截图/记录 policy loss、entropy，验证是否存在熵坍缩或 value 爆炸（Checklist②）。
2. **单鱼 VecEnv 雏形**：基于 `SingleFishEnv` 重写 `make_env`，让 `SubprocVecEnv` 中的每个环境只控制一条鱼，自然获得真实 reward/obs 对齐。短期内可以把 `num_envs` 提升到 128 以补足样本量。
3. **评估脚本**：实现 `evaluate_checkpoint.py`，给定模型路径与随机种子，输出 deterministic run 的终局存活数（Checklist③）。
4. **版本分支**：完成上面 TODO 后创建 `experiments/v3/train.py` + `experiments/v3/dev_v3.md`，并将新的 artifacts 存在 `experiments/v3/artifacts/`，遵循 SOP。

## 当前训练现象
- 100 次 iteration 内，`VecMonitor` 输出的 `Survival` 指标长期停留在 0.75~0.79，`Steps` 始终是 500，说明绝大多数 episode 都是跑满步数但没有明显提升。
- `num_alive` 在日志中也稳定在 18~19，表面上看像是训练趋于平稳，但实际生存鱼数被压缩成平均值，掩盖了真正的表现差异。

## 根因分析
1. **观测与动作错位**：`VectorizedFishEnv.step` (见 `experiments/v2/train.py:103-150`) 仍旧用同一个动作去控制所有存活小鱼，并且只返回第一条活鱼的观测。随着小鱼死亡，这个“第一条”不断换人，策略网络始终面对一个不连续、无法追踪身份的观测流，却承担着所有鱼的平均奖励，导致策略梯度几乎退化成噪声。（尚待在离线 TODO 中彻底解决，可考虑拆成单鱼 env）
2. **统计数据被平均化**（已修复）：`VectorizedFishEnv` 现在把实时、平均、终局指标分开写回 `info`，`CheckpointCallback` 也会分别汇总/打印/存盘。
3. **奖励规模过大**（已修复）：通过 `REWARD_SCALE` 将奖励缩小 0.1 倍，便于 PPO 计算优势。

## 建议的调试与日志手段
- **区分实时与终局指标**：在 `VectorizedFishEnv` 把 `info['avg_num_alive']` 和 `info['final_num_alive']` 分开传给 `VecMonitor`，并在 `CheckpointCallback` 里显示两者（新增 `avg_alive`/`final_alive` 字段）。
- **TensorBoard 监控**：启用 `model.set_logger` 或 `logger.record`，汇报 `survival_rate`, `final_num_alive`, `entropy_loss`, `policy_loss`, `value_loss` 等，观察是否出现熵坍缩或 value 爆炸。
- **定期评估**：写一个 `evaluate_checkpoint.py`，每隔 N iteration 用 `model.predict(deterministic=True)` 在固定种子下跑 5~10 个 episode，输出真实的平均存活条数，避免训练日志被噪声误导。
- **采样验证**：在 `CheckpointCallback` 中缓存最近一个 `info['final_num_alive']`，触发阈值（如 <5）时 dump 一帧环境状态或动作序列，确认策略是否发生崩溃。

## 改进方向
1. **重新封装单鱼环境**：用 `SingleFishEnv` 或者把 `FishEscapeEnv` 拆成 `num_fish` 个 agent；每个并行环境只负责一条鱼，动作与奖励保持一一对应。然后用 `SubprocVecEnv` 组合 64~128 个独立环境训练 PPO，避免共享动作带来的 credit assignment 问题。
2. **保留真实的 `num_alive`**：让 `info['num_alive']` 始终表示当前实际存活数，把平均值写到 `info['avg_num_alive']`，`CheckpointCallback` 记录 `final_num_alive` 作为核心指标。
3. **压缩奖励尺度**：把 `FishEscapeEnv` 中的生存+距离奖励除以常数（如 10），或在 `VectorizedFishEnv` 返回 reward 之前做标准化，保证 `advantage` 的量级更适中，提高策略梯度的信噪比。
4. **更真实的动作模型**：允许每条鱼根据自己的观测选择动作。可以把 `VectorizedFishEnv` 改为一次返回 `obs` 的 batch，并让 `policy` 输入批量观测后输出批量动作（需要自定义 `VecEnv`).
5. **课程式训练**：先在易版环境（鱼少、捕食者慢）训练，逐步提升难度；或定期重置捕食者速度/重力，增加策略泛化能力。

## 下一步可验证的 Checklist
- [x] 修改环境后先运行 5 iteration，确认 `info` 中三类指标（当前、平均、终局）都能在日志里看到，且数值不同。（证据：`experiments/v2/artifacts/logs/train_v2_iter100.log` 多处包含 `num_alive`、`avg_num_alive`、`final_num_alive`）
- [ ] 打开 TensorBoard，检查 `loss/entropy_vs_iteration` 是否保持在合理范围（>0.5）。
- [ ] 用评估脚本对比旧版/改进版在同一随机种子下的 `final_num_alive`，确认改动是否让分布更集中。
