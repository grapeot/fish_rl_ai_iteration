# dev_v7

## 启动
- 延续 dev_v6 的结论：虽然邻居特征和两阶段 curriculum 把 25 鱼 survival 稳定在 ~72%，但 10% 小鱼仍在 3 步内阵亡，说明开局姿态/初始速度存在系统性短板。
- dev_v6 提醒：下一轮需要在环境里引入“随机旋转+开局逃逸脉冲”并进一步增强邻居密度相关特征，以减少拥挤区的猜测式决策。
- 本轮聚焦：优化出生态势（spawn jitter + outward boost）、增加邻居速度/散度观测，并把 curriculum 改为三阶段渐进 (15→20→25 鱼) 以平滑 LR 适应。

## 观察
- 早期死亡集中在 <20 步，最糟糕 3 步即被捕，推测当前 reset 将速度初始化为低速随机方向，缺少“离中心逃逸”信号，导致贴脸捕食者时无反应。
- 邻居统计仍只包含密度+平均相对位置+最近邻距离，缺乏对邻居速度趋势（扩散 vs. 聚拢）的观测，策略难以在拥挤区采取差异化动作。
- Curriculum 在 iteration 13 切换 15→25 鱼时出现 survival 跳变，佐证需要更细颗粒度的 schedule 和/或切换时的学习率衰减控制。

## 实验计划
1. **环境 Spawn 增强**：给 `FishEscapeEnv` 新增 `initial_escape_boost` 选项，reset 时对每条鱼施加朝远离捕食者（圆心）方向的初始脉冲，并允许一定角度抖动，验证是否能缓解首 20 步死亡。
2. **邻居速度/散度特征**：在邻居观测中追加 `neighbor_speed_mean`、`neighbor_speed_std`、`neighbor_divergence`（沿相对向量的速度投影），并在训练脚本中默认开启，观察曲线震荡是否下降。
3. **三阶段 Curriculum**：在 `experiments/v7/train.py` 中改为 `(15,8)+(20,8)+(25,8)`，保持 `num_envs=128`，配合 cosine LR；训练完成后同样导出生存曲线、死亡直方图与多鱼视频，所有 artifacts 写入 `experiments/v7/artifacts/`。

## 运行记录
- **2025-11-11 00:05~00:39 (UTC-8)**：`source venv/bin/activate && python experiments/v7/train.py --total_iterations 24 --curriculum "15:8,20:8,25:8" --num_envs 128 --eval_multi_fish 25 --eval_multi_episodes 8 --video_num_fish 25 --run_name dev_v7_spawn_boost_schedule --seed 52`
  - 阶段 1（迭代 1~8, 15 fish）：survival 稳定在 0.65~0.69，初始逃逸脉冲后 `first_death_step` 提前到 ~40 但平均死亡降至 10 fish。
  - 阶段 2（迭代 9~16, 20 fish）：sr≈0.68，`final_alive` 抬升至 13.7，曲线几乎无跳变。
  - 阶段 3（迭代 17~24, 25 fish）：最佳 iteration 23 达到 `sr=69.6% / final_alive=17.4`，最终收敛 iteration 24 时 `sr=66.9% / final_alive=16.7`。
  - 日志：`experiments/v7/artifacts/logs/dev_v7_spawn_boost_schedule.log`；TensorBoard：`experiments/v7/artifacts/tb_logs/dev_v7_spawn_boost_schedule/`；Checkpoints：`experiments/v7/artifacts/checkpoints/dev_v7_spawn_boost_schedule/`（含 eval/death_stats/ckpt）。

## 结果与产出
- **训练指标**：24 iteration 全程 500 步；best survival 69.6% (iter 23)，最终 66.9%；`avg_num_alive` 平滑提升但仍低于 dev_v6 的 0.72，高并行时 reward 下降 ~3%。
- **死亡分布**：`death_stats.json` 显示 p10=83.8、早期死亡 (<150 步) 降至 14%（vs v6 的 21%），表明 spawn boost 有效推迟首批阵亡。
- **评估**：单鱼 deterministic (25 fish ×5) 平均 `final_alive=13.8`；多鱼评估 (25×8) 平均 `final_alive=12.6`，波动区间 [6,16]，仍存在极端失败 episode。
- **Artifacts**：
  - 曲线：`experiments/v7/artifacts/plots/dev_v7_spawn_boost_schedule_survival.png`
  - 死亡直方图：`experiments/v7/artifacts/plots/dev_v7_spawn_boost_schedule_death_histogram.png`
  - GIF：`experiments/v7/artifacts/media/dev_v7_spawn_boost_schedule_curve.gif`
  - 多鱼视频：`experiments/v7/artifacts/media/dev_v7_spawn_boost_schedule_multi_fish_eval.mp4`
  - 日志/ckpt/评估/JSON 见 `experiments/v7/artifacts/checkpoints/dev_v7_spawn_boost_schedule/`

## 学习 & 下一步计划
1. **Spawn Boost 只解决超早期死亡**：p10 从 3→84 步，但多鱼评估仍在 200~250 步出现群体崩塌，说明中段布局/协同仍依赖邻居策略；下一轮需要在 reward 或 policy 中显式鼓励“密度扩散”，例如引入 `neighbor_divergence` 的正向奖励或熵调制。
2. **邻居速度特征尚未被充分利用**：TensorBoard 中 `custom/final_num_alive` 仍在 16~17 徘徊，未突破 dev_v6；可以考虑把新特征开关化（用于 ablation），或在模型侧添加更宽的第一层以吸收新增维度。
3. **Curriculum 17→20 的过渡仍有掉速**：虽然三阶段消除了剧烈跳变，但 iteration 17 仍出现 reward dip；下一轮可尝试在阶段切换处执行短暂 replay-freeze 或使用 `--lr_schedule warm_cosine`（先线性降速再余弦），并把 `fish_schedule=[(15,6),(20,9),(25,9)]` 缩短 warm-up 以节约算力。
