# dev_v9

## 启动
- 继承 dev_v8 的 divergence 奖励 + 密度 penalty 组合，iteration 13~24 已把 final_alive 推至 19.7，但早期 p10 掉到 26 步，说明密度惩罚在阶段 1 过强。
- 当前工作假设：对密度 penalty 进行迭代/阶段调度 + 提高 divergence_coef 可同时保留中后期优势并抑制初期早死；同时补充 per-phase early death logging 以验证分布。
- 目标：构建可交接的 dev_v9 baseline（≥128 env）并输出完整日志/可视化，作为 dev_v10 的新起点。

## 观察
- dev_v8 记录显示 iteration 17 reward dip 已消失，但单鱼 reward 方差拉大，提示 penalty 对弱个体影响更大。
- TensorBoard 中新增的邻居速度分布仍波动大，策略未显著使用；需要更直接的 logging（如 first_death_step 直方图）来佐证。
- 多鱼 eval 最低 survival 18/25，仍有 28% episode 在 250 步前坍塌，说明 divergence 奖励仍不足够保持全程散度。

## 实验计划
1. **Run A：密度 penalty ramp** —— 在 `train.py` 新增 per-phase penalty schedule，阶段 1 关闭，阶段 2 线性 ramp 到 0.08，阶段 3 0.08 固定；记录 first death histogram。
2. **Run B：divergence reward 扩大 + logging** —— 在 Run A 基础上，将 `divergence_reward_coef` 从 0.15 提到 0.22，重点观察 `custom/final_alive` 与 early death 统计是否改善。
3. **可视化/媒体** —— 训练后生成 survival 曲线 PNG + 多鱼 eval 视频，拷贝至 `experiments/v9/artifacts/plots|media/`，并将 run 命令、指标、下一步计划写入本文档。

## 运行记录
- 2025-11-11 00:05~00:21 (UTC-8)：`source venv/bin/activate && python experiments/v9/train.py --run_name dev_v9_density_ramp --num_envs 128 --curriculum 15:6,20:9,25:9 --policy_hidden_sizes 384,384 --divergence_reward_coef 0.15 --density_penalty_coef 0.0 --density_penalty_phase_targets 0.0,0.08,0.08 --density_penalty_ramp_phases 2 --eval_multi_episodes 8 --eval_multi_fish 25 --video_num_fish 25 --seed 72`
  - 128 并行环境、24 iteration，阶段 2 ramp (迭代 7~15) 线性抬升 penalty，阶段 3 固定 0.08。TensorBoard/日志/ckpt 位置：`experiments/v9/artifacts/tb_logs/dev_v9_density_ramp/`、`experiments/v9/artifacts/logs/dev_v9_density_ramp.log`、`experiments/v9/artifacts/checkpoints/dev_v9_density_ramp/`。

## 结果与产出
- 训练曲线：`experiments/v9/artifacts/plots/dev_v9_density_ramp_survival.png`；first-death 直方图：`experiments/v9/artifacts/plots/dev_v9_density_ramp_death_histogram.png`；曲线 GIF：`experiments/v9/artifacts/media/dev_v9_density_ramp_curve.gif`。
- 最佳迭代 16：`sr=69.2% / final_alive=17.31 / first_death=10.9`；阶段 3 平均 (迭代 16~24) `final_alive=17.02±0.18 / sr=68.1% / first_death=8.39`，表明 ramp 后 reward dip 仍消失，但 first_death 中位数仅 11.6 步 (p10=6.5)，早期仍偏脆弱。
- 与 dev_v8 对比：终局活鱼 17→17.0 基本持平，但 early death 提前（v8 p10≈26 步）。密度 penalty 在阶段 2 ramp 仍导致 iteration 8~10 的 first_death 掉到 5~7 步，需要更晚生效或调低目标。
- 多鱼 eval (25 fish ×8) `final_alive=13.25±2.5`，最低 episode 8 条，`min_survival_rate=32%`；`avg_reward=0.431`，明显逊于 dev_v8 (19.7)。单鱼 eval `reward=244.6±28.9 / final_alive=17.6`，说明单体策略仍稳定，群体坍塌来自协同不足。
- 媒体：`experiments/v9/artifacts/media/dev_v9_density_ramp_multi_fish_eval.mp4`；多鱼 death 统计 JSON 位于 `experiments/v9/artifacts/checkpoints/dev_v9_density_ramp/death_stats.json`（survival_fraction=0.41, mean_death_step=287）。

## Learning & 下一步计划
1. **推迟/分段 penalty ramp**：阶段 2 的 ramp 迭代仍过早，first_death p10=6.5。下一轮尝试仅在 iteration ≥10 或进入阶段 3 后开启 penalty，或把 ramp 目标降至 0.05 再在阶段 3 提升。
2. **Run B：抬升 divergence 奖励 + 监控**：在 ramp 调整基础上把 `divergence_reward_coef` 提到 0.22，并加入 `custom/min_survival_rate` logging（多鱼 eval）以监控最差 episode。
3. **可视化补充**：当前 plots 仅含 survival/final_alive，下一轮需要生成 first-death vs iteration 折线或把数值写入 `experiments/v9/artifacts/plots/first_death_iter.png`，方便直观看到 ramp 影响。
