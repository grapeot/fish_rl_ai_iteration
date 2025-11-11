# dev_v10

## 启动
- 延续 dev_v9 的 PPO baseline（128 env / 24 iterations / curriculum 15→20→25 fish），上一轮最佳迭代 16 仅达到 `final_alive≈17`，但 first_death p10≈6.5，显著早于 dev_v8。
- dev_v9 已实现 density penalty ramp + first-death 直方图 + 多鱼 eval 视频，产出路径位于 `experiments/v9/artifacts/`，可作为本轮复现实验的对照。
- 本轮目标：推迟密度 penalty 的生效时间、抬升 divergence 奖励，并补充 first-death 折线与 `min_survival_rate` 指标，力求把阶段 3 的 early death 推回 ≥15 步且多鱼 min survival ≥50%。

## 观察
- 阶段 2 ramp（迭代 7~15）仍会把 first_death 拉低到 5~7 步，说明 penalty 早期介入抑制了逃逸启动；需要把 penalty 延迟到阶段 3 或至少降低阶段 2 目标。
- divergence_reward_coef=0.15 下，多鱼 eval `min_survival_rate=32%`，最差 episode 8/25 存活，表明散度奖励仍不足支撑群体分散。
- 当前日志仅有 first_death 直方图，缺乏 iteration 曲线；多鱼 eval 也没有聚合的 min survival 记录，导致下一轮难以快速对比。

## 实验计划
1. **Run A：阶段 3 才启用 penalty** —— 维持 dev_v9 其它超参，设置 `density_penalty_phase_targets=0.0,0.0,0.08` 且只对 phase 3 进行 ramp（迭代 16~24 线性升至 0.08），确认 first_death 折线是否回升；同步生成 first_death vs iteration PNG。
2. **Run B：divergence 奖励 0.22 + 群体指标** —— 在 Run A 设置上把 `divergence_reward_coef` 提升到 0.22，训练结束后统计多鱼 eval 的 `min_survival_rate`（写入 JSON & dev_v10.md）并再次绘制 death histogram。
3. **媒体与交付** —— 每个 run 输出 TensorBoard、日志、ckpt、曲线、first-death 折线、death histogram 与多鱼 mp4 到 `experiments/v10/artifacts/{tb_logs,logs,checkpoints,plots,media}/`，并把关键指标/路径写入本文档供 dev_v11 启动。

## 运行记录
- 2025-11-11 00:28~00:36 (UTC-8) `source venv/bin/activate && python experiments/v10/train.py --run_name dev_v10_runA_delay_penalty --num_envs 128 --curriculum 15:6,20:9,25:9 --policy_hidden_sizes 384,384 --divergence_reward_coef 0.15 --density_penalty_coef 0.0 --density_penalty_phase_targets 0.0,0.0,0.08 --density_penalty_ramp_phases 3 --eval_multi_episodes 8 --eval_multi_fish 25 --video_num_fish 25 --seed 91`
  - 最佳迭代 17：`final_alive=16.56 / survival_rate=66.2% / first_death=14.5`；阶段 3 平均 `final_alive=16.14±0.28 / sr=64.6% / first_death_mean=8.1 (p10=4.7)`，first-death 折线虽整体上扬，但 ramp 开启后仍瞬间跌回 <10 步。
  - 多鱼 eval (25×8) `avg_final_survival=66% / min_final_survival=40% / avg_min_survival=66%`；death histogram `p10=29`，`early_death_fraction_150=25%`；`mean_death_step=367`。
  - 产出：TB `experiments/v10/artifacts/tb_logs/dev_v10_runA_delay_penalty/`、日志 `experiments/v10/artifacts/logs/dev_v10_runA_delay_penalty.log`、ckpt `experiments/v10/artifacts/checkpoints/dev_v10_runA_delay_penalty/`、曲线 `experiments/v10/artifacts/plots/dev_v10_runA_delay_penalty_survival.png`、first-death 折线 `experiments/v10/artifacts/plots/dev_v10_runA_delay_penalty_first_death.png`、death histogram `experiments/v10/artifacts/plots/dev_v10_runA_delay_penalty_death_histogram.png`、媒体 `experiments/v10/artifacts/media/dev_v10_runA_delay_penalty_{curve.gif,multi_fish_eval.mp4}`。
- 2025-11-11 00:38~00:44 (UTC-8) `source venv/bin/activate && python experiments/v10/train.py --run_name dev_v10_runB_divergence_022 --num_envs 128 --curriculum 15:6,20:9,25:9 --policy_hidden_sizes 384,384 --divergence_reward_coef 0.22 --density_penalty_coef 0.0 --density_penalty_phase_targets 0.0,0.0,0.08 --density_penalty_ramp_phases 3 --eval_multi_episodes 8 --eval_multi_fish 25 --video_num_fish 25 --seed 123`
  - 最佳迭代 16：`final_alive=17.65 / survival_rate=70.6% / first_death=10.1`；阶段 3 平均 `final_alive=16.73±0.48 / sr=66.9% / first_death_mean=7.5 (p10=5.0)`，散度奖励提升带来了更高终局活鱼，但 first_death 仍在 ramp 开启即回落。
  - 多鱼 eval (25×8) `avg_final_survival=70.5% / min_final_survival=68% / avg_min_survival=70.5%`；death histogram `p10=12.9 / survival_fraction=70.5% / early_death_fraction_150=20.5%`；`mean_death_step=395`，min survival 拉升到 17/25。
  - 产出：TB `experiments/v10/artifacts/tb_logs/dev_v10_runB_divergence_022/`、日志 `experiments/v10/artifacts/logs/dev_v10_runB_divergence_022.log`、ckpt `experiments/v10/artifacts/checkpoints/dev_v10_runB_divergence_022/`、曲线 `experiments/v10/artifacts/plots/dev_v10_runB_divergence_022_survival.png`、first-death 折线 `experiments/v10/artifacts/plots/dev_v10_runB_divergence_022_first_death.png`、death histogram `experiments/v10/artifacts/plots/dev_v10_runB_divergence_022_death_histogram.png`、媒体 `experiments/v10/artifacts/media/dev_v10_runB_divergence_022_{curve.gif,multi_fish_eval.mp4}`。

## 结果与产出
- `dev_v10_runA_delay_penalty` 证明“仅阶段 3 ramp”可以保持终局 16+ 鱼，但 first_death p10 仍 <5 步，提示 penalty 即便推迟也会瞬间伤害开局策略。
- `dev_v10_runB_divergence_022` 将最终存活抬到 17.6 且多鱼 min survival=68%，相比 v9 (32%) 大幅改善，但阶段 3 first-death 曲线均值仅 7.5，未达到 ≥15 步目标。
- 新增的 first-death 折线 + `eval_multi_summary.json` 让早期崩溃与群体最差表现可以直接复盘；所有 plots/gif/mp4 已存放在 `experiments/v10/artifacts/{plots,media}/` 并纳入版本控制。
- death histogram 显示 Run B 的生存尾部更长（survival_fraction 70.5%），但 p10=12.9 仍远低于上轮（v8 p10≈26），需要额外机制防止 ramp 刚启用时的团灭。

## Learning & 下一步计划
1. **阶段 3 热身** —— 继续保持阶段 2 penalty=0，但在阶段 3 先以 0.0→0.04 的缓 ramp（或在迭代 18 之后才升至 0.08），并在 penalty 更新同时注入额外熵奖励来避免策略瞬间收紧。
2. **早期逃逸保护** —— 针对 iteration 16 之后 first_death <10 的问题，考虑在 penalty ramp 期间追加“first death clip”奖励或暂时提升 `initial_escape_boost`，并跟踪 `first_death_step` 与 penalty value 的相关性（可在日志中输出二者对）。
3. **群体指标对齐** —— `min_final_survival=40% → 68%` 说明 divergence=0.22 有效，但仍观察到 death histogram p10=12.9；下一轮可以尝试将 divergence_coef 维持 0.22 的同时新增 neighbor-variance 特征或增大 `neighbor_radius`，并把 `eval_multi_summary.json` 中的指标同步至 TensorBoard（custom/min_survival_rate）。
