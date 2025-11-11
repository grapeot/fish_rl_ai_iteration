# dev_v15

## 启动
- 回顾 2025-11-11 完成的 dev_v14，总结：Phase 3 plateau 仍不足、penalty ramp 区在 `coef>0.05` 时生存率坍缩；escape boost gating 有改善但缺乏细粒度控制。
- 本轮目标：让 `penalty=0.06/0.08` 段拿到 ≥3/2 个迭代样本，配合 gating 与新增日志，验证高 penalty 稳态是否能把 `first_death_p10` 拉回 ≥9。
- 工作文档同步记录命令/指标/产物，确保 dev_v15 可以直接被 dev_v16 使用。

## 观察
- dev_v14 中一旦 ramp 到 `density_penalty>=0.06`，`first_death_mean` 从 ~10 直接跌到 <7，说明缓冲区太短；Phase 5 只有 2 iter，在 penalty=0.06 尚未稳定时就切到 0.08。
- Phase 3 早期（iter11/19）已跌破 `first_death_p10<8`，推断 escape boost 长期保持 0.75/0.8 可能让鱼群过于激进，需要在中期 penalty<0.04 时稍降，避免提前触发大规模死亡。
- 目前 `training_stats.pkl` 才能看到 penalty/boost 时间线，缺少轻量 JSON/文本导致复盘成本高，违背上一轮“Penalty per-iteration 追踪”的 TODO。

## 实验计划
1. **Phase 5/6 加长**：把 `total_iterations` 提到 26，curriculum 调整为 `15:6,20:7,25:4,25:4,25:3,25:2`，对应 `penalty_targets=0.0,0.02,0.04,0.05,0.06,0.08`，保证 `coef=0.06` 有 3 iter、`coef=0.08` 有 2 iter，并让 Phase 4 维持 0.05 过渡。
2. **Penalty Trace Logging**：在 callback 中新增 `schedule_trace.json`（iteration→density_penalty/entropy/escape_boost），跑完即可无需 Python 解 pickle，就能快速定位 ramp 断点。
3. **Gating 微调**：将 `escape_boost_phase_speeds` 扩展到 Phase 3~6（例如 `3:0.78,4:0.8,5:0.82,6:0.82`），并保留 Phase 1-2 的默认 0.75，目标是在 penalty<0.04 时避免过冲，进入高 penalty 时提供额外逃逸。
4. **完整训练 + 记录**：以上改动集成进 `experiments/v15/train.py`，以 `--num_envs 128 --n_steps 512 --total_iterations 26` 跑满大规模实验；日志、TB、plots、媒体统一落在 `experiments/v15/artifacts/`。

## 运行记录
- 2025-11-11 02:26 PST 完成 `dev_v15_runA_phase6_plateau_full`（命令如下，已在运行前 `source venv/bin/activate`）：

  ```bash
  python experiments/v15/train.py \
    --run_name dev_v15_runA_phase6_plateau_full \
    --num_envs 128 --total_iterations 26 \
    --curriculum 15:6,20:7,25:4,25:4,25:3,25:2 \
    --n_steps 512 --policy_hidden_sizes 384,384 \
    --divergence_reward_coef 0.22 \
    --density_penalty_phase_targets 0.0,0.02,0.04,0.05,0.06,0.08 \
    --density_penalty_ramp_phases 3,4,5,6 \
    --density_penalty_phase_plateaus 0,0,4,3,2,1 \
    --entropy_coef_schedule 16-20:0.03,21-27:0.02 \
    --escape_boost_phase_speeds 3:0.78,4:0.8,5:0.82,6:0.82 \
    --eval_multi_fish 25 --eval_multi_episodes 8 \
    --video_num_fish 25 --video_max_steps 500 --video_fps 20 \
    --seed 213
  ```
- 产物：
  - 日志 `experiments/v15/artifacts/logs/dev_v15_runA_phase6_plateau_full.log`
  - TensorBoard `experiments/v15/artifacts/tb_logs/dev_v15_runA_phase6_plateau_full/`
  - Checkpoints+stats+trace `experiments/v15/artifacts/checkpoints/dev_v15_runA_phase6_plateau_full/`（含 `schedule_trace.json`）
  - Plot `experiments/v15/artifacts/plots/dev_v15_runA_phase6_plateau_full_*.png`
  - 媒体 `experiments/v15/artifacts/media/dev_v15_runA_phase6_plateau_full_{curve.gif,multi_fish_eval.mp4}`

## 结果 & 学习
- Phase 5 (`density_penalty≈0.06`) 现有 2 个迭代，`first_death_step` 提升至 10.32→13.21（schedule trace 平均 11.77），但 Phase 6 的 `0.08` 目标未真正执行——迭代 26 结束后才会把 penalty 推到 0.08，导致本轮仍无 `coef=0.08` 数据。
- Phase 4~5 ramp 期间的 `survival_rate` 基本维持在 0.64~0.66，但 `first_death` 在 `penalty=0.05` 时仍只有 8.x，说明 plateau 拉长没有解决 ramp 前的脆弱窗口。
- 单鱼 deterministic eval：平均奖励 241.1，`final_num_alive=15.8`，表明单体场景保持稳定。
- 多鱼 eval（8×25 鱼）恶化：`avg_final_survival_rate=43% / min=28%`，死亡分布 `p10=58.9`，`early_death_fraction_150=22.5%`，比 dev_v14 的 78.5% 明显下降，推测是更高 penalty + boost 组合使鱼群更早聚集触发密度惩罚。
- 新的 `schedule_trace.json` 满足“无需解 pickle 即可追踪 penalty/entropy/boost”的需求，可直接 grep 到 iteration→系数的变化。

## 下一步
1. **真实执行 `penalty=0.08`**：要么把 Phase 6 时长增加到 ≥3 iter，要么调整 scheduler（例如在 `_on_rollout_start` 就应用下一次 penalty），否则永远得不到 0.08 段样本。
2. **Phase 4/5 稳定化**：目前 `penalty=0.05` 仍只有 `first_death≈8`、多鱼生存 <50%。需要结合 `schedule_trace` 和 TensorBoard 梳理是否是 escape boost 0.82 过大，或迭代 19 之前的 entropy taper 不足，建议下轮尝试 `boost_phase_speed=5:0.8` 或引入 penalty-aware boost。
3. **多鱼监控增强**：为调试 25 鱼崩溃，考虑在训练中周期性运行小批 `eval_multi`（或记录 `survival_rate` 的分布），并把关键指标写入 `artifacts/checkpoints/.../eval_multi_history.json`，这样 dev_v16 可以快速定位 ramp 中的失败点。
