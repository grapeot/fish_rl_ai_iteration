# dev_v17

## 启动
- 继承 dev_v16 已实现的真实 Phase6 penalty（迭代 27~30 的 `density_penalty=0.08`）与 `schedule_trace.json` 日志链路，继续以 25 条鱼、128 并行环境作为 baseline。
- dev_v16 的多鱼终局（8×25）平均存活率 78.5%，但 ramp 后期 `first_death_p10` 仍跌到 ≈9，说明 penalty=0.06 以上的稳定化不足。
- `eval_multi_history` 揭示迭代 12→28 之间的快速劣化；本轮目标是围绕 Phase5/6 的 ramp 策略和 escape boost clamp 做更精细的调节，并把多鱼探针信号直接沉淀到 plots/TensorBoard。

## 观察
- Phase6 在 dev_v16 中仅 4 iter 且 ramp 立即推到 0.08，导致高 penalty 区没有缓冲；Phase5 plateau 只有 2 iter，同样过短。
- 固定 `escape_boost_penalty_caps`（0.05→0.8）无法跟随 `first_death_p10` 的实时波动；迭代 28 之前的崩溃窗口需要自适应 clamp。
- 多鱼探针虽已经 JSONL 化，但缺失 per-iteration histogram / TB scalar，同步难度仍高。

### dev_v17_runA_phase6_buffer（2025-11-11 PST）
- **运行命令**：

  ```bash
  python experiments/v17/train.py \
    --run_name dev_v17_runA_phase6_buffer \
    --divergence_reward_coef 0.22 \
    --multi_eval_interval 4 \
    --multi_eval_probe_episodes 5 \
    --density_penalty_phase_plateaus 0,0,3,3,4,1,1 \
    --escape_boost_penalty_caps 0.05:0.8,0.075:0.78 \
    --entropy_coef_schedule 16-24:0.03,25-32:0.02 \
    --seed 214
  ```
- **关键指标**：
  - 完整训练 30 iteration，`num_envs=128, n_steps=512`，Phase6 拆为 0.065（2 iter）+0.08（2 iter）。
  - 终局多鱼 deterministic（5×25 鱼）：`avg_final_survival_rate=84.8%` / `min=76%`，`death_stats.p10=54.4`，`early_death_fraction_150=13.6%`（`experiments/v17/artifacts/checkpoints/dev_v17_runA_phase6_buffer/eval_multi_summary.json`）。
  - 单鱼 deterministic：平均奖励 241.2，`final_num_alive=15.8`（`experiments/v17/artifacts/checkpoints/dev_v17_runA_phase6_buffer/eval_single_fish.json`）。
  - 训练阶段最后 5 iter `first_death_step` 平均 7.9；`first_death_p10` 由于统计接口 bug 一直被写成 1，导致自适应 clamp 未实际触发。
  - 期中多鱼探针（迭代 4→28）新增 7 组直方图：迭代 24 之前 `avg_final_survival_rate≈0.89`，到迭代 28 降到 0.864，`death_stats.p10` 仍 <120，提示 Phase6 高 penalty 依旧脆弱。
- **观察**：
  - Phase6 拆分后主训练的 `survival_rate` 波动明显小于 dev_v16，且 `multi_eval_history` 不再在 0.06 区间瞬间崩盘，但高 penalty（0.065+0.08）段的 `first_death` 仍卡在 8 左右，没有跨入两位数。
  - 期中探针曲线与 `plots/dev_v17_runA_phase6_buffer_multi_eval_hist/*.png` 能定位 ramp 末端的死亡分布；`death_hist` 显示依旧存在 <100 step 的密集死亡尾巴。
  - `first_death_p10` 在 `schedule_trace.json` 中全为 1，说明当前 `ep_info_buffer` → percentile 的实现存在数据对齐 bug；也因此 `adaptive_boost` 回调未写入任何日志，需要下一轮修正后再评估 clamp 策略。
- **产物路径**：
  - Checkpoints & 日志：`experiments/v17/artifacts/checkpoints/dev_v17_runA_phase6_buffer/`（含 `schedule_trace.json`, `training_stats.pkl`, `eval_multi_history.jsonl`）。
  - TensorBoard：`experiments/v17/artifacts/tb_logs/dev_v17_runA_phase6_buffer/`（`multi_eval_probes/` 下同步了探针标量）。
  - Plot：`experiments/v17/artifacts/plots/` 下的 `*_survival.png`, `*_first_death.png`, `*_penalty_vs_*`, `dev_v17_runA_phase6_buffer_multi_eval_timeline.png` 以及 `dev_v17_runA_phase6_buffer_multi_eval_hist/*.png`。
  - 媒体：`experiments/v17/artifacts/media/dev_v17_runA_phase6_buffer_curve.gif`、`experiments/v17/artifacts/media/dev_v17_runA_phase6_buffer_multi_fish_eval.mp4`。
  - 文本日志：`experiments/v17/artifacts/logs/dev_v17_runA_phase6_buffer.log`。

## 实验计划
1. **修复 first_death_p10 统计**：在 rollout callback 内存储每个 episode 的 `first_death_step` 或直接读取 `death_timesteps`，避免当前全为 1 的 bug，确保 adaptive clamp 依据真实分布决策。
2. **验证 adaptive clamp 生效路径**：在修复统计后重跑短程实验，观察 `adaptive_boost` 日志能否在 Phase5/6 主动把 base speed 限制到 0.78，并评估是否需要更激进的上限（例如 0.76）。
3. **Phase6 迭代再拉长**：考虑在 0.065 段追加 ≥2 plateau iter 或维持 penalty=0.065 的 longer window，再进入 0.08，以期 `multi_eval_history` 在迭代 28 后不再掉到 `p10≈110`。
4. **多鱼探针评估增强**：把高 penalty 区间的探针 episodes 提至 8，并输出 `early_death_fraction_100` 等更敏感指标，配合现有 histogram 以快速判断 ramp 末端的崩溃模式。
