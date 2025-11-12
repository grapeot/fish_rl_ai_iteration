# dev_v39

## 启动
- 2025-11-12 23:55 PST 依照 SOP 复盘 `dev_v38` 的 Learning/下一步，确认当前阻塞在 early-death gate variance、NE stage穿插不够、tail warning 迟滞以及 step-one 数据尚未分类型聚合。
- 本轮目标：在不牺牲 run reproducibility 的前提下，把 penalty gate 调整为 stage-aware、放大 multi-eval 样本，扩展 NE-only stage 时间，并输出面向 stage_type 的 step-one 汇总，力求为解锁 stage1 提供可信信号。

## 观察（承接 dev_v38）
1. `rolling_early_death_median≈0.115` 长期高于 0.10 阈值，尤其 iter40/48 虽 avg_final>0.65 但仍被判失败，说明 multi-eval 方差尚未被压平。
2. 单一 `stage0:NE-only` 仅覆盖前 9 iter，`step_one_top_share` 峰值却落在 main 段 (0.312)，提示 NE 病灶在 main 阶段重新堆积，需再插入一段 NE-only。
3. `tail_stage_warn_ratio=0.65`、`patience=3` 导致 iter59 即便跌到 0.648 也未触发 warning，缺乏“将耗尽”提醒。
4. 虽然 `step_one_clusters.jsonl` 已存 `tail_stage_type`，但缺少自动按 stage_type 聚合/可视化流程，只能手动 grep，效率低。

## 实验计划
1. **Gate 稳定化**：在 `train.py` 中引入 `--penalty_gate_success_early_death_by_stage`（例如 `ne:0.13,main:0.10`），并把 multi-eval probe/episodes 提升到 16/32，减小 early-death 中位数噪声。
2. **NE stage 扩展**：把 `tail_seed_stage_spec` 改为 `8(ne)+6(ne)+其余 main)`，并在 `TailStageTracker` 里记录 `remaining_by_type`，方便判断 NE 队列是否溢出；默认 `--num_envs 128`、`--total_iterations 60` 不变。
3. **Tail warning 调整**：缩短 `tail_stage_warn_patience` 至 2，并把阈值提升到 0.68，同时把 warning 事件写入 `eval_multi_history.jsonl` 以供复盘。
4. **Step-one by stage**：实现 `step_one_stage_summary.json` 生成器 + TensorBoard `custom/step_one_top_share_<type>`，让下一轮可以直接看到 NE vs main 的聚类分布。

## 实验记录

### dev_v39_stageaware_gate_v1（中断）
- **命令**：与计划一致（128 env × 60 iter，multi-eval 16/32，stage-aware gate），run_name=`dev_v39_stageaware_gate_v1`。
- **状态**：CLI 600 s 超时在 iter≈40 被动终止；日志、TensorBoard 仍保留于 `experiments/v39/artifacts/*/dev_v39_stageaware_gate_v1*` 供排查。
- **收获**：确认 `TailStageTracker` 的 `remaining_by_type` 与 `tail_stage_warning_events` 在中途已写入 `schedule_trace.json`，step-one JSON 也包含 `tail_stage_type`。

### dev_v39_stageaware_gate_v1_fast（中断）
- **命令**：在保持 60 iter 的前提下降低 `n_steps=96 / n_epochs=4`（run_name=`..._fast`），希望缩短 wall time。
- **状态**：30 min 超时依然卡在 iter≈40，说明瓶颈主要来自多阶段 curriculum 与 multi-eval，而非单次 rollout 尺度；该 run 的 artifacts 同样留存。

### dev_v39_stageaware_gate_v1_longrun（128 env × 60 iter，完成）
- **命令**（耗时约 45 min）：
  ```bash
  python experiments/v39/train.py --run_name dev_v39_stageaware_gate_v1_longrun \
    --total_iterations 60 --num_envs 128 --n_steps 128 --batch_size 1024 --n_epochs 5 --learning_rate 2.5e-4 \
    --curriculum 15:9,20:10,25:9,25:8,25:8,25:8,25:8 \
    --multi_eval_interval 8 --multi_eval_probe_episodes 16 --multi_eval_seed_base 411232 \
    --eval_multi_fish 96 --eval_multi_episodes 32 \
    --penalty_gate_phase_allowance 4,5,6,7 --penalty_gate_required_successes 2 \
    --penalty_gate_success_step_one 18 --penalty_gate_success_step_one_ratio 0.02 \
    --penalty_gate_success_early_death 0.10 --penalty_gate_success_early_death_window 3 \
    --penalty_gate_success_early_death_by_stage ne:0.13,main:0.10 \
    --penalty_gate_success_p10 70 --penalty_gate_success_median_window 4 \
    --penalty_gate_freeze_iterations 5 --penalty_gate_success_freeze_iterations 2 \
    --penalty_gate_failure_p10 95 --penalty_gate_failure_min_final 0.5 --penalty_gate_failure_tolerance 2 \
    --tail_seed_stage_spec '8:experiments/v34/artifacts/checkpoints/tail_seed_mix_v34_ne.json|type=ne;6:experiments/v34/artifacts/checkpoints/tail_seed_mix_v34_ne.json|type=ne;15:experiments/v34/artifacts/checkpoints/tail_seed_mix_v34_ne.json|type=main;24:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage1.json|type=main;16:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage1.json|type=main;25:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage2.json|type=main;20:experiments/v33/artifacts/checkpoints/tail_seed_mix_v33_stage3.json|type=main' \
    --tail_force_reset_steps 224 --tail_replay_count 2 --video_num_fish 48 \
    --tail_stage_warn_ratio 0.68 --tail_stage_warn_patience 2 --skip_final_eval --seed 411232
  ```
- **On-policy 60 iter 总结**：最近 10 iter 平均 `survival_rate=0.854`、`first_death_step≈31`（受 224 步截断影响）、`step_one_death_ratio=0.020`。Penalty gate 仍停留在 stage0，但 NE 阶段使用 0.13 阈值后未再成为首要瓶颈，真正触发 `failure_hold` 的是 iter48（`first_death_p10=82.5` + `step_one_ratio=0.0299`）。
- **Multi-eval（96 fish × 32 epi）**：

  | iter | stage | avg_final | min_final | early_death₁₀₀ | step_one_ratio | first_death_p10 |
  | --- | --- | --- | --- | --- | --- | --- |
  | 8  | ne   | 0.845 | 0.760 | 0.099 | 0.0202 | 102.5 |
  | 16 | ne   | 0.779 | 0.688 | 0.092 | 0.0176 | 112.0 |
  | 24 | main | 0.699 | 0.521 | 0.092 | 0.0228 | 109.0 |
  | 32 | main | 0.767 | 0.625 | 0.094 | 0.0182 | 108.0 |
  | 40 | main | 0.791 | 0.688 | 0.092 | 0.0208 | 106.0 |
  | 48 | main | 0.764 | 0.531 | 0.111 | 0.0299 | 82.5 |
  | 56 | main | 0.785 | 0.615 | 0.100 | 0.0202 | 99.0 |

  NE-only phase借助 0.13 阈值实现 `rolling_early_death_median≈0.092`，但 main 阶段在 iter48 出现 `first_death_p10` 暴跌，导致 gate 卡住。
- **新 telemetry**：
  - `schedule_trace.json` 现包含 `tail_stage_remaining_by_type`（iter60 时 `{'ne':0,'main':9728}`），并在 iter59 记录首个 `tail_stage_warning_events`（ratio=0.667、label=stage4）。
  - `step_one_stage_summary.json`：共 230 个 step-one 事件，NE 占 25.2%（迭代 8~16，均速 1.51），main 占 74.8%（迭代 ≥24，均速 1.21）。
  - TensorBoard 添加 `custom/step_one_top_share_ne/main` 与 `..._ne_high_speed_share_*`，可复查 stage-type 分布。
- **Artifacts**：
  - Logs: `experiments/v39/artifacts/logs/dev_v39_stageaware_gate_v1_longrun.log`
  - Checkpoints & JSON（含 `schedule_trace.json`、`step_one_stage_summary.json`、`eval_multi_history.jsonl`）: `experiments/v39/artifacts/checkpoints/dev_v39_stageaware_gate_v1_longrun/`
  - TensorBoard: `experiments/v39/artifacts/tb_logs/dev_v39_stageaware_gate_v1_longrun/`
  - Plots: `experiments/v39/artifacts/plots/dev_v39_stageaware_gate_v1_longrun_*.png`
  - Media (curve GIF + multi-eval mp4): `experiments/v39/artifacts/media/dev_v39_stageaware_gate_v1_longrun_*`

## Learning / 下一步
1. **Gate 仍被 main 阶段早逝牵制**：NE 阶段采用 0.13 阈值后不再触发 failure，但 main 阶段在 iter48 的 `first_death_p10=82.5`、`early_death₁₀₀=0.111` 直接导致 `failure_hold`。下一轮打算把 multi-eval episodes 再提升（例如 20/40）并区分 main 阶段的 success/failure window，避免单次尖峰拉低长期趋势。
2. **NE 阶段覆盖时间仍不足**：`remaining_by_type` 表明 iter32 之后 NE 队列已枯竭；此后 step-one 事件 100% 来自 main（占 74.8%）。考虑在 `tail_seed_stage_spec` 中穿插第三段 `type=ne`（例如迭代 30~36）或在 `TailSeedCycler` 中支持按类型循环。
3. **Tail warning 触发过晚**：新的 0.68 阈值 + patience=2 只在 iter59 告警一次。可尝试提高阈值到 0.7 并在 `schedule_trace` 中记录“警告持续 iter 数”，方便下一轮做更灵敏的提示。
4. **Stage-aware step-one 分析待自动化**：虽已生成 `step_one_stage_summary.json`，但尚未把 NE/main 的 share 直接可视化。建议新增脚本把该 JSON 拟合成 SVG（加到 `artifacts/plots`），并把 `ne` vs `main` 的 `step_one_top_share` 一同写进 `dev_v40` 文档模板。
