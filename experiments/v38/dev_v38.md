# dev_v38

## 启动
- 2025-11-12 23:40 PST 重新阅读 `SOP.md` 与 `experiments/v37/dev_v37.md` 的 “Learning/下一步”，确认 v37 已经完成 tail stage warning 与 Penalty gate instrumentation，但 success 阈值偏严。
- 本轮聚焦 v37 留下的四个痛点：Penalty gate 卡在 stage0、NE counter mix 仍无法压降高速簇、tail warning 触发滞后以及 step-one logging 未带上 tail stage 标签。
- 目标：在 `experiments/v38/train.py` 中落实上述改动，复现实验并确保 artifacts / 文档齐全，为下一轮提供可复查的曲线与媒体。

## 观察
1. v37 主跑 `dev_v37_tail_warn_gate_v1` 的 `rolling_early_death_median` 稳定在 0.108，距 success 阈值 0.09 仅差 0.018，但因窗口较短 + 阈值苛刻无法解锁 gate。
2. NE counter mix 虽降到 15%，`step_one_top_share` 仍高达 0.20，`step_one_ne_high_speed_share` 也维持在 0.18~0.2，说明 stage 配比需要更细化（例如独立 NE-only stage）。
3. tail stage warning 功能已上线，但 ratio 只降到 0.62 未触发阈值 0.5，导致实际操作上仍缺乏“即将耗尽”的提醒。
4. `schedule_trace.json` 中已有 `tail_stage_label`，但 step-one 的 clusters 与 deaths 里没有 stage 标签，无法直接判断某个簇是否来自特定 stage。

## 实验计划
1. **Penalty gate**：把 `penalty_gate_success_early_death` 阈值放宽到 0.10，并把 `penalty_gate_success_median_window` 改为 4；同时维持 multi-eval interval=8，以稳定 rolling median。
2. **Tail stage spec**：复制 v37 stage 配方，新增 `stage0:NE-only`（迭代占比 8）用于集中消化 `tail_seed_mix_v34_ne.json`，并在 metadata 里加上 `stage_type`（"ne", "main"）。
3. **Tail warning**：默认把 `--tail_stage_warn_ratio` 提升到 0.65，`--tail_stage_warn_patience` 设为 3，确保在耗尽 35% 缓冲区前给予提示。
4. **Step-one logging**：在记录 step-one deaths / clusters 时附加当前 `tail_stage_label` 与 `stage_type`，并把该信息写入 `step_one_clusters.jsonl` 与 `step_one_worst_seeds.json`，方便后续 cross-reference。
5. **主实验**：128 env × 60 iter，沿用 v37 其他超参，run name `dev_v38_gate_relax_ne_stage_v1`；所有输出写入 `experiments/v38/artifacts/`，包括 TensorBoard、plots、媒体。

## 实验记录

### dev_v38_gate_relax_ne_stage_v1（中断）
- 128 env × 60 iter 同步配置，运行到 iter44（00:28 PST）时 CLI 两次超时被动终止；保留日志 / TB / media 方便对比，但不再作为 v38 结论。
- 产物（不推荐复用）：`experiments/v38/artifacts/logs/dev_v38_gate_relax_ne_stage_v1.log`、`.../tb_logs/dev_v38_gate_relax_ne_stage_v1/`、`.../media/dev_v38_gate_relax_ne_stage_v1_iter*.mp4`。

### dev_v38_gate_relax_ne_stage_v1_full（128 env × 60 iter，2025-11-12）

- **运行命令**
  ```bash
  python experiments/v38/train.py \
    --run_name dev_v38_gate_relax_ne_stage_v1_full \
    --total_iterations 60 --num_envs 128 --n_steps 128 \
    --batch_size 1024 --n_epochs 5 --learning_rate 2.5e-4 \
    --curriculum 15:9,20:10,25:9,25:8,25:8,25:8,25:8 \
    --multi_eval_interval 8 --multi_eval_probe_episodes 12 --multi_eval_seed_base 411232 \
    --eval_multi_fish 96 --eval_multi_episodes 24 \
    --penalty_gate_phase_allowance 4,5,6,7 --penalty_gate_required_successes 2 \
    --penalty_gate_success_step_one 18 --penalty_gate_success_step_one_ratio 0.02 \
    --penalty_gate_success_early_death 0.10 --penalty_gate_success_early_death_window 3 \
    --penalty_gate_success_p10 70 --penalty_gate_success_median_window 4 \
    --penalty_gate_freeze_iterations 5 --penalty_gate_success_freeze_iterations 2 \
    --penalty_gate_failure_p10 95 --penalty_gate_failure_min_final 0.5 --penalty_gate_failure_tolerance 2 \
    --tail_seed_stage_spec '8:experiments/v34/artifacts/checkpoints/tail_seed_mix_v34_ne.json|type=ne;15:experiments/v34/artifacts/checkpoints/tail_seed_mix_v34_ne.json|type=main;24:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage1.json|type=main;16:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage1.json|type=main;25:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage2.json|type=main;20:experiments/v33/artifacts/checkpoints/tail_seed_mix_v33_stage3.json|type=main' \
    --tail_force_reset_steps 224 --tail_replay_count 2 --video_num_fish 48 \
    --tail_stage_warn_ratio 0.65 --tail_stage_warn_patience 3 \
    --skip_final_eval --seed 411232
  ```
- **训练概览**
  - On-policy 最后 10 iter 平均 `survival_rate=0.847`，强制 224 步截断使得 `first_death_p10` 固定在 1；`avg_num_alive` 稳定在 21.3~21.5。
  - Tail 队列新增 `stage_type` telemetry：`ne` 阶段消耗 9 iter，随后 50 iter 处于 `main`；`tail_queue_remaining_ratio` 在 iter59 首次跌到 0.648，但因 `patience=3` 未触发 warning（`tail_stage_warnings=[]`）。
  - `step_one_top_share` 峰值 0.312（iter40，stage_type=main），`step_one_ne_high_speed_share` 峰值 0.375（iter32）；NE-only 阶段（iter≤9）最高 share 0.24。
  - Penalty gate 仍停留在 stage0：iter40 / 48 / 56 的 multi-eval 先后触发 `failure_hold`，`rolling_early_death_median` 始终 ≈0.115，未达到放宽后的 0.10 阈值。
- **Multi-eval（96 fish / 24 epi）**

  | iter | tail_stage_type | avg_final | min_final | early_death₁₀₀ | step_one_ratio | first_death_p10 |
  | --- | --- | --- | --- | --- | --- | --- |
  | 8  | ne   | 0.714 | 0.552 | 0.136 | 0.0217 | 70.0 |
  | 16 | main | 0.445 | 0.281 | 0.167 | 0.0191 | 65.1 |
  | 24 | main | 0.480 | 0.250 | 0.141 | 0.0165 | 64.0 |
  | 32 | main | 0.757 | 0.583 | 0.135 | 0.0252 | 58.0 |
  | 40 | main | 0.659 | 0.385 | 0.115 | 0.0139 | 76.0 |
  | 48 | main | 0.773 | 0.531 | 0.113 | 0.0208 | 81.0 |
  | 56 | main | 0.706 | 0.510 | 0.162 | 0.0252 | 68.1 |

  `eval_multi_history.jsonl` 现已把 `tail_stage_type` 写入每条记录，可直接筛选 NE-only vs main 阶段的表现。
- **Tail / Step-one instrumentation**
  - `schedule_trace.json` 新增字段：`tail_stage_type` + `tail_stage_label` 均随 iteration 写入；`tail_stage_warning_events` 为空，验证 patience=3 生效。
  - `step_one_clusters.jsonl` 与 `step_one_worst_seeds.json` 每条 sample 都带上 `tail_stage_label/type`，方便排查“stage0 的 NE 簇是否穿越到 stage2”。
  - `media/dev_v38_gate_relax_ne_stage_v1_full_iter*_tail_rank*.mp4` 对应 multi-eval tail 样本，`dev_v38_gate_relax_ne_stage_v1_full_curve.gif` 汇总 survival & penalty 走势。
- **Penalty gate 回顾**
  - iter40: `avg_final=0.659` 但 `early_death₁₀₀=0.115`，success streak 归零。
  - iter48: `avg_final=0.773` 仍被 `early_death₁₀₀=0.113` 拖住；`step_one_death_ratio=0.0208` 未过 ratio 阈值。
  - iter56: `early_death₁₀₀` 反弹至 0.162，直接抹平前期改动，说明需要进一步提升 multi-eval 样本数或重新设定 success window。
- **产物**
  - Logs: `experiments/v38/artifacts/logs/dev_v38_gate_relax_ne_stage_v1_full.log`
  - Checkpoints（含 `schedule_trace.json`、`step_one_clusters.jsonl` 等）: `experiments/v38/artifacts/checkpoints/dev_v38_gate_relax_ne_stage_v1_full/`
  - TensorBoard: `experiments/v38/artifacts/tb_logs/dev_v38_gate_relax_ne_stage_v1_full/`
  - Plots: `experiments/v38/artifacts/plots/`（timeline / penalty 对比 / polar + death hists）
  - Media: `experiments/v38/artifacts/media/`（曲线 GIF + iter8-56 tail replays）

## Learning / 下一步
1. **Early-death gating仍未解锁**：`rolling_early_death_median≈0.115`，而 24/32/48 iter 的 `avg_final` 已趋稳，说明 variance 仍然过大。下一轮计划把 `multi_eval_probe_episodes` 提升到 16 或者增大 `eval_multi_episodes`→32，以降低 median 抖动，并考虑把 success 阈值拆分为“NE-only 阶段用 0.13、main 阶段用 0.10”。
2. **NE stage 需要更细颗粒控制**：现有 `stage0` 只撑了 9 iter，`step_one_top_share` 最高反而出现在 main 阶段 (0.312)。可尝试在 `tail_seed_stage_spec` 中加入第二段 `type=ne`（例如 8+6 iter）并在 `TailStageTracker` 中暴露 `remaining_by_type`，以便强制 NE 病灶贯穿到 iter32。
3. **Tail warning 需降低触发条件**：ratio 已到 0.648 仍未告警，建议把 `tail_stage_warn_patience` 降为 2，同时把阈值升到 0.68，或者在 `schedule_trace` 中记录“连续满足次数”，便于下一轮调整。
4. **Step-one → stage_type 关联待利用**：数据已写入 JSON / worst_seeds，但当前分析仍靠手工。下一轮应实现一个 summary 脚本，将 `step_one_top_share` 分别对 `ne` / `main` 聚合，或直接在 TensorBoard 中绘制 `custom/step_one_top_share_by_type`，从而更快评估 NE-only stage 的收益。
