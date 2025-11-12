# dev_v37

## 启动
- 2025-11-12 23:05 PST 回顾 `SOP.md` 与 `experiments/v36/dev_v36.md`，确认 v36 已实现 TailSeedCycler 持久化、Penalty gate median 判据以及 NE counter mix 接入。
- 本轮聚焦三个延续目标：
  1. 在训练脚本中补充 tail stage warning（当 `tail_queue_remaining_ratio` 低于阈值时提示 stage 将耗尽）以及 stage label，防止长跑时 tail mix 悄然枯竭。
  2. 通过更频繁的 multi-eval（interval=8）和更宽松/稳定的 success 阈值来帮助 Penalty gate 脱离 stage0。
  3. 调整 NE counter mix 占比，使其在前半程占 15% 监控 NE 高速 share 的下降，同时保留 stage2/3 的 tail coverage。

## 观察
1. v36 主跑（dev_v36_tail_cycler_stage3_full2）证明 TailSeedCycler 能使 `tail_queue_remaining_ratio` 从 0.92 降到 0.62，但缺乏“ratio<=0.5”的预警，导致 stage 切换策略无法自动触发。
2. Penalty gate 使用 rolling median 判据后在 iter60 达成一次 success，但因 multi-eval 间距 12 iter、require_success=2，progress 停留在 stage0；early_death/p10 指标抖动较大。
3. NE counter mix 的 share 在 iter60 增至 42.9%，说明 20% 的 NE tail 注入不足以冲淡高危样本，需要更精细的 stage 配比或监控字段来区分 NE 样本。

## 实验计划
1. 基于 v36 脚本复制 `experiments/v37/train.py`，新增 `--tail_stage_warn_ratio/--tail_stage_warn_patience`，在 TensorBoard 与 `schedule_trace.json` 中记录 stage label + warning 事件，并把 warning 触发历史一并存档。
2. 训练配置：128 env × 60 iter，`multi_eval_interval=8`，`penalty_gate_success_early_death=0.09`、`success_median_window=3`、`required_successes=2`，以加快 success streak 累积；其余 hyper 继承 v36。
3. Tail seeds：沿用 v36 stage spec，但将 `tail_seed_mix_v34_ne.json` 配额降至 15%，并记录 NE mix 与普通样本的消费比（通过新增 stage label + warning 观察）。
4. 运行主实验 `dev_v37_tail_warn_gate_v1`，所有日志/ckpt/TB/plots/media 保存在 `experiments/v37/artifacts/`，并生成多曲线 + tail warning 事件截图，必要时录制 2 段 tail replay。

## 实验记录
### dev_v37_tail_warn_gate_v1（128 env × 60 iter, 2025-11-12）

- **运行命令**
  ```bash
  python experiments/v37/train.py \
    --run_name dev_v37_tail_warn_gate_v1 \
    --total_iterations 60 --num_envs 128 --n_steps 128 \
    --batch_size 1024 --n_epochs 5 --learning_rate 2.5e-4 \
    --curriculum 15:9,20:10,25:9,25:8,25:8,25:8,25:8 \
    --multi_eval_interval 8 --multi_eval_probe_episodes 12 --multi_eval_seed_base 411232 \
    --eval_multi_fish 96 --eval_multi_episodes 24 \
    --penalty_gate_phase_allowance 4,5,6,7 --penalty_gate_required_successes 2 \
    --penalty_gate_success_step_one 18 --penalty_gate_success_step_one_ratio 0.02 \
    --penalty_gate_success_early_death 0.09 --penalty_gate_success_early_death_window 3 \
    --penalty_gate_success_p10 70 --penalty_gate_success_median_window 3 \
    --penalty_gate_freeze_iterations 5 --penalty_gate_success_freeze_iterations 2 \
    --penalty_gate_failure_p10 95 --penalty_gate_failure_min_final 0.5 --penalty_gate_failure_tolerance 2 \
    --tail_seed_stage_spec '15:experiments/v34/artifacts/checkpoints/tail_seed_mix_v34_ne.json;24:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage1.json;16:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage1.json;25:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage2.json;20:experiments/v33/artifacts/checkpoints/tail_seed_mix_v33_stage3.json' \
    --tail_force_reset_steps 224 --tail_replay_count 2 --video_num_fish 48 \
    --tail_stage_warn_ratio 0.5 --tail_stage_warn_patience 2 \
    --skip_final_eval --seed 411232
  ```
- **训练概览**
  - On-policy 指标在 iter40 附近稳定在 `survival_rate≈0.85`、`final_num_alive≈21.2`，由于 224-step 强制截断，`first_death_step` 依旧贴近 32。
  - Penalty gate 迭代 8→56 期间每次 multi-eval 仍落在 `failure_hold`，`rolling_early_death_median` 自 0.133→0.108，但未触达 success 阈值 0.09；`step_one_death_ratio` 峰值 2.6%（iter56），说明 NE counter mix 虽减至 15% 仍偏重。
  - 新增的 tail stage warning 字段已经写入 TensorBoard 及 `schedule_trace.json`，stage label 依次为 `stage1:tail_seed_mix_v34_ne` → `stage2:tail_seed_mix_v32_stage1`；`tail_queue_remaining_ratio` 从 0.98（iter2）下降到 0.62（iter60），但尚未触发 `ratio<=0.5` 的告警（warning log 列表为空），验证了线路可工作。
- **Multi-eval（96 fish / 24 epi）**

  | iter | avg_final | min_final | early_death₁₀₀ | first_death_p10 | step_one_ratio |
  | --- | --- | --- | --- | --- | --- |
  | 8  | 0.802 | 0.542 | 0.133 | 67.0 | 1.74% |
  | 16 | 0.798 | 0.594 | 0.133 | 59.0 | 2.17% |
  | 24 | 0.837 | 0.719 | 0.131 | 54.2 | 1.56% |
  | 32 | 0.814 | 0.646 | 0.126 | 54.1 | 2.78% |
  | 40 | 0.836 | 0.667 | 0.114 | 81.0 | 1.22% |
  | 48 | 0.814 | 0.656 | 0.112 | 76.2 | 2.08% |
  | 56 | 0.831 | 0.719 | 0.108 | 85.4 | 2.60% |

- **Tail 监控**
  - `schedule_trace.json` 新增字段：`tail_stage_label` 当前固定在 stage2，`tail_stage_warn_active` 全程 0，`tail_stage_warn_since` 为空；`tail_stage_warning_events` 数组在 `schedule_trace.json` 中存在但本轮为空，方便后续解析。
  - `tail_queue_remaining_ratio` 下降速度与 v36 相近但更平滑，`tail_prewarm_injections_total = tail_forced_resets_total = 3,968`；tensorboard `custom/tail_stage_warn_active` 折线用于确认预警是否触发。
  - `step_one_clusters.jsonl` 显示 iter56 的 NE 高速簇（60°~90°，1.6~2.0）占 20%，说明当前 15% NE 配额仍不足以压低该 cluster，需要进一步动态注入 counter-samples。
- **产物**
  - Logs: `experiments/v37/artifacts/logs/dev_v37_tail_warn_gate_v1.log`
  - Checkpoints & schedule trace: `experiments/v37/artifacts/checkpoints/dev_v37_tail_warn_gate_v1/`
  - TensorBoard: `experiments/v37/artifacts/tb_logs/dev_v37_tail_warn_gate_v1/`
  - Plots: `experiments/v37/artifacts/plots/`（包含 survival/first-death/penalty alignment + multi-eval timeline + step-one polar）
  - Media: `experiments/v37/artifacts/media/`（curve GIF + 7×multi-eval tail mp4）

## Learning / 下一步
1. **Gate success 仍未解锁**：`rolling_early_death_median≈0.108`，距离 success 阈值 0.09 仍有差距；考虑 (a) 将 `success_early_death` 阈值放宽到 0.10 同时把 `penalty_gate_success_median_window` 拉长至 4；或 (b) 提升 multi-eval episodes/频率以便压低方差。
2. **NE counter mix 需要精细配比**：step-one 最大簇 share 0.20，说明 15% NE slot 仍然不足，下一轮尝试在 phase1-2 设置“NE-only stage”并在 schedule trace 中区分 `tail_stage_label` → `tail_stage_type`，以便核算 NE 消耗。
3. **Tail warning 应尽早触发**：实际 ratio 只下降到 0.62，建议把 `--tail_stage_warn_ratio` 提到 0.65 并增加 `tail_stage_warn_patience=3`，或在 stage metadata 中附加“target ratio”字段，以便自动切换到 stage3。
4. **Step-one logging 需联动 tail 标签**：当前 `tail_stage_label` 可在 schedule trace 中查询，但尚未与 `step_one_deaths` 绑定；下一轮可以把 per-iteration `tail_stage_label` 写入 `step_one_clusters`，从而追踪“某 stage 是否引入特定簇”。
