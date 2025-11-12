# dev_v40

## 启动
- 2025-11-12  (UTC-8) 依 SOP 复盘 dev_v39 的 Learning/下一步，并盘点 `train.py` 改动、日志与 artifacts。
- dev_v39 已确认 NE 阶段的 stage-aware gate 生效，但 main 阶段的 `first_death_p10` 下坠仍卡住 gate；本轮聚焦 main 稳定性、NE 覆盖时长以及 tail warning 灵敏度。
- 目标：把 stage-aware gate 真正拓展到 main 段（提高 multi-eval 采样、引入 stage-specific window）、补足第三段 NE-only 课程，并把 step-one by stage 的可视化自动化，保证 run 全量 artifacts 写入 `experiments/v40/artifacts/`。

## 观察（承接 dev_v39）
1. main 阶段 `first_death_p10` 在 iter48 暴跌至 82.5，连带 `early_death₁₀₀=0.111`、`step_one_ratio=0.0299`，导致 gate 长期 failure_hold；multi-eval 16/32 样本不足以平滑波动。
2. `tail_stage_remaining_by_type` 显示 iter32 之后 NE 队列枯竭，step-one 事件 74.8% 集中在 main，说明追加的 2 段 NE 仍不够覆盖后半程。
3. 将 `tail_stage_warn_ratio`=0.68、patience=2 只在 iter59 告警一次；缺乏更早的“NE 阶段耗尽”提示。
4. `step_one_stage_summary.json` 已含可用数据但暂无自动绘图/引用路径，下一轮启动仍需手工 grep。

## 实验计划
1. **Stage-aware gate + multi-eval 提升**：在 `experiments/v40/train.py` 中保留 NE=0.13 阈值，新增 main 段独立窗口（示例：`penalty_gate_success_early_death_main_window=4`，`failure_window=3`）并把 `multi_eval_probe_episodes`/`eval_multi_episodes` 各增至 20/40 以降低方差；默认 `--num_envs 128`、`--total_iterations 60`。
2. **NE 覆盖延长**：扩展 `tail_seed_stage_spec`，在 iter30~36 再插入一段 `type=ne`（可循环使用 v34 NE checkpoints），并在 `TailSeedCycler` 中记录 `ne_reinjections` 指标输出到 `schedule_trace.json`。
3. **Tail warning 灵敏化**：将阈值提高至 0.70，记录 warning 累积时长（连续 iter 次数）写入 `tail_stage_warning_events` 以辅助调参。
4. **Step-one by stage 可视化**：新增 JSON→PNG/SVG 脚本或 TensorBoard scalar，把 `step_one_stage_summary` 自动绘制成 stacked bar，并将输出复制到 `experiments/v40/artifacts/plots/`，确保在 `dev_v40.md` 引用。

## 观察记录
- **On-policy**：`training_stats.pkl` 显示最近 10 iter 平均存活率 **0.849**（最低 0.837），step-one 死亡占比稳定在 **1.98%**，证明 main 阶段的 jitter 已被压缩，但 gate 仍未解锁到 stage1。
- **Multi-eval**：20/40 探针让迭代 56 的 main 段达到 `avg_final=0.768`、`first_death_p10=108.9`、`early_death₁₀₀=0.094`，相比 dev_v39 iter48 的 82.5 明显改善；然而 iter32/40 仍出现 `p10<90` 的 failure_hold，说明 main 段仍需额外稳态信号。
- **Instrumentation**：`schedule_trace.json` 现记录 `tail_ne_reinjections=2`，`tail_stage_warn_duration` 也写回 trace；`step_one_stage_summary` 透出 main vs NE 事件占比 **57.6% / 42.4%**，并由新生成的 `experiments/v40/artifacts/plots/dev_v40_main_rebalance_v1_step_one_stage_distribution.png` 可视化。
- **Tail warning**：新阈值 0.70 在 iter57 提前触发告警，duration 字段显示持续 4 iter（直到 iter60），为下一轮评估“NE 阶段快耗尽”提供可引用的数字。

## 实验记录

### dev_v40_main_rebalance_v1（128 env × 60 iter）
```
python experiments/v40/train.py --run_name dev_v40_main_rebalance_v1 \
  --total_iterations 60 --num_envs 128 --n_steps 128 --batch_size 1024 --n_epochs 5 --learning_rate 2.5e-4 \
  --curriculum 15:9,20:10,25:9,25:8,25:8,25:8,25:8 --multi_eval_interval 8 \
  --multi_eval_probe_episodes 20 --eval_multi_fish 96 --eval_multi_episodes 40 --multi_eval_seed_base 411232 \
  --penalty_gate_phase_allowance 4,5,6,7 --penalty_gate_required_successes 2 --penalty_gate_freeze_iterations 5 \
  --penalty_gate_success_step_one 18 --penalty_gate_success_step_one_ratio 0.02 \
  --penalty_gate_success_early_death 0.10 --penalty_gate_success_early_death_window 3 \
  --penalty_gate_success_early_death_by_stage ne:0.13,main:0.10 --penalty_gate_success_early_death_window_by_stage main:4 \
  --penalty_gate_success_p10 70 --penalty_gate_success_freeze_iterations 2 --penalty_gate_success_median_window 4 \
  --penalty_gate_failure_p10 95 --penalty_gate_failure_min_final 0.5 --penalty_gate_failure_tolerance 2 \
  --penalty_gate_failure_tolerance_by_stage main:3 --tail_stage_warn_ratio 0.70 --tail_stage_warn_patience 2 \
  --tail_seed_stage_spec '8:experiments/v34/artifacts/checkpoints/tail_seed_mix_v34_ne.json|type=ne;6:experiments/v34/artifacts/checkpoints/tail_seed_mix_v34_ne.json|type=ne;15:experiments/v34/artifacts/checkpoints/tail_seed_mix_v34_ne.json|type=main;6:experiments/v34/artifacts/checkpoints/tail_seed_mix_v34_ne.json|type=ne;24:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage1.json|type=main;16:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage1.json|type=main;25:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage2.json|type=main;20:experiments/v33/artifacts/checkpoints/tail_seed_mix_v33_stage3.json|type=main' \
  --tail_force_reset_steps 224 --tail_replay_count 2 --video_num_fish 48 --skip_final_eval --seed 411232
```

- **On-policy**：iter51-60 的均值 `survival=0.849`、`first_death_step≈402`，step-one 占比保持在 `1.98%`，说明额外 NE 阶段并未拖累主干训练速度。
- **Multi-eval 20/40 概览**：

| iter | stage | avg_final | min_final | early_death₁₀₀ | first_death_p10 | step_one_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 8  | ne   | 0.716 | 0.615 | 0.117 | 83.0  | 0.0208 |
| 16 | ne   | 0.553 | 0.417 | 0.099 | 101.0 | 0.0198 |
| 24 | main | 0.654 | 0.573 | 0.099 | 100.9 | 0.0151 |
| 32 | main | 0.668 | 0.594 | 0.107 | 87.0  | 0.0240 |
| 40 | main | 0.688 | 0.552 | 0.110 | 89.0  | 0.0260 |
| 48 | ne   | 0.721 | 0.594 | 0.105 | 95.0  | 0.0219 |
| 56 | main | 0.768 | 0.615 | 0.094 | 108.9 | 0.0198 |

  iter48 的第三段 NE-only 让 `tail_ne_reinjections` 升至 2，并在 iter56 main 段达到 `avg_final≈0.77`；但 iter32/40 仍触发 failure_hold，表明 main 阶段仍欠缺足够的 rolling p10 余量。
- **Gate 行为**：`penalty_stage_debug.jsonl` 显示 main 段应用了 `failure_tolerance=3`（新覆盖）后未再回退，但 success_streak 始终被`first_death_p10<100` 打断，gate 仍停在 stage0。
- **Tail 观测**：`schedule_trace.json` 记录了 iter57 的 warning（ratio=0.70，duration=4），以及 `tail_stage_reinjections_by_type.ne=2` 的新字段。
- **Step-one by stage**：`step_one_stage_summary.json` + `plots/dev_v40_main_rebalance_v1_step_one_stage_distribution.png` 给出了 main:NE = 163:120 的事件分布，较 dev_v39 的 75%:25% 有效提升 NE 覆盖。
- **Artifacts**：
  - 日志与指标：`experiments/v40/artifacts/logs/dev_v40_main_rebalance_v1.log`、`.../tb_logs/dev_v40_main_rebalance_v1/`、`.../checkpoints/dev_v40_main_rebalance_v1/penalty_stage_debug.jsonl`
  - 度量 JSON：`.../eval_multi_history.jsonl`、`.../schedule_trace.json`、`.../step_one_stage_summary.json`
  - 图像/媒体：`experiments/v40/artifacts/plots/` 下的 survival/first-death/step-one-stage 分布图，GIF + 14 个 mp4 位于 `experiments/v40/artifacts/media/`（含 iter56 tail 播放）。

## Learning / 下一步
1. **让 gate 真正进入 stage1**：目前 main 段 `first_death_p10` 平稳在 95~109 但仍被 `success_p10=70` + `median_window=4` 判为失败，考虑在 main 段单独设置 `success_first_death_p10`/`success_median_window` 或将 success 判据改为“最近 3 次均 >100”以积累 success_streak。
2. **补强 main 段稳定性**：iter32/40 的 `p10<90` 说明 tail seeds 仍包含过难样本；下一轮可尝试对 `tail_seed_mix_v32_stage1/2` 做速度聚类过滤，或在 main 段增加 1~2 个 “缓冲” checkpoints 以平滑重新插入的 NE block。
3. **Tail warning 可视化**：既然 `tail_stage_warn_duration` 已落地，应新增 TensorBoard 曲线或自动 PNG，将 warning duration 与 `remaining_ratio` 同图显示，便于判断阈值 0.70 是否仍偏晚；这也能为后续调参（例如动态阈值）提供依据。
