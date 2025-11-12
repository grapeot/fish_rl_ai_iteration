# dev_v41

## 启动
- 2025-11-12 (UTC-8) 依照 SOP 回顾 dev_v40。上一轮虽然在 main 段把 multi-eval 提升到 20/40，并实现 NE reinjection 指标，但 gate 依旧卡在 stage0，说明 success 判据仍未跟上 main 收敛速度。
- dev_v40 的学习要点：
  - main 段 `first_death_p10` 常态可到 95~109，但 rolling median 需要 ≥4 次合格才算成功，导致 success_streak 经常被单次抖动打断。
  - tail seed 中穿插的 v32/v33 高速 main checkpoints 偶尔让 iter32/40 出现 `p10<90` 的 failure_hold，暴露“NE→main”衔接仍过硬。
  - `tail_stage_warn_duration` 已写入 trace，但仍缺少可视化，下一轮复盘仍需手动 grep。

## 观察
1. **Gate 进度瓶颈**：main 段 early-death 已满足 0.10 的 stage override，`step_one_ratio≈0.02` 也达标，但 `success_p10=120` 且 median window=4 时，滚动窗口常被单次 89/90 的样本拖低；缺少 stage-aware `success_p10` 与窗口调节。
2. **Main 稳态信号不足**：tail seed spec 仍在 main 段重复使用 stage1/2 的高难 checkpoint，iter32/40 的 `failure_hold` 说明 reinjection 缺少“缓冲”样本，会在 gate 刚准备晋级时引入尖刺。
3. **Tail Warning 可视性**：虽然 trace 中已有 `tail_stage_warning_events` 与 `duration` 字段，但缺少 TensorBoard/PNG。下一轮若再遇 NE 枯竭，仍得翻 JSON 才能知道 warning 是否持续。

## 实验计划
1. **Stage-aware success_p10**：在 v41 训练脚本中新增 `--penalty_gate_success_first_death_p10_by_stage` 与 `--penalty_gate_success_p10_median_window_by_stage`，main 段使用阈值 100、窗口 3；其余阶段沿用默认 120/4，以便 main 段 rolling median 更快积累 success。
2. **Tail seed 平滑化**：构造新的 `tail_seed_stage_spec`，在 main/NE 交接之间插入“缓冲” checkpoint（例如 v33-stage2 的较慢样本），并保持 NE reinjection 频率≥2。本轮重点观察 iter24~40 的 multi-eval `p10` 是否仍掉到 90 以下。
3. **Tail warning 可视化**：在 v41 `train.py` 中输出 `tail_stage_warning.png`，展示 `tail_queue_remaining_ratio` 曲线 + warning duration 柱；在 dev 文档引用该图，后续迭代无需手动解析 trace 即可判断 warning 触发点。

## 运行记录
### dev_v41_stage_override_longrun_v1（128 env × 60 iter）
```
python experiments/v41/train.py --run_name dev_v41_stage_override_longrun_v1 \
  --total_iterations 60 --num_envs 128 --n_steps 128 --batch_size 1024 --n_epochs 5 --learning_rate 2.5e-4 \
  --curriculum 15:9,20:10,25:9,25:8,25:8,25:8,25:8 --multi_eval_interval 8 \
  --multi_eval_probe_episodes 20 --eval_multi_fish 96 --eval_multi_episodes 40 --multi_eval_seed_base 411232 \
  --penalty_gate_phase_allowance 4,5,6,7 --penalty_gate_required_successes 2 --penalty_gate_freeze_iterations 5 \
  --penalty_gate_success_step_one 18 --penalty_gate_success_step_one_ratio 0.02 \
  --penalty_gate_success_early_death 0.10 --penalty_gate_success_early_death_window 3 \
  --penalty_gate_success_early_death_by_stage ne:0.13,main:0.10 --penalty_gate_success_early_death_window_by_stage main:4 \
  --penalty_gate_success_p10 70 --penalty_gate_success_first_death_p10_by_stage main:100 \
  --penalty_gate_success_freeze_iterations 2 --penalty_gate_success_median_window 4 \
  --penalty_gate_success_p10_median_window_by_stage main:3 \
  --penalty_gate_failure_p10 95 --penalty_gate_failure_min_final 0.5 \
  --penalty_gate_failure_tolerance 2 --penalty_gate_failure_tolerance_by_stage main:3 \
  --tail_stage_warn_ratio 0.70 --tail_stage_warn_patience 2 \
  --tail_seed_stage_spec '10:experiments/v34/artifacts/checkpoints/tail_seed_mix_v34_ne.json|type=ne|label=ne_start;8:experiments/v34/artifacts/checkpoints/tail_seed_mix_v34_ne.json|type=ne|label=ne_boost;18:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage1.json|type=main|label=main_stage1a;12:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage1.json|type=main|label=main_stage1b;18:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage2.json|type=main|label=main_buffer;10:experiments/v34/artifacts/checkpoints/tail_seed_mix_v34_ne.json|type=ne|label=ne_refresh;14:experiments/v33/artifacts/checkpoints/tail_seed_mix_v33_stage3.json|type=main|label=main_stage3a;10:experiments/v33/artifacts/checkpoints/tail_seed_mix_v33_stage3.json|type=main|label=main_stage3b' \
  --tail_force_reset_steps 224 --tail_replay_count 2 --video_num_fish 48 --skip_final_eval --seed 411235
```
- **On-policy**：迭代 51-60 的平均在轨存活率 0.8616，`first_death_step≈38`（tail replay 覆盖导致一阶统计偏低），`step_one_ratio` 均值 0.0198，`logs` 详见 `experiments/v41/artifacts/logs/dev_v41_stage_override_longrun_v1.log`。
- **Multi-eval 20/40**（96 fish）：

| iter | stage | avg_final | min_final | early_death₁₀₀ | first_death_p10 | step_one_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | ne | 0.792 | 0.375 | 0.0943 | 108.0 | 0.0214 |
| 16 | ne | 0.851 | 0.688 | 0.0839 | 151.0 | 0.0203 |
| 24 | ne | 0.840 | 0.635 | 0.0625 | 165.5 | 0.0151 |
| 32 | main | 0.821 | 0.583 | 0.0844 | 116.0 | 0.0245 |
| 40 | main | 0.840 | 0.490 | 0.0917 | 107.9 | 0.0260 |
| 48 | main | 0.887 | 0.792 | 0.0750 | 338.5 | 0.0219 |
| 56 | main | 0.857 | 0.573 | 0.0870 | 219.9 | 0.0198 |

  - stage-aware rolling median成功抬升 main 段 `first_death_p10`，iter48/56 均 >200，但 `step_one_death_count` 长期维持在 38~50，超过 success 阈值 18，导致 gate 仍停留 stage0。
  - iter40 再次触发 `failure_hold`（min_final=0.49 < 0.5），说明 main-tail 缓冲仍不足以保证最差样本。
- **Tail 阶段 & warning**：新的 `tail_stage_warning.png`（`experiments/v41/artifacts/plots/dev_v41_stage_override_longrun_v1_tail_stage_warning.png`）展示 iteration47 剩余比率跌破 0.70，warning 持续 14 iter；`schedule_trace.json` 记录 `tail_ne_reinjections=1`，未达到目标≥2。
- **Step-one by stage**：`step_one_stage_summary.json` 显示 main:NE 事件 = 177:109（62%:38%），较 dev_v40 的 57.6%:42.4% 更偏向 main，说明新增缓冲段仍不足以覆盖 NE 尾巴。
- **Artifacts**：
  - Checkpoints/JSON：`experiments/v41/artifacts/checkpoints/dev_v41_stage_override_longrun_v1/`
  - 日志：`experiments/v41/artifacts/logs/dev_v41_stage_override_longrun_v1.log`
  - TensorBoard：`experiments/v41/artifacts/tb_logs/dev_v41_stage_override_longrun_v1`
  - 曲线 & tail warning：`experiments/v41/artifacts/plots/`（survival/first-death/penalty/tail_stage_warning 等 PNG，multi-eval hist 时间线）
  - 媒体：`experiments/v41/artifacts/media/dev_v41_stage_override_longrun_v1_curve.gif` 与 iter08-56 tail mp4。
- **Sanity runs**：`dev_v41_main_gate_buffer_v1`（iter≤16）与 `dev_v41_stage_override_v1`（iter≤32）因 CLI 超时提前结束，但其日志/plots 仍保留作为阶段性诊断（gate 行为与 tail warning 早期数据）。

## Learning / 下一步
1. **放宽 step-one success 条件**：main 段 multi-eval 已稳定 `p10>200`，但 `step_one_death_count` 长期 38~50 使 gate 永远无法积累 success_streak。下一轮需考虑：
   - 把 success 判据改为“仅检查 ratio<=0.02”，或
   - stage-aware 地把 `success_step_one` 提高到 ≥40，同时保持 ratio 限制，以免出现硬性瓶颈。
2. **强化 NE reinjection**：`tail_ne_reinjections=1` 且 warning (iter47-60) 持续 14 iter，说明新的 stage spec 仍在 main 阶段过早耗尽 NE。可在 stage5 之后再插入一段 v34 NE 或新增 v33 stage2 checkpoint，把 warning 持续时间压回 <5 iter。
3. **提高最差样本稳态**：iter40 因 `min_final<0.5` 触发 failure_hold，显示缓冲 checkpoint 仍不足。建议：
   - 在 main_stage1b 与 stage3 之间加入“中速 main”样本（例如对 v32_stage2 进行速度裁剪），
   - 或在 tail cycler 中对 main 段施加权重衰减，优先消费 NE block 之后的新 checkpoint，以避免再度将 hardest main 样本塞到 gate 刚解锁期。
