# dev_v35

## 启动
- 2025-11-12 18:10 PST 阅读 SOP 以及 dev_v34 结论：当前强制截断没有真正消费 tail 队列，Penalty gate 仍停留在 stage0，NE 高速样本占比在 iter60 飙升至 35% 以上。
- 本轮目标：落地“真正的 tail 注入机制”、放宽并改写 gate 指标逻辑、构建 NE counter-sample 管线，并补充 forced reset vs tail injection 监控。

## 观察
1. 强制 reset 仅改变 rollout 截断，`prewarm_override_ratio` 自 iter12 起仍为 0；tail seeds 1 万条几乎未被消费，`tail_queue_remaining_ratio`≈1.0。
2. Penalty gate 成功判据绑定训练端 `first_death_p10`，被 224-step 截断污染，导致 multi-eval 即便提升到 avg_final 0.61 也无法晋级。
3. NE hotspot 样本高度集中（step-one NE 高速 share 达 35.7%），缺乏反向采样（counter-sample）导致 curriculum 难以稀释风险区域。
4. 监控空缺：缺少 forced reset 后是否注入 tail 的 per-env 统计，schedule_trace 也无法定位未消费队列的 env。

## 实验计划
1. **Tail 注入机制**：在 env reset 钩子新增 `refill_tail_queue()` 或循环读取能力，并在 forced reset 后显式触发 tail seed 覆盖，目标把 `prewarm_override_ratio` 拉到 ≥0.3。
2. **Penalty gate 指标改写**：切换至 multi-eval early_death median≥双门槛 + `first_death_p10≥70`，并放宽 stage0 phase limit>3，避免长期 `failure_hold`。
3. **NE counter-sample**：从 dev_v34 step-one worst seeds 中提取 NE 高速样本，生成 `tail_seed_mix_v34_ne.json`，并在 v35 tail spec 中注入 ≥20% 份额。
4. **监控增强**：新增 per-env `prewarm_injected` 与 `forced_reset` 计数，写入 schedule_trace/tb，确保可定位未消费 tail seeds 的 env。
5. **主实验**：基于 v34 的 train 配置，启用新 tail 注入+gate+NE mix，128 env / n_steps 128 / total_iter≥60，输出统一写入 `experiments/v35/artifacts/`。

## 实验记录

### dev_v35_tail_loop_stage3_full1（128 env，n_steps=128，60 iter，tail_force_reset_steps=224，skip_final_eval）
- **命令**
  ```bash
  python experiments/v35/train.py \
    --run_name dev_v35_tail_loop_stage3_full1 \
    --total_iterations 60 --num_envs 128 --n_steps 128 \
    --batch_size 1024 --n_epochs 5 --learning_rate 2.5e-4 \
    --curriculum 15:9,20:10,25:9,25:8,25:8,25:8,25:8 \
    --multi_eval_interval 12 --multi_eval_probe_episodes 12 --multi_eval_seed_base 411232 \
    --eval_multi_fish 96 --eval_multi_episodes 24 \
    --penalty_gate_phase_allowance 4,5,6 --penalty_gate_required_successes 2 \
    --penalty_gate_success_step_one 18 --penalty_gate_success_step_one_ratio 0.02 \
    --penalty_gate_success_early_death 0.115 --penalty_gate_success_median_window 2 \
    --penalty_gate_freeze_iterations 5 --penalty_gate_success_freeze_iterations 2 \
    --penalty_gate_failure_p10 95 --penalty_gate_failure_min_final 0.5 --penalty_gate_failure_tolerance 2 \
    --tail_seed_stage_spec "24:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage1.json;16:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage1.json;20:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage2.json;20:experiments/v33/artifacts/checkpoints/tail_seed_mix_v33_stage3.json" \
    --tail_force_reset_steps 224 --tail_replay_count 2 --video_num_fish 48 \
    --skip_final_eval --seed 411232
  ```
- **训练概览**：完成 60 iter（约 20 分钟），128 env 在 224 步即被截断，训练端 `survival_rate` 稳定在 0.81~0.85，`first_death_p10` 因强制截断仍≈1。Penalty gate 在 iter12/24/36/48 的 multi-eval 依旧 `failure_hold`，stage limit 锁在 3。
- **Multi-eval（96 fish / 24 epi）**

  | iter | avg_final | min_final | early_death₁₀₀ | first_death_p10 | step-one ratio | NE高速 share |
  | --- | --- | --- | --- | --- | --- | --- |
  | 12 | 0.416 | 0.208 | 0.110 | 88.1 | 1.91% | 0.00 |
  | 24 | 0.363 | 0.177 | 0.121 | 68.0 | 2.08% | 8.3% |
  | 36 | 0.469 | 0.302 | 0.108 | 84.3 | 1.48% | 11.8% |
  | 48 | 0.567 | 0.344 | 0.076 | 122.1 | 2.60% | 6.7% |

- **Tail 监控**：`tail_forced_resets` 按期望在奇数 iter 固定 128，但新指标 `tail_prewarm_injections`、`prewarm_override_ratio` 在 iter≥9 迅速跌为 0，说明 forced reset 并未触发新的 tail 覆盖；`tail_queue_remaining_ratio` 仍 1.0，验证 env 重建后 `total_overrides` 被清零，TailStageTracker 无法反映真实消耗。
- **NE 热区**：iter36 `step_one_ne_high_speed_share` = 11.7%，iter48 降到 6.7%，但缺少新的 counter-sample mix，NE 仍反复出现。
- **产物**：
  - 日志：`experiments/v35/artifacts/logs/dev_v35_tail_loop_stage3_full1.log`
  - Checkpoints & stats：`experiments/v35/artifacts/checkpoints/dev_v35_tail_loop_stage3_full1/`（含 `model_iter_*.zip`、`model_final.zip`、`training_stats.pkl`、`schedule_trace.json`、`eval_multi_history.jsonl`、`step_one_worst_seeds.json`）
  - TensorBoard：`experiments/v35/artifacts/tb_logs/dev_v35_tail_loop_stage3_full1/`
  - Plots：`experiments/v35/artifacts/plots/` 下包含 survival/first-death/penalty 对齐、multi-eval timeline、step-one polar。
  - 媒体：`experiments/v35/artifacts/media/`（包含 tail 回放 mp4 以及 `dev_v35_tail_loop_stage3_full1_curve.gif`）。

## Learning / 下一步
1. **Tail 注入仍未生效**：新计数器显示 `tail_prewarm_injections=0`，forced reset 仅截断 episode，并未真正将 tail seed 重新压入队列；根因是 env 在切换 curriculum 阶段时被重建，`prewarm_predator_velocity_queue` 与 `total_overrides` 也被一并重置。需要：
   - 把 `tail_seed_sequences` 抽象为可持久的 `TailSeedCycler`，在阶段切换时继续沿用剩余队列，并将“增量 overrides”累加到全局 tracker，而非依赖 env 内部的 total 计数。
   - 在 `SingleFishEnv` 层保留 per-env pointer，并在 `reset()` 时显式写入 `enqueue_prewarm_velocity`（当前逻辑需确认 _tail_seed_library 在阶段 2+ 仍存在）。
2. **TailStageTracker 结果偏差**：由于每次重建 env 会把 `total_overrides` 清零，`tail_queue_remaining_ratio` 永远≈1。下一步改为使用 `overrides`（增量）来维护一个 callback 级别的累计 consumed 计数，从而正确显示阶段消耗，并把该值写回 `schedule_trace`。
3. **Penalty gate 仍卡在 stage0**：multi-eval iter48 已达到 early_death 0.076、first_death_p10 122，但 gate 仍触发 `failure_hold`。需要在 v36 把 success 判据切换到 multi-eval median + rolling early_death，并将 stage limit 与 phase allowance 解绑，避免 forced reset 训练信号污染后 gate 无法晋级。
4. **NE counter-sample**：iter24/36 在 NE 区域分别出现 8%/12% 的集中度，需要按照计划生成 `tail_seed_mix_v34_ne.json` 并在下一轮 tail spec 中注入 ≥20%，同时在 `export_tail_diagnostics` 结果中标注 NE cluster。
5. **实验扩展**：在 tail injection 修复后，再跑一次 60 iter 主实验并比较 `prewarm_override_ratio`、`tail_prewarm_injections` 与 multi-eval early death，确认 forced reset 真正把 tail 队列拉低到 `remaining_ratio≤0.7`。
