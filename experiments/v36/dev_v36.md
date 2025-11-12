# dev_v36

## 启动
- 2025-11-12 21:35 PST 阅读 `SOP.md` 与 `experiments/v35/dev_v35.md`，确认上一轮暴露的尾部注入失效、Penalty gate 长期 stage0、NE hotspot 未建立 counter-sample 以及监控缺口。
- 本轮目标：实现可跨 phase 持久的 tail seed 注入与消耗统计、改写 gate 成基于 multi-eval median 的双门槛逻辑，并补齐 NE counter mix 与监控数据，产出可交接的日志/曲线/媒体。

## 观察
1. **Tail 注入**：`tail_seed_sequences` 在 phase 切换时被重置，env 重建后不再携带任何 prewarm queue，导致 forced reset 只截断 episode；`tail_stage_tracker` 依赖的 `total_overrides` 也随 env 重建归零，`tail_queue_remaining_ratio` 永远≈1。
2. **Penalty gate**：success 判据依赖单次 multi-eval 的 `early_death_fraction_100` 与 `first_death_p10`，224-step 强制截断污染训练指标后 gate 长期 `failure_hold`，phase limit 锁在 3，无从验证高 penalty。
3. **NE Hotspot**：step-one 记录显示 NE 高速 share 在 iter24/36 达 8~12%，但缺乏来源于 dev_v34 worst seeds 的 counter-sample mix，tail spec 仍由 v32/v33 旧样本构成。
4. **监控空缺**：`tail_prewarm_injections_total`、`forced_resets_total` 同样随 env 重建而丢失进度，schedule trace 难以回溯“谁消费了 tail seeds”。

## 实验计划
1. **TailSeedCycler**：实现跨 env/phase 的 per-env pointer 与全局 consumption 计数，迭代中累積 overrides/injections/forced resets，并把 `remaining_ratio` 写回 schedule trace。
2. **Penalty gate 改写**：引入 multi-eval `early_death_fraction_100` 滚动 median（window≥2）+ `first_death_p10` median≥70 的成功判据，同时将 phase allowance 扩展到 4-7 以解除 stage0 限制。
3. **NE counter mix**：从 dev_v34 `step_one_worst_seeds` summary 中抽取 NE 高速簇，生成 `tail_seed_mix_v34_ne.json` 并在 `tail_seed_stage_spec` 中注入 ≥20% 份额。
4. **主实验**：以 `experiments/v36/train.py` 为基线，128 env / n_steps 128 / total_iter 60，启用 TailSeedCycler + 新 gate + NE mix；所有日志/ckpt/TB/plots/media 写入 `experiments/v36/artifacts/`，并提取关键指标（survival、first_death、tail consumption）。
5. **可视化/媒体**：沿用 plot + GIF + tail polar，必要时截取 tail replay mp4，复制到 `experiments/v36/artifacts/plots|media/`。

## 实验记录

### dev_v36_tail_cycler_stage3_full1（热身，128 env × 50 iter，超时停止）
- **命令**
  ```bash
  python experiments/v36/train.py --run_name dev_v36_tail_cycler_stage3_full1 \
    --total_iterations 60 --num_envs 128 --n_steps 128 \
    --batch_size 1024 --n_epochs 5 --learning_rate 2.5e-4 \
    --curriculum 15:9,20:10,25:9,25:8,25:8,25:8,25:8 \
    --multi_eval_interval 12 --multi_eval_probe_episodes 12 --multi_eval_seed_base 411232 \
    --eval_multi_fish 96 --eval_multi_episodes 24 \
    --penalty_gate_phase_allowance 4,5,6,7 --penalty_gate_required_successes 2 \
    --penalty_gate_success_step_one 18 --penalty_gate_success_step_one_ratio 0.02 \
    --penalty_gate_success_early_death 0.115 --penalty_gate_success_early_death_window 2 \
    --penalty_gate_success_p10 70 --penalty_gate_success_median_window 2 \
    --penalty_gate_freeze_iterations 5 --penalty_gate_success_freeze_iterations 2 \
    --penalty_gate_failure_p10 95 --penalty_gate_failure_min_final 0.5 --penalty_gate_failure_tolerance 2 \
    --tail_seed_stage_spec '20:experiments/v34/artifacts/checkpoints/tail_seed_mix_v34_ne.json;24:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage1.json;16:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage1.json;20:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage2.json;20:experiments/v33/artifacts/checkpoints/tail_seed_mix_v33_stage3.json' \
    --tail_force_reset_steps 224 --tail_replay_count 2 --video_num_fish 48 \
    --skip_final_eval --seed 411232
  ```
- **结果**：CLI 30 分钟超时在 iter≈50 停机，仍产出完整 checkpoint/tb/log，可用于比较。tail_queue_remaining_ratio 降到 0.68，multi-eval iter48 得分 (avg_final=0.884, min_final=0.813, early_death₁₀₀=0.090, p10=116)。记录在 `experiments/v36/artifacts/logs/dev_v36_tail_cycler_stage3_full1.log`，TensorBoard 在 `.../tb_logs/dev_v36_tail_cycler_stage3_full1/`。

### dev_v36_tail_cycler_stage3_full2（主跑，128 env × 60 iter）
- **命令**：同上，仅 `--run_name` 改为 `dev_v36_tail_cycler_stage3_full2`。
- **训练概览**
  - 60 iter 全部完成（约 32 min）。滚动 survival_rate ≈0.85，强制截断保持 224 步。
  - 新 TailSeedCycler 持续消费 tail seeds：`tail_queue_remaining_ratio` 从 0.92（iter10）降到 0.62（iter60），`tail_prewarm_injections_total = tail_forced_resets_total = 3,968`，确认 forced reset 后每次都注入新 tail。
  - Penalty gate 切换为“rolling early-death + median p10”，迭代 60 时第一次满足 success 判据（`rolling_early_death_median = 0.086`，`p10_median = 126`），但因 `required_successes=2` 仍处于 stage0 success_progress。
  - Tail mix：`tail_seed_mix_v34_ne.json`（20% 配额）已加入 stage spec。NE 高速 share 在 iter60 multi-eval 中升至 0.429，说明 counter mix 需要进一步配比或与 curriculum 绑定。
  - `training_stats.pkl` 位于 `experiments/v36/artifacts/checkpoints/dev_v36_tail_cycler_stage3_full2/`，TensorBoard 目录 `.../tb_logs/dev_v36_tail_cycler_stage3_full2/`。
- **Multi-eval（96 fish / 24 epi）**

  | iter | avg_final | min_final | early_death₁₀₀ | first_death_p10 | step-one ratio | NE高速 share |
  | --- | --- | --- | --- | --- | --- | --- |
  | 12 | 0.848 | 0.771 | 0.104 | 87.3 | 0.019 | 0.000 |
  | 24 | 0.886 | 0.802 | 0.085 | 141.6 | 0.020 | 0.087 |
  | 36 | 0.868 | 0.729 | 0.100 | 102.1 | 0.016 | 0.111 |
  | 48 | 0.884 | 0.813 | 0.090 | 116.0 | 0.027 | 0.065 |
  | 60 | 0.880 | 0.771 | 0.082 | 136.2 | 0.012 | 0.429 |

- **Tail 监控**
  - `tail_stage_index`：stage2（对应 NE mix stage），`tail_stage_progress=0.38`，`tail_queue_remaining=4096`（0.62 ratio）。
  - `tail_prewarm_injections` 在奇数 iter 固定 128、偶数 iter 为 0，与 `tail_force_reset_steps=224` 的节奏一致。
- **Artifacts**
  - Logs: `experiments/v36/artifacts/logs/dev_v36_tail_cycler_stage3_full2.log`
  - Checkpoints: `experiments/v36/artifacts/checkpoints/dev_v36_tail_cycler_stage3_full2/`（含 `model_final.zip`、`schedule_trace.json`、`step_one_worst_seeds.json`）
  - TensorBoard: `experiments/v36/artifacts/tb_logs/dev_v36_tail_cycler_stage3_full2/`
  - Plots: `experiments/v36/artifacts/plots/`（survival/first-death/penalty/step-one polar + multi-eval timeline）
  - Media: `experiments/v36/artifacts/media/dev_v36_tail_cycler_stage3_full2_curve.gif` 与 tail replay mp4（iter12/24/36/48/60 rank0/1）

## Learning / 下一步
1. **Tail 注入链路已闭环**：全局消费计数与 `tail_queue_remaining_ratio=0.62` 证明 TailSeedCycler 生效，下一步可以建立“阶段耗尽预警”（例如 ratio≤0.5 时切换下一 stage），以及让 `prewarm_override_ratio` 恢复数值化记录。
2. **Penalty gate 仍停留 stage0**：新的 median 判据在 iter60 获得首次 success_progress，但缺少连续成功。需：
   - 考虑降低 `success_early_death` 到 0.09 或延长 window=3，使 success 判据与 multi-eval 抖动更匹配。
   - 让 multi-eval 频率更高（例如每 8 iter）以便更快累计 success streak。
3. **NE counter mix 需要再调参**：NE 高速 share 在 iter60 反而升至 42.9%。下一轮拟：
   - 将 `tail_seed_mix_v34_ne` 与其它 stage 比例调至 15% 或在 curriculum 前半段限定 NE 混入，后半段恢复均衡。
   - 在 env reset 钩子里打标签，区分“来自 NE mix”与“普通 tail”，方便核算 counter-sample 的真实覆盖率。
4. **观测指标补充**：`prewarm_override_ratio` 仍为 None，应在 callback 内基于 episodes/overrides 增量计算；同时把 `tail_stage_tracker` 的 remaining_ratio 写入 TensorBoard 曲线，作为 tail 消耗健康度。
