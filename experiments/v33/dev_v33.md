# dev_v33

## 启动
- 2025-11-12 14:15 PST 复盘 `dev_v32` 记录：staged tail queue 4096 samples 7 iter 内耗尽，ratio gate 长期被 1.5% 锁死，step-one NE hotspot 在 iter29 再度抬头。
- 本轮目标：把 tail 预热延长到迭代 20+ 仍保持 `prewarm_override_ratio>0.25`，放宽 ratio gate + 增加观测性（队列剩余、阶段日志、NE hotspot 频率），确保 CLI 仍在 15 min 内完成 60 iter。

## 观察（承接 v32）
1. `prewarm_override_ratio` 在 iter7 后骤降，staged queue 需要补充/循环；缺少 per-iteration stage id / queue 长度追踪，无法量化何时耗尽。
2. Penalty gate 只看 step-one 绝对值 + 1.5% ratio，iter32 虽低于阈值仍因单次 spike 被锁，导致高 penalty phase 永远不开启。
3. `step_one_ne_high_speed_share` 仅在 multi-eval (16/32/48) 才有数值，无法捕捉训练过程中 NE 回弹路径。

## 实验计划
1. **Tail queue 延命**：
   - 新增 SW 低速 `tail_seed_mix_v33_stage3.json`，并将 stage spec 扩展为 3+ 阶段（总覆盖 ≥ 60 iter），记录 per-iteration stage id 与剩余容量。
   - 若 staged queue 仍在 iter20 前耗尽，则考虑在 env reset 时循环当前阶段样本。
2. **Penalty gate 放宽**：
   - `--penalty_gate_success_step_one_ratio` 提升到 0.02，并设 `required_successes=2`，保留 step-one 绝对值 18 作为硬阈值。
   - 观察 gate 日志（rolling_p10、event）确保 stage 能在 iter32 前至少晋级一次。
3. **可观测性**：
   - 在 `train.py` 统计中添加 `tail_stage_index/remaining`、训练迭代的 `step_one_ne_high_speed_share`。
   - Multi-eval 间隔缩短为 12 iter，确保 NE hotspot 监控 ≥5 次/全程。
4. **主实验**：
   - run `dev_v33_tail_refuel_stage3`：128 env, n_steps=128, total_iter=60, staged tail + ratio gate 新配置，`--skip_final_eval` 控制时长。
   - 日志/曲线/媒体全部写入 `experiments/v33/artifacts/`，同步复制关键图表与 tail 视频。

## 实验记录

### dev_v33_tail_refuel_stage3（128 env，n_steps=128，60 iter，skip_final_eval）
- **命令**
  ```bash
  python experiments/v33/train.py \
    --run_name dev_v33_tail_refuel_stage3 \
    --total_iterations 60 --num_envs 128 --n_steps 128 \
    --batch_size 1024 --n_epochs 5 --learning_rate 2.5e-4 \
    --curriculum 15:9,20:10,25:9,25:8,25:8,25:8,25:8 \
    --multi_eval_interval 12 --multi_eval_probe_episodes 12 --multi_eval_seed_base 411232 \
    --eval_multi_fish 96 --eval_multi_episodes 24 \
    --penalty_gate_phase_allowance 4,5,6 --penalty_gate_required_successes 2 \
    --penalty_gate_success_step_one 18 --penalty_gate_success_step_one_ratio 0.02 \
    --penalty_gate_freeze_iterations 5 --penalty_gate_success_freeze_iterations 2 \
    --penalty_gate_success_median_window 2 --penalty_gate_failure_tolerance 2 \
    --tail_seed_stage_spec "24:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage1.json;16:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage1.json;20:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage2.json;20:experiments/v33/artifacts/checkpoints/tail_seed_mix_v33_stage3.json" \
    --tail_replay_count 2 --video_num_fish 48 --skip_final_eval \
    --seed 411232
  ```
- **结果**
  - 训练 60 iter 完成（CLI 20 min 超时但 run 已写全量产物）；最后 10 iter `survival_rate=0.737±0.012`，`first_death_step≈32.4`。
  - Multi-eval（间隔 12 iter）汇总：

    | iter | avg_final | min_final | step_one_ratio | p10 | early_death₁₀₀ |
    | --- | --- | --- | --- | --- | --- |
    | 12 | 0.520 | 0.427 | 1.91% | 90.2 | 11.9% |
    | 24 | 0.496 | 0.354 | 1.91% | 77.1 | 12.2% |
    | 36 | 0.615 | 0.448 | 1.65% | 80.2 | 11.5% |
    | 48 | 0.591 | 0.396 | 2.60% | 59.1 | 13.4% |
    | 60 | 0.617 | 0.510 | 1.30% | 101.1 | 9.6% |
  - Penalty gate 在 12/24/36/48 iter 均触发 `failure_hold`（主要因为 `early_death_fraction_100` > 0.09），stage 始终停在 0；iter60 虽满足 step-one/p10 但 early-death 仍高，success streak 没有累积。
  - `step_one_ne_high_speed_share` 每次 multi-eval 记录为 `[0.00, 9.1%, 10.5%, 6.7%, 6.7%]`，NE hotspot 在 24~36 iter 明显回弹。
- **可观测性 & Tail 诊断**
  - 新 `tail_stage_tracker` 加载 4 段共 10,240 个 velocity slot，但全程仅消费 384 次 override（3.8%），`tail_stage_index` 未离开 stage1，`tail_queue_remaining_ratio` 在 iter10 以后常驻 1.0；`prewarm_override_ratio` 下降的根因是 episode reset 极少而非库存耗尽。
  - `prewarm_override_ratio` 仅在 iter4/8 达到 1.0，之后因 500-step 长局 -> reset 稀少，日志多为 `None/0`，说明需要强制触发 tail 注入或在 rollout 间复位 predator。
  - `penalty_stage_debug.jsonl` 与 `plots/dev_v33_tail_refuel_stage3_multi_eval_timeline.png` 显示 p10 median 在 iter48 掉到 69.6，直接触发 failure_hold；迭代 60 虽恢复但距离 success 连续两次仍差 1 次。
- **产物**
  - 日志：`experiments/v33/artifacts/logs/dev_v33_tail_refuel_stage3.log`
  - Checkpoints & 统计：`experiments/v33/artifacts/checkpoints/dev_v33_tail_refuel_stage3/`
  - TensorBoard：`experiments/v33/artifacts/tb_logs/dev_v33_tail_refuel_stage3`
  - Multi-eval 曲线/直方图：`experiments/v33/artifacts/plots/dev_v33_tail_refuel_stage3_multi_eval_timeline.png` 与 `plots/dev_v33_tail_refuel_stage3_multi_eval_hist/*.png`
  - Tail 媒体（per multi-eval 最差样本）：`experiments/v33/artifacts/media/dev_v33_tail_refuel_stage3_iter0{12,24,36,48,60}_tail_rank*.mp4`

## 后续计划
1. **Tail 注入逻辑**：基于新日志，问题在于 episode 少量 reset，需在 env reset 前硬插入 tail seeds（例如每个 rollout 强制重启若预热队列 >0，或实现 `--tail_seed_stage_refill_interval` 每 N iter 重新 push seeds）。
2. **Penalty gate 门槛**：将 `success_early_death` 上限放宽到 0.11~0.12，或改为 rolling median 判定；若 ratio 已稳定 ≤1.5%，可在 stage0 放开 phase_limit=4 以避免长期卡死。
3. **NE hotspot 稀释**：在 tail mix 中显式加入 “NE 高速 counter-samples” 或提升 multi-eval 频率至 10 iter，并把 `step_one_ne_high_speed_share` 写入 JSON / TensorBoard 以供自动告警。
