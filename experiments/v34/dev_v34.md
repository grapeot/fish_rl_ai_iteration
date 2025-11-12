# dev_v34

## 启动
- 2025-11-12 16:45 PST 快速复盘 `dev_v33`：tail staged queue 在 10k+ 容量下仅消费 3.8%，`prewarm_override_ratio` 在 iter8 之后掉到 0，说明瓶颈不是库存而是 episode reset 过少；Penalty gate 因 `early_death_fraction_100>0.09` 长期 failure_hold，阶段锁死在 0；NE 高速 hotspot 占比 6~10%，但缺少实时 counter-sample 注入路径。
- 本轮目标：建立强制 tail 注入机制（即便 episode 不 reset 也定期截断），放宽 penalty gate 的 early-death 判据并增加 median 过滤，继续追踪 NE hotspot，并以 128 env / 60 iter 主跑生成可交付 artifacts。

## 观察
1. **Tail 覆盖率**：`tail_stage_tracker` 显示 4 段合计 10,240 slots 几乎未用，`prewarm_override_ratio` 仅在 iter≤8 为 1.0，此后长时间为 0，意味着需要在 rollout 层面强制触发 reset 才能消费队列。
2. **Penalty gate**：即使 step-one ratio 稳定 ≤2%，`early_death_fraction_100` 仍在 0.11±0.02，当前 0.09 阈值+必要连胜导致高 penalty 永不启用；`failure_hold` 频率高说明需要更宽松或更有记忆性的 success 判据。
3. **NE Hotspot**：multi-eval NE 高速 share 在 24~36 iter 达到 10%，但 tail mix 没有针对性的 counter-sample；需要将该指标写入 JSON/TB 并结合 tail 强制 reset 以覆盖更多样本。

## 实验计划
1. **Tail 强制 reset**：在 `SingleFishEnv` 中增加 `tail_force_reset_steps` 支持，默认关闭；本轮设置 192~256 步强制截断，使每个 env 每~2 rollout 触发一次 reset，从而拉高 `prewarm_override_ratio` 至 ≥0.3。
2. **Penalty gate 放宽**：将 `success_early_death` 提升至 0.115，并把 `success_p10_median_window` 提到 2，确保需要连续两次 p10≥120 才视为成功，同时保留 `required_successes=2`；在文档中记录新的 gate 逻辑。
3. **NE Hotspot logging**：保持 `step_one_ne_high_speed_share` 写入 TB + JSON，并在 multi-eval summary 中额外生成 `ne_hotspot_share` 字段，方便 dev_v35 自动引用。
4. **主实验：`dev_v34_tail_force_reset_stage3`**
   - 128 env / n_steps 128 / total_iter 60 / tail staged seeds 同 v33。
   - 启用 `--tail_force_reset_steps 224`，`--penalty_gate_success_early_death 0.115`，`--penalty_gate_success_median_window 2`，其余保持 v33 超参数。
   - 日志、checkpoint、TensorBoard、plots、媒体写入 `experiments/v34/artifacts/`，并复制关键曲线+tail mp4。

## 实验记录
### dev_v34_tail_force_reset_stage3_full4（128 env，n_steps=128，60 iter，skip_final_eval）
- **命令**
  ```bash
  python experiments/v34/train.py \
    --run_name dev_v34_tail_force_reset_stage3_full4 \
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
- **训练概览**
  - 运行 19.9 min 完成 60 iter；`tail_force_reset_steps=224` 让 128 个 env 每两轮触发一次截断，`tail_forced_resets` 指标在奇数 iter 为 128、偶数 iter 为 0，滚动均值 ≈64。
  - 由于强制截断在 224 step 就重启，训练端 `first_death_p10` 被压成 1 step，`survival_rate` 最近 10 iter 平均 0.826（0.811~0.839），但 `prewarm_override_ratio` 仍在 iter12 后掉到 0，`tail_queue_remaining_ratio` ≈1.0 表示 1.0 万条 tail seed 依旧未消耗。
  - Penalty gate 仍停在 stage0：iter60 multi-eval `early_death_fraction_100=0.167`>0.115，触发 `failure_hold` 并把 phase limit 锁在 3。
- **Multi-eval (96 fish / 24 epi)**

  | iter | avg_final | min_final | early_death₁₀₀ | first_death_p10 | step-one ratio | NE高速占比 |
  | --- | --- | --- | --- | --- | --- | --- |
  | 12 | 0.417 | 0.104 | 0.164 | 64.1 | 2.08% | 0.00 |
  | 24 | 0.502 | 0.302 | 0.169 | 57.1 | 2.17% | 0.12 |
  | 36 | 0.536 | 0.344 | 0.187 | 51.1 | 1.39% | 0.13 |
  | 48 | 0.582 | 0.417 | 0.194 | 46.0 | 2.78% | 0.09 |
  | 60 | 0.615 | 0.500 | 0.167 | 69.1 | 1.22% | 0.36 |

  - 尾段生存率略升（avg_final→0.615，min_final→0.5），但早死率始终 >0.16，Penalty gate 没有一次成功晋级。
  - `step_one_ne_high_speed_share` 在 iter60 飙到 35.7%，明显高于 v33 的 6~10%，说明强制 reset 加剧了 NE 区域样本集中。
- **产物**
  - 日志：`experiments/v34/artifacts/logs/dev_v34_tail_force_reset_stage3_full4.log`
  - Checkpoints/统计：`experiments/v34/artifacts/checkpoints/dev_v34_tail_force_reset_stage3_full4/`（含 schedule_trace、multi-eval history、step-one worst seeds、model_iter_{5..60}.zip、model_final.zip）
  - TensorBoard：`experiments/v34/artifacts/tb_logs/dev_v34_tail_force_reset_stage3_full4/`
  - 曲线：`experiments/v34/artifacts/plots/` 下生成 survival、first_death、penalty 对齐、multi-eval timeline & hist、step-one polar。
  - 媒体：`experiments/v34/artifacts/media/` 含 `dev_v34_tail_force_reset_stage3_full4_curve.gif` 以及 iter12/24/36/48/60 的 tail 最差样本 mp4，均挂在 v34 artifacts 目录。
- **关键观察**
  1. *强制 reset vs. tail 消耗*：尽管平均强制截断 64 次/iter，`prewarm_override_ratio` 在 iter12 之后仍为 0，说明仅靠 truncate 并不会触发 tail seed 队列——需要在 reset 钩子里显式重注或循环队列，否则 tail_stage_tracker 永远停在 100%。
  2. *训练端 first-death 信号被污染*：`first_death_p10`=1 纯粹是因为被动截断，导致训练态度量失真；后续 gate/可视化必须改用 multi-eval 指标。
  3. *NE hotspot 恶化*： 强制 reset 让 NE 高速 share 在 iter60 提升到 35.7%，已经远高于目标区间，需要专门的 tail seed 或 curriculum 稀释。
  4. *Penalty gate 仍锁死*：早死率始终 ~0.17，即便放宽阈值仍 failure_hold，提示我们需要新的稳定性度量（例如 rolling median of early_death）或 staged curriculum 来降低高 penalty 的门槛。

## Learning / 下一步计划
1. **真正的 tail 注入机制**：在 `SingleFishEnv`/`FishEscapeEnv` 中添加 `refill_tail_queue()` 或“循环读取”能力，并在 forced reset 后强制压入 tail velocity，目标是把 `prewarm_override_ratio` 提升到 ≥0.3，`tail_queue_remaining_ratio` 能按阶段下降。
2. **改写 gate 指标**：将 penalty gate 的 success 判据改为 multi-eval early_death median + `first_death_p10`（>=70）双门槛，同时放宽 stage0 允许的 phase_limit>3，避免长期 stage-lock。
3. **NE counter-sample pipeline**：从 iter60 的 step-one worst seeds 中抽取 NE 高速样本，生成 `tail_seed_mix_v34_ne.json`，并在下一轮 tail stage spec 中混入 ≥20% NE counter-samples。
4. **监控 forced reset 效果**：新增 per-env `prewarm_injected` vs `forced_reset` stats，写入 schedule_trace，确认哪些 env 没有消耗 tail seeds，以便定位 bug。
5. **日志系统**：已修复 `callback._log` 在 finalize 后写入触发的 ValueError；后续需要把 skip-final-eval 的提示迁移到 finalize 之前，确保长跑不会因日志句柄关闭而抛异常。
