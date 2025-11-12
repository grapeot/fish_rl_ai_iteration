# dev_v30

## 启动
- 2025-11-11 15:40 PST 复盘 `SOP.md` 与 `experiments/v29/dev_v29.md`，继承 v29 聚焦的四个痛点：NE 高速捕食者仍占主导、Penalty gate 迟迟不解锁、tail prewarm 缺乏仪表化、期中 multi-eval 频率导致 CLI 超时。
- dev_v30 继续沿用 SOP：所有命令、假设、指标实时写在本文；产物统一落在 `experiments/v30/artifacts/` 下并被纳入版本控制。

## 观察（承接 v29）
1. **NE hotspot 未彻底稀释**：v29 虽把 90°–150°/speed>2.0m/s share 压到 ~25%，step-one cluster 仍由该象限+高速桶主导；prewarm 队列被 5 次迭代耗尽后，采样重新偏向 NE，策略缺乏覆盖其他象限。
2. **Penalty gate 仍锁死在 stage0**：即使放宽 `penalty_gate_required_successes=1`，`p10` 最高值只有 ~75，`early_death_fraction_100` 未能稳定低于 0.11，`penalty_stage_debug.jsonl` 全是 `failure_hold/locked`，density penalty 无法进阶。
3. **Tail prewarm 观测不足**：新增的 `prewarm_override_ratio` 显示尾队列在迭代 12 后就降为 0，但缺少图形化与阈值告警，也无法针对 NE 高速外的 seed 维持混合覆盖。
4. **Multi-eval 诱发超时**：128 env × 24 probe，interval=8 仍需 ~30 min 才 34 iteration，导致 runA 被 CLI kill；需要更激进地削减期中 probe，把深度评估放到训练后脚本。

## 实验计划
1. **NE hotspot & tail 覆盖**：
   - 制定“角度×速度”黑名单，运行时把 90°–150° 且 speed>1.8 的预 roll 直接重采，或至少把其权重调到 <0.05。
   - tail replay 中混入低速 NE 样本与其他象限 worst seeds，确保 `prewarm_override_ratio` 逐步下降而非瞬间耗尽；必要时在训练脚本里记录“NE high-speed share”。
2. **解锁 penalty gate**：
   - 进一步下调 `penalty_gate_success_step_one`（例如 12→10）、拉高 `penalty_gate_success_p10` 判定窗口的容忍度，并把 `success_p10_median_window` 改为 1；记录 gate event timeline。
   - 配合更高 `density_penalty_phase_targets`（最高 ≥0.06）验证是否能在 64 iteration 内推进至少一个阶段。
3. **可观测性提升**：
   - 补充 `prewarm_override_ratio`、step-one cluster top5 share 到 TensorBoard scalar，并产出 PNG 放在 `artifacts/plots/`。
   - 如需新的 JSON/tb 字段，文档中记录字段名与解释，确保下一轮可复现。
4. **运行节奏**：
   - 训练脚本默认 `--multi_eval_interval >=12`，只在关键节点触发简短 probe；训练完成后用单独 evaluate/可视化脚本写入 `artifacts/plots|media/`。
   - 每次实验至少 64 iteration（≥60），`--num_envs >=128`，产物包括日志、TB、checkpoint、plot、媒体。

（以下小节待实验推进后补全）

## 实验记录

### dev_v30_runA_gate_unlock（128 env，n_steps=256，迭代 31/64，CLI 10 min 超时）
- **命令**
  ```bash
  python experiments/v30/train.py --run_name dev_v30_runA_gate_unlock --total_iterations 64 --num_envs 128 --n_steps 256 \
    --curriculum 15:12,20:14,25:8,25:8,25:10,25:6,25:6 --multi_eval_interval 12 --multi_eval_probe_episodes 16 \
    --predator_pre_roll_steps 20 --predator_heading_bias '0-30:0.8,30-60:0.8,60-90:0.5,90-120:0.05,120-150:0.05,150-180:0.6,180-210:1.2,210-240:1.5,240-270:1.6,270-300:1.4,300-330:1.0,330-360:0.9' \
    --predator_pre_roll_speed_bias '0-1.0:1.8,1.0-1.6:1.3,1.6-2.0:0.7,2.0-2.6:0.3,2.6-3.2:0.15' \
    --density_penalty_phase_targets 0.0,0.02,0.04,0.05,0.055,0.06,0.065 --density_penalty_lock_value 0.05 --density_penalty_lock_until 24 \
    --penalty_gate_phase_allowance 4,5,6,7 --penalty_gate_success_step_one 10 --penalty_gate_success_early_death 0.11 --penalty_gate_success_p10 95 \
    --penalty_gate_failure_p10 90 --penalty_gate_failure_tolerance 3 --penalty_gate_lock_iteration 16 \
    --tail_seed_replay_path experiments/v29/artifacts/checkpoints/dev_v29_runB_gate_relax_mini/step_one_worst_seeds.json \
    --tail_seed_prewarm_iterations 8 --tail_replay_count 2 --eval_multi_fish 96 --eval_multi_episodes 48 --video_num_fish 48 --seed 301101
  ```
- **结果**：完成 31 iteration 后因 CLI 600 s 超时被杀；multi-eval 在 iter 12/24 触发（avg_final=0.79→0.38，p10=72.9→78.0，step_one=27→28），`step_one_top_share≈0.37`，NE 高速 share 仍 >0.2；`prewarm_override_ratio`=1.0（迭代 ≤11）后跌至 0。
- **关键信号**：`penalty_stage_debug` 连续 `failure_hold`，density penalty 被锁在 stage0；TB 已记录 `custom/step_one_top_share` 与 `custom/step_one_ne_high_speed_share`。
- **产物**：`logs/dev_v30_runA_gate_unlock.log`、`tb_logs/dev_v30_runA_gate_unlock/`、`checkpoints/dev_v30_runA_gate_unlock/`、`plots/dev_v30_runA_gate_unlock_multi_eval_timeline.png`、`media/dev_v30_runA_gate_unlock_iter012_tail_rank0_ep0.mp4` 等。

### dev_v30_runB_gate_squeeze（96 env，n_steps=192，迭代 36/64，CLI 超时）
- **命令**：与 runA 相同但将 `--num_envs 96　--n_steps 192 --n_epochs 6 --multi_eval_probe_episodes 12 --predator_pre_roll_steps 18` 等参数下调以减速。
- **结果**：完成 36 iteration；multi-eval iter12/24 的 avg_final=0.78→0.72，p10=72.9→73.1，step_one=31；NE share 仍 0.23±0.02；prewarm tail 6 个 iteration 内耗尽。CLI 于 600 s 处终止。
- **产物**：`logs/dev_v30_runB_gate_squeeze.log`、`tb_logs/dev_v30_runB_gate_squeeze/`、`plots/...runB...`、`media/...runB...`。

### dev_v30_runC_fast64（64 env，n_steps=128，迭代 48/64，CLI 超时）
- **命令**：`--num_envs 64 --n_steps 128 --n_epochs 4 --multi_eval_probe_episodes 10` 以进一步压缩单次 rollout。
- **结果**：完成 48 iteration；multi-eval iter12/24/36 的 avg_final=0.78→0.74→0.51，p10=73→71→110.8（出现首次 >110 的窗口），step_one=16；NE share 在 iter36 ≈0.09，但迭代 48 后日志显示重新攀升。仍因 CLI 超时被停止。
- **产物**：`logs/dev_v30_runC_fast64.log`、`plots/dev_v30_runC_fast64_multi_eval_timeline.png`、`media/dev_v30_runC_fast64_iter036_tail_rank0_ep0.mp4` 等。

### dev_v30_runD_fast64_shortsteps（64 env，n_steps=96，迭代 36/64，CLI 超时）
- **命令**：在 runC 基础上把 `n_steps` 再压到 96、`n_epochs`=3，锚定同一 tail 配置。
- **结果**：迭代 36 时 `p10≈112.8`、`step_one≈25`，multi-eval 只记录两次（iter12/24）；训练仍需 ~10 min，手动 run 仍被 CLI 杀掉。
- **产物**：`logs/dev_v30_runD_fast64_shortsteps.log`、`plots/...runD...`、`media/...runD...`。

### dev_v30_runE_fast64_bg（64 env，n_steps=96，n_epochs=3，后台运行完成 64/64）
- **命令**：
  ```bash
  # 通过 nohup 在后台运行以规避 CLI 10 min 限制
  python experiments/v30/train.py --run_name dev_v30_runE_fast64_bg --total_iterations 64 --num_envs 64 --n_steps 96 --n_epochs 3 \
    --curriculum 15:12,20:14,25:8,25:8,25:10,25:6,25:6 --multi_eval_interval 12 --multi_eval_probe_episodes 10 \
    --predator_pre_roll_steps 16 --predator_heading_bias '0-30:0.9,30-60:0.8,60-90:0.4,90-120:0.04,120-150:0.04,150-180:0.6,180-210:1.3,210-240:1.4,240-270:1.5,270-300:1.3,300-330:1.0,330-360:0.85' \
    --predator_pre_roll_speed_bias '0-1.0:1.6,1.0-1.4:1.3,1.4-1.8:0.8,1.8-2.4:0.35,2.4-3.0:0.15' \
    --density_penalty_phase_targets 0.0,0.02,0.04,0.05,0.055,0.06,0.065 --density_penalty_lock_value 0.048 --density_penalty_lock_until 20 \
    --penalty_gate_phase_allowance 4,5,6,7 --penalty_gate_success_step_one 10 --penalty_gate_success_early_death 0.11 --penalty_gate_success_p10 95 \
    --penalty_gate_failure_p10 90 --penalty_gate_failure_tolerance 3 --penalty_gate_lock_iteration 14 \
    --tail_seed_replay_path experiments/v29/artifacts/checkpoints/dev_v29_runB_gate_relax_mini/step_one_worst_seeds.json \
    --tail_seed_prewarm_iterations 6 --tail_replay_count 2 --eval_multi_fish 96 --eval_multi_episodes 32 --video_num_fish 32 --seed 301105
  ```
- **训练概况**：后台执行约 26 min 完成 64 iteration；checkpoint 每 5 iter 保存，`tensorboard` 记录 `custom/step_one_top_share` / `custom/step_one_ne_high_speed_share`；`logs/dev_v30_runE_fast64_bg.log`、`logs/dev_v30_runE_fast64_bg.stdout` 保留实时输出。
- **关键指标**：
  - 期末 multi-eval（iter=60）`avg_final=0.855`、`p10=111.5`、`early_death_fraction_100=0.095`、`step_one_death_count=18`（gate 仍未解锁，受 `step_one<=10` 限制）。
  - `schedule_trace` 中 `step_one_top_share` 平均 0.267，NE 高速 share 平均 0.136，但 iter=60 回升到 0.333；`prewarm_override_ratio` 在 iter≤11 保持 1.0，之后降为 0——tail 队列仍一次性耗尽。
  - `eval_multi_fish.json`（32 EP）显示 `avg_final_survival_rate=0.818`，`death_stats` 中 `mean_death_step=422.4`，`early_death_fraction_100=0.133`。
  - `penalty_stage_debug` 记录 iter12/24/36 的 `failure_hold`，iter48/60 仅 `noop`，stage 仍锁在 phase 3。
- **产物**：
  - 日志：`experiments/v30/artifacts/logs/dev_v30_runE_fast64_bg.log`、同名 `.stdout`；TensorBoard：`tb_logs/dev_v30_runE_fast64_bg`。
  - Checkpoints/评估：`checkpoints/dev_v30_runE_fast64_bg/`（`model_iter_05..60.zip`、`eval_multi_summary.json`、`schedule_trace.json` 等）。
  - 可视化：`plots/dev_v30_runE_fast64_bg_survival.png`、`..._first_death.png`、`..._multi_eval_timeline.png`、`..._step_one_heading_polar.png`、`..._penalty_vs_entropy.png`、`..._death_histogram.png`。
  - 媒体：`media/dev_v30_runE_fast64_bg_iter012/024/036/048/060_tail_*.mp4` + `media/dev_v30_runE_fast64_bg_curve.gif`。

## Learning / 下一步计划
1. **Gate 仍被 step-one 阈值卡死**：iter60 multi-eval 已满足 `p10=111.5`、`early_death_fraction_100=0.095`，但 `step_one_death_count≈18` 未降到门限 10。需要：
   - 重新评估 gate 目标，给 stage0→1 提供“先降 NE share 再降 step_one”的双阶段策略；或改成“step_one≤16 且 NE share<0.15”触发阶段提升。
   - 增加 `step_one_death_count` 的 EMA/报警，写入下一轮文档以便快速定位 gate 卡点。
2. **NE hotspot 仅阶段性缓解**：新加的 TB scalar 表明在 iter12/24/36/48 NE share 分别为 0.111/0.056/0.091/0.087，但 iter60 回升到 0.333。需要：
   - 在敌人预 roll 中加入角度×速度黑名单或 rejection sampling，避免阶段末尾因 tail 耗尽而重新倾向 NE；
   - 将 `step_one_clusters.jsonl` 的 top bucket 自动导出成 CSV/plot（当前 polar plot 只能人工读取）。
3. **Tail prewarm 依旧一次性烧光**：`prewarm_override_ratio` 仍在迭代 11 后清零。下一轮考虑：
   - 为 tail 维持一个按 NE/其他象限/速度分组的 reservoir，并在每次 `consume_prewarm_stats` 时按比例补充；
   - 在训练脚本中增加 `custom/prewarm_override_ratio_window`，以及当 ratio<0.2 时自动切换至“低 NE 权重”模式。
4. **运行节奏**：后台运行证明在 64 env 时 64 iteration 需要 ~26 min；若 CLI 交互式执行仍有 10 min 限制，下一轮需继续使用后台/分阶段脚本，并在 `dev_v31.md` 明确如何监控背景作业（PID、日志路径等）。

- **下一轮待办**：
  1. 实现 NE 黑名单采样或 speed-angle 权重自动降噪，验证是否能把 `step_one_ne_high_speed_share` 稳定压在 0.15 以下。
  2. 设计 tail reservoir（允许分批补充 worst seeds），并在 `schedule_trace` 中记录“累计注入次数/当前存量”。
  3. 调整 gate 条件（例如 `step_one<=16` 或“连续两次 step_one<=18 且 NE share<0.15”），并增加当 gate 锁死>3 次时的自适应提示。
  4. 继续用 runE 的 checkpoint（`model_iter_60.zip`）做单独 evaluate/可视化，确认策略在高并发配置下的真实存活率，并在下一轮对比。

