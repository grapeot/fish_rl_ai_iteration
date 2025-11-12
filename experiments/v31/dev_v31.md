# dev_v31

## 启动
- 2025-11-12 10:05 PST 复盘 `SOP.md`、`experiments/v30/dev_v30.md` 以及 v30 artifacts，继续延伸“降低 NE 高速垄断 + 解锁 penalty gate”主线。
- 沿用 v30 的记录格式：所有命令、日志、plot、媒体统一落在 `experiments/v31/artifacts/`，并在本文实时补全。
- v31 的阶段目标：
  1. 持续降低 NE（90°–150°）高速 bucket (>1.8 m/s) 的 step-one 覆盖占比。
  2. 在 64+ 迭代窗口内推进 penalty gate 至至少 stage1，并保留可靠的 gate timeline。
  3. 减少 multi-eval timeout：训练期仅保留稀疏 probe，训练后单独运行可视化脚本。

## 观察（承接 v30）
1. **NE hotspot 仍高**：runA 把 90°–150°/speed>2.0 share 压到 ~20% 但依旧成为 step-one top cluster，tail 预热耗尽后又回升，说明 prewarm seed 与预 roll bias 仍偏 NE。
2. **Penalty gate 锁死在 stage0**：即使下调 success step-one 门槛到 10，multi-eval timeline 还是重复 `failure_hold`，density penalty 没能越过 0.02，导致阶段切换无望。
3. **Tail prewarm 观测不足**：`prewarm_override_ratio` 监控到迭代 11 就跌到 0，原因是 tail seeds 被快速消费且没有混入慢速/非 NE 样本；缺乏“prewarm queue 组成”与 `step_one_ne_high_speed_share` 的联合可视化。
4. **Multi-eval 仍触发 CLI 超时**：128 env × 24 probe 导致 runA/runB 在 31/36 iteration 被 600s 限制杀掉，需要彻底推迟深度评估。

## 实验计划（v31）
1. **Tail & pre-roll 去偏**
   - 生成新的 `tail_seed_mix_v31.json`：从 v29 worst seeds 中剔除 NE 高频 + 高速 (>1.8 m/s) 样本，按 40% NE 低速、60% 其他象限 worst seeds 混排，确保 `prewarm_override_ratio` 平滑下降。
   - 在 `train.py` 默认参数里把 `predator_heading_bias`/`speed_bias` 改成强抑制 90°–150° 且 speed>1.6 的配置，同时在 multi-eval summary 中持续记录 `step_one_ne_high_speed_share`。
2. **Penalty gate 推进**
   - 将 `penalty_gate_success_step_one` 降到 9，`success_p10` 窗口采用单次 metric，`density_penalty_phase_targets` 拉升到 0.065 以上并锁定 iteration<=24；在 `penalty_stage_debug.jsonl` 中写入 gate timeline。
   - 迭代初期用 `n_steps=256`、`num_envs=128`，若 CLI 超时预期 < 10 分钟则维持 64 iteration，否则切换 96 env + `n_steps=192` 并增加 checkpoint 频率。
3. **运行节奏与可观测性**
   - `--multi_eval_interval 16`，probe episode=12，只记录必要指标；训练结束后调用 `visualize.py` 生成 death histogram/step-one polar 并存到 `artifacts/plots/`。
   - 每轮运行前固定 `--seed 311031` 和 `--multi_eval_seed_base`，方便比较 step-one cluster；媒体录制合并到 `artifacts/media/`。

(后续实验在“实验记录”小节补充)

## 实验记录

### dev_v31_runA_tailmix（128 env，n_steps=256，目标 64 iteration，CLI 15min 超时）
- **命令**
  ```bash
  python experiments/v31/train.py --run_name dev_v31_runA_tailmix --total_iterations 64 --num_envs 128 \
    --n_steps 256 --batch_size 2048 --n_epochs 6 --learning_rate 2.5e-4 \
    --curriculum 15:10,20:12,25:10,25:8,25:8,25:8,25:8 --multi_eval_interval 16 --multi_eval_probe_episodes 12 \
    --eval_multi_fish 96 --eval_multi_episodes 48 --multi_eval_seed_base 311031 --seed 311031 \
    --predator_pre_roll_steps 20 --predator_heading_bias '0-30:1.2,30-60:1.0,60-90:0.6,90-120:0.05,120-150:0.04,150-210:0.95,210-240:1.3,240-270:1.5,270-300:1.3,300-330:1.0,330-360:0.9' \
    --predator_pre_roll_speed_bias '0-1.0:1.8,1.0-1.6:1.4,1.6-2.0:0.65,2.0-2.4:0.3,2.4-3.0:0.15' \
    --density_penalty_phase_targets 0.0,0.02,0.04,0.05,0.055,0.06,0.065 --density_penalty_lock_value 0.05 --density_penalty_lock_until 24 \
    --penalty_gate_success_step_one 9 --penalty_gate_success_early_death 0.11 --penalty_gate_success_p10 95 \
    --penalty_gate_failure_p10 90 --penalty_gate_failure_tolerance 3 --penalty_gate_phase_allowance 4,5,6,7 \
    --tail_seed_replay_path experiments/v31/artifacts/checkpoints/tail_seed_mix_v31.json --tail_seed_prewarm_iterations 12 \
    --tail_replay_count 2 --video_num_fish 48 --video_max_steps 500 --video_fps 20
  ```
- **结果**：迭代推进到 48/64 时 CLI 因 900s 超时终止；训练日志、checkpoint、TB 均已写入 `experiments/v31/artifacts/`。因 `load_tail_seed_velocities` 尚不识别 `velocities` 字段，tail 预热在开局直接失效（log 中出现 `[tail-prewarm] no predator_velocity samples`）。
- **指标**：multi-eval（迭代 16/32）平均终局存活 `0.85→0.80`，`early_death_fraction_100=0.06→0.076`，step-one 死亡 `18→20`；`step_one_clusters.jsonl` 显示 NE≥1.8m/s share 由 22% 上扬到 30%。
- **产物**：日志 `experiments/v31/artifacts/logs/dev_v31_runA_tailmix.log`、TB `.../tb_logs/dev_v31_runA_tailmix/`、plot `.../plots/dev_v31_runA_tailmix_multi_eval_timeline.png`、媒体 `.../media/dev_v31_runA_tailmix_iter032_tail_rank1_ep11.mp4`。

### dev_v31_runB_tailmix（128 env，n_steps=192，60 iteration 完成，CLI 在收尾评估阶段超时）
- **命令**：同 runA，但 `--total_iterations 60 --n_steps 192 --batch_size 1536 --n_epochs 5 --curriculum 15:9,20:10,25:9,25:8,25:8,25:8,25:8`。
- **运行情况**：所有 60 轮 rollout 结束，checkpoint `model_iter_60.zip` 与 `training_stats.pkl` 均落盘；CLI 在 best-checkpoint 复核阶段（约 25 分钟）被杀，但不会影响 artifacts。多鱼 probe（迭代 16/32/48）平均终局存活 `0.921/0.911/0.916`，`early_death_fraction_100=0.072/0.084/0.073`，step-one 死亡 `18/24/21` 仍高于成功门槛 9，因此 penalty gate 始终停留在 stage0（`penalty_stage_debug.jsonl` 中全是 `locked/noop`）。
- **NE hotspot**：`step_one_clusters.jsonl` 显示 NE≥1.8m/s share 由迭代16的 22% → 32 的 29% → 48 的 4.8%，证明新的 heading/speed bias+tail seed mix 能在后期显著稀释 NE 高速团，但早期仍会回弹。
- **Tail 预热**：自定义 `experiments/v31/artifacts/checkpoints/tail_seed_mix_v31.json` 经 `velocities` 字段被正确加载；不过 `training_stats.pkl` 记录 `prewarm_override_ratio` 在迭代 10 前就跌到 0，说明 12*128=1536 个样本仍不足，后续需要按阶段动态补充。
- **训练指标**：最后 10 轮平均 `survival_rate=0.744`，`first_death_step≈37.3`，`density_penalty_coef` 维持 0.05；multi-eval `first_death_p10` 被 clamp 到 500（未触发死亡），但 step-one 死亡数量成为 gate 主要瓶颈。
- **艺术品**：
  - 日志 `experiments/v31/artifacts/logs/dev_v31_runB_tailmix.log`、TB `.../tb_logs/dev_v31_runB_tailmix/`。
  - 曲线与可视化：`.../plots/dev_v31_runB_tailmix_survival.png`、`..._first_death.png`、`..._penalty_vs_first_death.png`、`..._step_one_heading_polar.png`。
  - 媒体：tail 视频 `experiments/v31/artifacts/media/dev_v31_runB_tailmix_iter048_tail_rank0_ep1.mp4`、合成 `dev_v31_runB_tailmix_curve.gif`。
  - Checkpoint/日志：`experiments/v31/artifacts/checkpoints/dev_v31_runB_tailmix/`（含 schedule_trace、penalty_stage_debug、multi-eval history 等）。

## 下一步
1. **延命 tail queue**：重新生成 `tail_seed_mix_v31`（或 v31b）时将 `velocities` 拆成阶段化批次（例如前 12 iteration 40% NE 低速 + 60% 其他象限，迭代 12 之后再补充一批），或在训练脚本内支持按 iteration 动态注入 tail seeds，避免 `prewarm_override_ratio` 在 10 以内清空。
2. **Gate 触发逻辑**：step-one 死亡一直稳定在 18~24，说明“<=9”门槛过低；需要在 v32 中提升阈值或改成“step-one per 1000 fish”比率，以免即使 multi-eval `p10=500` 也无法晋级。并补充 `step_one_ne_high_speed_share` 的 TB 曲线，方便和 gate 时间线对齐。
3. **Probe 节奏**：两次 run 都在 CLI 超时时被杀（runB 已结束但收尾评估过长），需要拆分 best-checkpoint 复核到单独脚本或减少 `eval_multi_episodes`，并考虑在 `train.py` 中新增 `--skip_final_eval` 选项。
4. **NE hotspot 早期压制**：虽然迭代 48 share 已降到 4.8%，但迭代 32 仍有 29%；可尝试在 curriculum 前半段单独指定“黑名单重采”的 pre-roll 逻辑（需在环境内实现），或在 tail mix 中人工插入南/西象限慢速 worst seeds。
