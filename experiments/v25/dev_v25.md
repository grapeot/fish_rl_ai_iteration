# dev_v25

## 启动
- 2025-11-11 10:05 PST 复盘 `SOP.md` 与 `experiments/v24/dev_v24.md`，确认 v24 虽然压低了 step-one / early-death，但 `first_death_p10` 始终卡在 105–135，密度惩罚从未解锁。
- v25 目标：在不牺牲防守指标的前提下放宽 success gate，通过 rolling p10 判别推进 penalty stage；同时补齐 tail 事件日志与 pre-roll 扰动观测，让 ramp 调整有据可循。
- 本轮所有代码、日志、媒体落在 `experiments/v25/`，训练并行度遵守 ≥128 env 约束。

## 观察
- Gate 成功条件中过高的 `p10>=150` 使 success streak 反复清零，stage 长期停在 0，导致 density_penalty_coef 恒为 0.04。
- pre-roll 与 spawn jitter 已显著压低 `step_one`（迭代 60 仅 3 条鱼死亡）与 `early_death_fraction_100`（≈0.078），说明开局保护足够，但 tail 事件在某些种子集中爆发，拖累 p10。
- deterministic eval (48EP) 与 multi-probe (24EP) 指标高度一致，提示样本方差不是主要瓶颈；缺失的是对 tail 场景与 gate 决策的可观测性。

## 实验计划
1. **Rolling gate success**：将 `success_first_death_p10` 降到 140，并要求最近 2 次 multi-eval 都满足 success 条件后才提级，以便真正触发 stage1 ramp；trace 写入 `stage_debug.jsonl`。
2. **Tail event 聚合**：自动解析 multi-eval `step_one_deaths`，按 predator velocity / angle 聚类，输出 `artifacts/checkpoints/.../step_one_clusters.json`，用于复盘反复出现的坏 seed。
3. **Pre-roll 观测**：在 env reset 阶段记录 predator pre-roll 的角度与速度采样分布，写入 `pre_roll_stats.jsonl` 方便比较 jitter 是否覆盖 360°。
4. **主线 run (`dev_v25_runA_gate_ramp`)**：复制 v24 超参，保持 `--num_envs 128`、`--total_iterations 64`、强化 logging；若 stage1 解锁，继续监控 penalty 与 escape boost 互动。
5. 若时间允许，追加 `dev_v25_runB_tail_focus`，调高 `predator_spawn_jitter_radius=2.0` + acceleration noise 验证 tail 是否改善。

## 运行记录

### dev_v25_runA_gate_ramp（2025-11-11 08:27–08:30 PST，abort）
- 命令：`python experiments/v25/train.py --run_name dev_v25_runA_gate_ramp --num_envs 128 --total_iterations 64 --curriculum 15:12,20:14,25:8,25:8,25:10,25:6,25:6 --multi_eval_interval 4 --multi_eval_probe_episodes 24 --eval_multi_episodes 48 --best_checkpoint_eval_episodes 32 --video_num_fish 32 --seed 924`
- 在迭代 4 刚触发第一轮 multi-eval 时被旧版本 `evaluate_multi_fish` 缺少 `pre_roll_stats` 变量的 `NameError` 终止；日志与少量 tb 事件保留在 `experiments/v25/artifacts/logs/dev_v25_runA_gate_ramp.log`、`experiments/v25/artifacts/tb_logs/dev_v25_runA_gate_ramp/` 供查错。

### dev_v25_runA_gate_ramp_full（2025-11-11 08:30–08:33 PST，abort）
- 同上命令，修补部分 instrumentation 但依旧在 multi-eval 阶段因为相同 bug 退出；保留的空 checkpoint 目录 `experiments/v25/artifacts/checkpoints/dev_v25_runA_gate_ramp_full/` 仅作溯源，不纳入分析。

### dev_v25_runA_gate_ramp_main（2025-11-11 08:41–09:08 PST，主线 run）
- 命令：`nohup python experiments/v25/train.py --run_name dev_v25_runA_gate_ramp_main --num_envs 128 --n_steps 256 --total_iterations 64 --curriculum 15:12,20:14,25:8,25:8,25:10,25:6,25:6 --multi_eval_interval 4 --multi_eval_probe_episodes 24 --eval_multi_episodes 48 --best_checkpoint_eval_episodes 32 --video_num_fish 32 --seed 924 > codex_runs/dev_v25_runA_gate_ramp_main.stdout`。
- 为规避 CLI 120s 限制改用 `nohup` 持续 27 分钟；`n_steps` 降到 256 以保证 64 iteration 可在单轮完成（`--num_envs 128` 仍满足 SOP）。所有产物写入：
  - Log/TB：`experiments/v25/artifacts/logs/dev_v25_runA_gate_ramp_main.log`、`experiments/v25/artifacts/tb_logs/dev_v25_runA_gate_ramp_main/`
  - Checkpoints & JSON：`experiments/v25/artifacts/checkpoints/dev_v25_runA_gate_ramp_main/`
  - Plot/GIF：`experiments/v25/artifacts/plots/dev_v25_runA_gate_ramp_main_*.png`、`experiments/v25/artifacts/media/dev_v25_runA_gate_ramp_main_curve.gif`
  - 视频：`experiments/v25/artifacts/media/dev_v25_runA_gate_ramp_main_multi_fish_eval.mp4`、`..._best_iter_16_multi.mp4`
  - 新增 instrumentation：`penalty_stage_debug.jsonl`、`pre_roll_stats.jsonl`、`step_one_clusters.jsonl` 均位于同一 checkpoint 目录。
- 最终 deterministic multi-eval（25 鱼 ×48EP）：`avg_final_survival_rate=0.744`、`min_final_survival_rate=0.52`，死亡统计 `p10=66`、`early_death_fraction_100=0.141`（见 `eval_multi_summary.json` 与 `death_stats.json`）。
- Gating：把 `success_p10` 放宽到 140 仍未达到，`penalty_stage_debug.jsonl` 显示迭代 4–64 全是 `failure_hold`，`phase_limit` 固定在 3，`density_penalty_coef` 因此没有超过 0.04。
- Multi-eval best checkpoint仍落在迭代 16（`best_checkpoint_iteration=16`，`avg_final_survival_rate=0.695`，详见 `eval_multi_best_iter_16_summary.json`），说明高迭代虽然提升均值，但尾部 `p10` 仍拖累 gate。
- Step-one 聚类：`step_one_clusters.jsonl` 的累积 share 显示 0°–60°、速度 1.2–1.6 的 predator heading 占 60%+ 的 Step=1 死亡，指向 spawn jitter 仍偏向正 x / 东北象限；`pre_roll_stats.jsonl`（迭代 60）证实 pre-roll 角度分布均匀但 `spawn_radius_p90≈1.49`，仍有大量 predator 从中心偏 1m 范围直接冲向鱼群。
- 关键截图/曲线：
  - Survival & reward：`experiments/v25/artifacts/plots/dev_v25_runA_gate_ramp_main_survival.png`
  - First-death / penalty：`..._first_death.png`、`..._penalty_vs_first_death.png`
  - Multi-eval 时间线：`experiments/v25/artifacts/plots/dev_v25_runA_gate_ramp_main_multi_eval_timeline.png`
  - 媒体：`experiments/v25/artifacts/media/dev_v25_runA_gate_ramp_main_multi_fish_eval.mp4`

## 本轮观察
- `success_p10>=140` 依旧过高，迭代 60 的 `p10` 也仅 98；stage 长期停在 0，`density_penalty_coef` 从未解锁，`escape_boost` 也因 adaptive clamp 在 iteration 59 被迫回落至 0.76，进一步压低 phase6 防守。
- 新的 `pre_roll_stats.jsonl` 表明 pre-roll 角度 jitter 的确覆盖 ±17°，`spawn_radius_mean≈0.83、p90≈1.49`，说明随机偏移已接近 1.6m 上限；但 `step_one_clusters` 聚类仍显示 0°–60° 的捕食者速度向量贡献了 60% 以上的 Step=1 死亡，意味着 tail 事件更多来自速度向量而非出生位置，需要在 env 内对 `predator_pre_roll_angle_jitter` 或 `predator_pre_roll_speed_jitter` 做非均匀扰动。
- `penalty_stage_debug.jsonl` 让 gate 状态更易追踪：多次 `failure_hold` 几乎都在 `p10<110` 触发，而 `early_death_fraction_100` 已稳定在 0.10 附近，证实当前 bottleneck 只剩尾部生存。
- `multi_eval_history` 与 `step_one_deaths` 结合 `step_one_clusters.jsonl` 可以快速定位重复 seed（例如 iteration 48/52 中同一 Predator heading），减少手工 grep。

## 下一步计划
1. **调低 success 指标 / 使用 rolling median**：把 `success_p10` 再降到 130，并在 gate 中加入 rolling median（例如最近 3 次 probe 的 `median(p10)>=130`）以避免单次抖动；若仍无法超过 0.04 penalty，就考虑把 `failure_p10` 改为 110，减少频繁 `failure_hold`。
2. **非对称 pre-roll 扰动**：基于 `step_one_clusters` 的 0°–60° 热区，将 pre-roll 角噪声改成“上半圆优先”，或在 `FishEscapeEnv._apply_predator_pre_roll` 中引入倾向性旋转，以强制更多 180°/240° 入侵；同时把 `predator_spawn_jitter_radius` 扩展到 2.0 并记录对 `spawn_radius` 分布的影响。
3. **Tail 诊断脚本**：利用 `pre_roll_stats.jsonl` + `step_one_deaths` 输出 “worst seeds” 的 `death replay list`，并在 `experiments/v25/artifacts/plots/` 自动生成 `step_one_heading_polar.png`，让下一轮可直接根据雷达图调参。
4. **Escape boost – penalty 联动**：重新审视 `AdaptiveBoostController`，使其读取多次 probe 的 `p10`（而非即时值）并把 clamp 下限绑到 penalty stage，避免在 60 iteration 靠一次坏样本就把 boost 重置为 0.76。
