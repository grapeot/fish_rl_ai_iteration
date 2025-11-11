# dev_v24

## 启动
- 2025-11-11 09:15 PST 阅读 `SOP.md` 与 `experiments/v23/dev_v23.md`，确认 v23 未能真正触发 penalty stage，`density_penalty_coef` 全程停留在 0.04。
- v24 目标：实现“放宽 gate + 冻结缓冲”策略、补强 pre-roll 扰动与 eval 规模，验证 penalty ramp 是否能稳定推升 `first_death_p10` 并抑制 tail 事件。
- 复用 v23 的训练骨架复制到 `experiments/v24/train.py`，所有日志/模型/媒体写入 `experiments/v24/artifacts/`，并确保 `--num_envs >=128` 与 ≥60 iteration 完整 run。

## 观察
- v23 的 gate 条件 (`step_one<=5 & early_death_fraction_100<=0.1 & p10>=120`) 过严，导致 success streak 反复清零；`penalty_gate.stage` 始终为 0，无从验证双阶段 ramp。
- deterministic eval 方差大（24EP 仅 0.777，`min_final_survival_rate=0.2`），与 multi-eval 60 的 0.64 差距明显，说明样本数不足且 seed 影响大。
- `early_death_fraction_100≈0.12` 与 `p10≈77` 表示 step-one 保护依然薄弱，仅延长 penalty lock 不足以保护 Phase6。

## 实验计划
1. **Gate 放宽 + 缓冲**：把成功条件改为 `step_one<=7`、`early_death_fraction_100<=0.12`、`first_death_p10>=150`。加入“成功后冻结 1 次迭代”机制；若失败则 stage 回滚，但保留最近一次成功指标供分析。
2. **Penalty 双阶段软解锁**：stage0 维持 0.04；stage1 设 plateau (0.05, iter 29-32)；stage2 线性 ramp 至 0.06。记录每次 stage 变化和冻结窗口到 `schedule_trace.jsonl`。
3. **开局扰动强化**：增加 pre-roll 步数与 `predator_speed_jitter` 上限，并引入 `predator_spawn_jitter_radius` 扰动；记录到 config。
4. **Eval 扩充**：multi-eval probe=24EP，best checkpoint eval=32EP，final deterministic eval=48EP + 32 fish 视频。把关键曲线/视频复制到 `experiments/v24/artifacts/plots|media/`。
5. **主线 run (`dev_v24_runA_soft_gate`)**：`--num_envs 128`、`--total_iterations 64`、`--multi_eval_interval 4`、设定新 gate 与扰动配置。日志、tb、检查点路径与命名沿用 SOP。

## 运行记录

### dev_v24_runA_soft_gate_abort（2025-11-11 07:28–07:40 PST，超时终止）
- **命令**：`python experiments/v24/train.py --run_name dev_v24_runA_soft_gate --num_envs 128 --total_iterations 64 --curriculum 15:12,20:14,25:8,25:8,25:10,25:6,25:6 --multi_eval_interval 4 --multi_eval_probe_episodes 24 --eval_multi_episodes 48 --best_checkpoint_eval_episodes 32 --video_num_fish 32 --seed 924`
- 由于 CLI 默认 120s 超时，进程在 iteration≈32 被 kill；为保留上下文，所有输出转存到 `experiments/v24/artifacts/*/dev_v24_runA_soft_gate_abort*` 供参考，不参与后续分析。

### dev_v24_runA_soft_gate（2025-11-11 07:44–08:15 PST，主线 run）
- **命令**：`python experiments/v24/train.py --run_name dev_v24_runA_soft_gate --num_envs 128 --total_iterations 64 --curriculum 15:12,20:14,25:8,25:8,25:10,25:6,25:6 --multi_eval_interval 4 --multi_eval_probe_episodes 24 --eval_multi_episodes 48 --best_checkpoint_eval_episodes 32 --video_num_fish 32 --seed 924`
- **资源配置**：`n_steps=512`、`batch_size=1024`、`penalty_gate_success_step_one<=7`、`success_early_death<=0.12`、`success_p10>=150`、`success_freeze=1 iteration`、`penalty_gate_freeze=4 iterations`、`predator_pre_roll_steps=16`、`predator_pre_roll_speed_jitter=0.2`、`predator_spawn_jitter_radius=1.6`。
- **Multi-eval 探针**（24EP, 每 4 iter）：iteration=56 达成 `avg_final_survival=0.767`、`step_one=5`、`early_death=0.083`、`p10=116.9`；iteration=60 进一步压到 `step_one=3`、`early_death=0.078` 但 `p10=135.3 (<150)`，gate 仍停在 stage0（phase_limit=3, penalty=0.04）。iteration=64 方差回升（`avg_final_survival=0.732`、`p10=105.6`、`step_one=12`）。
- **训练后 deterministic eval**（48EP, 25 fish）：`avg_final_survival_rate=0.7625`、`min_final_survival_rate=0.56`、`median_min_survival=0.76`，死亡统计 `p10=107.9`、`early_death_fraction_100=0.0925`。JSON：`experiments/v24/artifacts/checkpoints/dev_v24_runA_soft_gate/eval_multi_summary.json` 与 `death_stats.json`。
- **Best checkpoint (iter 60)**：`model_iter_60.zip` + `multi_eval_best_iter_60*.json`，32EP 复核 `avg_final_survival=0.751`、`min_final_survival=0.56`，并导出视频 `experiments/v24/artifacts/media/dev_v24_runA_soft_gate_best_iter_60_multi.mp4`（32 fish deterministic）。
- **媒体/曲线**：训练曲线 `experiments/v24/artifacts/media/dev_v24_runA_soft_gate_curve.gif`，最终 32 fish 视频 `.../media/dev_v24_runA_soft_gate_multi_fish_eval.mp4`，多评估时间线 `experiments/v24/artifacts/plots/dev_v24_runA_soft_gate_multi_eval_timeline.png`，死亡直方图 `.../plots/dev_v24_runA_soft_gate_death_histogram.png` 等。
- **日志与板**：文本日志 `experiments/v24/artifacts/logs/dev_v24_runA_soft_gate.log`，TensorBoard `experiments/v24/artifacts/tb_logs/dev_v24_runA_soft_gate/`，schedule trace `experiments/v24/artifacts/checkpoints/dev_v24_runA_soft_gate/schedule_trace.json`。

## 本轮观察
- 尽管加入 success 缓冲后 `step_one` 与 `early_death` 迅速达标，但 `first_death_p10` 仍停留在 105–135 区间，导致 penalty gate stage 始终为 0，`density_penalty_coef` 自始至终锁在 0.04，无法验证后续 0.05/0.06 ramp。
- iteration=60 的探针已经把 `step_one=3`、`early_death=0.078` 压低，但 success streak 受 `p10>=150` 约束被重置；说明当前的 pre-roll + spawn jitter 设置改善开局防护，却对 tail 事件贡献有限。
- deterministic 48EP eval 的 `p10=107.9` 与 probe 相近，表明问题并非样本不足，而是特定 seeds 下 predator still rushing center：`step_one` 记录显示同一 predator vel 重复出现在 3+ fishes。
- 多次 adaptive boost 在 iteration ≥53 将 escape boost 锁回 0.76，提示 `first_death_p10` 低迷触发了低速 clamp；需要把 boost 控制与 penalty gate 联动，否则 escape boost oscillation 可能压制本就脆弱的 phase6。

## 下一步计划
1. **调低 success p10 或增加平滑指标**：尝试 `success_p10>=140` 或使用 rolling median(`p10>=140` for 2 consecutive probes) 以便真正进入 stage1，结合 `last_success` 记录验证安全性。
2. **扩展开局扰动 & logging**：在 `fish_env` 内记录 pre-roll predator 角度/速度分布，确认是否仍存在特定方向集中导致 p10 <120；必要时增大 `predator_spawn_jitter_radius` 到 2.0 并加入 pre-roll acceleration noise。
3. **Tail 事件追踪**：从 `eval_multi_history.jsonl` 自动提取 step-one 重复 seed，写入 `experiments/v24/artifacts/checkpoints/.../step_one_clusters.json` 供下一轮针对性回放；配合额外 64EP deterministic eval (seed-fixed) 评估方差。
4. **Penalty ramp instrumentation**：在 scheduler 中增设 `stage_debug.jsonl`，记录每次 `advance/failure_hold` 的 `p10/early_death`，确保下一轮能快速判断 gate 梯度，而不必手动解析日志。
