# dev_v23

## 启动
- 阅读 `experiments/v22/dev_v22.md`（时间戳：2025-11-11 08:00 PST）后确认 v22 主线 run 在 `density_penalty_lock_until=28` + `predator_pre_roll_speed_jitter` 的组合下，step-one 死亡能压到 4，但 Phase6 解锁后的 `first_death_p10` 仍跌至 ~65，`avg_final_survival_rate=0.665` 低于 v21（0.86）。
- best checkpoint (`experiments/v22/artifacts/checkpoints/dev_v22_runB_penalty_lock28/model_iter_28.zip`) 复核 24EP 仅 `avg_final_survival_rate=0.625`、`early_death_fraction_100≈0.10`，说明 penalty 解锁策略仍需加长或分段；该 checkpoint 作为 v23 热启动基准。
- checkpoint 精简逻辑（仅在满足 `step_one<=8` & `early_death_fraction_100<=0.1` 时写 best）已验证无误，可沿用到 v23 以保持 IO 可控。

## 观察
- penalty lock 解除瞬间 `density_penalty_coef` 抬升至 0.065，escape boost clamp 剧烈跳变，`first_death_p10` 直接腰斩；需要“软解锁”及失败回滚机制。
- 12-step pre-roll + speed jitter 把 step-one 均值维持在 4~7，但 Phase6 对 `early_death_fraction_100` 依旧敏感，表明 tail event 控制逻辑需要再加强。
- multi-eval 仍用 16EP，方差大导致 best checkpoint 判定不稳；上一轮建议提高到 ≥24EP，并加上 `min_final_survival_rate>0.5` 的硬阈值。

## 实验计划
1. **双阶段 penalty 解锁**：保留 `lock_until_iter=28` 的 0.04，之后分两段 ramp（0.04→0.05 plateau 29-32，>32 线性升至 0.06），且要求连续两次 multi-eval 满足 `step_one<=5` & `early_death_fraction_100<=0.1` 才允许进入下一段。
2. **Penalty 回滚/冻结**：若 multi-eval 期间 `first_death_p10<120` 或 `min_final_survival_rate<0.5`，立刻回退到上一档 penalty 并记录冻结窗口 (≥2 iteration)，防止 Phase6 崩盘。
3. **Eval 加强**：把 multi-eval episode 数从 16 提到 24，best checkpoint eval 调到 32EP，另做一次 32 fish deterministic 录像以检查高载荷行为。
4. **主线 run (`dev_v23_runA_dual_unlock`)**：在复制自 v22 的脚本上实现上述逻辑，用 `--num_envs 128`、`--total_iterations 60`、`--multi_eval_interval 4`、`--seed` 固定 (待定)，所有日志/plots/媒体写入 `experiments/v23/artifacts/`，并生成 penalty vs first-death 曲线及 multi-eval 时间线。

## 运行记录

### dev_v23_runA_dual_unlock（2025-11-11 07:02–07:26 PST，主线）
- **命令**：`python experiments/v23/train.py --run_name dev_v23_runA_dual_unlock --num_envs 128 --total_iterations 60 --multi_eval_interval 4 --multi_eval_probe_episodes 24 --eval_multi_episodes 24 --best_checkpoint_eval_episodes 32 --video_num_fish 32 --seed 923`
- **Artifacts**：日志 `experiments/v23/artifacts/logs/dev_v23_runA_dual_unlock.log`，TensorBoard `experiments/v23/artifacts/tb_logs/dev_v23_runA_dual_unlock/`，检查点+统计 `experiments/v23/artifacts/checkpoints/dev_v23_runA_dual_unlock/`（含 `model_final.zip`、`training_stats.pkl`、`schedule_trace.json`、`eval_multi_history.jsonl`、`eval_multi_summary.json`、`death_stats.json` 等），曲线 `experiments/v23/artifacts/plots/dev_v23_runA_dual_unlock_*.png`（含 survival / first-death / penalty 对齐 / penalty vs entropy / death histogram / multi-eval timeline + `multi_eval_hist/` 直方图），媒体 `experiments/v23/artifacts/media/dev_v23_runA_dual_unlock_curve.gif` 与 `experiments/v23/artifacts/media/dev_v23_runA_dual_unlock_multi_fish_eval.mp4`（32 fish deterministic 录像）。
- **探针 / Multi-eval**：iteration=60 的最后一次 multi-eval（24 EP）给出 `avg_final_survival_rate=0.845`、`min_final_survival_rate=0.64`、`first_death_p10≈212`、`step_one_death_count=10`、`early_death_fraction_100=0.078`；但在 iteration=56 之前多次出现 `p10<80` 的 failure，因此 penalty gate 的 success streak 始终被中断，`penalty_gate.stage` 一直锁在 0（`density_penalty_coef`=0.04）。
- **训练后 eval（24 EP deterministic）**：`avg_final_survival=0.777`、`min_final_survival=0.20`、`median_min_survival=0.82`；死亡统计 `p10=77`、`early_death_fraction_100=0.12` (`experiments/v23/artifacts/checkpoints/dev_v23_runA_dual_unlock/death_stats.json`)，说明 tail 事件在 phase6 仍未控制。
- **Best checkpoint gating**：`step_one<=5 & early_death<=0.1` 的组合始终未同时满足（可见 `eval_multi_history.jsonl` 中每次 `penalty_gate.event` 均为 `failure_hold`/`noop`，stage=0），因此未生成 `multi_eval_best_iter_*.json` / best checkpoint 视频。
- **Media**：训练曲线 GIF `...curve.gif`，以及补跑的 32 fish deterministic 视频 `...multi_fish_eval.mp4`（`record_multi_fish_video` 直接加载 `model_final.zip` 生成）。

## 本轮观察
- penalty gate 的 success 条件过于苛刻，iteration=60 的 probe 虽达到 `p10>200`、`step_one=10`，但因前一次失败导致 success streak=0，最终两个阶段均未解锁；`density_penalty_coef` 全程维持 0.04，双阶段策略未被真正验证。
- final eval 的 `min_final_survival_rate=0.2` 与 multi-eval 60 的 0.64 差异较大，说明 24 EP deterministic 仍有大方差；需要增强 sample count（>32 EP）或在 eval 中固定 seed 以便对比。
- `early_death_fraction_100≈0.12` 与 `p10=77` 远低于解锁阈值 120，说明延长锁窗并不能单靠 penalty 控制，需要更强的开局扰动 / Step=1 保护（或直接把成功条件放宽到 `step_one<=7` + `p10>=180`）。
- best checkpoint gating没有命中，因此自动 checkpoint 精简逻辑没有发挥作用，`model_iter_5/10/...` 仍按间隔落盘，占用了较多磁盘。

## Learning / 下一步
- **放宽 gate 条件 + 引入连续成功缓冲**：改为 `step_one<=7`、`early_death_fraction_100<=0.12`、`first_death_p10>=150`，并允许 “成功→冻结 1 次” 的缓冲，以避免单次 failure 把 stage 拉回 0。
- **Penalty stage 预热**：在 stage0→stage1 transition 前增加额外 logging，观察 `p10`/`step_one` 是否稳定≥2 次；若连续两次成功，则直接把 `phase_limit` 提到 4，并把 `freeze_iterations` 提升到 4，确保 ramp 有足够时间生效。
- **开局防护**：结合 multi-eval history，可看到 step-one 死亡仍集中在特定 seeds；计划在 `FishEscapeEnv` 中增加 pre-roll 速度扰动上限（例如 0.2）以及额外 4 步 pre-roll，配合更激进的 `predator_spawn_jitter_radius`。
- **评估增强**：将训练后 deterministic eval 提至 48 EP，并附带 96EP 的快速 smoke（不写入文档）以分辨 tail event；同时保留 32 fish 视频用于交付。
