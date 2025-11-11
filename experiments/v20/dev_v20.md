# dev_v20

## 启动
- dev_v19_runA_low_speed_plateau_v2（2025-11-11）把 `adaptive_boost_low_speed` 压到 0.76 并延长 Phase5 plateau，`early_death_fraction_100` 从 0.224 降至 0.113，但 deterministic eval 的 `min_final_survival_rate` 仍只有 0.48，`first_death_p10` 长期锁死在 1。
- step=1 碰撞样本集中在捕食者起点附近 `(0.15, 0.005)`，`eval_step_one_deaths.json` 与 `multi_eval_history` 证实绝大多数鱼在刚开局即被吞噬，说明初始化几何而非逃逸速度是主要瓶颈。
- clamp 虽能把 `escape_boost_speed` 拉回 0.76，但 Phase5/6 penalty ramp（0.08 阶梯）太快，刚降速就把 `density_penalty` 提升导致 clamp 无法稳定；多鱼探针 12EP 样本方差仍大，需要更长 deterministic eval（≥20 EP）挑 checkpoint。

## 观察
- Adaptive clamp 与 penalty ramp 未能协同：Phase5 结束前 penalty 已爬到 0.08，刚触发 clamp 的低速没等验证就被新的高 penalty 带回失控区，`first_death_p10` 曲线与 penalty 曲线合拍震荡。
- step=1 死亡空间簇在捕食者起点 0.2 半径内，鱼群初始半径 8 的随机采样并未避开中心；spawn jitter 或 pre-roll 可能在不动鱼的情况下把捕食者先推离中心。
- 多鱼 deterministic eval (12EP) 在 iter16 最佳，但 `avg_final_survival_rate` 只有 ~0.60，且 `early_death_fraction_100` 在 iter24 以后重新抬头，提示 checkpoint 选择应盯住 step-one death 计数最低段；需要自动化 “best early-death checkpoint”。

## 实验计划
1. **Spawn jitter + pre-roll**：在 `FishEscapeEnv` 中加入捕食者初始位移 (`predator_spawn_jitter_radius`) 与只移动捕食者的 pre-roll (`initial_pre_roll_steps`)，默认 dev_v20 运行设置为 radius=0.8、pre_roll=6，期待 step=1 death 显著下降。
2. **更激进 clamp**：把 `adaptive_boost_low_speed` 降到 0.74，并将 penalty ramp plateau 配置改为 Phase5/6 至少 6/4 iter，保证 clamp 有时间评估；同时保留 upper=0.80 便于恢复。
3. **Checkpoint 复核**：训练脚本末尾对 `model_iter_{best}`（按 `early_death_fraction_100` & step-one death 最低迭代）额外运行 ≥20 EP deterministic eval，并产出曲线/媒体到 `experiments/v20/artifacts/plots|media/`，把运行命令、指标、artifact 路径写进下文运行记录。

## 运行记录

### dev_v20_runA_spawn_jitter（2025-11-11 04:41–04:43 PST，未完成）
- **命令**：`python experiments/v20/train.py --run_name dev_v20_runA_spawn_jitter --num_envs 128 --checkpoint_interval 2 --multi_eval_interval 4 --eval_multi_episodes 16 --best_checkpoint_eval_episodes 24 --adaptive_boost_low_speed 0.74 --density_penalty_phase_plateaus "0,0,6,6,8,4,4" --predator_spawn_jitter_radius 0.8 --predator_pre_roll_steps 6 --seed 311`
- **状态**：CLI 在迭代 7 前超时，仍保留早期 checkpoint/log（`experiments/v20/artifacts/checkpoints/dev_v20_runA_spawn_jitter/`、`logs/dev_v20_runA_spawn_jitter.log`、`plots/dev_v20_runA_*`）。不计入正式结论，仅作为新脚本 smoke test（确认 pre-roll/jitter 能跑通）。

### dev_v20_runB_spawn_jitter（2025-11-11 04:44–05:00 PST）
- **命令**：同 RunA，但 `--run_name dev_v20_runB_spawn_jitter`；完整 30 iteration @ `--num_envs 128`。
- **日志/曲线**：`logs/dev_v20_runB_spawn_jitter.log`、TensorBoard `tb_logs/dev_v20_runB_spawn_jitter/`、曲线 `plots/dev_v20_runB_spawn_jitter_*.png`、动态曲线 `media/dev_v20_runB_spawn_jitter_curve.gif`。
- **期末 deterministic eval (16×EP, 25 fish)**：`avg_final_survival_rate=0.795`、`min_final_survival_rate=0.64`，详见 `checkpoints/dev_v20_runB_spawn_jitter/eval_multi_summary.json`；死亡统计 `death_stats.json` 给出 `p10=79.1`、`early_death_fraction_100=0.12`（dev_v19 baseline 0.113），median 500 step。
- **Best iteration 选择**：`multi_eval_history.jsonl` 的迭代 20 拥有 `step_one_death_count=6`、`early_death_fraction_100=0.093`（density penalty=0.04、escape_boost=0.74）。训练脚本自动生成 `multi_eval_best_iter_20.json`（记录 heuristics）与 checkpoint `model_iter_20.zip`。
- **Best checkpoint 复核 (24×EP)**：`eval_multi_best_iter_20_summary.json`：`avg_final_survival_rate=0.793`、`min_final_survival_rate=0.64`、`early_death_fraction_100=0.093`，对应视频 `media/dev_v20_runB_spawn_jitter_best_iter_20_multi.mp4`（同目录 `.../multi_fish_eval.mp4` 保存末迭代 16EP 画面）。
- **Step=1 诊断**：`eval_step_one_deaths.json` 仅记录 11 个事件，较 v19 (≈30) 明显下降；坐标分布在捕食者偏移后的 1.6~2.2 半径处，说明 pre-roll+jitter 已把冲突向外推，但仍需 further jitter/inertia 以避免个别鱼仍贴边。
- **Artifacts 总览**：`checkpoints/dev_v20_runB_spawn_jitter/`（checkpoint/timeline/JSON）、`plots/dev_v20_runB_spawn_jitter_multi_eval_timeline.png`（迭代 4→28 `avg_final` 稳定在 0.77~0.82，同时 step-one 曲线在 Phase5 降到 6）、`plots/dev_v20_runB_spawn_jitter_death_histogram.png`（死亡主要集中在 500 step）。
- **关键观察**：
  - spawn jitter + 6-step pre-roll 把死亡 10 分位推迟到 79 step，同时 `step_one_death_count` 降至一位数，但 Phase6 ramp 仍把 clamp 逼回 0.80（`adaptive_boost` 几乎每次 multi eval 都被触发）。
  - `early_death_fraction_100` 在 iter 28 penalty=0.065 时回升到 0.153，证明 plateau 仍不够长；最稳定的迭代集中在 penalty=0.04~0.05 区间（iter 16~22）。
  - best checkpoint 自动评估 + 媒体输出流程已经跑通，文档化和可交接性明显提升。

## Learning / 下一步
- **延长 Phase5/6 plateau**：在当前 `0,0,6,6,8,4,4` 基础上，再把 Phase6 plateau 调至 ≥6（或直接锁 0.04 penalty 到 iter 24），避免 penalty 把 clamp 推回 0.80；可结合 `penalty_entropy.png` 观察 ramp 触发点。
- **更强 spawn jitter / pre-roll**：虽然 step=1 事件下降到 11 次，但仍发生在捕食者预热轨迹附近，建议：1) 让 pre-roll 步骤指数式拉长（6→10），2) 给捕食者初速增加 `np_random.uniform(-0.3,0.3)` 角度让轨迹多样化。
- **自动 early-death gating**：目前 best checkpoint依据 multi eval history（每 4 iter）；下一步可在训练期中直接对 `step_one_death_count` 设阈值（<=8）才保存额外 checkpoint，并在 `multi_eval_history` 中写入 `best_model_path` 以便下轮直接加载。
- **评估覆盖**：继续沿用 24EP deterministic 复核，同时为 `eval_multi_best_iter_*` 生成密度热力/step-one散点（可复用 `eval_step_one_deaths.json`），帮助 next iteration 直接验证 spawn jitter 参数。
