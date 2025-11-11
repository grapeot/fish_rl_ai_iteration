# dev_v21

## 启动
- dev_v20 在 2025-11-11 完成 `dev_v20_runB_spawn_jitter`，`--num_envs 128` 共 30 iteration，`avg_final_survival_rate≈0.79`、`early_death_fraction_100≈0.093`（best iter 20）。
- spawn jitter（0.8 半径）+ 6 step pre-roll 把 step=1 死亡从 ~30 降到 11，验证几何初始化可显著改善开局聚集死亡。
- 但 density penalty phase plateaus `0,0,6,6,8,4,4` 仍过短，Phase6 penalty≈0.065 时 `early_death_fraction_100` 回升到 0.153，adaptive boost clamp 被迫回到 0.80，导致 Phase5/6 末尾震荡。
- 自动 best-checkpoint 流程 + 24EP deterministic 复核已在 v20 打通，`media/dev_v20_runB_spawn_jitter_best_iter_20_multi.mp4` 可用；v21 需要延续并沉淀指标写法。

## 观察
- penalty ramp 与 clamp 仍互相抢占：即便第一死亡显著下降，Phase6 penalty 一旦超过 0.05 就把 escape boost 拉回 0.80，造成 `first_death_p10` 曲线回弹。
- 捕食者 pre-roll 只移动 6 step，仍有 11 个 step=1 死亡集中在半径 1.6~2.2；说明 jitter 成功把 cluster 往外推，但 pre-roll 时长与初速扰动还不足以让捕食者脱离鱼群表面。
- multi eval 机制运转正常，不过 best checkpoint 只根据 rolling 24EP `early_death_fraction_100`，尚未对 `step_one_death_count` 设硬阈值，导致个别 iteration 虽早期表现好但 step=1 death 偶发。

## 实验计划
1. **延长 penalty plateau**：在 dev_v21 将 Phase5/6 plateau 延长至 `0,0,6,10,10,6,6`（或等效逻辑）并允许锁定 penalty=0.04 直到 iter≥24，确保 clamp 在低速下收敛。
2. **加强捕食者扰动**：默认 `predator_spawn_jitter_radius=1.2`、`predator_pre_roll_steps=10`，并给 pre-roll 期间的速度方向加入 `±0.3rad` 抖动（若脚本支持），以进一步压低 step=1 死亡。
3. **Step-one gating checkpoint**：训练过程中仅当 `step_one_death_count<=8` 且 `early_death_fraction_100<=0.1` 时才更新 best checkpoint，并在 multi-eval history 中写入 `best_model_path` 字段，方便下一轮直接引用。
4. **运行规模**：维持 `--num_envs 128`、≥30 iteration；若 plateau 延长导致时间增加，可在 smoke 阶段先跑 10 iter 验证 schedule，再跑完整实验。
5. **产出要求**：所有日志、tb、plots、媒体写入 `experiments/v21/artifacts/`，并为 penalty vs first-death 追加曲线，完成后在 “运行记录” 中填入命令、指标与 artifact 路径。

## 运行记录

### dev_v21_runA_penalty_lock（2025-11-11 05:11–05:12 PST，smoke）
- **命令**：`python experiments/v21/train.py --run_name dev_v21_runA_penalty_lock --num_envs 128 --total_iterations 30 --multi_eval_interval 4 --multi_eval_probe_episodes 16 --eval_multi_episodes 16 --best_checkpoint_eval_episodes 24 --adaptive_boost_low_speed 0.74 --density_penalty_phase_plateaus "0,0,6,10,10,6,6" --density_penalty_lock_value 0.04 --density_penalty_lock_until 24 --predator_spawn_jitter_radius 1.2 --predator_pre_roll_steps 10 --predator_pre_roll_angle_jitter 0.3 --seed 321`
- **状态**：CLI 默认 120s 超时在 iter=6 终止，仅验证新的 penalty lock / pre-roll 参数；Artifacts：`experiments/v21/artifacts/logs/dev_v21_runA_penalty_lock.log`、`tb_logs/dev_v21_runA_penalty_lock/`、`checkpoints/dev_v21_runA_penalty_lock/model_iter_5.zip`。
- **观察**：pre-roll+jitter 运行正常，step=1 死亡在 smoke 阶段已降至个位数；后续 runB/runC 也只跑到 iter≈6 即捕获 `build_vectorized_env` 缺少角度噪声参数的 TypeError，用于修脚本但不计入结论。

### dev_v21_runD_penalty_lock（2025-11-11 05:18–05:29 PST）
- **命令**：`python experiments/v21/train.py --run_name dev_v21_runD_penalty_lock --num_envs 128 --total_iterations 30 --multi_eval_interval 4 --multi_eval_probe_episodes 16 --eval_multi_episodes 16 --best_checkpoint_eval_episodes 24 --adaptive_boost_low_speed 0.74 --density_penalty_phase_plateaus "0,0,6,10,10,6,6" --density_penalty_lock_value 0.04 --density_penalty_lock_until 24 --predator_spawn_jitter_radius 1.2 --predator_pre_roll_steps 10 --predator_pre_roll_angle_jitter 0.3 --seed 611`
- **指标**：16EP deterministic eval `avg_final_survival_rate=0.838`、`min_final_survival_rate=0.76`、`death_stats.p10=90.5`、`early_death_fraction_100=0.105`（见 `experiments/v21/artifacts/checkpoints/dev_v21_runD_penalty_lock`）。
- **Artifacts**：`logs/dev_v21_runD_penalty_lock.log`、`tb_logs/dev_v21_runD_penalty_lock/`、plots `plots/dev_v21_runD_penalty_lock_*.png`、媒体 `media/dev_v21_runD_penalty_lock_curve.gif` / `media/dev_v21_runD_penalty_lock_multi_fish_eval.mp4`。
- **观察**：multi-eval 在 iter=16 达成 `step_one_death_count=8`、`early_death_fraction_100=0.082`，但 `checkpoint_interval=5` 未持久化 `model_iter_16.zip`，gating 无法输出 best checkpoint；决定改为 per-iter checkpoint。

### dev_v21_runE_penalty_lock（2025-11-11 05:31–05:43 PST，主线）
- **命令**：`python experiments/v21/train.py --run_name dev_v21_runE_penalty_lock --num_envs 128 --total_iterations 30 --checkpoint_interval 1 --multi_eval_interval 4 --multi_eval_probe_episodes 16 --eval_multi_episodes 16 --best_checkpoint_eval_episodes 24 --adaptive_boost_low_speed 0.74 --density_penalty_phase_plateaus "0,0,6,10,10,6,6" --density_penalty_lock_value 0.04 --density_penalty_lock_until 24 --predator_spawn_jitter_radius 1.2 --predator_pre_roll_steps 10 --predator_pre_roll_angle_jitter 0.3 --seed 777`
- **训练轨迹**：penalty lock 让 `density_penalty_coef` 在 iter≤24 全程保持 0.04；multi-eval step-one 计数在 iter24 降至 6，`early_death_fraction_100=0.0575`。
- **最终评估**：16EP deterministic eval `avg_final_survival_rate=0.8475`、`min_final_survival_rate=0.64`、`death_stats.p10=310.7`、`early_death_fraction_100=0.0575`（`checkpoints/dev_v21_runE_penalty_lock/eval_multi_summary.json`、`death_stats.json`）。
- **Best checkpoint (gated)**：`multi_eval_best_iter_24.json` 记录 `best_model_path=.../model_iter_24.zip`，24EP 复核 `avg_final_survival_rate=0.86`、`min_final_survival_rate=0.64`，step-one 死亡仅 6（`eval_multi_best_iter_24_summary.json`）；视频 `experiments/v21/artifacts/media/dev_v21_runE_penalty_lock_best_iter_24_multi.mp4`。
- **Artifacts 汇总**：日志 `experiments/v21/artifacts/logs/dev_v21_runE_penalty_lock.log`、TensorBoard `experiments/v21/artifacts/tb_logs/dev_v21_runE_penalty_lock/`、plots `experiments/v21/artifacts/plots/dev_v21_runE_penalty_lock_*.png`、媒体 `experiments/v21/artifacts/media/dev_v21_runE_penalty_lock_curve.gif` / `..._multi_fish_eval.mp4`、history `checkpoints/dev_v21_runE_penalty_lock/eval_multi_history.jsonl`、step-one 样本 `.../eval_step_one_deaths.json`。
- **观察**：锁住 penalty 时 `p10` 可维持在 400+，解锁至 Phase6 ramp 后降到 ≈310；`checkpoint_interval=1` 虽增存储但保证 gating 完成，history 现含 `best_model_path` 用于下一轮热启动。

## Learning / 下一步
- 延长锁定窗口：iter25 解锁后 `p10` 迅速掉到 ~310，考虑把 `density_penalty_lock_until` 提高到 ≥28 或为 Phase6 再追加 2 iteration plateau，让 clamp 能在 0.74 更长时间内工作。
- 捕食者仍会在部分 seed 中贴边，iter12~16 的 step-one 激增（14~15）说明仅靠 10-step pre-roll 仍不足；计划在 v22 试验更长 pre-roll（12）并给大鱼初速绝对值也加 jitter，以进一步分散开局碰撞。
- checkpoint per-iter 虽解决 gating，但出现 30 份模型；下一轮需要在脚本里判断只有当 multi-eval 通过阈值时才额外保存 best checkpoint，以减小 IO 与存储。
