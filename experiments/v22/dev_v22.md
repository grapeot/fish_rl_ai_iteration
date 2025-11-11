# dev_v22

## 启动
- 复盘 2025-11-11 完成的 `dev_v21_runE_penalty_lock`：`density_penalty_lock_value=0.04` 在 iter≤24 维持 `p10≈400`，解锁后跌到 ≈310；24EP gated best checkpoint (`model_iter_24.zip`) 给出 `avg_final_survival_rate≈0.86`、step-one 死亡 6。
- 捕食者 spawn jitter (1.2 半径) + 10-step pre-roll 将 `step_one_death_count` 降至 6~8，但部分 seed 在 iter12~16 仍会回弹到 14+，显示 pre-roll 距离仍不足。
- checkpoint 每 iter 写一份虽保证 gating，但占用 30 份模型；需要压缩策略，例如仅在满足 `step_one_death<=8` & `early_death_fraction_100<=0.1` 时额外保存 best。

## 观察
- penalty lock 结束得太早：iter25 解锁后 clamp 立即把 escape boost 推高，`first_death_p10` 降至 310 左右，说明 Phase6 仍需要 3~4 iteration 的低速缓冲。
- 捕食者 pre-roll 轨迹仍贴近鱼群球面，推测 pre-roll 步数和速度扰动都不够，导致 jitter 只在鱼群壳层附近扩散，step=1 死亡尾部仍难压到 <5。
- best checkpoint gating 逻辑可行，但每 iter checkpoint 占 IO；需要把“per-iter保存”改成“常规间隔 + 条件命中时补存 best”，并把 `best_model_path` 继续写进 history 供下一轮热启动。

## 实验计划
1. **延长 penalty lock**：在 `experiments/v22/train.py` 里把 `density_penalty_lock_until` 提升到 ≥28，并把 Phase6 plateau 延长到 12 iter，使 escape boost clamp 在 0.74 低速区更久，观察 `first_death_p10` 能否保持 >350。
2. **强化捕食者扰动**：将 `predator_pre_roll_steps` 提升至 12，并为 pre-roll 初速度添加 `speed_jitter_ratio≈0.15`（若需实现额外参数），让捕食者更大概率脱离鱼群表层，目标把 `step_one_death_count` 降到 ≤5。
3. **精简 checkpoint**：保留 `checkpoint_interval=5` 作为基础，仅当 multi-eval 通过 gating (`step_one_death<=8` & `early_death_fraction_100<=0.1`) 时额外保存 best checkpoint，同时把路径+指标写入 history/summary。
4. **主线实验**：以 `--num_envs 128`、≥30 iteration 做完整 run，并生成 penalty vs first-death 曲线与 multi-eval 时间线，全部落在 `experiments/v22/artifacts/`；必要时推一个 smoke run 验证 schedule，但最终需有完整 run 的日志/曲线/媒体。

## 运行记录

### dev_v22_runA_penalty_lock28（2025-11-11 06:03–06:04 PST，smoke）
- **命令**：`python experiments/v22/train.py --run_name dev_v22_runA_penalty_lock28 --num_envs 128 --total_iterations 30 --multi_eval_interval 4 --multi_eval_probe_episodes 16 --eval_multi_episodes 16 --best_checkpoint_eval_episodes 24 --checkpoint_interval 5 --seed 922`
- **状态**：CLI 默认 120s 超时在 iter=6 终止，但验证了新的 penalty lock=28、生效的 pre-roll speed jitter 及自动 best checkpoint 保存（iter4 时 `step_one=5`）。Artifacts：日志 `experiments/v22/artifacts/logs/dev_v22_runA_penalty_lock28.log`、TB `experiments/v22/artifacts/tb_logs/dev_v22_runA_penalty_lock28/`、plots `experiments/v22/artifacts/plots/dev_v22_runA_penalty_lock28_*`。
- **观察**：锁窗期内 `density_penalty_coef` 稳定 0.04，step-one 死亡快速回到 5~6；自动 gating 在首次满足阈值时写出 `model_iter_4.zip`，证明 checkpoint 精简逻辑可用，再开主线 run。

### dev_v22_runB_penalty_lock28（2025-11-11 06:06–06:16 PST，主线）
- **命令**：`python experiments/v22/train.py --run_name dev_v22_runB_penalty_lock28 --num_envs 128 --total_iterations 30 --multi_eval_interval 4 --multi_eval_probe_episodes 16 --eval_multi_episodes 16 --best_checkpoint_eval_episodes 24 --checkpoint_interval 5 --seed 123`
- **Artifacts**：日志 `experiments/v22/artifacts/logs/dev_v22_runB_penalty_lock28.log`，TensorBoard `experiments/v22/artifacts/tb_logs/dev_v22_runB_penalty_lock28/`，曲线 `experiments/v22/artifacts/plots/dev_v22_runB_penalty_lock28_survival.png` / `..._first_death.png` / `..._penalty_vs_first_death.png`，multi-eval timeline `experiments/v22/artifacts/plots/dev_v22_runB_penalty_lock28_multi_eval_timeline.png`（直方目录 `..._multi_eval_hist/`)，媒体 `experiments/v22/artifacts/media/dev_v22_runB_penalty_lock28_curve.gif`、`..._multi_fish_eval.mp4`、`..._best_iter_28_multi.mp4`，检查点+指标位于 `experiments/v22/artifacts/checkpoints/dev_v22_runB_penalty_lock28/`。
- **训练轨迹**：延长 lock 让 iter≤28 的 `density_penalty_coef` 全程保持 0.04，multi-eval step-one 记数从 9→6→4（迭代 12/24/28）。iter=28 自动满足 gating (`step_one=4`,`early_death_fraction_100=0.0925`) 并仅保存 `model_iter_28.zip` 作为 best（记录 `multi_eval_best_iter_28.json`）。
- **最终评估（全量 16EP deterministic）**：`avg_final_survival_rate=0.665`、`min_final_survival_rate=0.40`、`early_death_fraction_100=0.125`、`p10=64.9`，详见 `.../death_stats.json`。
- **Best checkpoint 24EP 复核（iter28）**：`avg_final_survival_rate=0.625`、`min_final_survival_rate=0.44`、`early_death_fraction_100≈0.10`，summary / history /视频分别在 `eval_multi_best_iter_28_summary.json`、`multi_eval_best_iter_28.json`、`media/dev_v22_runB_penalty_lock28_best_iter_28_multi.mp4`。
- **观察**：锁窗有效降低 step-one（迭代 28 时仅 4 次），但 Phase6 解锁后 penalty 立刻涨至 0.065，`p10` 从 158 跌到 64.9，最终 `avg_final_survival_rate` 仍低于 v21（0.86→0.665）。pre-roll speed jitter 把 step-one 均值压到 <7，不过 early-death 尾部依旧敏感，提示需要更平滑的解锁/再锁或 adaptive penalty 控制。

## Learning / 下一步
- **延长或分段解锁**：当前 lock=28 仍不足，一旦 Phase6 打开 penalty，`first_death_p10` 直接跌破 70；下一轮尝试把 Phase6 plateau 拆成双段（如 0.04→0.05→0.06）并引入 “解锁需连续两次 multi-eval 通过” 的逻辑，避免一次解锁导致逃逸速度立刻被 clamp。
- **Step-one 精细 gating**：speed jitter 把 `step_one_death_count` 稳定在 4~7；可追加 “只要 step-one<=5 就冻结 penalty 提升” 的策略，让低 step-one 区间再多跑几次 multi-eval。
- **评估指标回升**：即便 best iter=28 指标合格，24EP 复核平均只有 0.625。需要引入 min-survival >0.5 的硬阈值，或增大 multi-eval EP (→24) 以降低方差；同时考虑把 deterministic eval 的 `num_fish` 提到 32，检测密度 penalty 在高载荷下的表现。
- **后续工程**：保留 `experiments/v22/artifacts/checkpoints/dev_v22_runB_penalty_lock28/model_iter_28.zip` 作为 v23 热启动；若要对比锁窗策略，先将 `multi_eval_history.jsonl` 中 iter 20/24/28 数据可视化到 `experiments/v22/artifacts/plots/dev_v22_runB_penalty_lock28_multi_eval_timeline.png`，下一轮直接引用。
