# dev_v16

## 启动
- 回顾 2025-11-11 完成的 dev_v15：Phase 5 延长后 `first_death` 有提升，但 Phase 6 的 `density_penalty=0.08` 实际上没有经历任何完整迭代，导致目标区间完全缺失。
- dev_v15 的多鱼评估（8×25）平均终局存活率 43%，显著低于 dev_v14（≈78%）；`first_death_p10` 仍落在 8 左右，说明 penalty>0.05 之前仍会触发早崩。
- 新增的 `schedule_trace.json` 成功降低复盘成本，是本轮必须延续的日志形态。

## 观察
- Ramp 逻辑在迭代末才应用下一阶段 penalty，意味着 Phase N 的 target 会在 N+1 的第一迭代才生效；当总迭代数恰好等于 Phase 6 结束时，`penalty=0.08` 永远不会命中训练主循环。
- Phase 4/5 期间 escape boost 0.82 与 penalty ramp 同步上升，TensorBoard 显示 `survival_rate` 虽稳定在 0.65 左右，但 `first_death` 低于 9，推断是 boost 过大导致鱼群在 ramp 初期就触发密度惩罚。
- 多鱼评估只在训练结束后执行一次，缺少 ramp 过程中“早期崩盘”的定位线索，这也是 dev_v15 未能解释 25 鱼崩溃根因的主要障碍。

## 实验计划
1. **Phase 6 真实执行**：将 Phase 6 的迭代数提升到 ≥4，并把 `total_iterations` 扩展到 30，确保 scheduler 至少在第 27~30 迭代期间维持 `density_penalty=0.08`，同时把 `_on_rollout_end` 的 penalty 更新提前到下一次 rollout 之前。
2. **Penalty-aware Boost**：在 `escape_boost_phase_speeds` 之外，新增一个 penalty→boost 裁剪逻辑：当密度 penalty ≥0.05 时把 boost 限制在 0.8，Phase 3/4 额外加入 0.76/0.78 过渡，目标是抑制 ramp 前的早崩。
3. **多鱼监控**：在训练 callback 中每 `k` 次迭代（初期设 4）触发一次 25 鱼 × 4 episodes 的快速评估，把 `survival_rate`、`first_death`、死亡分布写入 `artifacts/checkpoints/<run>/eval_multi_history.jsonl`，并同步生成简单折线图，方便在下一轮直接复盘。
4. **完整实验**：延续 dev_v15 的其它超参（`num_envs 128`、`n_steps 512`、`policy 384×384`），运行至少 30 iteration；若时间不足先跑 10 iteration sanity run，再复刻完整实验。

## 运行记录
- 2025-11-11 02:38~02:49 PST 完成 `dev_v16_runA_penalty_cap_probe`（default curriculum 6+7+4+4+5+4=30 iter，`--num_envs 128 --n_steps 512 --policy_hidden_sizes 384,384 --divergence_reward_coef 0.22`）。
- 关键命令：

  ```bash
  python experiments/v16/train.py \
    --run_name dev_v16_runA_penalty_cap_probe \
    --num_envs 128 --total_iterations 30 \
    --curriculum 15:6,20:7,25:4,25:4,25:5,25:4 \
    --divergence_reward_coef 0.22 \
    --density_penalty_phase_targets 0.0,0.02,0.04,0.05,0.06,0.08 \
    --density_penalty_ramp_phases 3,4,5,6 \
    --density_penalty_phase_plateaus 0,0,3,2,2,1 \
    --entropy_coef_schedule 16-24:0.03,25-32:0.02 \
    --escape_boost_phase_speeds 3:0.76,4:0.78,5:0.82,6:0.82 \
    --escape_boost_penalty_caps 0.05:0.8 \
    --multi_eval_interval 4 --multi_eval_probe_episodes 3 \
    --eval_multi_fish 25 --eval_multi_episodes 8 \
    --video_num_fish 25 --video_max_steps 500 --video_fps 20 \
    --seed 214
  ```
- 产物：
  - Checkpoints+日志：`experiments/v16/artifacts/checkpoints/dev_v16_runA_penalty_cap_probe/`（含 `training_stats.pkl`、`schedule_trace.json`、`eval_multi_history.jsonl`）。
  - Plot：`experiments/v16/artifacts/plots/` 下的 `*_survival.png / *_first_death.png / *_penalty_vs_* / *_multi_eval_timeline.png`。
  - 媒体：`experiments/v16/artifacts/media/dev_v16_runA_penalty_cap_probe_{curve.gif,multi_fish_eval.mp4}`。
  - 多鱼探针曲线：`eval_multi_history.jsonl` + `plots/dev_v16_runA_penalty_cap_probe_multi_eval_timeline.png` 已入库。
  - 终局多鱼评估（8×25）摘要：`avg_final_survival_rate=78.5% / min=68%`（`eval_multi_summary.json`）。
  - 单鱼 deterministic eval：平均奖励 250±43，`final_num_alive` 均值 16.4（`eval_single_fish.json`）。

## 结果 & 学习
- **Penalty=0.08 得到真实迭代**：Phase6 拉长 + 在 `_on_rollout_start` 更新 penalty 后，迭代 30 的 `density_penalty` 为 0.08（日志 `density_penalty_update iteration=030`），同迭代 `first_death=9.9`、`survival_rate=66%`，确认高 penalty 已进入训练而非只在结束后生效。
- **Penalty-aware boost 缓冲 Phase5**：Phase3/4/5 的 base speed = 0.76/0.78/0.82，但当 penalty≥0.05 自动 clamp=0.8。迭代 22~24 (`penalty=0.05`) 的 `first_death` 稳定在 11~12，相比 dev_v15 的 <9 有明显回升。
- **高 penalty 段仍掉队**：多鱼探针在迭代 28 (`penalty=0.06`) 记录到 `avg_final_survival_rate=66.7% / min=44% / early_death_fraction_150=22.7%`，说明 clamp=0.8 无法阻止 ramp 后期的密度冲击，尤其在 Phase6 ramp 到 0.06 之前。
- **探针提供定位**：`eval_multi_history.jsonl` 覆盖迭代 4→28，早期 (iter12) `avg_final_survival_rate=88% / early_death_fraction_150=6.7%`，到 iter28 迅速恶化，下一轮可针对该窗口插入额外 logging（例如 penalty-aware entropy）。
- **终局指标**：
  - `first_death` 后五迭代平均 9.13，峰值 37.8 仅出现在迭代 2。
  - 多鱼 (8×25) `avg_final_survival_rate=78.5%`、`min=68%`、`death_stats_p10=83.8`、`early_death_fraction_150=14.5%`。
  - 单鱼 deterministic eval 平均奖励 250，说明单体仍稳定，问题集中在密度 ramp。

## 下一步
1. **延缓 Phase6 ramp**：基于 `eval_multi_history`，考虑把 Phase6 分拆为 `penalty=0.065`（2 iter）+ `0.08`（2 iter），或将 Phase5 plateau 从 2→4，确保 `penalty≥0.06` 前至少获得 4 次迭代稳定化。
2. **自适应 boost clamp**：利用 schedule trace 的 rolling `first_death`，在 `p10<12` 时将 clamp 降到 0.78；`p10>15` 再回升到 0.8，避免固定阈值导致 Phase6 不够保守。
3. **探针加密 + 可视化**：把 `--multi_eval_probe_episodes` 提升到 5，并在 callback 内对 `eval_multi_history` 自动生成 per-phase death hist（保存到 `artifacts/plots/phase6_death_hist_iterXX.png`），方便下一轮定性观测。
4. **多鱼日志并行**：尝试在训练循环内追加 `eval_multi_history.jsonl`→TensorBoard 的同步（例如记录 `eval_multi/p10`），让 ramp 期间的崩溃可以直接在 TB 上看到，无需离线解析。
