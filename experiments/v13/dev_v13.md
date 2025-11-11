# dev_v13

## 启动
- 复盘 dev_v12：entropy cushion 把阶段 3 的 `first_death_mean` 拉到 ~10.5，但仍无法在 penalty≥0.05 时维持 12 步，multi-fish survival 47.5% 且 death histogram p10=19.7。
- dev_v12 的 Learning 指向三件事：① 阶段 3 需要更长 plateau（持续 0.02~0.04）延缓 ramp；② entropy 需在迭代 21+ 回落以避免阶段 4 过度探索；③ 如 plateau 仍不足，再考虑临时提升 `escape_boost_speed`。
- dev_v13 目标：先验证 Run B（extended plateau + 温和 ramp + entropy taper）的可行性，再视情况叠加 escape boost gating。

## 观察
- 基线命令：`dev_v12_runA_entropy_cushion_fix`（num_envs=128，density ramp 3/4 → 0.08，ent cushion 0.03）。迭代 16~19 `first_death_mean=10.53`，迭代 22 后跌回 7~10；multi eval `avg_final_survival=47.5% / min=36%`。
- penalty vs first-death 显示一旦 penalty>0.05 就骤降，entropy 0.03 贯穿导致阶段 4 小鱼无法聚拢；plateau=2 只延迟 ramp 起点，未能缓 ramp 斜率。

## 实验计划
1. **Run B – Extended Plateau Ramp**：基于 dev_v12 runA 复刻参数，但把 phase targets 调整为 `0.0,0.02,0.04,0.08` 并设置 `phase_plateaus=0,0,4,2`、`ramp_phases=3,4`，期望在 penalty≤0.05 内把 `first_death_mean` 推至 ≥12。
2. **Entropy Taper**：应用 `entropy_coef_schedule=16-20:0.03,21-24:0.02`，观测阶段 4 survival 是否回升到 ≥65%。
3. **Stretch – Escape Boost Gating**：若上述两点仍无法让 `first_death_p10` ≥10，则在 plateau 期间短暂将 `escape_boost_speed`=0.8，并记录 on/off scalars 以供下一轮对比。

## 运行记录
- 2025-11-11 01:39~01:46 (UTC-8) `source venv/bin/activate && python experiments/v13/train.py --run_name dev_v13_runB_plateau_taper --num_envs 128 --curriculum 15:6,20:9,25:4,25:5 --total_iterations 24 --n_steps 512 --policy_hidden_sizes 384,384 --divergence_reward_coef 0.22 --escape_boost_speed 0.75 --density_penalty_coef 0.0 --density_penalty_phase_targets 0.0,0.02,0.04,0.08 --density_penalty_ramp_phases 3,4 --density_penalty_phase_plateaus 0,0,4,2 --ent_coef 0.02 --entropy_coef_schedule 16-20:0.03,21-24:0.02 --eval_multi_episodes 8 --eval_multi_fish 25 --video_num_fish 25 --video_max_steps 500 --seed 213`
  - Phase 3 (`iter 16-20`) `first_death_mean=10.43 / p10=9.15 / final_alive_mean=17.47`；plateau ≥4 iter 保证 penalty≤0.02 时波动 <±1 步。
  - Phase 4 (`iter 21-24`) 在 plateau 结束后 ramp=0.04→0.06→0.08：`first_death_mean=10.03`，但 `iter24` penalty=0.06 时跌至 4.17 步；entropy taper 保持 `survival_rate≈69.7% / final_alive≈17.4`。
  - 单鱼 deterministic eval (`experiments/v13/artifacts/checkpoints/dev_v13_runB_plateau_taper/eval_single_fish.json`)：平均 `reward=258.1 / final_alive=18.4`。
  - 多鱼 eval (`experiments/v13/artifacts/checkpoints/dev_v13_runB_plateau_taper/eval_multi_summary.json`)：`avg_final_survival=51.5% / min=32% / median_min_survival=56%`；death stats (`experiments/v13/artifacts/checkpoints/dev_v13_runB_plateau_taper/death_stats.json`) `p10=3.8 / p25=200.8 / survival_fraction=51.5% / early_death_fraction_150=24%`。
  - 产物：日志 `experiments/v13/artifacts/logs/dev_v13_runB_plateau_taper.log`、TensorBoard `experiments/v13/artifacts/tb_logs/dev_v13_runB_plateau_taper/`、检查点+stats `experiments/v13/artifacts/checkpoints/dev_v13_runB_plateau_taper/`、曲线 `experiments/v13/artifacts/plots/dev_v13_runB_plateau_taper_{survival,first_death,penalty_vs_first_death,penalty_vs_entropy,death_histogram}.png`、媒体 `experiments/v13/artifacts/media/dev_v13_runB_plateau_taper_{curve.gif,multi_fish_eval.mp4}`。

## 结果 & 学习
- Phase 3 plateau 延长后，`first_death_p10` 从 v12 的 8.45 提升到 9.15，说明保持 penalty=0.02 确实给策略更多探索空间，但均值仍停留在 10.4 步，未触及 ≥12 的目标。
- Phase 4 ramp 被拆成 plateau(2) + ramp，但由于阶段总长度只有 5 iter，0.04→0.08 仍在 3 iter 内完成，`iter24` penalty=0.06 即触发 first-death 崩塌 (4.17)，多鱼 survival 也只回升到 51.5%（虽高于 v12 的 47.5%，但 `min_final_survival` 反而掉到 32%）。
- entropy taper (21+ 降至 0.02) 让 `survival_rate` 没有因为 ramp 而暴跌（phase4 平均 69.7%），说明 taper 有助于保持群体收敛，但 death histogram `p10=3.8` / `early_death_fraction_150=24%` 暗示梯度仍过于陡峭。
- 密度/entropy overlay (`experiments/v13/artifacts/plots/dev_v13_runB_plateau_taper_penalty_vs_entropy.png`) 显示 penalty=0.06 时 entropy 已经回落，所以问题主要来自 penalty ramp，而非熵噪声。

## 下一步计划
1. **延长 Phase 4 或拆出 Phase 5**：把 curriculum 改成 `15:6,20:9,25:4,25:3,25:2`（或类似）以获得 ≥5 iter ramp，只在最后 2 iter 抬到 0.08，确保 penalty=0.06 有足够训练步数验证稳定性。
2. **Escape Boost Gating**：在 ramp 区间 (iter 21-24) 动态提高 `escape_boost_speed=0.8` 并新增 TB scalar（boost_on/off），检验其对 `first_death_p10` 的缓冲作用；如果有效，记录 gating 策略以便下一轮固化到训练脚本。
3. **更细粒度的 penalty logging**：当前 stats 只记录 iteration 对应值，后续可在 callback 中把 ramp 更新日志写入 JSON（iteration→penalty）方便散点分析 penalty>0.05 区域的 collapse 频率。
