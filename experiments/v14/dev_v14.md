# dev_v14

## 启动
- 复盘 dev_v13：延长 Phase 3 plateau + entropy taper 让 `first_death_p10` 从 8.45 升至 9.15，但均值仍停在 ~10.4，距离 ≥12 的目标尚有差距。
- Phase 4 ramp 仍过陡：0.04→0.08 仅 3 iter，`penalty=0.06` 时 `first_death` 直接坍缩至 4.17，`min_final_survival` 也跌到 32%。
- entropy taper（21+ 降到 0.02）验证有效，可保持 `survival_rate≈70%`，说明问题主要来自 penalty ramp，而非噪声。

## 观察
- `density_penalty` 只要超过 0.05，`first_death` 与多鱼 survival 都会断崖式下滑，说明策略在高密度约束下缺少缓冲期。
- Phase 4 只有 5 iter，使 plateau/ramp 都很拥挤；需要拆出 Phase 5 或拉长 Phase 4，让 penalty=0.06 有足够 sample 数。
- Escape boost gating 仍未实现，缺乏额外的“紧急逃逸”支撑，导致 ramp 后期的第一批死亡无法被延后。

## 实验计划
1. **Phase 4/5 Reshape**：採用 `curriculum=15:6,20:9,25:4,25:3,25:2`，把 `density_penalty_phase_targets` 扩展到 `0.0,0.02,0.04,0.06,0.08`，并设 `phase_plateaus=0,0,4,2,1` + `ramp_phases=3,4,5`，确保 penalty=0.06 区段至少训练 3 iter。
2. **Entropy Taper 延续**：继续使用 `16-20:0.03,21-27:0.02`，若 Phase 5 仍不稳再考虑 Phase 5 尾声降到 0.015。
3. **Escape Boost Gating**：在 Phase 4-5 ramp 区间把 `escape_boost_speed` 提至 0.8，并在 callback 记录 `boost_state` scalar，验证其对 `first_death_p10` 的缓冲作用；若效果正面，准备固化为默认策略。

## 运行记录
- 2025-11-11 01:57~02:04 (UTC-8) 运行 `dev_v14_runA_phase5_gate`
  - 命令：
    ```
    source venv/bin/activate && python experiments/v14/train.py \
      --run_name dev_v14_runA_phase5_gate --num_envs 128 --curriculum 15:6,20:9,25:4,25:3,25:2 \
      --total_iterations 24 --policy_hidden_sizes 384,384 --divergence_reward_coef 0.22 \
      --density_penalty_phase_targets 0.0,0.02,0.04,0.06,0.08 --density_penalty_ramp_phases 3,4,5 \
      --density_penalty_phase_plateaus 0,0,4,2,1 --entropy_coef_schedule 16-20:0.03,21-27:0.02 \
      --escape_boost_speed 0.75 --escape_boost_phase_speeds 4:0.8,5:0.8 \
      --eval_multi_episodes 8 --eval_multi_fish 25 --video_num_fish 25 --video_max_steps 500 \
      --n_steps 512 --seed 213
    ```
  - 产物：日志 `experiments/v14/artifacts/logs/dev_v14_runA_phase5_gate.log`、TensorBoard `experiments/v14/artifacts/tb_logs/dev_v14_runA_phase5_gate/`（含 `custom/escape_boost_speed` scalar）、检查点+stats `experiments/v14/artifacts/checkpoints/dev_v14_runA_phase5_gate/`、曲线 `experiments/v14/artifacts/plots/dev_v14_runA_phase5_gate_{survival,first_death,penalty_vs_first_death,penalty_vs_entropy,death_histogram}.png`、动画 `experiments/v14/artifacts/media/dev_v14_runA_phase5_gate_curve.gif`、多鱼视频 `experiments/v14/artifacts/media/dev_v14_runA_phase5_gate_multi_fish_eval.mp4`。

## 结果 & 学习
- Phase 3（iter16-19, penalty=0.02 plateau）`first_death_mean=10.59 / p10=6.01`，比 v13 的 `p10=9.15` 反而退步，说明新增 Phase 5 并未给 penalty≤0.02 足够的巩固窗口（受 Phase 2 提前 ramp 影响，iter11/19 已出现 <8 的 first_death）。
- Phase 4（iter20-22, penalty ramp→0.04, escape boost=0.8）表现明显好于 v13：`first_death_mean=10.00 / p10=7.55` 且训练日志中迭代 21/22 分别达到 11.29/12.10，佐证 gating 能在 penalty≤0.04 时承压。
- Phase 5 只有 2 iter，统计仍停留在 `penalty=0.06`（scheduler 在 iter24 结束时才切到 0.08）。`first_death` 立刻降到 10.21→6.56，`p10≈6.93`，虽然 multi eval `avg_final_survival=78.5% / min=64% / median_min=78%`（上一轮为 51.5% / 32% / 56%），但 `first_death≥12` 的目标依旧未达。
- `death_stats.json` 显示 `p10=13.6, survival_fraction=78.5%, early_death_fraction_150=18%`，对比 v13 的 `p10=3.8 / early_death_fraction_150=24%`，说明更长 plateau + gating 明显减少极端早死，只是 ramp 至 0.06 仍过快。
- 单鱼 deterministic eval：平均 `reward=287.6 / final_alive=20.6`，证明策略在单体场景已基本稳定；瓶颈主要来自高 penalty 条件。
- 记录的 `custom/escape_boost_speed` scalar（0.75→0.8）在 TensorBoard 中可直观看到 gating 切换时间点，可作为后续调参的对照。

## 下一步
1. **延长 Phase 5/+Phase 6**：至少给 `penalty=0.06` 3 iter、`penalty=0.08` ≥2 iter，可尝试 `25:4,25:3,25:3,25:2`（总 26 iter）或保持 24 iter 但把 Phase 3/4 各挪 1 iter 到 Phase 5，确保真正看到 penalty≥0.06 的稳态行为。
2. **Penalty per-iteration 追踪落地**：目前只有 `training_stats.pkl` 内部记录。添加 JSON dump（iteration→penalty, escape_boost）便于无 python 环境时快速评估 ramp 区崩溃点，也满足 v13 的 logging TODO。
3. **Gating 策略再细分**：Phase 3 已出现 first-death <8，考虑引入 `escape_boost_phase_speeds=3:0.78` 或按 penalty 阈值动态 boost（仅在 penalty≥0.04 时升至 0.8），避免早期迭代过度推进鱼群导致 Phase 3 p10 下降。
