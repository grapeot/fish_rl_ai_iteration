# dev_v12

## 启动
- dev_v11 的 phase-split ramp 虽然让阶段 3 的 first-death 均值提高到 ~9.6 步，但 ramp 尾段 (penalty≈0.08) 仍触发 7 步左右的崩塌，说明 penalty 热身与探索能力仍然脱节。
- dev_v11 Learning 中的 Run B 提案（迭代 16~24 提升 `ent_coef` 或 `escape_boost_speed`）被视为下一轮的首要验证点，以便在 penalty ramp 过程中保留更多策略多样性。
- 同时需要更灵活的 penalty gating：在阶段 3 前半段保持 0.02~0.04 的 plateau，再在阶段 4 ramp 至 0.08，以避免瞬时梯度过陡。

## 观察
- 基线参考 dev_v11_runA_phase_split：`num_envs=128`、`curriculum=15:6,20:9,25:4,25:5`、`divergence_reward_coef=0.22`、`escape_boost_speed=0.75`、density penalty ramp phases 3/4 从 0 → 0.08。
- 阶段 3 平均 `first_death_mean=9.6`、`first_death_p10=6.6`；迭代 22 以后随 penalty 升至 0.07~0.08，first-death 重新跌回 7 步。
- 多鱼 eval (25×8) 获得 `avg_final_survival=86% / min_final_survival=76%`，death histogram `p10=48.9` 显示后半程更稳定，但缺少 TB 中的 penalty vs first-death 可视化联动。

## 实验计划
1. **Run A：Entropy Cushion Ramp** —— 在 density penalty ramp (迭代 16~24) 内把 `ent_coef` 提升至 0.03，并维持 `escape_boost_speed=0.75`，验证阶段 3 的 `first_death_mean` 能否突破 12 步、`min_final_survival` 是否保持 ≥80%。
2. **Run B：Plateaued Penalty Ramp** —— 若 Run A 有效，再引入阶段 3 plateau（例如 4 迭代维持 penalty=0.02，再 ramp 至 0.06）并在阶段 4 ramp 0.06→0.08，观察 penalty 梯度降低是否能稳定 first-death ≥12 步。
3. **可观测性** —— 在训练脚本中新增 ent_coef scheduler logging + penalty plateau 指标，把“迭代→ent_coef/density penalty”同步到 TensorBoard，并输出 overlay 图到 `experiments/v12/artifacts/plots/`。

## 运行记录
- 2025-11-11 01:15~01:21 (UTC-8) `dev_v12_runA_entropy_cushion` —— 首次尝试在阶段 3 使用 `entropy_coef_schedule=16-24:0.03`，但旧的 density scheduler 公式让阶段 4 的 ramp 最高只到 0.072；完整 artifacts（曲线、GIF、mp4）仍保存在 `experiments/v12/artifacts/*/dev_v12_runA_entropy_cushion*`，仅作对照。
- 2025-11-11 01:22~01:28 (UTC-8) `source venv/bin/activate && python experiments/v12/train.py --run_name dev_v12_runA_entropy_cushion_fix --num_envs 128 --curriculum 15:6,20:9,25:4,25:5 --total_iterations 24 --n_steps 512 --policy_hidden_sizes 384,384 --divergence_reward_coef 0.22 --escape_boost_speed 0.75 --density_penalty_phase_targets 0.0,0.0,0.04,0.08 --density_penalty_ramp_phases 3,4 --density_penalty_phase_plateaus 0,0,2,0 --ent_coef 0.02 --entropy_coef_schedule 16-24:0.03 --eval_multi_episodes 8 --eval_multi_fish 25 --video_num_fish 25 --seed 212`
  - 阶段 3 (迭代 16~19) `first_death_mean=10.53 / p10=8.45`，单次最好是迭代 18 (`first_death=13.42`)；进入迭代 21 以后随 penalty>0.05 再度跌回 7~10 步。
  - 最佳迭代 22：`final_alive=17.45 (69.8%) / first_death=6.83`；多鱼 eval `avg_final_survival=47.5% / min_final_survival=36% / early_death_fraction_150=22.5%`，death histogram `p10=19.7 / p25=201`。
  - 单鱼 deterministic eval `avg_final_alive=17.6 / avg_reward=249.3`；TensorBoard 与 stats 写入 `experiments/v12/artifacts/tb_logs/dev_v12_runA_entropy_cushion_fix/` 与 `.../checkpoints/dev_v12_runA_entropy_cushion_fix/training_stats.pkl`。
  - 产出：日志 `experiments/v12/artifacts/logs/dev_v12_runA_entropy_cushion_fix.log`、ckpt `.../checkpoints/dev_v12_runA_entropy_cushion_fix/`、曲线 `.../plots/dev_v12_runA_entropy_cushion_fix_{survival,first_death}.png`、penalty/entropy overlay `..._penalty_vs_entropy.png`、death histogram `..._death_histogram.png`、GIF `experiments/v12/artifacts/media/dev_v12_runA_entropy_cushion_fix_curve.gif`、多鱼 mp4 `experiments/v12/artifacts/media/dev_v12_runA_entropy_cushion_fix_multi_fish_eval.mp4`。

## 结果与观察
- ent cushion 让阶段 3 的 first-death 出现 13.4 步的峰值，但均值只有 10.5，仍未达到 ≥12 的目标；penalty plateau=2 只能推迟 ramp 开始，无法阻止迭代 21~24 在 penalty≥0.05 时迅速坍塌（`penalty_vs_first_death.png` 显示 0.05 之后即跌入 6~9 步区间）。
- 多鱼表现显著劣化（avg survival 47.5%），主要因为 penalty ramp 在阶段 4 结束前就达到 0.08，配合高 entropy 导致鱼群在末段仍保持扩散而无法聚在一起逃逸；death histogram 的 `p10=19.7` 说明早期死亡比例仍偏高。
- 新增的 observability：`penalty_vs_entropy.png` 提供了 density 与 entropy 的双轴对照，TB 中也能看到 `custom/entropy_coef` 的阶梯；验证 plateau / ramp 调整时可以直接对齐 `iteration -> (penalty, entropy, first_death)`。

## Learning & 下一步计划
1. **Run B：Extended Plateau & gentler ramp** —— 将阶段 3 plateau 延长到 4~5 迭代 (`density_penalty_phase_plateaus=0,0,4,2`)，并把阶段 4 ramp 拆成两段 (0.04→0.06→0.08)，观察 first-death 是否能在 penalty≥0.05 前维持 ≥12，同时监控多鱼 survival。
2. **Entropy taper** —— 当前 `entropy_coef` 始终 0.03 贯穿 ramp，导致阶段 4 长尾 survival 掉到 47%；下一轮尝试 schedule `16-20:0.03,21-24:0.02` 或改为基于 penalty 的 hook，以免阶段 4 仍过度探索。
3. **Escape boost gating** —— 如果延长 plateau 仍无法把 first-death 抬到 12，可在 ramp 区间临时把 `escape_boost_speed` 提升至 0.8 并在 TB 中记录“boost on/off”，评估其对 early death 分位点 (p10) 的影响。
