# dev_v11

## 启动
- dev_v10 的 Run A/B 均证明阶段 3 (迭代 ≥16) 的 density penalty 一旦 ramp 起步就会把 `first_death` 拖回 <10 步，即便阶段 1/2 保持 0；Ramp 的瞬时梯度太陡，需要把阶段 3 拆成“热身 + 正式 ramp”并延迟到迭代 18 以后再升至 0.08。
- divergence 奖励提升到 0.22 可把 `min_final_survival` 从 40% 拉到 68%，但 first-death 仍崩在 7~10 步，说明“逃逸热身”与 penalty 的联动尚未建立；本轮要在 penalty ramp 期间额外注入探索（entropy boost 或更强的初始逃逸速度）。
- dev_v10 新增的 first-death 折线/直方图已投入使用，但 min survival 仍只保存在 JSON；TensorBoard 里缺乏 `custom/min_survival_rate` 等指标，不利于下一轮复盘，需要把多鱼 eval 摘要同步进 TB，并绘制 penalty vs first-death 的对照图。

## 观察
- 当前基线（dev_v10_runB）: `num_envs=128`, `curriculum=15:6,20:9,25:9`, `divergence_reward_coef=0.22`, density penalty 在迭代 16~24 ramp 到 0.08；阶段 3 平均 `first_death_mean=7.5 / p10=5.0`，多鱼 eval `avg_final_survival=70.5% / min_final_survival=68%`。
- ramp 生效那一迭代，first-death 曲线立刻跌落，且 `density_penalty` 与 `first_death` 的相关性尚未量化（日志里只有时间序列）；需要新增 stats 来记录每次 penalty 更新时的实时 value，以便和 first-death 对齐。
- 多鱼 eval 的 JSON 摘要已经含有 `avg_min_survival_rate` 等字段，但没有进入 TensorBoard；后续迭代想看趋势只能手工比对 md 文档。

## 实验计划
1. **Run A：Phase-split warm ramp** —— 把 curriculum 拆成 `15:6,20:9,25:4,25:5`，对应的 density 目标为 `0.0,0.0,0.04,0.08` 且仅对阶段 3/4 做 ramp；保持 `divergence_reward_coef=0.22`、`num_envs=128`、`n_steps=512`，检查迭代 16~19 的 first-death 是否能稳定在 ≥12 步并绘制 penalty vs first-death 图。
2. **Run B：Penalty entropy cushion** —— 基于 Run A，再在 penalty ramp 区间临时把 `ent_coef` 提升至 0.03（或等效地把 `escape_boost_speed` 提升至 0.8`），比较 first-death 与 min survival 的提振幅度；一旦确认有效，再决定是否需要引入 neighbor variance 特征。
3. **可观测性交付** —— 在 `train.py` 中把 density penalty 值写入迭代 stats 并生成 penalty-first_death 对照图；训练完毕后向 TensorBoard 写入 `eval_multi/min_final_survival_rate` 等指标，所有 plots/gif/mp4 存入 `experiments/v11/artifacts/{plots,media}/`。

## 运行记录
- 2025-11-11 00:54~01:03 (UTC-8) `source venv/bin/activate && python experiments/v11/train.py --run_name dev_v11_runA_phase_split --num_envs 128 --curriculum 15:6,20:9,25:4,25:5 --density_penalty_phase_targets 0.0,0.0,0.04,0.08 --density_penalty_ramp_phases 3,4 --divergence_reward_coef 0.22 --policy_hidden_sizes 384,384 --eval_multi_episodes 8 --eval_multi_fish 25 --video_num_fish 25 --escape_boost_speed 0.75 --seed 211`
  - 最佳迭代 22：`final_alive=17.44 / survival_rate=69.8% / first_death=7.0`；阶段 3 (16~24) 平均 `final_alive=16.96 / first_death_mean=9.59 / p10=6.56`，比 dev_v10 的 7.5 有所回升但仍低于 12 步目标。
  - 多鱼 eval (25×8)：`avg_final_survival=86.0% / min_final_survival=76% / avg_min_survival=86%`；death histogram `p10=48.9 / survival_fraction=86% / early_death_fraction_150=11.5%`，尾部显著长于 dev_v10。
  - 产出：TB `experiments/v11/artifacts/tb_logs/dev_v11_runA_phase_split/`（post_eval 含 min survival scalar）、日志 `experiments/v11/artifacts/logs/dev_v11_runA_phase_split.log`、ckpt `experiments/v11/artifacts/checkpoints/dev_v11_runA_phase_split/`、曲线 `experiments/v11/artifacts/plots/dev_v11_runA_phase_split_{survival,first_death,penalty_vs_first_death}.png`、death histogram `experiments/v11/artifacts/plots/dev_v11_runA_phase_split_death_histogram.png`、GIF `experiments/v11/artifacts/media/dev_v11_runA_phase_split_curve.gif`、多鱼视频 `experiments/v11/artifacts/media/dev_v11_runA_phase_split_multi_fish_eval.mp4`。

## 结果与观察
- phase-split ramp + `escape_boost_speed=0.75` 把 Stage3 first-death 均值推至 9.6 步，penalty vs first-death 图显示在迭代 16~18 ramp 前半段仍有 12~15 步的平台，但进入迭代 22 以后随 penalty=0.07~0.08 再度滑落至 7 步，需要额外的 entropy/boost 才能稳定维持。
- 多鱼表现显著改善：`avg_final_survival` 从 dev_v10_runB 的 70.5% → 86%，`min_final_survival` 也提升到 76%，death histogram 的 `p10=48.9` 说明长尾死亡集中在 50 步附近而非开局。
- 新增的 TensorBoard post_eval 指标 (`eval_multi/min_final_survival_rate` 等) 与 penalty-first_death plot 均已写入 artifacts，后续复盘可直接在 TB 中对比多轮结果。

## Learning & 下一步计划
1. **Penalty entropy cushion (Run B)** —— 在迭代 16~24 把 `ent_coef` 动态抬到 0.03，或让 `escape_boost_speed=0.8` 并仅在 penalty ramp 区间生效，评估是否能把阶段 3 的 `first_death_mean` 拉到 ≥12。
2. **Penalty gating vs iteration** —— 当前 ramp 仍是线性函数，计划在 `train.py` 中加入“迭代偏移”/plateau 控制（例如在阶段 3 前半段保持 0.02，阶段 4 再 ramp 0.02→0.08），并把该参数记入命令行，方便做 ablation。
3. **群体特征拓展准备** —— 若 Run B 仍无法超过 12 步，需要考虑在 `FishEscapeEnv` 中引入 neighbor variance/relative heading 特征，并在 dev_v11 文档中列出所需的 logging（例如 penalty value vs neighbor variance 的散点），为 dev_v12 做准备。
