# dev_v8

## 启动
- 承接 dev_v7 的 spawn boost 和三段 curriculum，虽然首 80 步死亡显著下降，但 iteration 17 切换阶段仍伴随 reward dip，提示 schedule/学习率还需更平滑。
- dev_v7 的邻居速度/散度观测已加入，但 `custom/final_num_alive` 仍压在 17 条上下，说明策略对新增特征的利用有限，需要更强的表示能力或直接引导“密度扩散”行为。
- 多鱼评估在 200~250 步依旧出现群体性崩溃，结合 death histogram 余弦状尾巴，判断需要额外的中段协作奖励来鼓励局部散开、维持正散度。

## 观察
- p10 死亡步数由 3→84 证明初始逃逸有效，但 150~250 步仍占 60% 的死亡事件，凸显中后期协同仍是瓶颈。
- iteration 17 附近 reward dip 与 `avg_num_alive` 突降一致，推测是 20→25 fish 切换造成的分布漂移；需要更细粒度 curriculum 或学习率 window。
- 新增的邻居速度统计在 TensorBoard 上波动大、缺少明显相关性，暗示策略网络可能未能吸收额外维度，或缺乏直接奖励推动其意义。

## 实验计划
1. **局部散度奖励 (Run A)**：在 `FishEscapeEnv` 内新增 `divergence_reward_coef` 与 `density_penalty_coef`，鼓励邻居速度向外、惩罚局部高密度；默认设置 `coef=0.15 / penalty=0.08 / target=0.45`，验证中段死亡是否下降。
2. **Warm Cosine LR + 分段 curriculum (Run A/B)**：启用 `warm_cosine` (前 20% 线性到 0.5×lr，再 cosine) 并缩短阶段为 `15:6,20:9,25:9`，减轻 iteration 17 震荡，同时保持 `num_envs=128` 满载。
3. **网络容量扩展 (Run B)**：把 policy/vf MLP 调整为 `[384, 384]` 并对比默认宽度，确认邻居特征是否需要更大容量；若 Run A 成功则在 Run B 进一步尝试更强 penalty。

## 运行记录
- 2025-11-10 23:42~23:51 (UTC-8)：`source venv/bin/activate && python experiments/v8/train.py --run_name dev_v8_divergence_warmcos --num_envs 128 --total_iterations 24 --curriculum 15:6,20:9,25:9 --eval_multi_episodes 8 --divergence_reward_coef 0.15 --density_penalty_coef 0.08 --density_target 0.45 --policy_hidden_sizes 384,384 --seed 72`
  - Warm cosine LR + 3 阶段 curriculum，128 并行环境，迭代 24。checkpoint/日志/TensorBoard 分别位于 `experiments/v8/artifacts/checkpoints/dev_v8_divergence_warmcos/`、`experiments/v8/artifacts/logs/dev_v8_divergence_warmcos.log`、`experiments/v8/artifacts/tb_logs/dev_v8_divergence_warmcos/`。

## 结果与产出
- 训练指标：最佳 iteration 13 达到 `sr=70.6% / final_alive=14.1`；阶段 3（迭代 16~24）平均 `final_alive=17.0±0.3`，iteration 24 收敛于 `sr=68.5% / final_alive=17.1`，未再出现 dev_v7 iteration 17 的 reward 掉崖。
- 多鱼评估（25 fish ×8 episode）：`final_alive` 均值 19.75（区间 18~22），`avg_reward=0.51`，比 dev_v7 记录的 12.6 大幅抬升，说明 divergence reward 把 200~250 步后的群体坍塌推迟。
- Death stats：79% 样本坚持到 500 步，`mean_death_step=417`，<150 步早死率 15.5%（略高于 v7 的 14%），p10 掉到 26 步，提示边缘个体在初期被 penalty 推得过散，需要更柔和的 density 策略。
- 单鱼 eval (25 fish ×5) 最终存活 14~19；多鱼/单鱼 JSON、checkpoint、`training_stats.pkl` 均保存在 `experiments/v8/artifacts/checkpoints/dev_v8_divergence_warmcos/`。
- 可视化：`experiments/v8/artifacts/plots/dev_v8_divergence_warmcos_survival.png`（曲线）与 `..._death_histogram.png`；媒体 `experiments/v8/artifacts/media/dev_v8_divergence_warmcos_curve.gif`、`.../dev_v8_divergence_warmcos_multi_fish_eval.mp4` 已更新，可直接嵌入汇报。

## 学习 & 下一步计划
1. **密度惩罚需要分阶段调度**：当前固定 `0.08` 在阶段 1 就生效，导致 p10=26 步；下一轮尝试在 15→20 fish 前线性 ramp（或只对迭代 ≥10 生效），并记录 `first_death_step` 直方图验证。
2. **散度奖励有效提高终局活鱼**：final_alive 均值 17→19.7，但单鱼 reward 方差增大，后续可做 Run B（更高 `divergence_reward_coef` + 软 density）与 baseline 384/256 对比，确认提升源自奖励而非网络容量。
3. **可观测性还需补充**：TB 中 `custom/avg_num_alive` 持续轻微震荡；计划在 train.py 中 dump per-phase early death 统计，并将多鱼 eval 的 `min_survival_rate` 写入 TensorBoard，方便后续快速定位崩溃区间。
