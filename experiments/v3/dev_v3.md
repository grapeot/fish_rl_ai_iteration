# dev_v3

## 启动
- 延续 dev_v2 的基线：100 iteration、32 并行环境的 VectorizedFishEnv 在 `final_num_alive≈17/25`、存活率 69% 左右停滞，Reward 缩放 (`REWARD_SCALE=0.1`) 虽平滑了 ADV，但未触及动作共享导致的 credit assignment 问题。
- v2 已将 `num_alive/avg_num_alive/final_num_alive` 分离并写入日志，`CheckpointCallback` 也会把三类指标及 `training_stats.pkl` 持久化，为本轮复用与追加监控提供了数据通路。
- 本轮目标：把 SOP 中“单鱼 env”与“TensorBoard 实测”真正落地，输出可复现的 dev_v3 文档 + 训练脚本 + artifacts，使下一轮能直接基于 per-agent POV 继续优化。

## 观察
- 日志显示 survival 与 `final_num_alive` 曲线高度一致，说明绝大多数死亡发生在尾段；共享动作导致局部逃跑信号被平均化，`VecMonitor` 中的 `avg_num_alive` 无法揭示个体差异。
- TensorBoard 尚未开启验证，`entropy` 是否坍缩不明；log 里虽然能看到 `custom/avg_survival_rate` 等指标，但缺少实际图形和截图。
- 评估脚本缺失，无法给某个 checkpoint 做 deterministic 对比，导致每轮“最终指标”可能依赖单次随机波动。

## 实验计划
1. **训练脚本 v3**：复制 v2 逻辑，改为 SingleFish 配置（每个 SubprocVecEnv 只控制 1 条鱼，动作-奖励一一对应），默认 `--num_envs=128`，附带 TensorBoard logdir（v3 专属）与统计导出。
2. **可观测性**：训练结束后自动输出 `training_stats.pkl`、Matplotlib 曲线（PNG）与 GIF（media 目录），满足“plots/media”留档要求，同时验证 TensorBoard 目录是否有数据。
3. **实验运行**：在 128 并行环境下跑 ≥15 iteration（根据时间窗口选择 20 iteration），记录命令、关键指标（survival/final_alive/reward）与 artifacts 路径，并初步评估 deterministic survival 以确认 per-agent 设置是否改善信噪比。
4. **文档更新**：把观察、指标和下一步假设写回本文件，包括：是否需要进一步增加观测维度、是否需要专门的评估脚本或更长训练。

## 运行记录
- **2025-11-10 22:29 (run=dev_v3_iter20_env64_r3)**：`source venv/bin/activate && python experiments/v3/train.py --total_iterations 20 --num_envs 64 --num_fish 1 --run_name dev_v3_iter20_env64_r3`；日志 `experiments/v3/artifacts/logs/dev_v3_iter20_env64_r3.log`，TensorBoard `experiments/v3/artifacts/tb_logs/dev_v3_iter20_env64_r3`。训练 20 rollout（每个 rollout=64 env × 512 steps）共 655,360 steps，迭代耗时 ~62s；`training_stats.pkl` + checkpoints 位于 `experiments/v3/artifacts/checkpoints/dev_v3_iter20_env64_r3/`。
- 自动生成的可视化：曲线 `experiments/v3/artifacts/plots/dev_v3_iter20_env64_r3_survival.png`，动图 `experiments/v3/artifacts/media/dev_v3_iter20_env64_r3_curve.gif`，供下轮直接引用进报告或 PPT。
- deterministic 评估：`experiments/v3/artifacts/checkpoints/dev_v3_iter20_env64_r3/eval_results.json`，5 个 episode 全部 500 步存活，奖励 351~364，表明单鱼设定下策略已学会保持最大生存时间。

## 指标总结
- 训练日志：最优迭代（#11）`survival_rate=88% / final_num_alive=0.88 / avg_reward≈304 / steps=446`，最终迭代（#20）回落至 `survival_rate=73%`，显示在高熵阶段波动仍然明显。
- 单鱼 PPO 的 deterministic run 可稳定满血通关，reward 提升到 350+；但 `training_stats.pkl` 中的平均 `final_num_alive` 仍围绕 0.75±0.05，说明 rollout 之间存在长尾 episode（被捕食者早期击杀）。
- `TensorBoard` 成功记录 `custom/avg_survival_rate`、`policy_loss`、`entropy_loss` 等指标；打开 `tensorboard --logdir experiments/v3/artifacts/tb_logs/dev_v3_iter20_env64_r3` 可看到熵在 0.6~0.8 区间，未出现彻底坍缩。

## 下一步
- [ ] 将 `SingleFishEnv` 结果推广回多鱼场景：需要批量观测 -> 批量动作的自定义 VecEnv，验证单鱼策略能否迁移至 25 条鱼共享策略。
- [ ] 扩展评估脚本：支持读取 checkpoint，批量运行 `num_fish>1` 配置并输出 `final_num_alive` 直方图，补足“多人协同”指标。
- [ ] 在 `media/` 目录补充真实渲染视频：利用 `FishEscapeEnv(render_mode="rgb_array")` + `imageio` 录制 deterministic episode，给下一轮肉眼观测策略行为。
- [ ] 如需进一步训练，可把 `total_iterations` 提升至 50，并加入余弦学习率/entropy schedule，缓解 20 iteration 后指标回落问题。
