# dev_v5

## 启动
- 延续 dev_v4 目标：让共享策略在 25 条鱼场景下仍具稳定存活率，并补齐完整 artifacts（媒体、曲线、评估）。
- dev_v4 的单鱼训练可在 24 iteration 内维持 0.7~0.9 survival，但多鱼评估只剩 19~22 条鱼存活，缺少视频与更长评估统计，导致协同能力与失效模式仍不透明。
- 本轮以“多鱼训练 signal + 媒体观测”为主题：在训练阶段就轮询不同小鱼观测，并默认输出 mp4/更长评估，为 dev_v6 可直接复现。

## 观察
- dev_v4 artifacts 完整但 `SingleFishEnv` 仍只返回第一条鱼的观测，训练样本高度相关，无法覆盖群体多样性；多鱼评估只是在推理时堆叠策略。
- deterministic 单鱼评估存在长尾失败（5 条轨迹有 1 条 0 存活），说明当前 PPO 在高方差样本上依旧不稳，需要更均匀的观测采样与更频繁 checkpoint。
- 媒体缺口：目前 plots/GIF 只展示统计曲线，没有直接渲染；下一轮的调参/观察会因缺少视频证据而受阻。

## 实验计划
1. 复制 v4 训练脚本为 `experiments/v5/train.py`，加入“轮询/随机”观测采样模式（默认 round-robin），reward 改为全鱼平均数，让单策略在训练时就看到多鱼分布。
2. 训练配置延续：`--num_envs 128`、`--total_iterations 24`，但默认 `num_fish=25` 并把 eval episodes 扩至 ≥8，同时产出多鱼 mp4（来自 deterministic run）。
3. 训练完毕后，将命令与关键指标写入本文档，并把曲线 PNG、GIF、mp4 分别放入 `experiments/v5/artifacts/plots|media/`，形成 dev_v6 可复现的交接包。

## 运行记录
- **2025-11-10 22:34~22:57 (UTC-8)**：`source venv/bin/activate && python experiments/v5/train.py --total_iterations 24 --num_envs 128 --num_fish 25 --eval_episodes 8 --eval_multi_fish 25 --eval_multi_episodes 5 --run_name dev_v5_iter24_env128_rr --obs_sampling round_robin --video_num_fish 25 --video_max_steps 500 --video_fps 20`
  - 迭代日志：`experiments/v5/artifacts/logs/dev_v5_iter24_env128_rr.log`
  - Checkpoints & 统计：`experiments/v5/artifacts/checkpoints/dev_v5_iter24_env128_rr/`（含 `model_iter_5/10/15/20`、`model_final.zip`、`training_stats.pkl`、单/多鱼 JSON）
  - TensorBoard：`experiments/v5/artifacts/tb_logs/dev_v5_iter24_env128_rr/`
  - Plot：`experiments/v5/artifacts/plots/dev_v5_iter24_env128_rr_survival.png`
  - 曲线 GIF：`experiments/v5/artifacts/media/dev_v5_iter24_env128_rr_curve.gif`
  - 多鱼渲染：`experiments/v5/artifacts/media/dev_v5_iter24_env128_rr_multi_fish_eval.mp4`

## 结果与产出
- **训练趋势**：24 iteration 全部跑满 500 步，平均 survival rate 0.716；最佳迭代 (12) 达到 `sr=75.6%`，最终迭代 `sr=74.8% / final_alive=18.7 / avg_reward=263.9`，round-robin 采样使曲线波动较 dev_v4 更平滑但仍有锯齿。
- **单鱼 deterministic (8 ep)**：`final_num_alive` 范围 18~23，平均 19.9；全部 episode 跑满 500 步，reward 均值 ≈277。相比 dev_v4 的一次崩溃，长尾失败暂未再现。
- **多鱼 deterministic (25 fish × 5 ep)**：`final_num_alive` 范围 18~22，均值 20；最差 survival rate 0.72，最好 0.88。策略能拖到终局但仍有 ~20% 鱼在 500 步前被捕（集中在同一时段）。
- **媒体**：`dev_v5_iter24_env128_rr_multi_fish_eval.mp4` 展示全 25 鱼轨迹，便于人工观察拥挤区；`*_curve.gif` 与 `*_survival.png` 提供统计曲线。

## 学习 & 下一步计划
1. **提升协同多样性**：虽然 round-robin 采样覆盖了不同鱼位姿，但 PPO 仍对群体密度没有直接输入。下一轮可在观测中加入“邻近鱼数量/相对位置”或借助 attention pooling，以区分靠近捕食者 vs. 靠近边界的鱼。
2. **早期失败定位**：多鱼评估里 0.72 survival 的 episode 集中在前 150 步内快速减员，需要在 `FishEscapeEnv` 中记录每鱼死亡时间并输出 histogram，方便定向调参（例如加强离散减速动作）。
3. **调参与 curriculum**：当前 `learning_rate=3e-4` 下 survival 仍呈锯齿，可尝试 cosine LR/entropy decay 或 curriculum（先 10 fish 再 25 fish）以减小波动；同时在 `train.py` 中开放 `--num_fish_schedule` 以便自动递增。
