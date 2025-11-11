# dev_v4

## 启动
- 延续 dev_v3 的单鱼基线：128 并行 env + `SingleFishEnv` 训练能在 20 iteration 内把 deterministic survival 拉到 500 步，`final_num_alive≈0.75` 说明 rollout 仍有长尾失败。
- dev_v3 `下一步` 中的首要任务是把单鱼策略迁回多鱼场景，并补上 checkpoint 级别的 deterministic 评估 / 媒体；SOP 也要求每轮 artifacts（plots/media/logs）齐套并能复现。
- 本轮 dev_v4 以“验证共享策略能否驱动 25 条鱼协同”为主题，目标是让训练脚本自带多鱼评估、媒体导出与日志模板，方便后续 dev_v5 接着扩展。

## 观察
- dev_v3 虽然在单鱼模式下达到 350+ reward，但 `training_stats.pkl` 中的 `final_num_alive` 仍波动 0.7~0.9，且尚未有真实渲染视频可供肉眼审查策略。
- 缺乏多鱼评估脚本导致无法判断单鱼 policy 的迁移效果；评估 JSON 仅覆盖单鱼 deterministic run。
- artifacts 结构已经清晰（logs/tb_logs/plots/media），但 v4 目录尚无脚本与文档，需要先补充模板再跑实验。

## 实验计划
1. 复制 v3 训练脚本到 `experiments/v4/train.py`，把 run 名称/标题/输出目录改成 v4，并新增：
   - 多鱼评估函数（直接在 `FishEscapeEnv(num_fish=25)` 中让策略对每条鱼独立决策），结果写入 `eval_multi_fish.json`。
   - 自动生成曲线 PNG + GIF + 媒体目录占位，确保所有 artifacts 都位于 `experiments/v4/artifacts/`。
2. 激活 `venv`（若不存在则用 `uv venv venv` 创建）并用 `uv pip install -r requirements.txt` 确认依赖，之后运行 v4 训练：
   - 默认 `--num_envs=128 --total_iterations=24`，run_name `dev_v4_iter24_env128_multi_eval`。
   - 将日志、TensorBoard、checkpoints、plot、gif、multi_fish 评估结果分别写入对应子目录。
3. 训练完成后，将命令、关键指标（单鱼 sr/final_alive，多鱼 deterministic 最终存活）、输出路径、媒体位置记录到 `experiments/v4/dev_v4.md` 的运行记录/指标部分，并按 SOP 整理下一步计划。

## 运行记录
- **2025-11-11 10:34 (local)**：`source venv/bin/activate && python experiments/v4/train.py --total_iterations 24 --num_envs 128 --num_fish 1 --run_name dev_v4_iter24_env128_multi_eval --eval_multi_fish 25 --eval_multi_episodes 5 --eval_episodes 5`
  - 满足 SOP 并行度（128 env）与 24 rollout（共 1,572,864 steps）。
  - 日志：`experiments/v4/artifacts/logs/dev_v4_iter24_env128_multi_eval.log`；TensorBoard：`experiments/v4/artifacts/tb_logs/dev_v4_iter24_env128_multi_eval/`。
  - Checkpoints + stats：`experiments/v4/artifacts/checkpoints/dev_v4_iter24_env128_multi_eval/`（含 model_iter_5/10/15/20、final、training_stats.pkl、eval JSON）。
  - 可视化：Plot `experiments/v4/artifacts/plots/dev_v4_iter24_env128_multi_eval_survival.png`，GIF `experiments/v4/artifacts/media/dev_v4_iter24_env128_multi_eval_curve.gif`。

## 结果与产出
### 指标
- 单鱼训练序列：24 iteration 中最佳 `survival_rate=91%`（iteration 1），末迭代 `survival_rate=79% / final_num_alive=0.79 / reward≈271`，波动集中在 0.70~0.86 区间。
- Deterministic 单鱼评估（5 ep）：前 4 条轨迹稳定 500 步满血（reward 350~356），第 5 条早期失误直接死亡，暴露 rollout 长尾风险。
- 新增多鱼评估（25 鱼 × 5 ep）：最终存活均值 19.6 条，范围 [17,22]，最差 `min_survival_rate=0.68`，说明共享策略可支撑 75~88% 群体但仍有 17~25% 被早期吃掉。
- 平均 reward（多鱼）0.47~0.62，对应 per-fish 总回报 234~309；全部 episode 均跑满 500 步，证明策略能拖延时间但协同不足。

### artifacts
- 日志：`experiments/v4/artifacts/logs/dev_v4_iter24_env128_multi_eval.log`
- TensorBoard：`experiments/v4/artifacts/tb_logs/dev_v4_iter24_env128_multi_eval/`
- Checkpoints & 统计：`experiments/v4/artifacts/checkpoints/dev_v4_iter24_env128_multi_eval/`
- Plot：`experiments/v4/artifacts/plots/dev_v4_iter24_env128_multi_eval_survival.png`
- GIF：`experiments/v4/artifacts/media/dev_v4_iter24_env128_multi_eval_curve.gif`
- 单/多鱼评估：`.../eval_single_fish.json`、`.../eval_multi_fish.json`

## 学习 & 观察
- `SingleFishEnv` 底模在迭代 1 即达到 0.91 sr，后续迭代呈锯齿，说明 PPO 仍受高方差样本影响；需要引入更平滑的 LR/entropy schedule 或者加大 `checkpoint_interval` 便于早停对比。
- 多鱼评估显示：即使直接复制单鱼策略，19~22 条鱼仍能存活，但 3~8 条鱼会在 500 步前死亡；推测原因在于共享策略对局部拥挤/捕食者方向缺乏差异化响应，后续需让训练阶段就处理批量观测。
- 单鱼 deterministic run 的一次崩溃提示需要更强的评估统计（>=10 ep）以及失效样本可视化，避免凭单次成功曲线误判泛化能力。

## 下一步计划
1. 把 `SingleFishEnv` 扩展成“批量取样”模式：每个 Subproc env 在 step 内顺序遍历 base env 的多条鱼，让策略对多观测做决策，逐步逼近真正的多鱼协同训练。
2. 在训练结束后调用渲染脚本生成 `experiments/v4/artifacts/media/*.mp4` 真机画面，结合当前 GIF（纯曲线）供人工审查策略动作。
3. 增强评估脚本：对单鱼/多鱼各跑 ≥10 episode，输出 `final_num_alive` 分布及失败案例的动作序列，为 dev_v5 调参提供数据驱动依据。
