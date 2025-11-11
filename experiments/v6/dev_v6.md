# dev_v6

## 启动
- 延续 dev_v5 的方向：共享策略已经在 25 鱼环境下达到 ~0.75 survival，但缺少对局部密度的直接观测，导致协同行为仍靠随机采样来“猜”。
- dev_v5 学习列表提醒：需要把邻居/拥挤度特征纳入观测、记录每条鱼的死亡时间并可视化长尾失败，还要探索更平滑的学习率或简单的 curriculum，以减小 24 iteration 曲线的锯齿。
- 本轮 dev_v6 的核心是：在不破坏已有流水线的前提下，为策略增加“邻居感知 + 早期死亡诊断”的能力，并验证一个最小 curriculum（前半程少鱼 warm-up，后半程恢复 25 鱼）。

## 观察
- dev_v5 round-robin 采样让 25 鱼训练趋稳，但 `final_num_alive` 仍集中在 18~22，最差 episode 在前 150 步就折损 25% 的鱼，说明策略对局部拥挤和捕食者切入角缺乏针对性。
- 训练曲线仍呈锯齿，iteration 12 之后 survival 从 0.75 回落到 0.71，再反弹；缺少细粒度的死亡分布数据，很难判断是早期崩溃还是末段连锁反应。
- 媒体/plots 已齐，但没有“死亡时间直方图 / 失败片段”，无法快速定位问题阶段；同时单一学习率 3e-4 可能过大，缺乏 warm-up。

## 实验计划
1. **观测增强**：修改 `FishEscapeEnv` 支持可选的邻居特征（邻近鱼数量、平均相对位置、最近邻距离等），并在 `experiments/v6/train.py` 默认开启，保持观察维度自适应。
2. **指标记录**：在环境中跟踪每条鱼的死亡步数；训练回调和评估脚本读取该数据，生成 `death_histogram.png` 与文本摘要，写入 `experiments/v6/artifacts/plots/`。
3. **轻量 curriculum / LR 调整**：在训练脚本中加入两阶段流程（前 12 iteration 使用 15 鱼 + 稍大的 entropy，后 12 iteration 切回 25 鱼并启用余弦 LR decay），确保 `--num_envs >= 128`；所有 run 输出继续落在 `experiments/v6/artifacts/` 并记录命令、指标、媒体。

## 运行记录
- **2025-11-10 23:15~23:19 (UTC-8)**：`source venv/bin/activate && python experiments/v6/train.py --total_iterations 24 --num_envs 128 --curriculum "15:12,25:12" --eval_multi_episodes 8 --run_name dev_v6_curriculum_neighbors --seed 42`
  - Phase A（iteration 1~12, 15 fish warm-up）：平均 survival 0.70，best=0.704（iter11），`final_num_alive≈10.6`；Phase B（iteration 13~24, 25 fish）继续同一模型，best=0.737（iter17，final_alive 18.4），最终 iteration=24 时 `sr=0.720/final_alive=18.0`。
  - 训练日志：`experiments/v6/artifacts/logs/dev_v6_curriculum_neighbors.log`；TensorBoard：`experiments/v6/artifacts/tb_logs/dev_v6_curriculum_neighbors/`；Checkpoints & 评估：`experiments/v6/artifacts/checkpoints/dev_v6_curriculum_neighbors/`（含 `model_iter_5/10/15/20`, `model_final.zip`, `eval_*.json`, `death_stats.json`）。
  - 可视化：曲线图 `experiments/v6/artifacts/plots/dev_v6_curriculum_neighbors_survival.png`；死亡直方图 `experiments/v6/artifacts/plots/dev_v6_curriculum_neighbors_death_histogram.png`；GIF `experiments/v6/artifacts/media/dev_v6_curriculum_neighbors_curve.gif`；多鱼渲染 `experiments/v6/artifacts/media/dev_v6_curriculum_neighbors_multi_fish_eval.mp4`。
  - 代码：`fish_env.py` 新增邻居特征/死亡时间追踪；`experiments/v6/train.py` 引入 curriculum、cosine LR、死亡直方图导出与媒体管线，所有变更均记录于本次提交。

## 结果与产出
### 指标
- **训练曲线**：24 iteration 全部跑满 500 步；迭代 1~12（15 fish）把 sr 从 0.66 提至 0.70，随后切换 25 fish 后在 iteration 17 达到最高 `sr=73.7%/final_alive=18.4`，收敛时稳定在 `sr≈72%` 且无暴跌。
- **单鱼 deterministic (5 ep)**：`final_alive` ∈ [19,24]，均值 21.4；平均 reward 300.3，所有轨迹均跑满 500 步。
- **多鱼 deterministic (25 fish × 8 ep)**：`final_alive` 均值 19.6（Std 1.8），最差 episode 17 条（survival 0.68），最好 22 条（0.88）；平均总奖励 270.5。
- **死亡分布**：基于 200 条鱼样本，78.5% 存活至 500 步，仍有 21% 在 150 步内阵亡；p10≈3 步表明仍存在“开局即被咬”的群体。

### artifacts
- 日志：`experiments/v6/artifacts/logs/dev_v6_curriculum_neighbors.log`
- TensorBoard：`experiments/v6/artifacts/tb_logs/dev_v6_curriculum_neighbors/`
- Checkpoints & 评估：`experiments/v6/artifacts/checkpoints/dev_v6_curriculum_neighbors/`
- 曲线 / 死亡图：`experiments/v6/artifacts/plots/dev_v6_curriculum_neighbors_survival.png`，`..._death_histogram.png`
- 媒体：`experiments/v6/artifacts/media/dev_v6_curriculum_neighbors_curve.gif`，`..._multi_fish_eval.mp4`

## 学习 & 下一步计划
1. **早期死亡仍集中在前 20 步**：死亡直方图揭示 10% 鱼在 3 步内被捕，怀疑开局出生角度固定导致“贴脸刷新”。下一轮需要在环境里加入随机旋转/初始逃逸脉冲，或强制第一帧执行离散规避动作并记录其效果。
2. **邻居特征虽减小波动，但对拥挤区尚无差异化策略**：阶段二的 `final_alive` 波动 17~18 条，仍未突破 20+；应进一步引入“邻居速度散度/密度”或 attention pooling，并在 TensorBoard 中单独记录 `neighbor_density` 的分布。
3. **Curriculum 切换存在指标断层**：iteration 13 出现 sr 跳变，说明切换到 25 鱼时同一优化器仍需短暂再适应。可尝试在切换时降低学习率或热启动价值网络、甚至在训练脚本中加入 `--fish_schedule=[(15,8),(20,8),(25,8)]` 逐级增加，以平滑阶段过渡。
