# dev_v19

## 启动
- dev_v18 修复了 `first_death_p10` 与 AdaptiveBoostController，但即使 clamp 会把 `escape_boost_speed` 拉回 0.78，Phase6 高 penalty 段仍在 step=1 出现死亡峰，`first_death_p10` 长期=1。
- dev_v18 的 Learning 建议进一步降低 clamp 速度（≤0.76）、延长 Phase5 plateau（例如 `0,0,4,4,6,2,2`），并针对 step=1 的死亡记录空间位置信息，用于排查是否是初始重叠或捕食者冲撞。
- 多鱼探针与期末 deterministic eval 差异大（0.625 vs 0.472），提示 checkpoint 选择与评估集不足，需要更稳定的迭代指标（例如迭代 28 clamp 生效时的 `early_death_fraction_100`）。

## 观察
- Adaptive clamp 现在会重复发令，但 `escape_boost_speed` baseline 仍以 0.80 起跳，导致 clamp 在 Phase5/6 反复震荡却无法压住 step=1 的死亡尖峰。
- `early_death_fraction_100` 等新指标已写入 JSON/TensorBoard，可用来定位 ramp 崩溃，但缺乏空间位置信息；需要把 step=1 死亡的捕食者/鱼位置写到 `episode_info` 方便分析。
- Phase5 ramp 期间 penalty 只有 4 iter plateau，clamp 降速后依旧过快升到 0.08，使得 `death_stats.p10` 在 0.065 段迅速跌到个位。

## 实验计划
1. **更激进的 clamp**：把 `adaptive_boost_low_speed` 降到 0.76，维持 `high_speed=0.80`，滑窗=4 iter，`p10` 目标区间 12~18；确认 `escape_boost_speed` 在 Phase5/6 大部分时间锁在 0.76~0.78。
2. **延长 ramp plateau**：将 `density_penalty_phase_plateaus` 更新为 `0,0,4,4,6,2,2` 并保持 Phase5 penalty=0.065 至少 6 iter，减少 clamp 震荡后的立即升档。
3. **记录早逝位置信息**：在 `FishEscapeEnv` / 训练回调中抓取 step=1 死亡的捕食者索引与鱼坐标，写入 `schedule_trace.json` & `multi_eval_history`，为后续 spawn jitter 或 pre-roll 提供依据。
4. **强制 checkpoint 复核**：提高多鱼探针的 deterministic eval EP 数（≥10）并在迭代 28±2 时额外保存 checkpoint，确保 `early_death_fraction_100` 最低点的模型被长期保留。

## 运行记录

### dev_v19_runA_low_speed_plateau（2025-11-11 03:32–03:55 PST，未完成）
- **命令**：

  ```bash
  python experiments/v19/train.py \
    --run_name dev_v19_runA_low_speed_plateau \
    --num_envs 128 \
    --checkpoint_interval 2 \
    --multi_eval_interval 4 \
    --multi_eval_probe_episodes 12 \
    --eval_multi_episodes 12 \
    --adaptive_boost_low_speed 0.76 \
    --adaptive_boost_upper_p10 18.0 \
    --seed 219
  ```
- **状态**：CLI 超时在 post-eval 之前，`checkpoints/dev_v19_runA_low_speed_plateau/` 仅包含迭代级 checkpoint 与 `schedule_trace.pkl`，缺失 `death_stats` / `eval_multi_summary`；留作原始 log 参考，不计入 v19 结论。

### dev_v19_runA_low_speed_plateau_v2（2025-11-11 03:57–04:26 PST）
- **命令**：同上，只是 `--run_name dev_v19_runA_low_speed_plateau_v2`。开启 128 env + 512 step rollout，`checkpoint_interval=2` 保留了迭代 2–30 的快照。
- **关键指标**（12× deterministic 多鱼评估 @25 fish）：
  - `avg_final_survival_rate=0.603`, `min_final_survival_rate=0.48`（`checkpoints/.../eval_multi_summary.json`）。
  - `death_stats.p10=71.6`, `early_death_fraction_100=0.113`，`survival_fraction=0.62`（`.../death_stats.json`）。
  - `multi_eval` 滑窗中最佳迭代在 iter16：`avg_final_survival_rate≈0.593`，`early_death_fraction_100=0.127`；日志 `eval_multi_history.jsonl` 记录了每个迭代的 `step_one_death_count` 与位置样本。
  - `first_death_p10` 仍锁在 1（`schedule_trace.json`），说明 Phase5/6 仍存在 step=1 崩溃，需要额外 spawn jitter 或更长 pre-roll。
- **step=1 观测**：`eval_step_one_deaths.json` 捕获 30 条 step=1 碰撞，全部集中在捕食者起点 `(0.15, 0.005)` 周围，鱼坐标呈扇形围绕圆心；`multi_eval_history.jsonl` 中同样追加 `step_one_deaths` 列表，可直接喂入 notebook 做空间聚类。
- **产物路径**：
  - 日志 `experiments/v19/artifacts/logs/dev_v19_runA_low_speed_plateau_v2.log`，TensorBoard：`experiments/v19/artifacts/tb_logs/dev_v19_runA_low_speed_plateau_v2/`（`PPO_1` + `multi_eval_probes` + `post_eval`）。
  - Checkpoints/评估：`experiments/v19/artifacts/checkpoints/dev_v19_runA_low_speed_plateau_v2/`（`model_final.zip`、`model_iter_02~30.zip`、`eval_multi_fish.json`、`eval_multi_summary.json`、`death_stats.json`、`eval_step_one_deaths.json`、`eval_multi_history.jsonl`、`schedule_trace.json`）。
  - 图表：`experiments/v19/artifacts/plots/dev_v19_runA_low_speed_plateau_v2_*.png`（survival/first-death/penalty 对齐、多鱼 timeline、per-probe histogram）。
  - 媒体：`experiments/v19/artifacts/media/dev_v19_runA_low_speed_plateau_v2_curve.gif` 与 `.../dev_v19_runA_low_speed_plateau_v2_multi_fish_eval.mp4`。
- **观察**：
  - Adaptive clamp 在 Phase5/6 几乎每个迭代都把速度压回 0.76，但 `first_death_p10` 没有回升，暗示问题来自初始化（step=1 碰撞）而非逃逸速度；`step_one_deaths` 的位置簇证明大部分鱼仍贴着捕食者起点。
  - `early_death_fraction_100` 从 dev_v18 的 0.224 降到 0.113，Phase5 plateau 延长有效减缓 ramp；不过 deterministic eval 的 `min_final_survival_rate` 仍只有 0.48，说明 clamp 仍不足以在高 penalty 段维持 >0.6 的底线。
  - `multi_eval_history` 现可直接筛选迭代（iter16/20/24/28）来挑 checkpoint；`model_iter_28.zip` 对应 clamp 生效区，可用于下一轮更长 eval（≥20 EP）。

## Learning / 下一步
- **定位 step=1 崩溃根因**：利用 `eval_step_one_deaths.json` + `multi_eval_history` 的坐标，统计与捕食者的极坐标分布；若全部集中在半径 <0.5，可在 v20 中为初始帧添加 5–10 step pre-roll 或随机偏移大鱼初始位置。
- **Clamp + penalty 同步**：当前 clamp 在 Phase5 刚降速就被 ramp 到 0.08 的 penalty 抹平，可尝试在迭代 20~26 将 `density_penalty` 暂停在 0.065，并把 `adaptive_boost_low_speed` 再降到 0.74 只针对 Phase6。
- **评估对齐**：给 `model_iter_16/24/28` 单独跑 20 EP deterministic eval，并把 `early_death_fraction_100` 画到 `multi_eval_timeline.png` 里，帮助下一轮直接挑选“最稳 checkpoint”；如需更快迭代，可引入自动脚本在训练末尾对 `step_one_death_count` 最少的迭代额外保存 `model_iter_best.zip`。
