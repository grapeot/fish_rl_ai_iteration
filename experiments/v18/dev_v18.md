# dev_v18

## 启动
- dev_v17 的 runA（2025-11-11 PST）在 25 条鱼、128 并行环境下把 Phase6 penalty 拆成 0.065→0.08，并保持 `divergence_reward_coef=0.22`，终局多鱼存活率 84.8%，但 ramp 高段依旧脆弱。
- `schedule_trace.json` 中的 `first_death_p10` 始终为 1，导致 adaptive clamp 未触发；需要在本轮把统计链路修好，否则 Phase6 clamp 实验无法推进。
- 多鱼探针 histogram 显示 penalty≥0.065 时 <100 步的死亡尾部仍密集，需延长 Phase5/6 的缓冲窗口，并把探针指标补齐（`early_death_fraction_100` 等）。

## 观察
- Phase6 ramp 拆分虽然平滑了 survival 波动，但 `first_death_step` 长期卡在一位数，说明高 penalty 段仍过激；`first_death_p10` 的统计 bug 隐藏了真实分布，adaptive clamp 无法响应。
- `escape_boost_penalty_caps` 目前是静态 (0.05→0.8, 0.075→0.78)，即使 ramp 弹性更好，也没办法随 `death_stats.p10` 调整；需要验证 adaptive clamp 生效后是否要把 clamp 上限降到 ≤0.76。
- 多鱼探针（迭代 4→28）虽然写入 JSONL + plots，但缺失 `early_death_fraction_100/150`、per-iteration histogram 指标，定位崩溃仍需人工比对。

## 实验计划
1. 修复 `first_death_p10` 统计：直接从 rollout 终止帧读取 `info["death_timesteps"]`，缓存 per-episode first-death，并写入 TensorBoard + `schedule_trace.json`，确保 adaptive clamp 读取真实值。
2. 验证 adaptive clamp：在 Phase5/6 启用自适应窗口（滑窗=4 iter, 目标 p10=12~18），运行短程 sanity check，确认 `escape_boost_speed` 会被 clamp 到 ≤0.78，并在日志中记录触发点。
3. 延长 Phase5/6 ramp：将 penalty plateau 改为 `0,0,4,4,4,2,2`，保持 0.065 至少 4 iter，再升 0.08 2 iter，观察 `multi_eval_history` 是否仍掉到 `p10≈110` 以下。
4. 强化多鱼探针：把 `multi_eval_probe_episodes` 提升到 8，额外计算 `early_death_fraction_100/150`，输出到 plots 及 `multi_eval_history.jsonl`，并同步到 TensorBoard。

## 运行记录

### dev_v18_runA_adaptive_clamp（2025-11-11 03:29–03:37 PST）
- **命令**：

  ```bash
  python experiments/v18/train.py \
    --run_name dev_v18_runA_adaptive_clamp \
    --num_envs 128 \
    --divergence_reward_coef 0.22 \
    --density_penalty_phase_plateaus 0,0,4,4,4,2,2 \
    --escape_boost_penalty_caps 0.05:0.8,0.075:0.78 \
    --multi_eval_interval 4 \
    --multi_eval_probe_episodes 8 \
    --adaptive_boost_upper_p10 18.0 \
    --seed 214
  ```
- **关键指标**：期末 deterministic 多鱼评估（5 EP）`avg_final_survival_rate=0.536 / min=0.28`，`death_stats.p10=31.6`，`early_death_fraction_100=0.224`（`experiments/v18/artifacts/checkpoints/dev_v18_runA_adaptive_clamp/death_stats.json`）。多鱼探针(`eval_multi_history.jsonl`)记录到迭代 28 仍有 `avg_final_survival_rate≈0.58`，但 `death_stats.p10` 在 ramp 段快速跌至 <40。
- **日志与产物**：
  - 训练日志 `experiments/v18/artifacts/logs/dev_v18_runA_adaptive_clamp.log`，TensorBoard `experiments/v18/artifacts/tb_logs/dev_v18_runA_adaptive_clamp/`（含 `multi_eval_probes/` 标量）。
  - Checkpoint/trace：`experiments/v18/artifacts/checkpoints/dev_v18_runA_adaptive_clamp/`（`schedule_trace.json`, `training_stats.pkl`, `eval_multi_history.jsonl`, `model_iter_*.zip`）。
  - 曲线/直方图：`experiments/v18/artifacts/plots/dev_v18_runA_adaptive_clamp_*.png` 与 `.../dev_v18_runA_adaptive_clamp_multi_eval_hist/iter*.png`；视频/动图：`experiments/v18/artifacts/media/dev_v18_runA_adaptive_clamp_curve.gif`、`..._multi_fish_eval.mp4`。
- **观察**：`first_death_p10` 统计已恢复（`schedule_trace` 中 `first_death_sample_size`=100），真实值确实锁死在 1。由于 `AdaptiveBoostController` 在早期（base speed=0.75）就缓存了 `target=0.78`，后续阶段即便 `phase_base=0.82` 也不会再次发指令，导致 Phase5/6 clamp 仍固定在 0.8；log 中没有 `adaptive_boost iteration=*` 记录。高速段的 `early_death_fraction_100` 维持在 0.22 左右，验证 ramp 延长尚不足以稳定 Phase6。

### dev_v18_runB_adaptive_clamp（2025-11-11 03:41–03:50 PST）
- **命令**：与 runA 相同参数，唯一差别是修复了 `AdaptiveBoostController`（移除 `_last_command` 抑制），run 名 `dev_v18_runB_adaptive_clamp`。
- **关键指标**：
  - 训练期中多鱼探针（8 EP）在迭代 28 达到 `avg_final_survival_rate=0.625`，`death_stats.p10=88.9`，`early_death_fraction_100=0.14`，对应 `experiments/v18/artifacts/plots/dev_v18_runB_adaptive_clamp_multi_eval_timeline.png` 与 hist (`.../multi_eval_hist/dev_v18_runB_adaptive_clamp_iter028_death_hist.png`)。
  - 期末 deterministic 多鱼评估（5 EP）`avg_final_survival_rate=0.472 / min=0.40`，`death_stats.p10=8.6`，`early_death_fraction_100=0.208`（`experiments/v18/artifacts/checkpoints/dev_v18_runB_adaptive_clamp/death_stats.json`）。单鱼评估 `avg_reward=233.9`，`final_num_alive≈15.0`（`eval_single_fish.json`）。
- **日志与产物**：同 runA 的目录结构，对应 runB 前缀（log / checkpoints / plots / media / TB）。`experiments/v18/artifacts/logs/dev_v18_runB_adaptive_clamp.log` 中在 iter 22/27/29 出现 `adaptive_boost iteration=***`，表明 clamp 会在 `first_death_p10` 滑窗均值低于 12 时把 `escape_boost_speed` 从 0.8 拉回 0.78，并在下一 iter 重新应用（`escape_boost_apply iteration=023/026/028/030 speed=0.780`）。
- **观察**：
  - Adaptive clamp 修复生效：`schedule_trace.json` 显示 Phase5/6 `escape_boost_speed` 在 0.78/0.80 之间跳动，且 `first_death_sample_size` 持续=100。虽然训练期中 clamp 降低了 `early_death_fraction_100`（0.22→0.14），但只要进入 0.065 penalty 段就再次出现 step=1 的死亡峰，导致 `first_death_p10` 仍维持 1。
  - 期末 deterministic 5 EP 的存活率显著低于探针（0.472 vs 0.625），说明当前 eval 采样不足且对 `escape_boost_speed` 状态敏感；需要 either 选取最佳迭代 checkpoint（例如 iter28 clamp 生效时）或增大 eval EP 数。
  - 新增的 `death_stats.early_death_fraction_1xx` 已写入 JSON+TensorBoard，可据此快速定位高 penalty 崩溃点。

## Learning / 下一步
- **Clamp 策略**：保持 `AdaptiveBoostController` 的重复发令逻辑，但需要更激进的低速（尝试 `--adaptive_boost_low_speed=0.76`）以及更长的 Phase5 plateau（例如 `0,0,4,4,6,2,2`），确保在 penalty ≥0.065 时 clamp 常驻。
- **早逝定位**：在 `FishEscapeEnv` 中临时记录 step=1 死亡的空间位置/捕食者索引，写入 `episode_info`，以确认是初始重叠还是 predator 加速度；若是初始化问题，可在下一轮加入 spawn jitter 或 pre-roll。
- **评估与模型选择**：
  1. 在训练结束后对 clamp 最稳定的迭代（例如迭代 28，可通过降低 `--checkpoint_interval` 或额外 `model.save` 保留）运行 ≥10 EP deterministic eval，以验证多鱼探针与最终评估之间的落差。
  2. 把 `eval_multi_summary` 的 `early_death_fraction_100` 直接绘制到 `multi_eval_timeline`，并在 `dev_v19` 中根据该指标挑选 checkpoint。
