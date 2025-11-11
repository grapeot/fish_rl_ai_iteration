# dev_v29

## 启动
- 2025-11-11 14:17 PST 阅读 `SOP.md` 与 `experiments/v28/dev_v28.md`，确认 v28 的结论聚焦在 NE 高速捕食者热点、过严的 gate 成功判定、tail prewarm 缺乏可观测性以及 multi-eval 过密导致的超时。
- 本轮沿用 SOP：所有日志/曲线/媒体写入 `experiments/v29/artifacts/` 子目录，并即时在本文同步假设、实验命令与指标。

## 观察
1. **NE 高速热点尚未解除**：虽然 v28 将 90°–150° 方向的 spawn 占比从 ~37% 压到 ~23%，但 step-one cluster 的最高桶依旧是 90°–120°/speed>2.0 m/s（33% share），说明 prewarm 仍主要喂给最坏场景，策略缺乏覆盖面。
2. **Penalty gate 长期锁死**：`success_p10_median_window=3` 且需连续两次命中，导致即便出现 `p10=121 & early_death=0.086` 也被下一次 failure 抵消，density penalty 无法提升至 ≥0.05。
3. **Tail prewarm 不可观测**：目前只依赖 mp4 猜测 prewarm 是否生效，缺少 `prewarm_velocity_override` 的命中计数，难以判断 worst seeds 注入率。
4. **Multi-eval 频率拖慢训练**：128 env × 24 probe × interval=4 直接把 runtime 拉至 30 min，并屡次因 CLI 超时在 iteration ≈24 停止，没能完成 ≥60 iteration 基线。

## 实验计划
1. **加大 NE 约束与 tail coverage**：对 pre-roll heading/speed bias 做更激进的 NE 高频惩罚，必要时引入“拒绝采样”或在 tail prewarm 队列中加入控制比例的低速 NE 样本，目标是把 step-one cluster 的 NE 高速 share 压到 <20%。
2. **放宽 gate 成功窗口**：将 `success_p10_median_window` 从 3 改为 1，并允许单次满足 `p10≥95 & early_death<0.11` 就临时开放更高 density penalty stage，以验证 penalty 是否能继续演进。
3. **增加 prewarm 可观测性**：在环境或训练脚本里记录 `prewarm_velocity_override` 命中的 episode/比例，写入 JSON/TensorBoard，便于判断 tail seeds 覆盖度。
4. **削减训练内 multi-eval 占比**：先以 `--multi_eval_interval 8`（或关闭）跑满 64 iteration，再另行执行独立 evaluate，多余的可视化脚本放入 `experiments/v29/artifacts/plots|media`，确保 128 env baseline 能在 30 min 内完成。

## 运行记录

### dev_v29_runA_gate_relax（128env，n_steps=256，total_iterations=64，CLI 30 min 于 iter≈34 被动中断）
- **命令**
  ```bash
  python experiments/v29/train.py \
    --run_name dev_v29_runA_gate_relax --total_iterations 64 --num_envs 128 --n_steps 256 \
    --curriculum 15:12,20:14,25:8,25:8,25:10,25:6,25:6 --multi_eval_interval 8 \
    --multi_eval_probe_episodes 20 --multi_eval_seed_base 2025111101 \
    --predator_pre_roll_steps 18 \
    --predator_heading_bias '0-30:0.3,30-60:0.2,60-90:0.1,90-120:0.05,120-150:0.05,150-180:0.4,180-210:1.4,210-240:1.7,240-270:1.8,270-300:1.8,300-330:1.5,330-360:0.6' \
    --predator_pre_roll_speed_bias '0-1.0:1.8,1.0-1.6:1.3,1.6-2.4:0.7' \
    --density_penalty_phase_targets 0.0,0.02,0.04,0.05,0.052,0.055,0.058 \
    --density_penalty_lock_value 0.05 --density_penalty_lock_until 24 \
    --penalty_gate_phase_allowance 4,5,6,7 --penalty_gate_required_successes 1 \
    --penalty_gate_success_step_one 9 --penalty_gate_success_early_death 0.11 --penalty_gate_success_p10 95 \
    --penalty_gate_failure_p10 90 --penalty_gate_failure_tolerance 2 \
    --tail_seed_replay_path experiments/v27/artifacts/checkpoints/dev_v27_runC_phasefloor_96env/step_one_worst_seeds.json \
    --tail_seed_prewarm_iterations 6 --tail_replay_count 2 --eval_multi_fish 96 --eval_multi_episodes 48 --seed 2911
  ```
- **指标与观察**
  - 完成 34 iteration 后因 30 min 超时被杀进程；phase3（25 鱼）尚未收敛到新的 penalty stage。
  - 期中 multi-eval（迭代 8/16/24/32）`avg_final≈0.83→0.75→0.75→0.80`，`p10` 最多 74.9（iter8），`early_death_fraction_100` 最低 0.115；`step_one_death_count` 始终 ≥19，`penalty_stage_debug.jsonl` 只记录 `failure_hold`/`locked`，说明放宽后的 gate 仍卡在 stage0。
  - 新增的 `prewarm_override_ratio` 记录在 `schedule_trace.json`：迭代 2/4/6/8/10 的 ratio=1.0（256 或 128 条 episode 全部命中 tail seeds），从迭代 12 起降为 0，验证 tail 队列在 5 个迭代内被完全耗尽。
  - `dev_v29_runA_gate_relax_multi_eval_timeline.png` 与 `.../multi_eval_hist/*.png` 显示 90°–150° 高速区虽被压低到 ~25%，但 p10 未能走出 70 档，仍难以触发 gate。
- **产物**
  - 日志：`experiments/v29/artifacts/logs/dev_v29_runA_gate_relax.log`
  - TensorBoard：`experiments/v29/artifacts/tb_logs/dev_v29_runA_gate_relax`
  - Checkpoints/JSON：`experiments/v29/artifacts/checkpoints/dev_v29_runA_gate_relax/`
  - 曲线/直方图：`experiments/v29/artifacts/plots/dev_v29_runA_gate_relax_multi_eval_timeline.png`、`.../multi_eval_hist/`
  - Tail 媒体：`experiments/v29/artifacts/media/dev_v29_runA_gate_relax_iter032_tail_rank0_ep13.mp4` 等 8 个片段

### dev_v29_runB_gate_relax_mini（128env，n_steps=192，total_iterations=64，完成）
- **命令**
  ```bash
  python experiments/v29/train.py \
    --run_name dev_v29_runB_gate_relax_mini --total_iterations 64 --num_envs 128 --n_steps 192 \
    --curriculum 15:12,20:14,25:8,25:8,25:10,25:6,25:6 --multi_eval_interval 8 \
    --multi_eval_probe_episodes 16 --multi_eval_seed_base 2025111102 \
    --predator_pre_roll_steps 18 \
    --predator_heading_bias '0-30:0.3,30-60:0.2,60-90:0.1,90-120:0.05,120-150:0.05,150-180:0.4,180-210:1.4,210-240:1.7,240-270:1.8,270-300:1.8,300-330:1.5,330-360:0.6' \
    --predator_pre_roll_speed_bias '0-1.0:1.8,1.0-1.6:1.3,1.6-2.4:0.7' \
    --density_penalty_phase_targets 0.0,0.02,0.04,0.05,0.052,0.055,0.058 \
    --density_penalty_lock_value 0.05 --density_penalty_lock_until 24 \
    --penalty_gate_phase_allowance 4,5,6,7 --penalty_gate_required_successes 1 \
    --penalty_gate_success_step_one 9 --penalty_gate_success_early_death 0.11 --penalty_gate_success_p10 95 \
    --penalty_gate_failure_p10 90 --penalty_gate_failure_tolerance 2 \
    --tail_seed_replay_path experiments/v27/artifacts/checkpoints/dev_v27_runC_phasefloor_96env/step_one_worst_seeds.json \
    --tail_seed_prewarm_iterations 5 --tail_replay_count 2 --eval_multi_fish 96 --eval_multi_episodes 40 --seed 2912
  ```
- **指标与观察**
  - 运行时长 ~40 min；所有 7 phase 完成 64 iteration。multi-eval（迭代 8/16/24/32/40/48/56/64）显示：
    - iter8：`avg_final=0.726`、`p10=128`、`early_death=0.079`，但 `step_one=29` > gate 阈值 9，仍无法解锁 stage1。
    - iter48：`avg_final=0.680`、`p10=97`、`early_death=0.103`，满足重新放宽的 p10/early 条件，但 `step_one=30` 继续触发 `failure_hold`。
    - iter64：`avg_final=0.666`、`p10=72`、`early_death=0.137`、`step_one=19`，gate 仍在 stage0，density penalty 被锁在 phase3（0.04）。
  - 事后 deterministic 评估（40×96 鱼，`eval_multi_summary.json`）：`avg_final=0.685`、`min_final=0.531`、`p10=75`、`early_death_100=0.131`（`death_stats.json`）。
  - `schedule_trace.json` 里的新仪表证明 tail prewarm 只覆盖前 4 个迭代：iter3/6/8/11 `prewarm_override_ratio=1.0`（hits=256/128），从 iter13 起 hits=0，remaining 60 iteration 完全没有再注入 worst seeds。
  - `step_one_clusters.jsonl`（iter64）头号桶为 **90°–120° @ 1.2–1.6 m/s，share=21.0%**，比 v28 的 33% 下来一些但仍高于目标 20%；`dev_v29_runB_gate_relax_mini_step_one_heading_polar.png` 仍显示 NE 密集。
  - `penalty_stage_debug.jsonl` 中所有事件都是 `locked` / `failure_hold`（rolling p10 128→99→59→63→97→60→72），说明放宽 median 窗口=1 + 单次成功仍不足以越过 `step_one` 硬阈值。
  - `pre_roll_stats.jsonl` 现已包含 `prewarm_override_count/ratio` 字段，可快速确认 multi-eval 阶段是否仍在注入 tail seeds（迭代 64 的 summary 显示 ratio=0）。
  - 媒体侧新增 `dev_v29_runB_gate_relax_mini_curve.gif`（训练指标动画）以及 iter8~64 的 tail mp4，便于对比 NE 失败案例。
- **产物**
  - 日志：`experiments/v29/artifacts/logs/dev_v29_runB_gate_relax_mini.log`
  - TensorBoard：`experiments/v29/artifacts/tb_logs/dev_v29_runB_gate_relax_mini`
  - Checkpoints/JSON：`experiments/v29/artifacts/checkpoints/dev_v29_runB_gate_relax_mini/`（含 `schedule_trace.json`、`eval_multi_history.jsonl`、`penalty_stage_debug.jsonl` 等）
  - 可视化：`experiments/v29/artifacts/plots/dev_v29_runB_gate_relax_mini_multi_eval_timeline.png`、`..._survival.png`、`..._step_one_heading_polar.png`、`..._death_histogram.png`
  - 媒体：`experiments/v29/artifacts/media/dev_v29_runB_gate_relax_mini_curve.gif` 及 10 组 tail mp4（iter8~64）

## Learning / 下一步计划
1. **step-one 阈值仍是 gate 的瓶颈**：即使 `p10≥128`、`early_death≤0.08`，`step_one_death_count` 也在 19~34 区间，导致放宽后的 gate 仍 100% 时间停留在 stage0。需要在 dev_v30 中：
   - 将 `step_one` 判据改为 rolling median / 百分位或针对 NE 桶的专门 limiter，而不是固定 ≤9。
   - 额外对 `step_one` 记录做分桶统计，允许 gate 按“NE share <X%”而非“绝对死亡数”解锁。
2. **Tail prewarm 队列过短**：新的 instrumentation 表明 tail seeds 只在前 ~5 个 iteration 生效。下一轮需要 either (a) 让 `prewarm_predator_velocity_queue` 循环复用，或 (b) 扩展 worst seed 文件（增加低速 NE 覆盖）并放宽 `--tail_seed_prewarm_iterations`，否则 instrumentation 永远读到 0。
3. **NE 高频仍占 21%**：`step_one_clusters` 显示 90°–120°/1.2–1.6 m/s 仍是头号桶。下一步要实现真正的“拒绝采样”（根据角度+速度联合权重重抽）或在 `FishEscapeEnv` 里引入 per-bin clip，确保 NE share <20%，否则 p10 改善空间有限。
4. **多次 `failure_hold` 仍拖慢 penalty**：虽然把 `multi_eval_interval` 提升到 8，运行仍花 40 min。下轮需拆分“训练 run（只保存 checkpoint）+ 独立 eval 脚本”，并考虑把 multi-eval seed 数降到 12，在单独脚本里再做高方差验证，以避免训练进程被 timeout。
5. **新增信号已验证可用**：`prewarm_override_ratio` 与 `pre_roll_stats_summary` 均能记录命中次数，下一轮可以把该信号写入 TB 曲线并作为 checklist，强制每轮实验都确认 tail 覆盖是否足够。
