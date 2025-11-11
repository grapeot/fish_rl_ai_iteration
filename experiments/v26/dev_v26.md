# dev_v26

## 启动
- 2025-11-11 11:05 PST 复盘 `SOP.md` 与 `experiments/v25/dev_v25.md`：v25 虽然补齐 `pre_roll_stats`、`step_one_clusters` 等 instrumentation，但 `success_p10>=140` 依旧导致 penalty stage 长期停在 0。
- v26 目标：让 penalty gate 真正触发阶段提升，同时把 tail 事件诊断与 escape boost 联动集成到主训练脚本，确保迭代可复现。
- 本轮所有代码、日志、媒体写入 `experiments/v26/`，训练并行度维持 ≥128 env。

## 观察
- Gate 阈值：v25 记录显示 multi-eval `p10` 在 60–100 徘徊，`success_p10=140` 过于激进，`failure_hold` 频繁触发，`density_penalty_coef` 从未超过 0.04。
- Tail 特征：`step_one_clusters` 暗示捕食者速度向量集中在 0°–60°，尽管 spawn 半径 jitter 接近上限，说明需对 pre-roll angle/speed 做非均匀扰动。
- Escape boost：`AdaptiveBoostController` 仍基于即时均值，当单次 probe 掉线时 clamp 直落 0.76，压缩 phase6 防守；需要引入 rolling median/阶段锁定以免回退。
- Logging：`pre_roll_stats.jsonl`、`penalty_stage_debug.jsonl`、`multi_eval_history` 已能定位问题，本轮需直接生成 tail polar plot 供肉眼确认。

## 实验计划
1. **Gate 松绑与滚动统计**：把 `success_p10` 降到 130，引入 rolling median（最近 3 次 probe 满足 `median(p10)>=130`）再加计成功；`failure_p10` 维持 110 但允许单次失败不立刻回滚，记录在 `penalty_stage_debug`。
2. **非对称 pre-roll 扰动**：在 env wrapper 中允许对 [0°, 90°] + [180°, 270°] 进行加权采样，扩展 `predator_spawn_jitter_radius=2.0`，并把实时分布写入 `pre_roll_stats`。
3. **Tail diagnostics 可视化**：复用 v25 聚类数据，自动生成 `step_one_heading_polar.png`（plots）与 worst seed 列表（logs/checkpoints），方便后续人工核查。
4. **主线 run `dev_v26_runA_gate_relax`**：基于 v25 主线超参，`--num_envs 128`、`--total_iterations 64`、`--multi_eval_interval 4`。若 gate 成功进入 stage1，则继续观察 penalty/boost 互动；如失败，再追加 `runB_tail_focus`。
5. **收尾**：整理命令、关键指标、artifact 路径到本文档，复制曲线/媒体到 `experiments/v26/artifacts/plots|media/`，并提交推送。

## 运行记录

### dev_v26_runA_gate_relax（2025-11-11 09:34–09:55 PST）
- **命令**：
  ```bash
  python experiments/v26/train.py \
    --run_name dev_v26_runA_gate_relax \
    --num_envs 128 --n_steps 256 --total_iterations 64 \
    --curriculum 15:12,20:14,25:8,25:8,25:10,25:6,25:6 \
    --multi_eval_interval 4 --multi_eval_probe_episodes 24 \
    --penalty_gate_success_p10 130 --penalty_gate_failure_p10 110 \
    --adaptive_boost_stage_floor 0:0.76,1:0.8,2:0.82 \
    --predator_spawn_jitter_radius 2.0 \
    --predator_heading_bias '0-90:0.8,180-270:1.4' \
    --seed 924
  ```
- **关键设定**：保持 `num_envs=128`、`n_steps=256` 满足 SOP；penalty gate 新增 rolling median（窗口 3）与 `failure_tolerance=2`；stage floor 约束 `escape_boost_speed≥0.76/0.8/0.82`（按 gate stage）；pre-roll 施加权重让 180°–210° 与 270°–300° 更常见，同时把 `predator_spawn_jitter_radius` 提到 2.0。
- **日志/产物**：主训练日志 `experiments/v26/artifacts/logs/dev_v26_runA_gate_relax.log`，TensorBoard `experiments/v26/artifacts/tb_logs/dev_v26_runA_gate_relax/`，checkpoint 与 JSON 位于 `experiments/v26/artifacts/checkpoints/dev_v26_runA_gate_relax/`；所有曲线在 `experiments/v26/artifacts/plots/`，媒体（loss 曲线 GIF、多鱼视频、best iter 视频）在 `experiments/v26/artifacts/media/`；step-one 极坐标图 `experiments/v26/artifacts/plots/dev_v26_runA_gate_relax_step_one_heading_polar.png`，worst seeds 摘要 `experiments/v26/artifacts/checkpoints/dev_v26_runA_gate_relax/step_one_worst_seeds.json`。
- **Deterministic multi-eval (25×48)**：`avg_final_survival_rate=0.739`、`min_final_survival_rate=0.44`，死亡分布 `p10=102`、`early_death_fraction_100=0.0925`（见 `eval_multi_fish.json` + `eval_multi_summary.json` + `death_stats.json`）。`step_one_death_count=8`，说明松绑 gate 后仍未触发 stage1（`density_penalty_coef` 全程卡在 0.04）。
- **Best checkpoint（iter 44）**：`avg_survival=0.765`、`p10=120.9`、`step_one=5`、`early_death_fraction_100=0.077`，对应数据 `multi_eval_best_iter_44.json` / `_summary.json`，复现视频 `media/dev_v26_runA_gate_relax_best_iter_44_multi.mp4`。虽然局部迭代达到 `p10>120`，rolling median 仍保持在 ≈107，gate 未解锁。
- **Gate/boost 观察**：`penalty_stage_debug.jsonl` 显示 stage=0 全程 `failure_hold`，median(p10) 峰值仅 111，`failure_streak` 压到 10 也只触发冻结不回退；自适应 boost 因 stage floor=0.76 再也不会跌破 0.76，但也无法升至 0.8 因为 gate 未晋级（阶段基线 0.82 被 adaptive clamp 迅速压回 0.76）。
- **Tail diagnostics**：`step_one_worst_seeds.json` 共采样 143 个 step=1 事件，前 3 个桶分别落在 180°–210°、120°–150°、270°–300°，占比 6.3%/6.3%/5.6%，说明加权 bias 成功把 0°–60° 热点拆散，但 NE 象限 (120°–150°) 仍保持 6% 以上份额；低速南向 (240°–270°、speed<0.4) 也出现 4.9% share，提示 spawn radius=2.0 让慢速捕食者贴近鱼群。Polar plot 与 JSON 现成可供后续定向回放；多鱼评估视频 `media/dev_v26_runA_gate_relax_multi_fish_eval.mp4` 捕捉了 tail 爆发场景。

## 本轮 Learning
- Rolling median + failure tolerance 抑制了伪晋级，但当前 `success_p10=130` 仍过高：12 次 multi-eval 的 median(p10) 在 95–111 区间摆动，即便 best iter 拉到 120.9 也不足以提升 stage，`density_penalty_coef` 因此持续锁定 0.04。
- `predator_heading_bias=0-90:0.8,180-270:1.4` + `spawn_jitter_radius=2.0` 把 step-one 聚类的峰值拉到 180°–210° 与 270°–300°，但 120°–150° 仍与 SW 象限同量级，说明需要更强的逆向采样或直接对 `predator_pre_roll_angle_jitter` 做非均匀分布；同时低速南向群（<0.4 m/s）开始贡献 5%+ 事件，意味着速度抖动也要伴随角度重权。
- `AdaptiveBoostController` 中的 stage floor 只有在 gate 晋级后才会释放到 0.8/0.82，现阶段与旧流程等价，仍会在探针失败后被 median→0.76 的逻辑压下；需要把 floor 绑定到“phase base”而非 gate stage，或在 stage=0 时也允许有限上调。
- 新的 tail polar/JSON 让最差 seed 可视化，但仍需要脚本把这些样本转成 deterministic replay（或放入 curriculum 的专门阶段），否则信息无法反馈到调参。

## 下一步计划
1. **再下调 gate 阈值或拆分阶段指标**：尝试 `success_p10=120` 并只在 median≥120 且 `early_death_fraction_100<0.09` 时解锁 stage1，同时把 `failure_p10` 改为 95，确保 `failure_streak` 真正被触发；必要时允许 stage1 只解锁到 `density_penalty_coef=0.05` 以免过冲。
2. **强化非对称 pre-roll**：基于 `step_one_worst_seeds.json` 再加一档偏置，如 (210°–300°):1.6、(30°–90°):0.4，并对 `predator_pre_roll_speed_jitter` 引入加权，使高角速度 (>1.2 m/s) 的出现概率提升 2×；同时把 `pre_roll_stats.jsonl` 做角度直方图，观察 bias 是否覆盖 210°–330°。
3. **Tail replay pipeline**：写一个简易脚本按 `step_one_worst_seeds` 中的 `predator_velocity`/iteration 触发 deterministic env reset，导出 mp4/GIF，供人工快速判定“需要调 angle 还是 speed”；这些 replay 也可用于离线 curriculum（先训练在最坏 heading，再恢复均匀分布）。
4. **Boost-floor 联动**：把 stage floor 解耦为“phase floor”，即阶段 N 的 escape boost 下限不低于 `phase_base_escape_boost_speed - 0.02`，并根据 multi-eval median 决定是否允许 adaptive clamp 触碰 floor；这样即便 gate 未解锁，phase5/6 也能保持 >0.78 的初速以抵抗 tail。
