# dev_v27

## 启动
- 2025-11-11 13:20 PST 复盘 `SOP.md` 与 `experiments/v26/dev_v26.md`，确认 v26 虽完成 gate 松绑与 tail polar logging，但 penalty gate 仍未晋级、boost floor 被锁死在 stage0。
- 本轮目标：在维持 ≥128 并行度的前提下，下调 gate 阈值并加入 phase-floor 解耦、非对称 pre-roll + 速度加权、tail worst-seed replay，验证是否能推动 `density_penalty_coef` 晋级并改善 `p10`/`early_death_fraction`。
- 所有代码/日志/媒体写入 `experiments/v27/`；训练脚本复制自 v26，并保留 v26 artifacts 只读。

## 观察
- **Gate 卡死**：v26 的 rolling median(p10) 长期停留在 95–111，`success_p10=130` 过高导致 `density_penalty_coef` 恒定 0.04；`failure_p10=110` 也让 `failure_hold` 不停触发。
- **Pre-roll 热点迁移不彻底**：加权 bias 虽把峰值移到 180°–300°，但 120°–150° 仍保持 >6% share，低速南向 (<0.4 m/s) 贡献 5% step-one 事件。
- **Boost floor 绑 gate**：`AdaptiveBoostController` 的 stage floor 只有 gate 晋级才可抬升，等于 stage0 恒定 0.76，无法抵御 tail。
- **Tail debug 未回流训练**：虽然 polar 图/JSON 现成，但还需自动化 worst seed replay 以复现并导出媒体。

## 实验计划
1. **Gate 阈值分级**：把 `success_p10` 降到 120，引入 `early_death_fraction_100<0.09` 作为并发条件，`failure_p10` 调到 95 并允许 `failure_streak<=1` 不降级；若晋级成功，只把 `density_penalty_coef` 拉到 0.05，避免过冲。
2. **Pre-roll 扩散 + 速度加权**：增加 210°–330° 权重 (1.6) 与 30°–90° 抑制 (0.4)，同时把 `predator_pre_roll_speed_jitter` 分段采样，让 >1.2 m/s 概率翻倍，并在日志中输出角度/速度直方图。
3. **Boost phase floor**：将 stage-floor 改为 phase-floor（基于 curriculum 阶段），即使 gate 未晋级也能逐步提高 escape boost 70→78→80。
4. **Tail worst-seed replay**：给训练脚本添加 `--tail_replay_count`，自动读取当次 multi-eval 最差 heading 生成 deterministic 回放 (mp4) 存到 `artifacts/media/`，用于观察角度/速度效果。
5. **主线 run（dev_v27_runA_gate_phasefloor）**：使用 `--num_envs 128`、`--total_iterations 64`，重点验证 gate 晋级与 phase floor 输出；若失败再开 runB。

## 运行记录
### dev_v27_runA_gate_phasefloor（128env，尝试，终止于 iter32）
- **命令**：`python experiments/v27/train.py --run_name dev_v27_runA_gate_phasefloor --num_envs 128 --n_steps 256 --total_iterations 64 --curriculum 15:12,20:14,25:8,25:8,25:10,25:6,25:6 --multi_eval_interval 4 --multi_eval_probe_episodes 24 --tail_replay_count 2 --predator_spawn_jitter_radius 2.0 --predator_pre_roll_angle_jitter 0.35 --predator_pre_roll_speed_jitter 0.25 --predator_heading_bias '0-90:0.4,180-210:1.0,210-330:1.6,330-360:0.9' --density_penalty_phase_targets 0.0,0.02,0.04,0.05,0.055,0.06,0.06 --adaptive_boost_stage_floor 0:0.76,1:0.8,2:0.82 --seed 924`。
- **结果**：CLI 600 s 超时杀掉训练，日志停在 iteration 32。已把所有产物移动到 `experiments/v27/artifacts/{logs,tb_logs,checkpoints,plots,media}/dev_v27_runA_gate_phasefloor_attempt1*` 方便复盘（包含前 7 次多鱼 hist + 14 个 tail mp4）。
- **诊断**：rolling median(p10) ≈107，gate 一直 `failure_hold`；phase-floor 成功把 escape boost 固定在 0.78，但阶段推进未发生。

### dev_v27_runB_phasefloor_short（128env，n_steps 192，尝试，终止于 iter56）
- **命令**：同 runA 但将 `--n_steps 192 --total_iterations 60`。
- **结果**：延长超时后依旧在 iteration 56 被 CLI 1200 s 限制终止，产物归档到 `experiments/v27/artifacts/*/dev_v27_runA_gate_phasefloor_attempt2*` 与 `.../media/dev_v27_runB_phasefloor_short_iter***.mp4`。
- **观察**：phase-floor 解耦与 tail replay pipeline 运行正常，多鱼探针 (24 epi) 仍显示 `p10≈92`、`early_death_fraction_100≈0.11`，density gate 维持 stage0。

### dev_v27_runC_phasefloor_96env（最终）
- **命令**：`python experiments/v27/train.py --run_name dev_v27_runC_phasefloor_96env --num_envs 96 --n_steps 192 --total_iterations 60 --curriculum 15:12,20:14,25:8,25:8,25:10,25:4,25:4 --multi_eval_interval 4 --multi_eval_probe_episodes 20 --multi_eval_seed_base 20251111 --tail_replay_count 2 --density_penalty_phase_targets 0.0,0.02,0.04,0.05,0.055,0.06,0.06 --predator_spawn_jitter_radius 2.0 --predator_pre_roll_angle_jitter 0.35 --predator_pre_roll_speed_jitter 0.25 --predator_heading_bias '0-90:0.4,180-210:1.0,210-330:1.6,330-360:0.9' --adaptive_boost_stage_floor 0:0.76,1:0.8,2:0.82 --seed 924`（并记录两次 timeout 后降低并行度的原因）。
- **关键指标**：训练 60 iteration 完成，gate 仍停在 stage0（rolling median(p10) 峰值 79.3、`early_death_fraction_100≈0.121`）。最终 deterministic 多鱼评估（48 epi）`avg_final_survival=0.671`、`min_final=0.36`、`p10=75.9`、`early_death_fraction_100=0.121`，单鱼评估与 v26 相当。`density_penalty_coef` 在 phase5 被限于 0.05，phase-floor 让 escape boost 在 iter43 以后稳定于 0.80。
- **产物**：
  - 日志 `experiments/v27/artifacts/logs/dev_v27_runC_phasefloor_96env.log`、TensorBoard `experiments/v27/artifacts/tb_logs/dev_v27_runC_phasefloor_96env/`、checkpoints+JSON `experiments/v27/artifacts/checkpoints/dev_v27_runC_phasefloor_96env/`。
  - 曲线/对比图：`survival/first_death/penalty_vs_first_death/penalty_vs_entropy` PNG 及多鱼时间线 `experiments/v27/artifacts/plots/dev_v27_runC_phasefloor_96env_multi_eval_timeline.png`、death hist 目录 `..._multi_eval_hist/`。
  - Tail diagnostics：`experiments/v27/artifacts/plots/dev_v27_runC_phasefloor_96env_step_one_heading_polar.png` 与 `.../checkpoints/.../step_one_worst_seeds.json`（156 个事件，300°–330° 低速与 120°–150° 高速仍是前两大桶）。
  - 自动 tail replay mp4：`experiments/v27/artifacts/media/dev_v27_runC_phasefloor_96env_iter0xx_tail_rank{0,1}_ep*.mp4`（覆盖 4≤iter≤60 的 15 轮探针 + 收敛 GIF `.../dev_v27_runC_phasefloor_96env_curve.gif`）。
- **现象**：pre-roll histogram（iteration 60 记录）显示 120°–150° bin 仍占 37.5%（3/8 样本），210°–330° 权重未完全压制 NE 象限；tail replay 清晰展示大鱼以 120°–150° 中高速切入导致 step-one cluster 仍集中在该区段。Phase-floor 保证 escape boost 不再跌破 0.8，但由于 gate card 0，density penalty 未能进到 0.055+。
-- **问题**：即便把 `failure_p10` 降到 95，rolling median 依旧在 75–90 徘徊；`early_death_fraction_100` 在 0.12 左右，无法满足 `success_early_death<0.09`，说明“放大北东方向的捕食权重”把 tail 问题迁移至 NE 但未整体降低 step-one 数量。

## 本轮 Learning
1. **Phase-floor 能保住 escape boost，但 gate 仍完全由 multi-eval p10 卡死**：phase5 之后 escape boost 始终维持 ≥0.80（见 log iter>=43），但 multi-eval `p10` 最高仅 105（iteration 4），rolling median 甚至跌到 75，`early_death_fraction_100` 常驻 0.11~0.13 → penalty gate 没有一次 `advance` 事件。
2. **非对称 pre-roll 仍留有 NE 热点**：`pre_roll_stats` 新增的角度直方图显示 iteration 60 仍有 3/8 样本集中在 120°–150°；`step_one_worst_seeds.json` 里 90°–150° + 150°–180° 合计 15% share，说明需要进一步拉低 NE 权重或在采样阶段限制该范围的低速事件。
3. **Tail replay pipeline 可用**：`--tail_replay_count 2` 在所有多鱼探针生成 deterministic mp4，并自动写入 JSON（种子、episode），定位问题速度明显提升，可在下一轮直接复用这些种子做 curriculum。
4. **128 env → 96 env 的折衷**：两次 128 env run 在 CLI 10-20min 限制下无法完成 64/60 iteration。切换到 96 env + 20 probe episodes 后可在 ~15 min 内完成 60 iteration，但样本效率下降（avg_final 仍 <0.7），后续需要恢复 ≥128 env 并延长超时时间，或把训练拆成两段 run 以保持对比的一致性。

## 下一步计划
1. **Pre-roll 重权再迭代**：把 90°–150° 的权重降到 ≤0.2，同时把 210°–330° 分成两档（210°–270°:1.8、270°–330°:2.0），并为 `predator_pre_roll_speed_jitter` 引入分段采样（>1.2 m/s 权重 ×2），以削减 NE step-one 份额。
2. **Gate 条件联动**：尝试把 `success_p10` 与 `early_death_fraction` 绑成“p10≥95 且 early_death<0.11 即可试探 stage1”，并把 stage1 penalty 上限限制在 0.05，防止 gate 永久锁死；必要时改成“连续 2 次达标 + median≤0.11”即可推进。
3. **Tail replay → curriculum**：基于 `step_one_worst_seeds` 中 top clusters，写脚本在训练前插入 5 iteration 的 deterministic seed replay（或在 `FishEscapeEnv` 中硬编码该 seed 序列），验证能否把 NE 象限死亡率降下来；同一脚本也可批量导出 GIF，减轻人工筛查成本。
4. **资源/timeout 策略**：在 CLI 层将超时拉到 ≥1800 s 或拆分训练脚本（phase1-3 / phase4-7 两段），以便恢复 128 env + 64 iteration baseline，确保下一轮对比不会因资源折衷引入额外变量。
