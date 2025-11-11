# dev_v28

## 启动
- 2025-11-11 15:05 PST 阅读 `SOP.md` 与 `experiments/v27/dev_v27.md`，确认 v27 解决了 phase-floor 与 tail replay 管线，但未能推动 density gate 进位。
- 本轮继续沿用 SOP：所有日志/曲线/媒体写入 `experiments/v28/artifacts/`，并在 `dev_v28.md` 内实时同步假设/实验记录。

## 观察
1. **Gate 仍由 multi-eval p10 卡死**：v27 的 rolling median(p10) 只有 75~105，`early_death_fraction_100` 长期 0.11~0.13，导致 gate 常驻 stage0，density penalty 无法爬到 0.055+。
2. **非对称 pre-roll 未抑制 NE 热点**：iteration 60 的 polar histogram 显示 120°–150° 仍占 37.5%，tail replay 的 worst seeds 也集中在 90°–150° 高速/低速混合区间。
3. **Tail replay 管线成熟但未反哺 curriculum**：`--tail_replay_count` 能稳定生成 mp4/JSON，却尚未用于训练阶段的 seed 预热。
4. **128env run 受 CLI 超时限制**：v27 两次 128env 运行被 600–1200s 超时中断，96env 虽能完成 60 iter 但指标回落（avg_final <0.7）。

## 实验计划
1. **Pre-roll 重权迭代**：将 90°–150° 权重降至 ≤0.2，同时拆分 210°–330° 为 210°–270°:1.8 与 270°–330°:2.0，并继续提升 >1.2 m/s 的采样概率，目标是压低 NE step-one 占比。
2. **Gate 条件联动**：把晋级条件放宽为 “p10≥95 且 early_death_fraction_100<0.11”，同时限制 stage1 penalty 不超过 0.05，避免 gate 长期封锁导致 penalty 无法演进。
3. **Tail replay → curriculum 预热**：在训练前注入 worst seeds deterministic replay（例如前 2 次 multi-eval 的 top cluster），观察是否能提前抑制 NE 死亡。
4. **资源与超时策略**：恢复 `--num_envs 128`、`--total_iterations 64` 基线，并设置 CLI 超时 ≥1800s；若仍受限，则拆成两段 run 但保持汇报一致。

## 运行记录
### dev_v28_runA_bias_prewarm（128env，n_steps 256，CLI 30min 超时于 iter≈24）
- **命令**
  ```bash
  python experiments/v28/train.py \
    --run_name dev_v28_runA_bias_prewarm --num_envs 128 --n_steps 256 --total_iterations 64 \
    --curriculum 15:12,20:14,25:8,25:8,25:10,25:6,25:6 \
    --multi_eval_interval 4 --multi_eval_probe_episodes 24 --multi_eval_seed_base 20251112 \
    --predator_spawn_jitter_radius 2.0 --predator_pre_roll_steps 20 \
    --predator_pre_roll_angle_jitter 0.35 --predator_pre_roll_speed_jitter 0.25 \
    --predator_heading_bias '0-30:0.4,30-90:0.2,90-150:0.18,150-180:0.6,180-210:1.2,210-270:1.8,270-330:2.0,330-360:0.8' \
    --predator_pre_roll_speed_bias '0-1.2:1.0,1.2-2.4:2.0' \
    --density_penalty_phase_targets 0.0,0.02,0.04,0.05,0.052,0.055,0.057 \
    --density_penalty_lock_value 0.05 --density_penalty_lock_until 24 \
    --penalty_gate_success_step_one 9 --penalty_gate_success_early_death 0.11 \
    --penalty_gate_success_p10 95 --penalty_gate_failure_p10 90 \
    --adaptive_boost_stage_floor 0:0.78,1:0.8,2:0.82 \
    --tail_replay_count 2 --tail_seed_replay_path experiments/v27/artifacts/checkpoints/dev_v27_runC_phasefloor_96env/step_one_worst_seeds.json \
    --tail_seed_prewarm_iterations 5 --eval_multi_fish 96 --eval_multi_episodes 48 --seed 924
  ```
- **状态**：训练在 iteration 24 附近被 CLI 30 min 超时中断，但 callback 已记录 5 次 multi-eval（迭代 4/8/12/16/20），并保存 `model_iter_20.zip`。
- **关键指标**
  - Multi-eval（迭代 20，文件 `experiments/v28/artifacts/plots/dev_v28_runA_bias_prewarm_multi_eval_timeline.png`）：`avg_final=0.53`、`min_final=0.10`、`p10=67`、`early_death_fraction_100=0.177`，最好的窗口是迭代 4（`p10=121`、`early_death=0.086`），但 rolling median=3 的 gate 仍然判定 “success 仅 1 次”→ stage 保持 0。
  - 手工离线评估（`manual_eval_multi_summary.json`)：载入 `model_iter_20.zip` 后 24×96 multi-fish 得到 `avg_final=0.53`、`min_final=0.14`、`p10=64`、`early_death_100=0.169`（文件 `.../manual_eval_multi_death.json`）。
  - Pre-roll histogram（`pre_roll_stats.jsonl`）显示 90°–150° 区间占比降至 **23%**（v27 为 37.5%），但 tail cluster 仍以 **90°–120°、speed>2.0 m/s** 为第一桶（`step_one_clusters.jsonl`，share 33%）。
  - Penalty gate 一直处于 `failure_hold`，`penalty_stage_debug.jsonl` 记录 freeze_until 频繁刷新，说明新的 success 条件（p10+early death）虽能短暂触发，但无法连续两次满足。
- **产物**：
  - 日志 `experiments/v28/artifacts/logs/dev_v28_runA_bias_prewarm.log`
  - 手工评估 `experiments/v28/artifacts/checkpoints/dev_v28_runA_bias_prewarm/manual_eval_multi_summary.json`、`manual_eval_multi_death.json`
  - 多图 `experiments/v28/artifacts/plots/dev_v28_runA_bias_prewarm_multi_eval_timeline.png`
  - Tail 媒体 `experiments/v28/artifacts/media/dev_v28_runA_bias_prewarm_iter020_tail_rank0_ep9.mp4` 等 10 个片段

### dev_v28_runB_bias_prewarm_short（128env，n_steps 192，CLI 20 min 超时于 iter≈24）
- **命令**：同 runA 但 `--n_steps 192 --total_iterations 60 --multi_eval_probe_episodes 20 --tail_seed_prewarm_iterations 4 --seed 925`。
- **状态**：迭代 20 被 25 min 超时终止；multi-eval 仍覆盖 4/8/12/16/20。
- **关键指标**
  - Multi-eval（迭代 20）`avg_final=0.43`、`min_final=0.29`、`p10=61`、`early_death=0.193`。最好的窗口为迭代 8（`p10≈104`、`early_death≈0.094`），但因下一次 multi-eval 立刻恶化，gate 继续锁在 stage0。
  - 手工评估 `manual_eval_multi_summary.json`：`avg_final=0.38`、`min_final=0.24`、`p10=60`、`early_death_100=0.192`。
  - Step-one cluster（迭代 20）最高桶转移到 **60°–90°/1.2–1.6 m/s**（17%），NE 高速段降到 13% share，说明缩短 `n_steps` + prewarm 对 cluster 有轻度分散作用，但生存率显著下降。
- **产物**：同 runA，对应所有 JSON/plot/mp4 均写入 `.../dev_v28_runB_bias_prewarm_short/`。

### 其他尝试
- **dev_v28_runC_bias_prewarm_fast**：多次尝试以 96 env / 64 env / 32 iteration 方案跑完一整个 curriculum，但在 CLI 20 min 限制下仍反复超时，`eval_multi_history.jsonl` 因重复 run_name 混入多次记录，仅保留 tail mp4 与调参日志留作参考。
- **dev_v28_runD_bias_prewarm_mini**：尝试把 multi-eval 关闭、迭代数降到 20 以验证脚本收尾逻辑，但 `training_stats.pkl` 未能写入（待排查 VecMonitor buffer 初始化），运行结果未计入正式结论。

## Learning / 下一步计划
1. **NE 热点削弱仍不彻底**：pre-roll speed bias 把 90°–150° spawn share 从 v27 的 37.5% 降到 ~23%，但 step-one cluster 的头号桶仍是 90°–120°/speed>2 m/s（33% share）。需要进一步为 NE + 高速组合设单独的拒绝采样或在 prewarm 队列里增加“低速 NE”占比，以避免 prewarm 反而喂给策略最坏场景。
2. **Gate 成功条件过于苛刻**：虽然迭代 4 就出现 `p10=121 & early_death=0.086`，但 rolling median=3 + “连续 2 次”要求导致立即被 failure_hold 抵消。建议在 v29 里把 `success_p10_median_window` 改成 1，并允许单次命中时临时开放 phase4→phase5（即 density 0.05），否则 gate 永远停在 stage0。
3. **Tail prewarm 需要显式可观测性**：目前只能从 mp4 推测预热是否生效，应在 `FishEscapeEnv` 中记录 `prewarm_velocity_override` 的命中次数，并把计数写入日志/JSON，方便量化“worst seed 覆盖率”。
4. **多评估占用训练时间**：128env × 24 probe × 4 轮 multi-eval 直接把 CLI runtime 拉到 30 min。下一轮应（a）把 `--multi_eval_interval` 提升到 8 或 0，训练后再单独运行 `evaluate_multi_fish`，或（b）将训练拆成两段脚本，确保 128env baseline 能跑满 ≥60 iteration。
