# dev_v42

## 启动
- 2025-11-12 (UTC-8) 参照 SOP 回顾 dev_v41，总结 gate 长期停在 stage0 的原因，并盘点 tail reinjection 与 warning 的数据。
- dev_v41 的 Learning 要点：
  - main 段 `first_death_p10` 已稳定 >200，但 `step_one_death_count` 固定阈值=18 导致 success 永远不成立，success streak=0。
  - `tail_ne_reinjections` 仅 1 次且 warning 持续 14 iter，说明新的 stage spec 仍无法在 main/NE 交界提供足够缓冲。
  - iter40 因 `min_final<0.5` 触发 failure_hold，暴露 main-tail 缓冲样本仍过“硬”，最差样本波动大。

## 观察
1. **Step-one success 阈值卡住 gate**：当前 gate 要求 `step_one<=18` + `ratio<=0.02`，但长程训练中 step-one 常在 38~50，导致所有 main 阶段 success 判定被否决，stage-aware `first_death_p10`/early-death 的改进无法体现。
2. **NE reinjection 不足**：tail schedule 在 iter47 之后耗尽 NE block，warning 持续 14 iter，仅记录到 1 次 reinjection，main 慢速缓冲依旧缺口。
3. **最差样本波动**：iter40 multi-eval `min_final=0.49` 触发 failure_hold，说明目前 tail 缓冲仍会在 gate 刚要晋级时引入 harder main checkpoint，需插入“中速 main”或动态权重。

## 实验计划
1. **Stage-aware step-one success**：在 v42 训练脚本引入 `--penalty_gate_success_step_one_by_stage`（main >=40）并沿用 ratio<=0.02，使 main 段 success 不被绝对阈值阻塞。
2. **强化 NE reinjection**：扩充 tail stage spec，在 main/NE 交界插入额外 v34/v33 NE block，并提升 `tail_ne_reinjections` 目标≥2；监控 warning 时长压到 <5 iter。
3. **缓冲最差样本**：在 main 阶段加入“中速 main” checkpoint（例如 v32_stage2 降速版），必要时添加 tail queue 权重衰减；重点观察 iter32~48 multi-eval 的 `min_final` 与 `failure_hold` 触发频次。

## 运行记录

### dev_v42_stage_buffer_v1（128 env × 24 iter，CLI 超时终止）
- **命令**：
  ```bash
  python experiments/v42/train.py --run_name dev_v42_stage_buffer_v1 --total_iterations 60 --num_envs 128 --n_steps 128 --batch_size 1024 --n_epochs 5 --learning_rate 2.5e-4 \
    --curriculum 15:9,20:10,25:9,25:8,25:8,25:8,25:8 --multi_eval_interval 8 --multi_eval_probe_episodes 20 \
    --eval_multi_fish 96 --eval_multi_episodes 40 --multi_eval_seed_base 411232 --penalty_gate_phase_allowance 4,5,6,7 \
    --penalty_gate_required_successes 2 --penalty_gate_freeze_iterations 5 --penalty_gate_success_step_one 18 \
    --penalty_gate_success_step_one_ratio 0.02 --penalty_gate_success_step_one_by_stage main:45 \
    --penalty_gate_success_early_death 0.10 --penalty_gate_success_early_death_window 3 \
    --penalty_gate_success_early_death_by_stage ne:0.13,main:0.10 --penalty_gate_success_early_death_window_by_stage main:4 \
    --penalty_gate_success_p10 70 --penalty_gate_success_first_death_p10_by_stage main:100 \
    --penalty_gate_success_freeze_iterations 2 --penalty_gate_success_median_window 4 \
    --penalty_gate_success_p10_median_window_by_stage main:3 --penalty_gate_failure_p10 95 \
    --penalty_gate_failure_min_final 0.5 --penalty_gate_failure_tolerance 2 --penalty_gate_failure_tolerance_by_stage main:3 \
    --tail_stage_warn_ratio 0.70 --tail_stage_warn_patience 2 --tail_seed_stage_spec "10:experiments/v34/artifacts/checkpoints/tail_seed_mix_v34_ne.json|type=ne|label=ne_warmup;8:experiments/v34/artifacts/checkpoints/tail_seed_mix_v34_ne.json|type=ne|label=ne_boost;16:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage1.json|type=main|label=main_stage1a;12:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage1.json|type=main|label=main_stage1b;16:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage2.json|type=main|label=main_midbuffer_a;12:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage2.json|type=main|label=main_midbuffer_b;10:experiments/v34/artifacts/checkpoints/tail_seed_mix_v34_ne.json|type=ne|label=ne_refresh_a;8:experiments/v33/artifacts/checkpoints/tail_seed_mix_v33_stage3.json|type=main|label=main_stage3a;8:experiments/v34/artifacts/checkpoints/tail_seed_mix_v34_ne.json|type=ne|label=ne_refresh_b;10:experiments/v33/artifacts/checkpoints/tail_seed_mix_v33_stage3.json|type=main|label=main_stage3b;8:experiments/v34/artifacts/checkpoints/tail_seed_mix_v34_ne.json|type=ne|label=ne_tail" \
    --tail_force_reset_steps 224 --tail_replay_count 2 --video_num_fish 48 --skip_final_eval --seed 421742
  ```
- **结果**：运行至 iter24 被 CLI 20 min 超时打断，日志/plots 仍保留（`experiments/v42/artifacts/logs/dev_v42_stage_buffer_v1.log`）。该 run 作为阶段性 sanity，未纳入正式指标。

### dev_v42_stage_buffer_v2（128 env × 60 iter）
- **命令**：与 v1 相同，仅更换 `--run_name dev_v42_stage_buffer_v2`、`--seed 421842`。运行耗时 ~45 min，完成 60 iter。
- **On-policy 指标（iter51-60 平均）**：存活率 0.847、`first_death_step=33.8`、唯一记录的 `step_one_ratio=0.0188`（迭代 56）。`experiments/v42/artifacts/logs/dev_v42_stage_buffer_v2.log` & `.../tb_logs/dev_v42_stage_buffer_v2/`。
- **Multi-eval（96 fish ×40 ep）**：

| iter | stage | avg_final | min_final | death_p10 | step_one | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 8  | NE   | 0.869 | 0.708 | 76.9 | 42 | 0.0219 |
| 16 | NE   | 0.843 | 0.740 | 71.0 | 40 | 0.0208 |
| 24 | NE   | 0.823 | 0.646 | 82.9 | 28 | 0.0146 |
| 32 | main | 0.787 | 0.531 | 92.9 | 46 | 0.0240 |
| 40 | main | 0.771 | 0.510 | 64.0 | 53 | 0.0276 |
| 48 | main | 0.739 | 0.510 | 63.0 | 42 | 0.0219 |
| 56 | main | 0.697 | 0.521 | 77.0 | 36 | 0.0187 |

  - stage-aware step-one 阈值解除“count<=18”的瓶颈，但 ratio>0.02 & `early_death_median≈0.126` 仍让 gate 在全部 7 次 multi-eval 触发 `failure_hold`（详见 `penalty_stage_debug.jsonl`）。
  - `min_final` 持续在 0.51~0.74，未达到 <0.5 的 failure_hold，但不足以提供 success streak。
- **Tail 行为**：
  - `schedule_trace.json` 显示 `tail_ne_reinjections=1`，比目标≥2 仍低但较 v41 的 1 次持平。
  - Warning 仅在 iter57-60 持续 4 iter（`dev_v42_stage_buffer_v2_tail_stage_warning.png`），较 v41 的 14 iter 大幅压缩。
  - `step_one_stage_summary.json`：main 占 61.7%（177/287），略高于 v41 (≈62%)，说明缓冲段仍主要由 main 事件驱动。
- **Artifacts**：
  - Checkpoints/JSON：`experiments/v42/artifacts/checkpoints/dev_v42_stage_buffer_v2/`（含 `model_iter_{5..60}.zip`, `schedule_trace.json`, `step_one_*.json`）。
  - Logs/TensorBoard：`experiments/v42/artifacts/logs/dev_v42_stage_buffer_v2.log`，`experiments/v42/artifacts/tb_logs/dev_v42_stage_buffer_v2/`。
  - Plots：`experiments/v42/artifacts/plots/` 下新增 `dev_v42_stage_buffer_v2_survival.png`, `..._penalty_vs_first_death.png`, `..._multi_eval_timeline.png`, `..._tail_stage_warning.png` 等。
  - Media：`experiments/v42/artifacts/media/dev_v42_stage_buffer_v2_curve.gif` & `..._iter0{08,16,...,56}_tail_rank{0,1}_ep*.mp4`，可直接复盘 tail 样本。

## 观察更新
1. **Gate 仍被 ratio + early-death 卡住**：虽然 main 阶段的 step-one count 已允许 ≤45，但 multi-eval 的 `step_one_ratio` 长期 0.022~0.028，且 `rolling_early_death_median≈0.126`，使 7 次 multi-eval 全部 `failure_hold`，stage 仍锁在 0。
2. **Tail warning 改善但 reinjection 仍不足**：new stage spec 把 warning 时长降到 4 iter，但 `tail_ne_reinjections=1`；iter57 warning 仍因 main_stage1b block 耗尽 NE 触发。
3. **最差样本仍停在 ~0.51**：`min_final` 没再跌破 0.5，但离成功阈值 (≥0.8) 差距明显；`dev_v42_stage_buffer_v2_multi_eval_hist/` 显示 main 段尾部仍集中在 high-speed 捕食者。

## 下一步计划
1. **放宽/分段 ratio 判据**：新增 `--penalty_gate_success_step_one_ratio_by_stage` 或在 main 段仅检查 `ratio<=0.025`，同时对 early-death 阈值做 stage-aware 改写（main:0.12）以避免 rolling median 长期 0.126 阻塞。
2. **强制 NE reinjection ≥2**：在 tail tracker 中加入显式 `--tail_ne_min_reinjections` 或将 stage7/9 的 NE block duplicate 并提高 `tail_replay_count` 到 3，确保 iter40 后仍有 fresh NE buffer。
3. **改善最差样本**：基于 v32_stage2 生成降速版 checkpoint（例如将 `velocities` 缩放 0.8）并插入 stage4.5，用以替换当前 main_stage3 的硬样本；同时评估 `multi_eval_seed` 滚动窗口以过滤反复失败的相同种子。
