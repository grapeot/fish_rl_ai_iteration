# dev_v32

## 启动
- 2025-11-12 13:30 PST 依据 `SOP.md` 与 `experiments/v31/dev_v31.md` 复盘 v31 结论：NE 高速 bucket 在迭代 32 前依旧回弹、penalty gate 被 step-one 死亡绝对值锁死、tail queue 在 10 迭代内耗尽、CLI 在收尾评估阶段频繁超时。
- v32 目标：
  1. 把 tail 预热做成 staged queue（前期偏 NE 低速+后期南/西慢速），保证 `prewarm_override_ratio` 在迭代 24 以内保持>0.25。
  2. 将 penalty gate 成功条件从“step-one <= 9”替换为“step-one ratio <= 1.5%”，并把该指标写入 TensorBoard 及 multi-eval jsonl。
  3. 通过 `--skip_final_eval` 等机制把长耗时的 best-checkpoint 复核/视频录制拆出去，主训练 run 必须在 15 分钟窗口内结束。

## 观察（承接 v31）
1. **Tail queue 早衰**：v31_runB 虽能加载 `velocities` 字段，但 12×128 样本在 10 iteration 内耗尽，`prewarm_override_ratio` 迅速跌至 0，导致 NE hotspot 在迭代 16-32 再度抬头。
2. **Penalty gate 绝对阈值过严**：step-one 死亡稳定在 18-24，绝对阈值 9 使 gate 永远停在 stage0，即便 multi-eval `first_death_p10` 已达到 500。
3. **收尾流程耗时**：两次 run 都卡在 best-checkpoint 评估和 48-episode deterministic eval，CLI 超时终止，训练主循环本身并未出错。
4. **NE hotspot 仅后期缓解**：`step_one_ne_high_speed_share` 在迭代 48 降到 4.8%，但在 32 仍有 29%，说明预热样本仍偏向 NE。

## 实验计划
1. **代码/工具改动**
   - 为 `train.py` 引入 `--tail_seed_stage_spec`，允许配置 `iterations:path` 阶段，使预热样本分批补充；生成 `tail_seed_mix_v32_stage1/2.json`，分别覆盖 NE 低速与南/西慢速。
   - 将 penalty gate 新增 `--penalty_gate_success_step_one_ratio`，默认 0 关闭；v32 运行时设为 0.015，并在 multi-eval entry/TB 中记录 `step_one_death_ratio`。
   - 新增 `--skip_final_eval`，默认执行旧流程；v32 主 run 开启该 flag，把 deterministic eval/视频延后处理。
2. **实验设计**
   - RunA（`dev_v32_stage_tail_ratio`）：沿用 v31_runB 的 hyper（128 env，n_steps=192，total_iterations=60），但开启 staged tail、ratio gate、skip final eval，`--tail_seed_stage_spec 16:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage1.json;16:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage2.json`，观察 `prewarm_override_ratio` 与 `step_one_death_ratio` 走势。
   - 若 RunA 仍在迭代 24 前出现 NE 回弹，则调整 stage2 配方（增大西南象限权重）并缩短 multi-eval interval 到 12 以快速验证。
3. **产物要求**
   - 训练日志、TB、plots、媒体统一写入 `experiments/v32/artifacts/`；将 multi-eval timeline、step-one polar、至少 1 个 tail 视频拷贝到 `plots/` 与 `media/`。
   - `dev_v32.md` 后续补充 run 命令、关键指标（survival、step_one_death_ratio、penalty stage）、artifact 路径及下一步计划。

## 实验记录

### dev_v32_stage_tail_ratio_r2（128 env，n_steps=128，60 iter，skip_final_eval）
- **命令**
  ```bash
  python experiments/v32/train.py --run_name dev_v32_stage_tail_ratio_r2 \
    --total_iterations 60 --num_envs 128 --n_steps 128 --batch_size 1024 --n_epochs 5 \
    --learning_rate 2.5e-4 --curriculum 15:9,20:10,25:9,25:8,25:8,25:8,25:8 \
    --multi_eval_interval 16 --multi_eval_probe_episodes 10 --eval_multi_fish 96 --eval_multi_episodes 24 \
    --multi_eval_seed_base 311232 --seed 311232 \
    --tail_seed_stage_spec '16:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage1.json;16:experiments/v32/artifacts/checkpoints/tail_seed_mix_v32_stage2.json' \
    --penalty_gate_success_step_one_ratio 0.015 --penalty_gate_success_step_one 18 \
    --tail_replay_count 2 --video_num_fish 48 --skip_final_eval
  ```
- **结果**
  - 60 轮全部完成；最后 10 轮平均 `survival_rate=0.720±0.007`，`first_death_step≈35.2`。（来源：`training_stats.pkl`）
  - 多次 probe（10×96 样本）：
    - Iter16：avg final survival 0.592，min final 0.458，`early_death_fraction_100=0.110`，`step_one_death_ratio=1.67%`（触发 failure_hold）。
    - Iter32：avg final 0.624，min final 0.552，`ratio=1.35%`（低于门槛，gate 仍因 freeze 未解锁）。
    - Iter48：avg final 0.605，min final 0.521，`ratio=2.29%` → 超过 1.5% 阈值，gate 保持 stage0。
  - `prewarm_override_ratio` 仅在 iter1/5 维持 1.0，iter7 之后迅速跌到 0，说明 staged queue 只撑过了 ~1 批 reset，后续仍靠实时采样。
  - `step_one_ne_high_speed_share`：iter13 为 6.2%，iter29 突增至 46.2%，iter45 回落到 27.3%；需要进一步抑制 mid-phase hotspot。
  - penalty_gate 日志显示 16/32/48 轮事件分别为 `failure_hold`、`noop`、`noop`，`rolling_p10` 约 88~91，仍未晋级。
- **产物**
  - 日志：`experiments/v32/artifacts/logs/dev_v32_stage_tail_ratio_r2.log`
  - 训练统计：`experiments/v32/artifacts/checkpoints/dev_v32_stage_tail_ratio_r2/training_stats.pkl`
  - Multi-eval 时间线 & hist：`experiments/v32/artifacts/plots/dev_v32_stage_tail_ratio_r2_multi_eval_timeline.png`、`.../dev_v32_stage_tail_ratio_r2_multi_eval_hist/`
  - 指标曲线：`experiments/v32/artifacts/plots/dev_v32_stage_tail_ratio_r2_survival.png`、`..._penalty_vs_first_death.png`、`..._step_one_heading_polar.png`
  - 媒体：`experiments/v32/artifacts/media/dev_v32_stage_tail_ratio_r2_iter048_tail_rank0_ep8.mp4`、`dev_v32_stage_tail_ratio_r2_curve.gif`
  - TB：`experiments/v32/artifacts/tb_logs/dev_v32_stage_tail_ratio_r2`
- **分析**
  - `--tail_seed_stage_spec` 虽能注入两个批次，但 32 iteration (≈4k 样本) 仍不足，迭代 7 以后几乎无 prewarm 覆盖，导致 iter29 再次出现 NE ≥1.8m/s 高速团。
  - ratio gate 监控生效：`step_one_death_ratio` 被写入日志/TB，但阈值 1.5% 对迭代 48 仍然严格，使 gate 长期锁在 stage0。
  - `--skip_final_eval` 把 CLI 总耗时压到 ~13 分钟，训练未再被收尾评估中断；后续如需 deterministic eval，可用独立脚本复用 `model_iter_60.zip`。

### dev_v32_stage_tail_ratio / r1（记录）
- 两次早期 run（`dev_v32_stage_tail_ratio` 与 `_r1`）仍沿用 `n_steps=192`，在迭代 38/45 前后因为 CLI 15 min 超时被杀，已保留部分 checkpoint、日志、媒体，命名规则同 run 名。后续调参（n_steps=128 + skip_final_eval）消除了该瓶颈。

## 后续计划（v33 移交草案）
1. **Tail queue 延命**：现有 staged mix 只有 4,096 个 velocity，实测在 ~7 iter 内耗尽；需要将 stage1/2 样本数量翻倍并允许 `tail_seed_stage_spec` 指定 3+ 阶段（或在 env 内周期性 refuel），目标是 `prewarm_override_ratio` 在 iter20 前保持 >0.25。
2. **Ratio gate 调参**：观测到 32 轮 ratio≈1.35%，48 轮又回到 2.29%；可尝试把阈值放宽到 2.0%，并改成 “连续 2 次满足” 才晋级，避免长期锁死在 stage0。
3. **Hotspot logging**：为了解 NE 回弹原因，需要输出 per-iteration 的 `tail_seed_stage_id` 与 “预热 velocity 队列剩余长度”；同时把 `step_one_ne_high_speed_share` 的采样频率提升到每次 multi-eval（目前 16/32/48 才有数据）。
