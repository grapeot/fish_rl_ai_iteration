# Round 2 v50 实验记录：整体奖励包对比与连续流重跑验证

## 1. 实验目标与环境设定

本轮在固定物理环境与固定基线下，比较两套**完整奖励包**（whole-reward-bundle）对策略表现的影响：

- **original**（`FishEscapeEnv` 原始逐鱼奖励）：
  - 存活步奖励：基础 +2，经 `REWARD_SCALE=0.1` 缩放后为 +0.2；
  - 距离奖励：视野外 +5（缩放后 +0.5），视野内按距离给奖励；
  - 额外塑形：边界惩罚、density penalty（coef 0.05）；
  - 死亡惩罚：缩放后 −50。
- **survival_only**：
  - 每存活步固定 +0.7；
  - 死亡惩罚 −50（一次性）；
  - 一次性移除距离、边界与密度全部塑形项。

两臂替换的是送入 PPO 的**整个奖励标量**：`survival_only` 的 +0.7 取自原始奖励的视野外常见基准（存活 2 + 远距离 5，乘 0.1），即保留原奖励形状而去掉视野内/边界/密度项。因此这是奖励水平与 shaping 结构的整体变化，**不是**保持原存活项不变、只删某一项的单项消融。

**环境与超参数约束**：

- 96 条鱼，固定焦点鱼身份，其余鱼 HOLD；
- 11 维局部观测（邻居特征关闭）；
- 物理、动作空间、出生分布不变，网络 384×384；
- Stable-Baselines3 PPO，$\gamma=0.99$、GAE $\lambda=0.95$、clip 0.2、ent_coef 0.02、lr 3e-4、n_epochs 10；
- 500 步超时用 bootstrap（显式训练惯例，不是唯一正确的有限时域目标）；
- 单 run 预算 614,400 步（200 iter × n_steps 512 × num_envs 6）。

**重复与随机种子设计**：

- 3 对配对重复（3 paired rep × 2 arms）；
- 同一 rep 内两臂初始模型权重与 worker 随机数池一致（`initial_policy_hash` 相同），策略分化后动作与 reset 时机不同，两臂并非重合轨迹；
- worker bank 6000011 / 6000021 / 6000031（各 rank 0..5，rep 间不相交），model seed 7000011 / 7000021 / 7000031；
- 单 run 约 1390 秒，分两波（每波 3 runs）执行。

## 2. 缺陷披露与连续流重跑

- **早期缺陷**：初版 pilot 脚本在每次环境 reset 时错误复用 `worker_seed`，每个 run 只反复重置为固定初始世界，违背连续流（continuous stream）设计。
- **修复逻辑**：显式传入 seed 时正常生效；否则 `worker_seed` 只在首次 reset 消费一次，之后 reset 由 `np_random` 持续推进。trainer 在 PPO 构造后把 VecEnv `_seeds` 置为 `[None]*n`。已在真实 `SubprocVecEnv` 下验证自动 reset 进入新 episode。
- **重跑范围**：在相同训练预算下从零重跑 6 次，使用全新 selection bank 500201（24 eps）与 report bank 500202（40 eps），修正产物归档于 `experiments/v50/artifacts/corrected_streams/`。
- **历史说明边界**：旧 pilot 数据仅在本地留存，不是公开有效候选集；公开候选不含其源码快照、日志或模型。旧 pilot 与 corrected 流共用 model seed、worker seed 与起始世界，训练分布不同，因此**不构成独立 replication**，两者之差**不是**隔离 reset 缺陷作用的因果实验，也不能据此说该缺陷此前毫无影响。正式结论完全依据 corrected 数据。

## 3. 基础设施与验证门禁

- **检查点与指标**：在 update 50、100、150、200 保存 checkpoint（保存发生在对应 PPO 更新**之后**），记录全部 200 次 update 的 metrics、完整环境配置（含 heading bias）、可移植 argv，并在每个 run 的 `config.json` 保留**运行时**训练源码 hash。
- **测试覆盖**：共 26 项（10 项语义 + 16 项验收门禁），全部通过。
- **report 门禁**：`v50_evaluate.py` 在传入 `--require-manifest` 时强制绑定：report seed bank、`neighbor` 标志与**完整** `env_config`（含 predator heading/speed bias）、每个 `*_selected`/`*_final` 臂的 checkpoint 路径及其 **policy tensor SHA-256**；缺臂、未知 run、重复臂名、env 参数或 seed 不符均 fail。模型身份用 `policy_tensor_sha256`（`policy.state_dict()` 排序键连续字节），因其对同权重可复现，比含时间戳的 zip 字节 hash 稳定。
- **analyzer**：要求每臂恰好是冻结的 40 个 report episode（缺一/重复 `(arm, seed)`/多一均 fail）；`final` 仅在 report 中真实存在，或 manifest 明示该 run `selected_is_final=True` **且 selected 与 final 的 policy tensor hash 相等**时才复用 selected 记录，否则 fail，不再静默回填。
- **元数据升级与时序**：manifest 的 `created_utc` 是**选择映射首次冻结**的时刻（`2026-10-09T10:02:16Z`）；report seeds、完整 env 配置与 final 身份字段是**首次 report 之后**补充的 v2 metadata enrichment，另有独立的 `bindings_enriched_utc` 记录。不声称所有强绑定在原始选择冻结时即已存在。后处理评估工具源码 hash 与运行时训练源码 hash 分开记录，未刷新训练历史。
- **门禁范围**：绑定检查在传入 `--require-manifest` 时运行（官方 report 编排器 `v50_report.py` 会传入）；不是对所有直接 evaluator 调用的无条件全局机制。

## 4. 评测结果与统计分析

基于 40 episode 的确定性评测（模型来自冻结选择清单，非挑选 report 最优）：

| 评估指标 | original (rep0 / rep1 / rep2) | survival_only (rep0 / rep1 / rep2) | 均值 (original vs survival_only) |
| :--- | :--- | :--- | :--- |
| **Selected** | 0.8721 / 0.8784 / 0.8943 | 0.9435 / 0.9323 / 0.9250 | **0.8816 vs 0.9336** (SD 0.0114 / 0.0093) |
| **Final** | 0.3935 / 0.8661 / 0.8932 | 0.9435 / 0.9357 / 0.9255 | **0.7176 vs 0.9349** |
| **训练流累计存活率** | 0.869 / 0.853 / 0.826 | 0.927 / 0.912 / 0.904 | — |

**核心统计差异**：

- **Selected**：survival_only 相比 original 提升 **+5.20 个百分点**（0.8816 → 0.9336；固定六模型、episode 配对 95% CI [+4.38, +6.03] 个百分点，40 episode 中 39 个为正）；三个 rep 的差值同为正向（+7.14 / +5.39 / +3.07 个百分点）。
- **Final**：差值为 **+21.73 个百分点**（95% CI [+20.12, +23.19] 个百分点，40 episode 全部为正），但**约 84.4% 的差值来自 original rep0**（rep0 差值 +55.0 点，rep1 +7.0 点，rep2 +3.2 点）。因此不能称所有 original run 都退化，也不能称 survival_only 一般稳定；大 final 差主要反映本轮 original rep0 的晚阶段退化。
- **基准参考**：fresh rules 评测 lead 0.9732、safe_top 0.9320、HOLD 0.7583。
- **统计边界**：3-rep SD 只是这三组训练结果的描述性离散度，**不是**训练随机性的抽样 CI；该 CI 条件在固定六模型上。episode 是统计单位，不把 96 条鱼当独立样本。

**退化观察**：original rep0 在同模型、两个不相交评估 bank 上都显示退化（selection bank 的 u50=0.8655、u200=0.3958；report bank 的 selected=0.8721、final=0.3935），这是本轮内的确认，不是两次独立训练 replication。该现象与 reward misalignment、优化不稳定、灾难性遗忘三种机制都相容，本轮**未识别**具体机制。

## 5. 决策边界与结论

- **实际选中更新步**：original u50 / u100 / u50；survival_only u200 / u50 / u150。
- **推进建议**：本设置下 survival_only 在三组 selected 上同向优于 original，支持将其作为下一代候选基准；不据此宣称奖励的普遍最优性或训练稳定性。
- **事实边界**：
  - 底层物理未改；收益是完整奖励包的整体替换，无法归因到 distance / boundary / density 中任何单项；
  - 只有 3 个训练 rep，final 差由单个 rep 主导；
  - 与 lead 规则基准（0.9732）仍有约 4 个百分点差距；
  - 不对所有随机种子、收敛 SOTA 或世界后段无风险作推断；
  - 后续信息与鲁棒性测试可直接基于本组固定模型开展。

## 6. 代码资产与复现流程

公共代码与数据资产：

- 核心源码：`v50_common.py`、`v50_env.py`、`v50_train.py`、`v50_evaluate.py`、`v50_select.py`、`v50_report.py`、`v50_analyze.py`、`v50_verify.py`；
- 测试：`tests/test_v50_semantics.py`、`tests/test_v50_acceptance_gates.py`；
- 运行工件：各 run 的 `config.json` / `run_summary.json` / `train_metrics.jsonl`，以及 `results/` 下的 `selection.jsonl`、`selection.summary.json`、`selection_manifest.json`、`report.jsonl`、`report.summary.json`、`analysis.json`、`post_run_source_hashes.json`（公共仓库不含模型权重 zip）。

**复现步骤**（仓库根目录）：

```bash
# 1. 环境与测试（复用既有 v48 venv）
source experiments/v48/.venv/bin/activate
python experiments/v50/tests/test_v50_semantics.py
python experiments/v50/tests/test_v50_acceptance_gates.py

# 2. 训练复现：3 rep × 2 arms，各 200 iter、614,400 steps
for rep in 0 1 2; do
  for arm in original survival_only; do
    python experiments/v50/v50_train.py \
      --replicate $rep --arm $arm --iterations 200 --num-envs 6 \
      --n-steps 512 --batch-size 1024 --n-epochs 10 --learning-rate 3e-4 \
      --ent-coef 0.02 --gamma 0.99 --gae-lambda 0.95 --clip-range 0.2 \
      --net-arch 384,384 --checkpoint-iters 50,100,150 \
      --out-dir experiments/v50/artifacts/reproduction/runs/rep${rep}_${arm}
  done
done

# 3. 选择：在 selection bank 上逐 run 选最优，写不可覆盖 manifest（先于 report）
python experiments/v50/v50_select.py \
  --runs-dir experiments/v50/artifacts/reproduction/runs \
  --out-jsonl experiments/v50/artifacts/reproduction/results/selection.jsonl \
  --manifest experiments/v50/artifacts/reproduction/results/selection_manifest.json

# 4. report：门控于 manifest
python experiments/v50/v50_report.py \
  --runs-dir experiments/v50/artifacts/reproduction/runs \
  --manifest experiments/v50/artifacts/reproduction/results/selection_manifest.json \
  --out experiments/v50/artifacts/reproduction/results/report.jsonl

# 5. 分析
python experiments/v50/v50_analyze.py \
  --runs-dir experiments/v50/artifacts/reproduction/runs \
  --report-summary experiments/v50/artifacts/reproduction/results/report.summary.json \
  --report-jsonl experiments/v50/artifacts/reproduction/results/report.jsonl \
  --manifest experiments/v50/artifacts/reproduction/results/selection_manifest.json \
  --out experiments/v50/artifacts/reproduction/results/analysis.json
```

`reproduction/` 为新目录，避免覆写既有归档；`config.json` 内的 `source_sha256` 是运行时训练源码 hash，不因事后修改评估脚本而刷新，评估工具的实际变更另记于 `post_run_source_hashes.json`。

训练源码（`v50_common.py`、`v50_env.py`、`v50_train.py`）保持运行时的字节快照不变，因此其 docstring 里可能仍带修正前的旧 bank 编号或示例路径；权威的 bank 定义与复现命令以本文档 §3/§6 与代码常量为准。评估/选择/报告/汇总/校验脚本的 docstring 已更新为 corrected 路径与 500201/500202。
