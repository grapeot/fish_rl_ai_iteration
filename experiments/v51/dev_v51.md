# v51 实验记录：v49 冻结模型的推理模式消融（确定性 vs 随机采样）

第 3 轮（共 10 轮）：比较同一冻结策略在确定性动作与随机采样动作下的存活表现。

## 1. 实验背景与问题设定

本轮实验为技术路线（10 轮规划）中的第 3 轮，完全独立于 v50 奖励函数实验。物理环境配置保持不变：11 维观测（关闭邻居观测 neighbor off）、96 条鱼群规模、每回合 500 步。本轮不对策略权重做重新训练，专注于评估 v49 产出的 6 个冻结检查点在推理阶段的行为表现：确定性预测（argmax）对比随机动作采样（stochastic sampling）。

评测复用 v49 的 6 个本地模型检查点：

- 阶段筛选检查点（selected stages）：
  - run1_sel：迭代轮次 150（实际完成 149 轮 PPO rollout 更新）
  - run2_sel：迭代轮次 100（实际完成 99 轮 PPO rollout 更新）
  - run3_sel：迭代轮次 50（实际完成 49 轮 PPO rollout 更新）
- 最终轮次检查点（final stages）：
  - run1_final、run2_final、run3_final（均为迭代轮次 200）

这 6 个模型的实际路径、SHA-256 与实际更新步数在评测前固化为 `artifacts/model_manifest.json`，并写入 `artifacts/config.json`。实际更新步数从每个 zip 的 `data['_n_updates'] // 10` 读出并与预登记值比对，不一致即中止；不是把 checkpoint 文件名当作实际更新数（v49 阶段标签 i50/i100/i150/final 分别对应 49/99/149/200 次已完成 PPO 更新）。

**历史局限与资产固化。** v49 的阶段选择是在 report 集生成之后按 selection 集每 run 均值 argmax 的事后结果，且三 run 的环境 RNG 流存在重叠（`BaseVecEnv.seed` 按 `seed+rank` 分配）。本轮按现状复用这 6 个模型，不回溯修改、不重新选择、不据本轮结果换模型。SHA 仅校验已评审的本地资产；重新训练无逐 bit 保证；模型权重（*.zip）不入库。

**源码快照范围。** `config.json` 的 `source_sha256` 是实验结束后修订源码的当前快照，不证明该版本生成了原始结果。`original_source_sha256` 与 `original_config_sha256` 保留首次审核时实际读到的原配置值；这也不构成独立证明评测前预登记。原始回合记录、模型清单和分析结果未因这次说明修订而重跑或改写。

## 2. 评测协议与随机数控制

- **评测场景与随机种子**：使用 40 个全新场景（主种子 510102；2 集调试用 510101）。动作流主种子 510201，按 `SeedSequence([master, episode_index, replicate_index])` 为每个 (episode, replicate) 派生独立动作流。该场景种子集与 v49 的 selection/report 银行（482101/482102）无交集；本轮不读取任何旧结果文件作为输入。
- **分支与隔离机制**：每个场景对比确定性预测（det）与 3 条独立随机动作流（stochastic sampling，`deterministic=False`）。环境世界 RNG 只由场景种子驱动（`env.reset(seed=scenario_seed)`）；stoch 臂每回合在 reset 之后调用一次 `torch.manual_seed(action_seed)`，不逐 step 重置。动作流 RNG 与环境重置 RNG 严格隔离：消耗 torch RNG 不会改变世界初态，det 臂不受任何 action/torch 种子影响。
- **统计分析方法**：场景内先对 3 条随机流存活率取均值（stoch_avg），再对 40 个场景做场景配对 Bootstrap，计算差值 `stoch_avg - det` 及其 95% 置信区间（10,000 次重采样，seed 510302）。统计单位是场景（episode），不是把 120 条随机回合或 96 条鱼当独立单位。模型固定，仅度量推理与场景变异，**不含训练不确定性**。
- **评测规模**：1040 回合（6 模型 ×（1 det + 3 stoch）× 40 场景 + 2 规则基线 × 40 场景），4 worker，torch 单线程，耗时 1611 秒（约 27 分钟，预算 ≤30 分钟）。3 条动作流是设计时冻结的方案，不是看结果后选的。
- **语义单测覆盖**：`tests/test_v51_semantics.py` 验证 7 项关键测试，全部通过：
  1. 相同动作种子重复产生完全相同动作序列；
  2. 变更动作种子产生不同的动作序列；
  3. 确定性预测（det）严格独立于动作采样随机种子与 torch RNG 状态；
  4. 环境重置 RNG 与动作流 RNG 相互隔离（消耗 torch RNG 或跑一个随机回合都不改变下次 `reset(seed)` 的世界初态）；
  5. 120 个派生动作流种子互不碰撞，且与本轮场景种子、v49 seed 集均无交集；
  6. 在真实训练模型上确定性预测与随机采样的动作序列确实不同（消融有功效）；
  7. 场景配对统计分析器逻辑，与独立从 raw JSONL 复算的结果一致。

这些测试使用真实环境、真实冻结 v49 检查点和真实随机初始化 PPO，不是自身构造的 mock 数组。

## 3. 评测结果

下表 Δ = `stoch_avg − det`，Δ 与 95% bootstrap CI 单位均为**百分点**。区间来自固定模型、固定 3-stream 方案下的场景配对重采样，逐模型未做多重比较校正，不含训练随机性估计。

| 模型 | det 存活率 | stoch_avg 存活率 | Δ（百分点） | 95% CI（百分点） | 点估计与区间 |
| :--- | ---: | ---: | ---: | :--- | :--- |
| run1_sel | 0.9247 | 0.8572 | −6.7535 | [−8.0992, −5.4427] | det 较优，CI 不跨 0 |
| run2_sel | 0.9023 | 0.8843 | −1.8056 | [−2.5781, −1.0330] | det 较优，CI 不跨 0 |
| run3_sel | 0.8742 | 0.8916 | +1.7361 | [+0.8941, +2.6042] | stoch 较优，CI 不跨 0 |
| run1_final | 0.8971 | 0.8864 | −1.0764 | [−1.9531, −0.1997] | det 较优，CI 不跨 0 |
| run2_final | 0.8544 | 0.8617 | +0.7292 | [−0.0260, +1.5799] | stoch 点估计较优，CI 跨 0 |
| run3_final | 0.8719 | 0.8847 | +1.2847 | [−0.5903, +3.4983] | stoch 点估计较优，CI 跨 0 |

同 bank 规则锚点：`rule_flee_lead = 0.96875`，`rule_hold = 0.7296875`。

## 4. 结果分析与边界说明

确定性动作点估计较优的有 **3 个**（run1_sel、run2_sel、run1_final），随机采样点估计较优的有 **3 个**（run3_sel、run2_final、run3_final）。其中区间排除 0 的：det 较优 3 个（run1_sel、run2_sel、run1_final），stoch 较优 1 个（run3_sel）；其余 2 个（run2_final、run3_final）点估计为正但区间跨 0。

- **采样没有全局一致收益，但对特定模型确实改变表现。** 6 个模型的差值点估计呈现 3 负 3 正。准确表述是“在六个冻结模型与本轮方案中未见普遍的随机采样收益”，**不能**写成“采样不改变存活率”。run1_sel 的随机采样较确定性下降约 6.75 个百分点、区间完全低于 0，这是模型内的实质变化，不能用模型间方向不统一抹去；这个量级也不应称作“小”。不能据此认为采样在任何意义上“无影响”。
- **有限重复性，不等于排除偶然性。** 3 条既定动作流在跨 40 场景的均值差符号上一致，这支持**有限的重复性**；但只有 3 条固定动作流、每场景内 stream 标准差约 0.017–0.022，样本有限，不足以强证明差异与“幸运轨迹”无关，也不足以证明该效应对广泛 action RNG 是一个稳定的模型属性。scenario CI 也不单独度量“增加动作流条数”带来的 Monte Carlo 不确定性。CI 跨 0 表示方向证据不足，不等于等效或无变化。
- **阶段差异仅为描述性统计。** 阶段筛选模型平均差值 −2.2743 个百分点、最终轮次模型平均差值 +0.3125 个百分点、交互项 +2.5868 个百分点。该数值仅描述这三组冻结模型（selection 检查点阶段不同且沿用 v49 事后选择），不能外推成“训练越久采样越有利”的训练规律，也不构成训练维度的统计置信。
- **归因边界与后续方向。** 现有数据不支撑对优化做病理诊断，也不构成部署采样的指导依据。动作分布均值均以存活个体为条件（幸存者条件量，非全体个体），且 HOLD 基线本身随时间非平稳，不能断言动作分布坍塌因果解释了全部现象。后续核心在于捕食者信息的有效利用（第 4 轮），而非全局推理参数调整。

本轮结果支持在有限范围内继续默认采用确定性动作、不改训练；若随机采样在某些模型上点估计更高，也不自动意味着应部署采样，报告同时给出均值增益、跨流离散与配对 CI。

## 5. 文件清单与复现验证

本次产出涉及以下代码、配置与记录（公开候选见交接清单）：

- 评测与冻结逻辑：
  - `experiments/v51/v51_common.py`
  - `experiments/v51/v51_eval.py`
  - `experiments/v51/v51_analyze.py`
  - `experiments/v51/v51_freeze.py`
- 语义单测套件：
  - `experiments/v51/tests/test_v51_semantics.py`
- 记录与产物：
  - `experiments/v51/dev_v51.md`
  - `experiments/v51/artifacts/config.json`
  - `experiments/v51/artifacts/model_manifest.json`
  - `experiments/v51/artifacts/results/fresh.jsonl`
  - `experiments/v51/artifacts/results/analysis.json`

依赖复用已有 v48 虚拟环境（`experiments/v48/.venv`，Python 3.12.9 / torch 2.13.0 / sb3 2.9.0），未新建环境。

完整复现命令（参数已按各脚本实际 argparse 核对；输出写到**新目录** `artifacts/repro/`，不覆盖已归档的 `artifacts/results/fresh.jsonl`）：

```bash
# 从 fish_rl 仓库根目录
source experiments/v48/.venv/bin/activate

# 0) 语义单测
python experiments/v51/tests/test_v51_semantics.py

# 1) 冻结六模型 manifest（路径 + sha256 + 实际更新步数），写到新目录不覆盖归档 config
#    依赖既有 v49 checkpoint：experiments/v49/artifacts/runs/seed49010{01,02,03}/checkpoints/
#    其中 selected 阶段为 model_iter_150/i100/i50，final 为 model_final（*.zip 不入库，需本地存在）
python experiments/v51/v51_freeze.py --out experiments/v51/artifacts/repro/config.json

# 2) 2 集调试/计时（可选）
python experiments/v51/v51_eval.py --scenarios debug --arms det stoch \
  --workers 2 --out experiments/v51/artifacts/repro/debug.jsonl

# 3) 40 场景正式评测：det(x1) + stoch(x3 streams) + 规则锚点，输出到新 outdir
python experiments/v51/v51_eval.py --scenarios fresh --arms det stoch rule \
  --rules hold flee_lead --workers 4 \
  --out experiments/v51/artifacts/repro/fresh.jsonl \
  --manifest-out experiments/v51/artifacts/repro/model_manifest.json

# 4) 分析
python experiments/v51/v51_analyze.py \
  --raw experiments/v51/artifacts/repro/fresh.jsonl \
  --manifest experiments/v51/artifacts/repro/model_manifest.json \
  --out experiments/v51/artifacts/repro/analysis.json
```

说明：`v51_freeze.py` 的 `--out` 默认为归档的 `artifacts/config.json`，复现时显式指定新目录（如上 `artifacts/repro/`）即可避免覆盖。`--models` 省略时默认全部 6 个模型；`--neighbor` 默认 off，与本轮 11 维观测一致。`artifacts/tests.log` 与模型 `*.zip` 均不在公开候选内。
