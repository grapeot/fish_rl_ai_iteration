# dev_v48: 鱼群避障世界规则与基线审查

## 1. 问题

本记录为前置世界规则与基线审查，不计入后续正式实验轮数，尚未决定采用三轮或十轮迭代方案。核心目标是评估当前物理仿真世界下的避障任务有效难度，并审查基线策略与环境语义的实现一致性。

审查中确认了一处独立的工程实现缺陷：旧版训练环境 `SingleFishEnv` 存在动作与奖励语义错位。训练过程中，环境根据轮换的单条鱼观测输出一个全群广播动作，并返回全群平均奖励；而在评估阶段，控制器采用逐鱼独立推断（`policy_per_fish`）。使用相同 checkpoint 评估时，逐鱼推断存活率为 0.9010，广播动作存活率为 0.7161，两者相差 18.49 个百分点。这一差异属于同一 checkpoint 下的执行口径差异，并非修复训练带来的因果收益，不能由此保证或预测修复训练后具体提升的数值。后续修复需要确保轨迹身份、动作执行、奖励分配与终止判定在整个 transition / return 上完全一致，不能仅简单改成逐鱼预测后继续轮换观测；该修复本轮仅记录待设计，不在本记录内实施。同时确立原则：不以训练代码的实现缺陷判定仿真世界规则本身不合理。

## 2. 方法

### 2.1 世界与物理规则
仿真环境为半径 10 的圆形舞台。每局运行 500 步，时间步长 $dt = 0.1$。捕食者运动受正 y 方向重力和圆形边界反弹支配，不读取鱼的状态。鱼群内部无物理碰撞体积。鱼的离散动作空间包含 5 种操作：加速、左转 30 度、右转 30 度、减速（当前速度乘以 0.9）、保持现有速度。保持动作指维持当前速度向量，并非静止。观测信息包括局部捕食者相对状态、自身绝对位置与速度向量，以及可选的邻居摘要。

### 2.2 评估集与统计口径
每局配置 96 条鱼。统计以整局（episode）为独立抽样单元，计算配对 Bootstrap 95% 置信区间。评测通过 `evaluate.py` 的 `--seeds <名称>` 选择种子集；种子集名称到 RNG seed 的映射定义在 `common.py` 的 `STAGE_SEEDS`：
- 报告集 `report`：RNG seed = 481002，共 40 局，作为本记录基线报告基准。
- 开发集 `dev`：RNG seed = 481001，共 12 局。
- 冒烟集 `smoke`：RNG seed = 481000，共 2 局。
- 历史复现集 `legacy_report`：RNG seed = 555002，共 40 局，仅用于历史基线对齐。

历史最佳模型 checkpoint 路径为 `experiments/v45/artifacts/checkpoints/ms_baseline_v42cfg_seed700000/model_iter_20.zip`（仓库相对路径）。该模型在历史复现集上的平均存活率为 0.8990，在报告集上为 0.9010。该权重文件未随本阶段候选提供。

### 2.3 评测策略臂与实验资产
实验代码与资产集中于 `experiments/v48/` 目录，包含 `common.py`、`evaluate.py`、`summarize_results.py`、`probe_policy_space.py`、`phase_analysis.py`、`predator_probe.py`、测试脚本 `tests/test_training_semantics.py` 与 `tests/test_env_decomposability.py`，以及输出目录 `artifacts/results/`。

除 PPO 策略（逐鱼推断 `policy_per_fish` 与广播执行 `policy_broadcast`）外，评估考察了多组对比规则。只有带避让判据的复杂规则才含固定手工阈值，恒定动作与随机动作不设阈值：
- 恒定与随机规则（无手工阈值）：保持动作（`rule_hold`）、恒定减速（`rule_decelerate`）、持续加速（`rule_accelerate`）、持续左转（`rule_turn_left`）、均匀随机（`rule_random`）。
- 逃逸规则（`rule_flee`）：基于与捕食者的背离向量叠加边界回避（含边界与朝向阈值）。
- 前瞻逃逸规则（`rule_flee_lead`）：在逃逸方向中引入捕食者可见速度的提前量（含同上前瞻阈值）。该规则没有减速分支。
- 固定安全区规则（`rule_safe_top`）：引导鱼游向预设上方目标点 $(0, -5.5)$，进入半径 1.5 距离阈值后持续减速，直至归一化速度 $\le 0.03$ 后切换为保持。

## 3. 结果

### 3.1 主表：最终平均存活率对比

在报告集 `report`（40 局）下的最终平均存活率如下。配对差值相对 `policy_per_fish`，取真实汇总值四舍五入；仅对承重结论给出 95% 配对 Bootstrap CI 与胜负比：

| 策略臂 | 最终存活率均值 | 相对 PPO (逐鱼) 配对差值 | 95% 配对 Bootstrap CI | 胜局比 (vs PPO) |
| :--- | :--- | :--- | :--- | :--- |
| `rule_flee_lead` | 0.9753 | +0.0742 | [+0.058, +0.091] | 39 / 40 |
| `rule_flee` | 0.9701 | +0.0690 | - | - |
| `rule_safe_top` | 0.9174 | +0.0164 | [-0.004, +0.036] | - |
| `policy_per_fish` (PPO) | 0.9010 | 基准 | - | - |
| `rule_decelerate` | 0.8302 | -0.0708 | - | - |
| `rule_turn_left` | 0.8279 | -0.0732 | - | - |
| `rule_random` | 0.7453 | -0.1557 | - | - |
| `rule_hold` | 0.7383 | -0.1628 | - | - |
| `policy_broadcast` (PPO) | 0.7161 | -0.1849 | - | - |
| `rule_accelerate` | 0.5370 | -0.3641 | - | - |

数据检验呈现三项核心特征：
1. `rule_flee_lead` 高于 PPO 逐鱼策略，配对差 +0.0742，95% CI [+0.058, +0.091]，在 40 局中有 39 局存活率更高。
2. `rule_safe_top` 相比 PPO 均值高出 0.0164，但置信区间跨越 0（CI [-0.004, +0.036]），不能宣称其显著优于 PPO。
3. 恒定减速存活率（0.8302）高于恒定保持（0.7383）。

### 3.2 分时存活率动态

| 策略臂 | Step 1 | Step 100 | Step 250 | Step 500 (最终) |
| :--- | :--- | :--- | :--- | :--- |
| `rule_flee_lead` | 0.9833 | 0.9818 | 0.9802 | 0.9753 |
| `rule_safe_top` | 0.9807 | 0.9174 | 0.9174 | 0.9174 |
| `policy_per_fish` | 0.9815 | 0.9201 | 0.9099 | 0.9010 |
| `rule_decelerate` | 0.9812 | 0.9036 | 0.8536 | 0.8302 |
| `rule_hold` | 0.9812 | 0.9031 | 0.8380 | 0.7383 |

在观测的 40 局样本中，`rule_safe_top` 在第 100 步之后存活率在样本内保持 0.9174，未记录到新增死亡。`rule_flee_lead` 在第 250 步后仍记录到少量死亡（0.9802 降至 0.9753）。

### 3.3 捕食者运动动力学分布

40 局中捕食者距世界中心半径 $r$ 的分段均值如下：
- 1-50 步：5.20
- 51-100 步：5.98
- 101-150 步：6.42
- 151-250 步：7.47
- 251-500 步：8.93

轨迹探针的后半程覆盖统计采用 250-500 步：90% 的局捕食者未进入 $r < 6$ 区域，60% 的局未进入 $r < 7$ 区域。上表最后分段采用 251-500 步，该分段每局最小半径的样本均值为 7.30。这些只是样本统计量，不构成所有轨迹的全局安全下界。

## 4. 限制

1. **测试集非完全隔离**：由于多种规则在分析过程中追加考查，报告集不再是从未接触过的隔离测试集。后续若产生承重性结论，需引入全新冻结的确认集。
2. **轨迹外推边界**：
   - 捕食者在样本分段平均上呈现外移趋势，但这仅代表样本分段平均，不等于每条轨迹半径单调递增，不能断言内圈绝对无风险。后半程每局最小半径的样本均值 7.30 亦非所有轨迹最小半径的理论下界，不存在已确立的全局安全下界。
   - `rule_safe_top` 在当前 40 局中第 100 步后未记录到死亡，不能外推至所有随机轨迹均永久安全。其到达目标后的减速停泊逻辑在物理上仍存在残余滑行。
   - `rule_flee_lead` 动作中约 92% 为保持（维持当前速度向量，非停止），即保持速度滑行；其实际速度大小需轨迹确认。无减速分支不代表必然处于高速，也不证明任务后期无需决策。该策略在后半程依然记录到少量死亡。
3. **首步死亡判定**：首步死亡不等于与动作无关的必然死亡。各策略臂在 Step 1 的存活率存在差异（0.9807 至 0.9833），不能将剔除首步死亡后的数据定义为真实可救援上限。
4. **结论适用范围**：当前实验表明，在现有状态分布下存在高效的简单避让启发式，固定区域规避策略可在样本后半程规避主要风险。这提示环境的空间覆盖与持续决策需求值得进一步审视，但不证明启发式达到全局最优，不代表所有初始状态都平凡，也不意味着强化学习方法在此任务中必然无效。

## 5. 复现

### 5.1 自动化测试
以下两份单元测试脚本已在本地实际运行并全部通过：
- `experiments/v48/tests/test_training_semantics.py`
- `experiments/v48/tests/test_env_decomposability.py`

### 5.2 运行环境快照
当前验证环境快照如下（仅作为现场运行状态记录，不保证与历史训练完全一致或锁定依赖）：
- Python 3.12.9
- torch 2.13.0
- stable-baselines3 2.9.0
- gymnasium 1.3.0
- numpy 2.5.1
- pygame 2.6.1

`requirements.txt` 列出运行所需包但使用宽泛下界，未锁定精确版本。评测路径只依赖上述基础包；语义测试还会加载仓库内已跟踪的 v45 / v47 训练源码，故其依赖按 `requirements.txt` 一并安装。

### 5.3 复现命令
从代码仓库根目录运行。规则臂不需要任何 checkpoint，可直接复现：

```bash
# 创建并激活虚拟环境
uv venv experiments/v48/.venv
source experiments/v48/.venv/bin/activate
uv pip install -r requirements.txt

# 运行语义与解耦测试
python experiments/v48/tests/test_training_semantics.py
python experiments/v48/tests/test_env_decomposability.py

# 规则臂评测（无 checkpoint，reference 设为 rule_hold）
python experiments/v48/evaluate.py --seeds report --reference rule_hold --workers 8 \
  --arm rule_hold \
  --arm rule_accelerate \
  --arm rule_turn_left \
  --arm rule_random \
  --arm rule_decelerate \
  --arm rule_flee \
  --arm rule_flee_lead \
  --arm rule_safe_top \
  --out experiments/v48/artifacts/results/rule_only.jsonl

# 后处理：汇总各阶段 summary，并做分时/阶段分析
python experiments/v48/summarize_results.py
python experiments/v48/phase_analysis.py \
  --per-episode experiments/v48/artifacts/results/phase_report2.jsonl \
  --out experiments/v48/artifacts/results/phase_analysis.json
```

PPO 臂需要额外提供历史 checkpoint，本阶段候选未包含该权重：

```bash
# P0：仅复现 PPO 臂需要额外 checkpoint（本阶段候选不含该文件）
python experiments/v48/evaluate.py --seeds report --reference policy_per_fish --workers 8 \
  --arm policy_per_fish=experiments/v45/artifacts/checkpoints/ms_baseline_v42cfg_seed700000/model_iter_20.zip \
  --arm policy_broadcast=experiments/v45/artifacts/checkpoints/ms_baseline_v42cfg_seed700000/model_iter_20.zip \
  --out experiments/v48/artifacts/results/report_ppo.jsonl
```

评测 PPO 臂依赖本地已存在的 checkpoint 文件 `experiments/v45/artifacts/checkpoints/ms_baseline_v42cfg_seed700000/model_iter_20.zip`。该权重文件的公开获取可用性未确认，复现依赖本地文件已就位；仅安装基础下限依赖无法保证与历史数字精确一致的复现。
