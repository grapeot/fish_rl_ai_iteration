# Round 7 v55 评测记录：捕食者初速度方向敏感性分析

## 实验概述与评测对象

本记录针对 Round 7（v55）开展捕食者初速度方向敏感性评测，评估冻结策略及基准规则在环境重置后的行为表现。

- **模型对象**：3 个已冻结的 PPO v50 修正版模型（`survival_only` 阶段最终第 200 次更新，累计 614,400 步）。
- **训练性质**：本轮为纯冻结评估，不包含任何微调或追加训练。
- **基准控制器**：3 个 PPO 模型（rep0、rep1、rep2）与 3 个规则基线（`rule_flee_lead`、`rule_safe_top`、`rule_hold`），共 6 个控制器。

## 评测配置与干预协议

- **评测场景**：40 个全新评测场景（种子序列 550102），另有 2 个调试场景（550101）仅用于吞吐测试，不进统计。每场景配置 96 条鱼，单局 500 步，关闭 11 维邻域特征。
- **旋转干预**：在标准 reset 完成后，仅对捕食者初速度向量 `predator_vel` 应用带符号置换矩阵，执行 0°、90°、180°、270° 旋转。
- **范数与恒等性**：旋转矩阵严格保持向量范数不变，0° 旋转严格为单位矩阵。
- **环境状态不变性**：鱼群初始状态、捕食者初始位置、环境时间步、RNG 内部状态及 pre-roll 完全保持不变，仅重算当前观测向量。
- **重力方向固定**：重力加速度在所有条件下严格固定为世界坐标 +y 方向，不随初速度做任何偏转。
- **干预边界说明**：本干预不是自然朝向重采样，不改变重力方向，也不改变速度标量大小。固定 +y 重力、固定初始位置与边界碰撞下，仅旋转 reset 后的初速度不构成后续整条捕食者轨迹的刚性旋转。
- **配对与计算开销**：各控制器在同一场景下均以自身 0° 结果为基线配对。6 控制器 × 4 方向 × 40 场景 = 960 局，4 工作进程总耗时 1838 秒。

## 评测数据与统计分析

统计方法采用两阶段计算：先在各场景内计算 3 个 PPO 模型的平均存活率，再通过 10,000 次场景 bootstrap 计算配对差值及 95% 名义置信区间。bootstrap 参数为固定种子 550301、percentile 区间；该区间为固定模型上的条件性名义区间，不是训练期置信区间。

| 控制器 / 评测组 | 0° (基准) | 90° | 180° | 270° |
| :--- | :--- | :--- | :--- | :--- |
| PPO (3 模型均值) | 0.9287 | 0.9235 | 0.8977 | 0.9176 |
| PPO 配对差值 vs 0° | 基准 | -0.0053 [-.0130, +.0024] | -0.0311 [-.0413, -.0207] | -0.0112 [-.0206, -.0020] |
| 规则基准: lead | 0.971 | 0.969 | 0.963 | 0.970 |
| 规则基准: safetop | 0.929 | 0.907 | 0.882 | 0.916 |
| 规则基准: HOLD | 0.755 | 0.746 | 0.746 | 0.765 |

同条件被动参照（事后补充）：对每个场景先平均 3 个 PPO，再减去**同一条件**下 `rule_hold` 的存活率，得到 PPO−HOLD 优势，再做配对 episode bootstrap。

| 同 bank 参照 | 均值 | Paired 95% CI |
| :--- | ---: | :--- |
| PPO−HOLD，0° | +0.1738 | [+0.1586, +0.1900] |
| PPO−HOLD，90° | +0.1771 | [+0.1603, +0.1944] |
| PPO−HOLD，180° | +0.1513 | [+0.1365, +0.1666] |
| PPO−HOLD，270° | +0.1527 | [+0.1423, +0.1633] |
| DiD：优势 180°−0° | -0.0225 | [-0.0391, -0.0063] |

## 结果特征与讨论

### 1. PPO 方向敏感性与异质性

- **180° 下降趋势**：180° 是四个**已测**点中的最差点，PPO 三模型聚合平均存活率下降 3.1 个百分点（-0.0311，95% CI [-0.0413, -0.0207]，40 场景中 32 个为负）。该 CI 是逐 contrast 的条件性区间，不是连续角度最差值的置信带。
- **模型间表现差异**：3 个 PPO 模型在 180° 下的配对差值分别为 -0.0003、-0.0216 与 -0.0714。若按本 bank、固定三模型、180° 相对各自 0° 的均值下降分解，rep2 约占 76.5%（约 7.14pp / 9.33pp），但 rep1 仍贡献约 23.2%、且其自身下降 CI 不含 0；这是均值下降分解，不是逐 episode 损失份额或训练总体贡献率。
- **统计解释边界**：rep0 差值虽接近于零且置信区间跨零，仅表明当前样本下未检出方向效应证据，不能判定为免疫或等价；没有预设的等价界限或等价检验。90° 聚合 CI 跨零同样不能写成没有影响。
- **覆盖局限**：测试的 4 个离散旋转点不代表连续 360° 覆盖，也不能代表自然朝向的分布外情况。

### 2. 规则控制器与物理动态

- **safetop 观测特征**：`rule_safe_top` 的目标点由 `r=0.55, angle=-90°` 给出，在世界坐标约为 **(0, -5.5)**，归一化 observation 坐标约为 **(0, -0.55)**，位于 -y 方向。safetop 在 180° 下录得 4.7 个百分点降幅（0.929 → 0.882，CI [-0.0677, -0.0297]），是方向敏感性最强的规则。
- **机制归因克制**：规则函数不读取重力方向，不能把下降写成依赖 gravity 方向的已证机制。固定目标与旋转后捕食者路径的相对几何可能解释损失，但本轮没有逐帧轨迹、命中位置或目标随条件旋转的对照，因此 fixed-world-target 致损仅列为**假设**，不排除后期转向控制的影响。
- **捕食者运动特性**：捕食者受固定 +y 重力影响的反弹运动是非平稳过程。
- **早期死亡分布**：3 个 PPO 模型在 40 个场景下的第 1 步死亡鱼数总计分别为 219（0°）、243（90°）、259（180°）和 238（270°）。所有死亡均保留在统计中。聚合差异主要在前 100 步形成（180° 相对 0° 在 step 1 为 -0.0035，到 step 100 已达 -0.0311，之后基本平台），但该时间分布不证明首步立刻撞鱼，也不排除后期转向控制的影响。
- **HOLD 基准对照**：HOLD 自身的 180°−0° 为 -0.0086（CI [-0.0266, +0.0102]）。PPO 绝对下降 3.1pp，扣除 HOLD 条件变化后相对优势缩小 2.25pp（DiD）。PPO 在这四个点仍优于同条件 HOLD，这是可支持的对照结果，不等于方向稳健的证明；DiD 是这个特定被动参照下的优势变化，不是对所有环境难度的因果消除。

### 3. 补充分析口径

同条件 PPO−HOLD 与 DiD 是在既有 raw 记录上补算的事后（post-hoc）对照分析，不启动新评测、不改模型/bank/阈值/评测记录，也不能追认成 pre-run primary 指标。

## 代码实现与模块构成

评测代码位于 `experiments/v55/` 目录下，主要组件包括：

- `v55_common.py`：公用配置、四个旋转矩阵与种子库。
- `v55_env.py`：post-reset 快照与初速度旋转、观测重算。
- `v55_verify.py`：模型绑定与配对记录一致性校验。
- `v55_freeze.py`：模型与配置冻结清单生成。
- `v55_evaluate.py`：多进程评测运行器。
- `v55_report.py`：按冻结清单编排 report 评测。
- `v55_analyze.py`：场景配对 bootstrap 与（可选）PPO−HOLD 事后对照输出。
- `v55_snapshot.py`：代码运行时哈希快照提取。
- `tests/`：`test_v55_semantics.py` 与 `test_v55_acceptance_gates.py`。

## 产物存证与测试验证

- **配置与结果清单**：
  - 冻结清单：`artifacts/frozen_config/v55_frozen_manifest.json`
  - 评测结果：`results/report.jsonl`、`results/report.summary.json`、`results/analysis.json`
  - 事后对照：`results/relative_hold_analysis.json`
  - 溯源存证：`results/post_run_source_hashes.json`、`results/post_review_source_hashes.json`、`results/run_provenance.json`
- **模型权重说明**：PPO 权重依赖本地 v50 检查点，模型 zip 包不公开分发。
- **自动化测试**：26 项自动化测试全部通过（10 项语义测试 + 16 项验收门禁）。测试覆盖矩阵范数保持、0° 恒等变换与整段 rollout 复现、重力未旋转、状态观测一致性、RNG/种子库隔离、朝向偏置绑定、遗漏角度/缺失种子/未知控制器/重复单元的拒绝，以及聚合与 DiD 的手工合成校验。单点测试通过不构成全步长形式证明。

## 复现指南与作者提示

冻结清单先于任何结果生成；report 评测门控于该清单。从仓库根目录执行以下命令（`--out` 指向独立输出目录，避免覆盖既有评测结果）：

```bash
# 1. 冻结三模型（路径 + policy tensor hash），先于任何结果
experiments/v48/.venv/bin/python experiments/v55/v55_freeze.py \
  --out experiments/v55/artifacts/frozen_config/v55_frozen_manifest.json

# 2. 测试套件
experiments/v48/.venv/bin/python experiments/v55/tests/test_v55_semantics.py
experiments/v48/.venv/bin/python experiments/v55/tests/test_v55_acceptance_gates.py

# 3. report 评测（门控于冻结清单；输出到全新目录）
experiments/v48/.venv/bin/python experiments/v55/v55_report.py \
  --manifest experiments/v55/artifacts/frozen_config/v55_frozen_manifest.json \
  --out <new_outdir>/report.jsonl --seeds report --workers 4

# 4. 分析（含可选的事后 PPO−HOLD 对照输出）
experiments/v48/.venv/bin/python experiments/v55/v55_analyze.py \
  --report-summary <new_outdir>/report.summary.json \
  --report-jsonl <new_outdir>/report.jsonl \
  --manifest experiments/v55/artifacts/frozen_config/v55_frozen_manifest.json \
  --out <new_outdir>/analysis.json \
  --relative-out <new_outdir>/relative_hold_analysis.json
```

> **作者说明**：复现评测前，请核实 `experiments/v55/v55_evaluate.py` 与 `v55_report.py` 的 argparse 参数定义并指定独立输出目录，避免覆盖既有评测结果。`report.summary.json` 与 `run_provenance.json` 中嵌入的是运行时的 7 项评测依赖源码哈希，不被任何事后修改刷新。
