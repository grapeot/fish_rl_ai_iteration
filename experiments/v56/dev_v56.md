# 开发者记录 v56：冻结 PPO 对捕食者初始速度缩放的泛化敏感度（第 8 轮，共 10 轮）

## 1. 实验设问与干预协议

本轮评估已冻结的 PPO 策略在测试期对捕食者（大鱼）初始速度缩放的泛化敏感度，量化把捕食者 post-reset 初速调慢或调快 25% 对鱼群存活率的边际影响。评测用已通过验收的 3 个 v50 修正版 `survival_only` FINAL 检查点（每模型 200 updates、614,400 steps），评测过程无二次训练。本轮不依赖第 7 轮（v55）结果，不使用 v54 或 v53，也不使用 `original` 臂。

评测在 40 个全新独立场景（主评测种子 560102，调试种子 560101）中展开，标准设定为 96 条鱼、500 步时长、11 维观测，邻居感知关闭。评测开始前先把三个模型路径、`policy_tensor_sha256`、世界配置与 seed bank 冻结进清单，并对照已合并的 v50 corrected selection manifest 交叉核对。

干预机制：

- 环境确定性 reset 后，快照 post-reset 世界（标称世界：`initial_escape_boost=True`、`escape_boost_speed=0.8`，含完整捕食者 heading/speed bias 与 pre-roll，每个场景的标称捕食者初速均值约 1.30）。
- 仅对捕食者初始速度向量乘以标量因子（0.75× / 1.0× / 1.25×），随后重算观测。
- 因子为正标量，方向（单位向量）保持，速度大小按比例缩放，1.0× 为精确恒等。
- 鱼体位置/速度、`fish_alive`、捕食者位置、timestep、pre-roll trace、env RNG 状态恢复后逐位一致。
- 未采用整回合恒定速度、也未改变捕食者最大速度；只对 reset 之后那一帧的捕食者速度向量做缩放，后续重力/反弹/碰撞物理不变，因此速度本身会按新初值演化出不同轨迹。
- 未重抽自然初速分布（那会改变 pre-roll 抽取序列与 RNG 消耗）。

零范数保护与术语：标量缩放（乘以正因子）在零速度处数学上仍有定义（结果仍为零）；无定义的是归一化方向（单位向量）与 applied/nominal 速度比，因为要除以范数。实现上当标称范数 ≤ 1e-12 时保留原向量（若输入的范数极小但不为零，也保留原向量），打 `zero_predator_norm` 标记，不做任何除法。40×6×3=720 个 episode 中零范数计数为 0；语义测试用合成零范数快照验证该保护保持零向量、置标记且不产生 NaN/inf。该术语问题不影响任何已报数值。

## 2. 评测规模与核心结果

评测覆盖 6 种控制器（3 个冻结 PPO 确定性模型、`rule_flee_lead`、`rule_safe_top`、`rule_hold` 基线）× 3 种速度条件 × 40 个场景，共计 720 个 episode。评测由 4 个 worker 并行执行，总耗时 1390 秒（约 23.2 分钟）。

3 个 PPO 模型平均最终存活率（每场景先对固定三 PPO 平均，再对 40 场景汇总）：

- 标称（1.0×）：0.92777778
- 慢速（0.75×）：0.92569444（较标称 **−0.20833 pp**，95% CI [−0.52951, +0.09549] pp）
- 快速（1.25×）：0.93064236（较标称 **+0.28646 pp**，95% CI [+0.03472, +0.54687] pp）

群体死亡率：标称 7.2222%、慢速 7.4306%、快速 6.9358%。

统计推断方法：置信区间由 10,000 次 episode 级 **two-sided** percentile bootstrap（随机种子 560301）计算，是 95% 双侧区间，不是单侧显著性检验。统计单位是 episode（场景），不把 96 条鱼当作独立样本；该估计固定了已有的 3 个模型，不反映模型训练过程的不确定性，也不是多重对比的同时区间。

逐模型副本（慢速 / 快速相对自身标称，pp）：rep0 −0.23437 / +0.28646；rep1 −0.07812 / +0.02604；rep2 −0.31250 / +0.54688。三个副本方向一致（慢速均略降、快速均略升），但敏感度不同，估计与区间异质。仅 rep2 的快速效应双侧 CI 不含 0（[+0.130208, +0.989583] pp）。区间不含 0 只说明在固定模型下点估计方向稳定，本身不等于实际重要性；区间跨 0 也不等于无影响。不能笼统地说每个副本的效应都落在噪声范围内。

## 3. 速度稳健性对比与归因边界

从原始 records 直接计算（非从 summary 复制），逐场景对固定三 PPO 求平均相对标称的差，再对 40 场景做配对 bootstrap：

- 慢速效应（0.75× − 1.0×）：**−0.20833 pp**，95% CI [−0.52951, +0.09549] pp（18 正 / 2 零 / 20 负）
- 快速效应（1.25× − 1.0×）：**+0.28646 pp**，95% CI [+0.03472, +0.54687] pp（23 正 / 1 零 / 16 负）
- 快慢不对称（快速效应 − 慢速效应）：**+0.49479 pp**，95% CI [+0.13889, +0.86806] pp（23 正 / 6 零 / 11 负）

归因边界：

- 数值范围必须收窄到这三个聚合 PPO **点估计**：慢速 −0.208333 pp、快速 +0.286458 pp、快慢不对称 +0.494792 pp。这个“小于 0.6 pp”的说法只适用于这三个聚合点估计，不适用于所有区间边界、不适用于单个模型副本、不适用于规则控制器、也不适用于单个场景。反例：rep2 快速 − 标称点估计 +0.546875 pp、区间上界 +0.989583 pp、快 − 慢点估计 +0.859375 pp；HOLD 慢速 − 标称 +0.729167 pp，safe_top 慢速 − 标称 −0.729167 pp；单场景层面慢 − 标称在 −3.125 到 +1.388889 pp 之间、快 − 标称在 −1.388889 到 +2.083333 pp 之间。
- 不能由此推断速度无关、等价（equivalence）或影响可忽略。区间不含 0 不等于实际重要性；区间跨 0 也不等于无影响。这是在这 40 个场景、3 个固定 FINAL 模型、post-reset 初速 ±25% 缩放范围内的一个小幅聚合均值变化描述。
- 快慢不对称是配对结果对比，不是因果分解；不能用两个独立 CI 相减代替。
- 单个模型副本的效应异质，三模型的方向一致性是描述性的，不代表训练随机性下的总体分布。

### 相对 HOLD 基准（新增，从原始数据计算，无重评）

对每个场景与条件，定义优势为固定三 PPO 的平均存活率减去同场景同条件的 HOLD 存活率；对该 40 元配对优势向量做同样的 two-sided percentile bootstrap（seed 560301）。最后一行对优势的配对差直接 bootstrap，不用两个区间端点相减：

- PPO − HOLD，0.75×：**+17.100694 pp**，95% CI [+15.616319, +18.645833] pp（40/0/0）
- PPO − HOLD，1.00×：**+18.038194 pp**，95% CI [+16.397569, +19.661458] pp（40/0/0）
- PPO − HOLD，1.25×：**+18.663194 pp**，95% CI [+16.788194, +20.546875] pp（40/0/0）
- 优势变化（1.25× 优势 − 1.00× 优势）：**+0.625000 pp**，95% CI [−0.946398, +2.326389] pp

三个条件下 PPO 都超过 HOLD。但快速条件的优势变化区间同时覆盖下降与上升，因此**不能**由快速条件的 PPO 绝对存活率上升推断其对 HOLD 的相对优势可靠扩大。HOLD 只是语境锚点，不是这里最强的规则：`rule_flee_lead` 在三个条件下的平均存活率都高于 PPO。这些基准对比是补充的原始数据推导分析，不是预设主终点。

## 4. 规则基线与时序动态

规则基线存活率均值（慢速 / 标称 / 快速）：

- HOLD：0.75469 / 0.74740 / 0.74401
- `rule_flee_lead`：0.97422 / 0.97188 / 0.97057
- `rule_safe_top`：0.91693 / 0.92422 / 0.92656

上述数值是各规则在各速度下的客观均值。它们之间的差值都是描述性观测，不是 equivalence 检验，也不能据此称某规则对速度扰动免疫或各规则等价。

阶段存活率时序（PPO 三模型平均）：

- 慢速：第 1 步 0.981944，第 100 步 0.926128，第 250 步 0.925694，第 500 步 0.925694。
- 标称：第 1 步 0.981684，第 100 步 0.928038，第 250 步 0.927865，第 500 步 0.927778。
- 快速：第 1 步 0.981684，第 100 步 0.931163，第 250 步 0.930642，第 500 步 0.930642。

三条件的死亡大多在第 100 步前形成，随后基本持平。以上只覆盖第 1/100/250/500 四个采样端点的聚合值，最大的条件间端点差在第 100 步约 0.503472 pp；这只支持端点层面的描述，不能推断每个中间步或整条轨迹都有同等小的界限。

## 5. 测试覆盖与数据契约审计

测试套件包含 26 个测试函数/检查（11 项语义测试函数，15 项准入验收测试函数），全部通过；每个函数内部含多项断言。测试覆盖有明确边界：

- 标称直测覆盖全部三个冻结模型在同一场景下的完整 episode，逐位比较动作序列与最终存活。
- 世界一致性测试断言位置/存活/捕食者位置/速度方向一致性、速度精确按因子缩放、pre-roll 与 RNG 未来一致。
- 观测测试在捕食者可见时逐条断言 `obs[8], obs[9]` 等于缩放后速度；不可见时断言通道 5..10 为零/不可见编码，捕获不可见编码仍为 0，不泄漏缩放速度。
- 零范数测试用合成零范数快照，断言三条件均保留零向量、置标记且不产生 NaN/inf。
- 统计单位测试用一个合成网格验证按场景聚合、快慢不对称等于两个配对效应之差，以及 PPO−HOLD 同条件优势与优势差的计算；门禁中另有廉价的不完整 grid 拒绝检查。
- 未对全部 40 场景 × 全部条件做穷尽逐位核验；上述为代表性单场景与合成聚合检查。

数据契约：

- 记录中的 `nominal_predator_speed` 是干预前的标称捕食者速度（三条件相同，均值约 1.30），是 pre-intervention 字段；`applied_predator_speed` 是缩放后的实际状态速度（均值约 0.975 / 1.300 / 1.625），有效值从状态读取，不从动作或标称字段推断。
- 配对分析核对为完整 40 个场景，单元格数据闭合，无丢弃 episode；分析脚本会用原始 records 复算校验 summary 中的均值，不一致即中止。
- 冻结机制在评测流程启动前生成并记录清单，并交叉核对已合并的 v50 manifest；它提供 `--allow-overwrite`，不是绝对密码学防篡改。
- 门禁独立校验条件集合与 speed factor map（不只依赖 common 常量），factor 值不符即拒绝。
- 运行期源码哈希子集保持原值不变、不刷新。需要说明：`results/post_run_source_hashes.json` 的覆盖范围只有评估时导入的 **7 个文件**（`v56_common.py`、`v56_env.py`、`v56_evaluate.py`、`v56_verify.py`、`v50_common.py`、`v50_env.py`、`fish_env.py`），**不含** `v56_freeze.py`、`v56_report.py`、`v56_analyze.py`、`v56_snapshot.py`、两个测试文件与 dev，也不是全部 v56 工件。它记录的是运行关联（end-of-run）哈希，不是开始前的加密锁定；完整审阅集合的准确性由本轮的候选文件 SHA-256 清单覆盖。
- 本轮为纯评测：在运行后新增的是从原始 records 计算的补充分析（`results/relative_hold_analysis.json`）与文案修订，没有重跑 720 episode，原始 `report.jsonl`、`report.summary.json`、`analysis.json` 字节保持不变；运行期与运行后源码哈希不因文案/分析脚本改动而刷新或回写。

## 6. 复现说明与局限性

工件位于 `experiments/v56/`：

- 核心代码：`v56_common.py`、`v56_env.py`、`v56_verify.py`、`v56_freeze.py`、`v56_evaluate.py`、`v56_report.py`、`v56_analyze.py`、`v56_snapshot.py`
- 测试：`tests/test_v56_semantics.py`、`tests/test_v56_acceptance_gates.py`
- 配置：`artifacts/frozen_config/v56_frozen_manifest.json`
- 结果：`artifacts/results/` 下的 `report.jsonl`、`report.summary.json`、`analysis.json`、`speed_robustness.json`、`relative_hold_analysis.json`、`post_run_source_hashes.json`、`run_provenance.json`

复现需在仓库根目录（fish_rl repo）执行，输出写入新的 `reproduction/` 目录，不覆写既有冻结归档：

```bash
# 0. 环境
source experiments/v48/.venv/bin/activate

# 1. 测试（语义 11 项 + 验收门禁 15 项）
python experiments/v56/tests/test_v56_semantics.py
python experiments/v56/tests/test_v56_acceptance_gates.py

# 2. 冻结三个模型（路径 + policy tensor hash）先于任何结果
python experiments/v56/v56_freeze.py \
  --out experiments/v56/artifacts/reproduction/frozen_config/v56_frozen_manifest.json

# 3. 评测（门控于冻结 manifest；40 场景 × 6 控制器 × 3 条件）
python experiments/v56/v56_evaluate.py --seeds report --workers 4 \
  --arm rep0_ppo=experiments/v50/artifacts/corrected_streams/runs/rep0_survival_only/checkpoints/model_final.zip \
  --arm rep1_ppo=experiments/v50/artifacts/corrected_streams/runs/rep1_survival_only/checkpoints/model_final.zip \
  --arm rep2_ppo=experiments/v50/artifacts/corrected_streams/runs/rep2_survival_only/checkpoints/model_final.zip \
  --arm rule_hold --arm rule_flee_lead --arm rule_safe_top \
  --out experiments/v56/artifacts/reproduction/results/report.jsonl \
  --require-manifest experiments/v56/artifacts/reproduction/frozen_config/v56_frozen_manifest.json
# 或经编排器：v56_report.py --manifest <manifest> --out <report.jsonl> --seeds report --workers 4

# 4. 分析（含原始数据速度稳健性视角、PPO−HOLD 基准视角与 summary-vs-raw 一致性校验）
#    这三个 --out 都写到新的 reproduction/ 目录，不覆写既有冻结结果
python experiments/v56/v56_analyze.py \
  --report-summary experiments/v56/artifacts/reproduction/results/report.summary.json \
  --report-jsonl experiments/v56/artifacts/reproduction/results/report.jsonl \
  --manifest experiments/v56/artifacts/reproduction/frozen_config/v56_frozen_manifest.json \
  --out experiments/v56/artifacts/reproduction/results/analysis.json \
  --speed-view-out experiments/v56/artifacts/reproduction/results/speed_robustness.json \
  --relative-hold-out experiments/v56/artifacts/reproduction/results/relative_hold_analysis.json

# 4b.（可选）只补 PPO−HOLD 基准视角，不重跑评测：同一 v56_analyze.py，指定
#     --relative-hold-out 即可；它只读原始 report，不修改 report/summary/analysis。

# 5. 运行后源码快照（scope 标注 post-run；覆盖范围见第 5 节，仅评估导入的 7 个文件）
python experiments/v56/v56_snapshot.py \
  --out experiments/v56/artifacts/reproduction/results/post_run_source_hashes.json
```

复现局限性：代码仓库不含预训练模型权重（公共候选不含 zip）。复现依赖本地已有的 v50 修正检查点——即 `experiments/v50/artifacts/corrected_streams/runs/rep{0,1,2}_survival_only/checkpoints/model_final.zip`，属于 zip 形式的本地依赖，不提供训练阶段的逐位一致性保证，也不附带模型权重许可证。本轮为纯评测：模型权重与 720 episode 原始数据不因文案或注释修改而重跑；运行后新增的补充分析（`relative_hold_analysis.json`）仅从原始 `report.jsonl` 计算，不修改原始记录。所有数值限定于 seed bank 560102 与这一固定模型集。
