# 开发者记录 v54：冻结 PPO 对初始游动速度缩放的泛化敏感度（第 6 轮，共 10 轮）

## 1. 实验设问与干预协议

本轮评估已冻结的 PPO 策略在测试期对鱼群初始游动速度缩放的泛化敏感度，量化初速削减对群体存活率的边际影响。评测用已通过验收的 3 个 v50 修正版 `survival_only` FINAL 检查点（每模型 200 updates、614,400 steps），评测过程无二次训练。本轮不依赖第 5 轮（v53），不使用旧 pilot，也不使用 `original` 臂。

评测在 40 个全新独立场景（主评测种子 540102，调试种子 540101）中展开，标准设定为 96 条鱼、500 步时长、11 维观测，邻居感知关闭。评测开始前先把三个模型路径、`policy_tensor_sha256`、世界配置与 seed bank 冻结进清单，并对照已合并的 v50 corrected selection manifest 交叉核对。

干预机制：

- 环境确定性 reset 后，快照 post-reset 世界（标称 boost 世界：`initial_escape_boost=True`、`escape_boost_speed=0.8`，鱼初速约 1.6，接近径向）。
- 仅缩放鱼体初始速度向量（1.0× / 0.5× / 0.0×），随后重算观测。
- 鱼体位置、`fish_alive`、捕食者位置/速度、pre-roll trace、env RNG 状态恢复后逐位一致。
- 未采用 `initial_escape_boost=False`：该标志会改变 reset 分布与 RNG 消耗序列，而本干预始终基于同一个 boosted 名义世界，只在 reset 之后缩放速度向量。

动力学与因果解释：

速度归零会清除鱼体的速度航向信息，同时把观测的速度通道 `obs[2], obs[3]` 归零。在环境动力学规则下，零速度时转向（turn）、减速（decelerate）、保持（HOLD）均不产生速度或位移，必须先执行前向加速沿 +x 建立速度后才能转向机动。此限制限定在 post-reset 首次非前向动作；一旦前向动作产生非零速度，之后仍可转向，因此不能描述为整回合无法转向。因此，该单轴状态干预同时改变了动量、速度观测、可恢复的朝向信息与首步有效控制能力，它估计的是这项 post-reset 干预的总效应，不能单独估计“纯动量赠送”的因果份额，也不等价于另一套 reset 分布的 `initial_escape_boost=False`。

## 2. 评测规模与核心结果

评测覆盖 6 种控制器（3 个冻结 PPO 确定性模型、`rule_flee_lead`、`rule_safe_top`、`rule_hold` 基线）× 3 种速度条件 × 40 个场景，共计 720 个 episode。评测由 4 个 worker 并行执行，总耗时 1386 秒。

3 个 PPO 模型平均存活率：

- 标称（1.0×）：0.93376736
- 减半（0.5×）：0.92378472（较标称 **−0.998264 pp**，95% CI [−1.440972, −0.564236] pp）
- 归零（0.0×）：0.88394097（较标称 **−4.982639 pp**，95% CI [−6.284939, −3.862847] pp）

群体死亡率：

- 整体死亡率由标称条件的 6.6233% 上升至归零条件的 11.6059%。
- 相对增幅为 75.23%（该比例为点估计描述性比率，非置信区间）。
- 3 个模型副本在归零条件下的降幅分别为 −7.96875 pp、−1.354167 pp 和 −5.625 pp（三者方向一致，敏感度不同）。

统计推断方法：置信区间由 10,000 次 episode 级 bootstrap（随机种子 540301）计算。统计单位是 episode（场景），不把 96 条鱼当作独立样本；该估计固定了已有的 3 个模型，不反映模型训练过程的不确定性，也不延伸成重新训练模型总体的 CI。

## 3. 相对基准对比与归因边界

在相同原始数据下，逐场景先对固定三 PPO 求平均 PPO 存活率，再减去同场景配对的 HOLD 存活率，得到各条件的优势，并对 40 个场景做配对 bootstrap：

- 标称初速优势（PPO − HOLD）：**+17.934028 pp**，95% CI [+16.215278, +19.626953] pp（40/0/0 正/零/负）
- 归零初速优势（PPO − HOLD）：**+1.831597 pp**，95% CI [+1.189236, +2.378472] pp（38/0/2）
- 差中差（归零优势 − 标称优势）：**−16.102431 pp**，95% CI [−17.855903, −14.279514] pp（0/0/40）

归因边界：

- 差中差是配对结果对比，不是因果中介分解。被动 HOLD 基线的存活率也随干预改变，在归零条件下提高，因此不把差中差解释为“初速贡献了约 16 pp 策略能力”。其区间也不能用两个独立 CI 相减代替。
- 归零条件仍有 0.88394 的绝对存活率，这是事实描述，不能据此推出策略不依赖初速、策略贡献已被隔离。
- 绝对存活率下降约 5 pp 与相对 HOLD 优势缩减约 16 pp 同时成立。仅凭较高的绝对存活率或约 5 pp 的降幅，不足以声称存活与初速无关。

## 4. 规则基线、时序动态与行为特征

规则基线存活率均值（标称 / 减半 / 归零）：

- HOLD：0.75442708 / 0.83307292 / 0.86562500
- `rule_flee_lead`：0.97343750 / 0.97213542 / 0.96015625
- `rule_safe_top`：0.92552083 / 0.92656250 / 0.92395833

上述数值是各规则在各速度下的客观均值。它们之间的差值都是描述性观测，不是 equivalence 检验，也不能据此称某规则对初速扰动免疫或各规则等价。

捕食者与物理动态：捕食者受重力与边界反弹机制驱动，不是静止实体。本轮评测未记录鱼群或捕食者轨迹，不能推断鱼群被甩向捕食者。

阶段存活率时序（PPO 三模型平均）：

- 标称：第 1 步 0.98133681，第 100 步 0.93437500，第 250 步 0.93394097，第 500 步 0.93376736。
- 归零：第 1 步 0.97916667，第 100 步 0.90850694，第 250 步 **0.88654514**，第 500 步 **0.88394097**。

多数归零−标称差到第 250 步已形成，可描述为损失集中在前段；第 250 步至第 500 步仍有继续死亡。现有 phase counts 不足以证明损失源于快速远离能力，更不能排除转向控制的作用。

策略动作分布：归零条件下，模型副本 2 的左转占比达 99.4581%、前向占比为 0%；副本 0 的右转占比约 92.12%。这些只描述动作输出分布的变化。鉴于零速度下转向为空操作，大量转向不能称为有效主动规避，也不能用输出分布漂移解释保留的存活率；没有轨迹核验时，不能把这类输出变化升格为已证明的 episode 级行为机制或死亡因果。

## 5. 测试覆盖与数据契约审计

既有测试套件包含 23 项断言（7 项语义检查，16 项准入验收检查），全部通过。测试覆盖有明确边界：

- 标称直测覆盖全部三个冻结模型在同一场景下的完整 episode，逐位比较动作序列与最终存活。
- 零速语义测试逐一执行全部五个动作（0/1/2/3/4）。
- 世界一致性测试断言位置/存活/捕食者/pre-roll 一致、速度精确缩放、RNG 未来一致。
- 统计单位测试用一个合成网格验证按场景聚合（先对固定三 PPO 求平均再配对 HOLD）与差中差等于两个配对优势之差。
- 未对全部 40 场景 × 全部条件做穷尽逐位核验；上述为代表性单场景与合成聚合检查。

数据契约：

- 汇总报告中的 `mean_init_fish_speed` 记录的是缩放前的标称初速（三条件均约 1.6），是干预前字段，不是干预后的实际速度。
- 配对分析核对为完整 40 个场景，单元格数据闭合，无丢弃 episode；分析脚本会用原始 records 复算校验 summary 中的均值，不一致即中止。
- 冻结机制在评测流程启动前生成并记录清单，并交叉核对已合并的 v50 manifest；它提供 `--allow-overwrite`，不是绝对密码学防篡改。
- 门禁独立校验条件集合与 velocity factor map（不只依赖 common 常量），factor 值不符即拒绝。
- 运行期源码哈希子集保持原值不变，运行后源码哈希（含本次 docstring/分析脚本改动）独立记录于 `results/post_review_source_hashes.json`，范围明确标注为 post-run analysis、not rerun，不刷新或伪造历史 runtime hash。

## 6. 复现说明与局限性

工件位于 `experiments/v54/`：

- 核心代码：`v54_common.py`、`v54_env.py`、`v54_verify.py`、`v54_freeze.py`、`v54_evaluate.py`、`v54_report.py`、`v54_analyze.py`、`v54_snapshot.py`
- 测试：`tests/test_v54_semantics.py`、`tests/test_v54_acceptance_gates.py`
- 配置：`artifacts/frozen_config/v54_frozen_manifest.json`
- 结果：`artifacts/results/` 下的 `report.jsonl`、`report.summary.json`、`analysis.json`、`relative_hold_analysis.json`、`post_run_source_hashes.json`、`post_review_source_hashes.json`、`run_provenance.json`

复现需在仓库根目录（fish_rl repo）执行，输出写入新的 `reproduction/` 目录，不覆写既有冻结归档：

```bash
# 0. 环境
source experiments/v48/.venv/bin/activate

# 1. 测试（语义 7 项 + 验收门禁 16 项）
python experiments/v54/tests/test_v54_semantics.py
python experiments/v54/tests/test_v54_acceptance_gates.py

# 2. 冻结三个模型（路径 + policy tensor hash）先于任何结果
python experiments/v54/v54_freeze.py \
  --out experiments/v54/artifacts/reproduction/frozen_config/v54_frozen_manifest.json

# 3. 评测（门控于冻结 manifest；40 场景 × 6 控制器 × 3 条件）
python experiments/v54/v54_evaluate.py --seeds report --workers 4 \
  --arm rep0_ppo=experiments/v50/artifacts/corrected_streams/runs/rep0_survival_only/checkpoints/model_final.zip \
  --arm rep1_ppo=experiments/v50/artifacts/corrected_streams/runs/rep1_survival_only/checkpoints/model_final.zip \
  --arm rep2_ppo=experiments/v50/artifacts/corrected_streams/runs/rep2_survival_only/checkpoints/model_final.zip \
  --arm rule_hold --arm rule_flee_lead --arm rule_safe_top \
  --out experiments/v54/artifacts/reproduction/results/report.jsonl \
  --require-manifest experiments/v54/artifacts/reproduction/frozen_config/v54_frozen_manifest.json
# 或经编排器：v54_report.py --manifest <manifest> --out <report.jsonl> --seeds report --workers 4

# 4. 分析（含原始数据相对 HOLD 视角与 summary-vs-raw 一致性校验）
python experiments/v54/v54_analyze.py \
  --report-summary experiments/v54/artifacts/reproduction/results/report.summary.json \
  --report-jsonl experiments/v54/artifacts/reproduction/results/report.jsonl \
  --manifest experiments/v54/artifacts/reproduction/frozen_config/v54_frozen_manifest.json \
  --out experiments/v54/artifacts/reproduction/results/analysis.json \
  --relative-hold-out experiments/v54/artifacts/reproduction/results/relative_hold_analysis.json

# 5. 运行后源码快照（scope 标注 post-run，不覆盖原 post_run_source_hashes.json）
python experiments/v54/v54_snapshot.py \
  --out experiments/v54/artifacts/reproduction/results/post_run_source_hashes.json
```

复现局限性：代码仓库不含预训练模型权重（公共候选不含 zip），复现依赖本地已有的 v50 修正检查点，不提供训练阶段的逐位一致性保证；本轮为纯评测，模型权重与 720 episode 原始数据不因文案或注释修改而重跑。
