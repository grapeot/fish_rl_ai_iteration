# 开发者记录 v57：训练期初始速度随机化的速度泛化对照（第 9 轮，共 10 轮）

## 1. 实验设问与动机

第 6 轮（v54）的同场景、固定模型配对评估给出：三个 v50 `survival_only` FINAL 策略的平均存活率在标称初速下为 0.93377、半速下为 0.92378、零初速下为 0.88394，归零相对标称下降约 4.98 pp。同时，PPO 相对同条件 HOLD 的优势从约 17.93 pp 缩至 1.83 pp。这些结果显示初速干预下的表现差异，不将较高的绝对存活率解释为初速不重要，也不分解初速对生存的因果贡献。

本轮（v57）检验的问题是：在训练期令策略每回合经历一个随机的初始速度缩放因子，能否改善策略在标称/半速/零速三档速度条件上的泛化表现，同时观察对标称条件的影响。这里不预设“应当改善”或“应当无损”，只按预登记方案比较。

## 2. 干预与对照设计

本轮唯一改变训练期初始速度分布的干预如下，其余设定与已验收基线严格对齐：

- **速度采样**：每次环境重置（reset）后，由一个独立的数据增强 RNG 在 $\{1.0, 0.5, 0.0\}$ 中均匀抽取一个全局速度缩放因子。
- **状态重算**：用该因子缩放全部鱼的初速度向量，随后重算 focal 观测。基础世界生成 RNG 不参与、也不被推进。增强 RNG 只消费一次独立抽样，连续两次抽样因子可能相同，不保证相邻回合因子不同。
- **对照配对**：3 个 treatment run 与已验收的 3 个 v50 修正版 `survival_only` control run 一一配对。两者共享初始策略权重与模型种子（`7000011`、`7000021`、`7000031`），以及连续 worker bank（`6000011`、`6000021`、`6000031` 各 +0..5）。配对仅约束初始权重与世界 worker bank，不约束轨迹采样与 reset 时序。
- **训练超参数**：总步数 614,400 = 200 updates × 512 × 6；96 条鱼；11 维无邻居观测；策略与值网络结构 $384\times384$；存活奖励 +0.7、死亡惩罚 −50、500 步超时 bootstrap、$\gamma=0.99$、熵系数 0.02、学习率 3e-4、batch 1024、PPO epochs 10。
- **因子覆盖**：3 个 treatment run 每 run 约 1300 次 reset，三个因子频次大致均衡（约各占 1/3），说明策略在整个训练期确实见到三档速度。因子频次是数据层面计数，不用于推断时间加权占据率。
- **运行特征**：3 个 treatment run 在 18-worker 单波次下各约 24 分钟（1443 / 1451 / 1437 s）。因数据增强介入，学习轨迹与 reset 时序并不同，本文只声称初始权重与 worker bank 相同，不声称训练轨迹相同。
- **训练指标**：训练累计 focal 存活率（`truncs / episodes` 的累计比例，非最终单次更新存活率）treatment 组为 0.909 / 0.906 / 0.917，control 组为 0.927 / 0.912 / 0.904。该指标为训练过程累计量，不是测试集评测结果。

## 3. 检查点选择与评测协议

为使双臂横向可比，本轮不使用 v50 仅依据标称条件择优的旧选择器，改用均衡选择机制，并在正式评测前冻结：

- **候选检查点**：每个 run 仅在 update 100（u100）与 update 200（u200，即该 run 的 final）两个检查点中择优。候选总数为 6 个 run × 2 检查点 = 12 个模型。
- **选择指标**：在标称/半速/零速三档条件的均值上取等权平均，取该均衡均值最高的检查点；平局取较早检查点。双臂标准完全一致。
- **选择规模**：新环境 bank，生成器 RNG 种子 `57010112`，24 个场景 × 3 档条件 × 12 个候选模型 = 864 条选模评估记录。24 是场景数，不是 episode 数；`57010112` 是 RNG 种子，不是场景数。
- **入选检查点**：control 组选定 rep0: u200、rep1: u200、rep2: u100；treatment 组三个 run 均选定 u100。
- **报告协议**：正式报告评测采用独立生成器 RNG 种子 `57010240`，40 个场景；调试种子 `57010101`。严格区分生成器种子与场景数（`57010112`/`57010240`/`57010101` 是 RNG 种子，24/40/2 才是场景数）。选择 bank 与报告 bank 场景种子互不重叠。
- **报告规模**：6 个选定策略模型 + 3 个规则基线（`rule_flee_lead`、`rule_safe_top`、`rule_hold`）× 3 档速度条件（Nominal 1.0、Half 0.5、Zero 0.0）× 40 个场景 = 1,080 episodes。未选中的 final 检查点不执行报告评测。
- **评测干预**：与 v54 完全匹配的 post-reset 鱼体速度缩放，直接复用 `v54_env` 的条件语义。

## 4. 评测结果与统计分析

### 4.1 规则基线

规则基线存活率（Nominal / Half / Zero）：`rule_flee_lead` 0.975 / 0.968 / 0.954；`rule_safe_top` 0.917 / 0.919 / 0.918；`rule_hold` 0.751 / 0.802 / 0.850。HOLD 随初始速度下降而改善，与 v54 一致。

### 4.2 策略模型配对对比

按场景配对 bootstrap（NumPy RNG 种子 `570301`，10,000 次重采样，每次对场景索引重采样并保留全部 3 条件与 6 个固定模型），treatment 3 模型均值与 control 3 模型均值之差如下。CI 仅代表固定 6 个选定模型在给定场景下的条件抽样不确定性，不代表训练或选模过程的不确定性。存活率为比例（0~1），乘以 100 为百分点（pp）。

| 评估条件 | Control 均值 | Treatment 均值 | 差值 (T − C) | 95% CI | 场景 胜/负/平 |
| :--- | ---: | ---: | ---: | ---: | :--- |
| Nominal (1.0) | 0.91901 | 0.91536 | −0.00365 | [−0.01224, +0.00521] | 16 / 22 / 2 |
| Half (0.5) | 0.92127 | 0.91406 | −0.00720 | [−0.01415, −0.00035] | 13 / 27 / 0 |
| Zero (0.0) | 0.87083 | 0.87396 | +0.00313 | [−0.00929, +0.01233] | 21 / 16 / 3 |
| 三条件等权均值 (Balanced) | 0.90370 | 0.90113 | −0.00258 | [−0.00775, +0.00220] | 20 / 20 / 0 |

统计解读边界：

1. **未检出均衡收益**：均衡差值 −0.00258，CI [−0.00775, +0.00220]，跨越 0。这是“在该范围内未检出收益”，与同一范围内的负效应、正效应都相容；不能引申为已证实中性、无害、干净的零效应或等效。
2. **未建立标称性能保持或非劣效**：标称差值 −0.00365，CI [−0.01224, +0.00521]，区间仍涵盖约 −1.224 pp 的损失。CI 不给该损失发生的概率，也不表示更大损失不可能。缺少预先声明的等效/非劣效边界与相应检验，不能声称标称性能得以保持，也不事后补 margin。
3. **保留 Half 条件的探索性负向区间**：Half 处理组相对 −0.00720，CI 严格落在负区间 [−0.01415, −0.00035]（27 个场景变差、13 个变好）。这是一项未经多重比较校正的次要探索性区间，应如实保留，不以“多重检验假阳性”或“replicate 间符号反转”为由淡化或抹除，也不将其上升为确认性机制断言。
4. **三条件等权均值是预登记选择与报告的主口径**，其数值与 v50 仅标称口径的选择器不可直接比较。

### 4.3 分 Replicate 明细与异质性

各 replicate 基于 2 个固定选定模型评估，反映模型间异质性，不代表训练总体分布。

| Replicate | 条件 | Control 均值 | Treatment 均值 | 差值 (T − C) | 95% CI | 符号 |
| :--- | :--- | ---: | ---: | ---: | ---: | :---: |
| rep0 | nominal | 0.93021 | 0.90833 | −0.02188 | [−0.03516, −0.00859] | − |
| rep0 | half | 0.92865 | 0.90885 | −0.01979 | [−0.03255, −0.00859] | − |
| rep0 | zero | 0.84948 | 0.86172 | +0.01224 | [−0.01250, +0.03073] | + |
| rep0 | balanced | — | — | −0.00981 | [−0.02240, +0.00061] | − |
| rep1 | nominal | 0.92161 | 0.93516 | +0.01354 | [−0.00286, +0.03281] | + |
| rep1 | half | 0.92344 | 0.93099 | +0.00755 | [−0.00313, +0.01849] | + |
| rep1 | zero | 0.91354 | 0.84948 | −0.06406 | [−0.08438, −0.04375] | − |
| rep1 | balanced | — | — | −0.01432 | [−0.02517, −0.00286] | − |
| rep2 | nominal | 0.90521 | 0.90260 | −0.00260 | [−0.01172, +0.00573] | − |
| rep2 | half | 0.91172 | 0.90234 | −0.00938 | [−0.02214, +0.00260] | − |
| rep2 | zero | 0.84948 | 0.91068 | +0.06120 | [+0.04193, +0.08073] | + |
| rep2 | balanced | — | — | +0.01641 | [+0.00738, +0.02517] | + |

各 replicate 跨条件的符号序列：nominal 为 −,+,−；half 为 −,+,−；zero 为 +,−,+。符号在不同 replicate 间发生反转，这是抽样变异与策略差异的体现，如实报告即可；它既不能证明、也不能排除某项潜在作用机制。

### 4.4 worst-condition 统计量

产物中的 `worst_condition` 值为 −0.02127，CI [−0.03212, −0.01328]。其精确定义是：**在每个场景下先分别计算三档条件的 treatment 组减 control 组组平均差值，取三者最小值，再跨 40 个场景求均值**，即“场景级最小配对条件反差均值（mean per-scenario minimum paired condition contrast）”。

它既不是三档条件平均差值的最小值（后者为 Half 的 −0.00720），也不是表现最差的单个 replicate，也不是任何单臂的绝对最低存活率。该探索性统计量不高于对应场景等权均值的平均值，但符号并无保证：三种条件差值全为正时，最小值也可以为正。它不能与均衡口径混读；此处保留其定义与本批负值，不将其替代主端点。

## 5. 速度敏感性（描述性，不作机制断言）

在相同的 v57 报告 bank 与同一 6 个固定选定模型下，臂内速度变化（相对各自标称）及配对交互项如下，均为描述性统计：

| 对比维度 | Control 臂内变化 | Treatment 臂内变化 | 配对交互项 (T − C) | 95% CI |
| :--- | ---: | ---: | ---: | ---: |
| Half − Nominal | +0.00226 | −0.00130 | −0.00356 | [−0.01476, +0.00599] |
| Zero − Nominal | −0.04818 | −0.04141 | +0.00677 | [−0.01224, +0.02031] |

zero − nominal 的模型平均差值：control −0.04818（CI [−0.06389, −0.03030]），treatment −0.04141（CI [−0.05069, −0.03273]）。half − nominal：control +0.00226（CI [−0.00634, +0.01267]），treatment −0.00130（CI [−0.00564, +0.00286]）。

无论 Half 还是 Zero，处理组相对对照组的速度交互项 CI 均跨越 0，交互效应未获确立。此处只报告同 bank、固定模型下的条件表现差异，不命名机制；既不能断言某项速度敏感性机制成立，也不能断言排除了训练/测试分布漂移。v54 的历史数值使用完全不同的场景 bank，且 v57 control rep2 选中 u100 而非全部 final 检查点，两代绝对数值不可直接相减推断“断层是否闭合”。

## 6. 工程验证与协议披露

### 6.1 测试

- **语义测试** 6 项，覆盖：增强仅改速度与 focal 观测且不动基础世界 RNG；增强 RNG 同种子可复现、异种子变化、逐回合变化；control 模式与已验收 v50 wrapper 逐步一致；奖励与 500 步超时 bootstrap 视界不变；真实 SubprocVecEnv 自动重置进入新回合且覆盖多于一个因子；评测条件与 v54 因子完全一致且标称路径逐 bit 复现直接基线。
- **验收门禁测试** 17 项（原 15 项 + 本次补强的 2 项因子映射变异拒绝），覆盖：缺失/未知/错误路径/错误哈希/错误环境/错误 bank/条件集合不符/错误速度因子映射均拒绝；真实 manifest 正向绑定；配对记录严格性（未知控制器/条件、重复单元、缺失场景、缺失条件单元均拒绝）；summary 与 raw 一致性守卫；均衡 CI 为场景配对均值而非两个 CI 相减。

### 6.2 门禁与校验器的功能边界（真实时间线）

必须区分两个不同职责的工具，且不得把报告后的补强粉饰为报告前的准入验收：

- **运行期评估身份门禁**（`v57_evaluate.py` 中的 `validate_manifest_gate`）：在每次评估运行前校验策略路径与张量哈希、环境配置、评估条件与种子列表。该门禁在报告运行时就已允许“仅含选定的 6 个模型、final 可选”的记录集合，报告运行确实带 `--require-manifest` 通过门禁后写出 1,080 行原始记录。
- **后置分析记录校验器**（`validate_paired_records`）：在分析阶段校验记录网格的完整性。报告产出后才发现当时的该函数要求同时涵盖所有选定的与所有 final 的模型，而报告按预登记只运行了选定模型，因而分析首次报错；随后只放宽了该校验器所需的模型集合（final 改为可选）。运行期身份门禁、物理环境、随机种子、模型权重与评测数据均未改动。

时间线（真实）：
- `12:02Z`（训练与报告前）通过了 6 项语义测试；
- `12:03Z` 当时跑门禁测试脚本只执行了 2 个合成测试，因为当时尚无 manifest，明确**跳过了 13 项清单与记录检查**；
- 首次**完整通过全部 15 项门禁**发生在报告产出并修复记录校验器之后的 `13:03Z`；
- 本次审查独立重跑，完整 6 语义 + 17 门禁全部通过。

因此，不得声称“全部 15 项在报告前通过”，也不得声称“所有评估源码哈希从未变化”。

### 6.3 冻结时序与不可覆写边界

- 冻结 manifest `artifacts/frozen_config/v57_frozen_manifest.json` 完整绑定了世界环境配置（含捕食者朝向与 pre-roll 速度偏置）、速度因子映射表、选定与最终检查点路径及张量哈希、两套生成 bank 与选择器逻辑。
- 冻结发生于生成报告**之前**（manifest `created_utc` `2026-10-09T12:44:59Z`，报告写毕 `13:02:56Z`）。其语义是“在生成报告前冻结选模结果”，不是在训练前对实验假设单独登记时间戳。
- `v57_select.py` 默认拒绝覆盖已存在的 manifest，但保留了 `--allow-overwrite` 开关，覆盖时不检查报告是否已存在、并保留原始 `created_utc`。本次无任何覆盖发生；本文如实区分“当前冻结事实”与“代码实际强制的边界”，不承诺无条件的不可覆写。

### 6.4 门禁加固与 post-review 哈希

- 本次按审查要求补齐 `velocity_factors` 映射表参数绑定：`validate_manifest_gate` 现在把实际使用的因子映射作为硬约束校验，篡改任何条件的因子（如把 half 的 0.5 改为 0.25、把 zero 的 0.0 改为 0.1）都会被拒绝。对应负例测试 `test_wrong_velocity_factor`、`test_wrong_velocity_factor_zero` 均已加入并通过。
- 该加固只增强当前校验器，不改变已冻结的因子值、模型或场景数字；冻结 manifest 文件哈希仍为 `ac3bcae513eb092d3d0a2f4cd2eafda01ff9f9ed019df80260022c0133b89ea7`。
- 运行期嵌入的源码哈希（各 run 的 `runtime_source_sha256`、报告的 `source_sha256`）保持原样，不刷新。报告期记录的 `v57_verify.py` 哈希为 `e92e89bb7ec00e4e30dabcce95e42360b8e3c0892321c154891678094c19db6c`（运行期身份门禁版本）。
- 源码快照分为三层且各自独立：`post_run_source_hashes.json`（报告期后、仅覆盖 common/env/train/evaluate/verify 与共享依赖）、`post_review_source_hashes.json`（本次审查后、覆盖完整 v57 公开工具链，见 §8），运行期哈希仍保留在 run/report 产物内、不被刷新。
- 工具绑定加固的元数据单独记录在 `artifacts/results/v57_manifest_binding_supplement.json`。该文件是独立补充，不是冻结 manifest 的一部分，不包含任何选择/模型/种子/因子/场景字段；冻结 manifest 与其 `created_utc` 均未改动。

## 7. 结论与后续规划

在当前算力预算与 6 个固定选定模型下，训练期随机化初始速度**未建立起可验证的三条件均衡性能提升**；三条件等权均值差 −0.00258，CI [−0.00775, +0.00220]。标称条件点估计为负（−0.00365，CI [−0.01224, +0.00521]），在缺乏显式非劣效检验的前提下不能声明标称无损；Half 条件存在探索性负向区间（−0.00720，CI [−0.01415, −0.00035]），如实保留。这些都不构成对该干预“中性/无害/等效”的确认，也不构成对其“有害”的确认。

第 10 轮（v58）按已预登记的独立确认方案执行，使用本轮的冻结输入，不做重新训练、重新挑选或替换模型。本轮不给出跨代速度敏感性的推断。

## 8. 代码资产、复现说明与哈希

### 8.1 工程资产（repo 相对路径）

- 模块与脚本：`experiments/v57/v57_common.py`、`v57_env.py`、`v57_train.py`、`v57_evaluate.py`、`v57_select.py`、`v57_report.py`、`v57_analyze.py`、`v57_verify.py`、`v57_snapshot.py`、`v57_post_review_snapshot.py`。
- 测试：`experiments/v57/tests/test_v57_semantics.py`、`tests/test_v57_acceptance_gates.py`。
- 冻结配置：`experiments/v57/artifacts/frozen_config/v57_frozen_manifest.json`。
- 结果产物：`experiments/v57/artifacts/results/{selection.jsonl, selection.summary.json, report.jsonl, report.summary.json, analysis.json, post_run_source_hashes.json, post_review_source_hashes.json}`。
- 训练数值产物：`experiments/v57/artifacts/runs/rep{0,1,2}_augmented_velocity/{config.json, run_summary.json, train_metrics.jsonl}`。
- 复用的 v50 数值资产直接引用公开资产，不重复存储。

### 8.2 复现命令（argparse 完整示例）

以下命令在 `fish_rl` 项目目录下执行，`.venv` 为 `experiments/v48/.venv`。三个 treatment run 使用全新输出目录（如未存在会自动创建）：

```bash
# 1) 训练 3 个 treatment run（每 run 6 env，一个 18-worker 波次）
for r in 0 1 2; do
  experiments/v48/.venv/bin/python experiments/v57/v57_train.py \
    --replicate "$r" --augment-velocity --iterations 200 --num-envs 6 \
    --n-steps 512 --batch-size 1024 --n-epochs 10 --learning-rate 3e-4 \
    --ent-coef 0.02 --gamma 0.99 --gae-lambda 0.95 --clip-range 0.2 \
    --net-arch 384,384 --checkpoint-iters 50,100,150 \
    --out-dir experiments/v57/artifacts/runs/rep${r}_augmented_velocity
done

# 2) 选模（预登记，写 selection.jsonl / selection.summary.json 与冻结 manifest）
experiments/v48/.venv/bin/python experiments/v57/v57_select.py \
  --out-jsonl experiments/v57/artifacts/results/selection.jsonl \
  --manifest experiments/v57/artifacts/frozen_config/v57_frozen_manifest.json \
  --workers 8

# 3) 报告评测（仅 6 个选定模型 + 3 规则基线；门禁要求 manifest）
experiments/v48/.venv/bin/python experiments/v57/v57_report.py \
  --manifest experiments/v57/artifacts/frozen_config/v57_frozen_manifest.json \
  --out experiments/v57/artifacts/results/report.jsonl --workers 8

# 4) 分析
experiments/v48/.venv/bin/python experiments/v57/v57_analyze.py \
  --report-summary experiments/v57/artifacts/results/report.summary.json \
  --report-jsonl experiments/v57/artifacts/results/report.jsonl \
  --manifest experiments/v57/artifacts/frozen_config/v57_frozen_manifest.json \
  --out experiments/v57/artifacts/results/analysis.json

# 5) 快照（post-run 与 post-review 分开记录）
experiments/v48/.venv/bin/python experiments/v57/v57_snapshot.py \
  --out experiments/v57/artifacts/results/post_run_source_hashes.json
experiments/v48/.venv/bin/python experiments/v57/v57_post_review_snapshot.py \
  --out experiments/v57/artifacts/results/post_review_source_hashes.json

# 6) 测试
experiments/v48/.venv/bin/python experiments/v57/tests/test_v57_semantics.py
experiments/v48/.venv/bin/python experiments/v57/tests/test_v57_acceptance_gates.py \
  --manifest experiments/v57/artifacts/frozen_config/v57_frozen_manifest.json
```

### 8.3 本地工件说明

模型权重 ZIP 仅存在于本地/私有环境，不入公开候选。训练脚本可自行生成这些权重（见 §8.2 第 1 步）；由于多 worker 与硬件调度不具逐 bit 确定性，重训不会得到相同字节，模型身份以 `policy_tensor_sha256`（基于排序后 policy `state_dict` 键与字节流）为准。冻结的 selected 模型路径与张量哈希已固化在冻结 manifest 中，供独立确认复现。

### 8.4 哈希范围

- 运行期训练源码哈希：各 treatment run 的 `config.json` / `run_summary.json` 内 `runtime_source_sha256`。
- 报告期评估源码哈希：`report.summary.json` 内 `source_sha256`。
- 报告期后快照：`post_run_source_hashes.json`（仅覆盖 common/env/train/evaluate/verify 与共享依赖）。
- 审查后完整工具链快照：`post_review_source_hashes.json`（覆盖全部 v57 模块、两个测试与本文档）。该文件的精确哈希清单在交付时另附。
- 工具绑定加固补充元数据：`v57_manifest_binding_supplement.json`（独立文件，非冻结字段）。
