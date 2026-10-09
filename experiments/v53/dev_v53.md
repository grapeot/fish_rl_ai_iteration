# dev_v53: 时域截断语义对比（round 5/10）——有限时域终态 vs 超时 Bootstrap

## 1. 问题与单轴控制

v50 修正版建立了 accepted baseline：`survival_only`（存活 +0.7 / 死亡 −50），
训练时存活到 500 步按 `truncated` 处理，SB3 在末步奖励上加 `γ·V(terminal_obs)`
（continuation / bootstrap 目标）。v53 检验一个单轴问题：**在物理、观测、动作、
奖励、优化器、预算全部冻结的前提下，只改变训练中第 500 步仍存活时的结束语义，
是否改变学习表现。**

- **control（timeout_bootstrap）**：直接复用已采纳的 v50 corrected
  `survival_only` 三个训练 run。存活到 500 步 → `truncated=True`，末步目标包含
  `γ·V(s_500)`。**不重训**。
- **treatment（finite_terminal）**：新训三个 v53 run。存活到 500 步 →
  真正的 `terminated=True`，末步目标只有当步存活奖励，结束后的未来价值为 0，
  不做 bootstrap。

两者是不同的时域累积目标定义，不是同一目标的两种实现。死亡优先级两组一致：若在
第 500 步因碰撞死亡，均判真正 terminal、结算 −50，不改成存活 horizon 终态。

固定项：96 鱼、固定 focal、其余鱼 HOLD、11 维无邻居局部观测、**不添加剩余时间特征**、
PPO 384×384、gamma 0.99、lr 3e-4、ent 0.02、n_steps 512 × 6 envs × 200 updates =
614,400 步、6 subproc worker 各 1 torch thread。

单轴配对：同一 replicate 内 control 与 treatment 共享 model seed 与连续 worker
bank，初始策略张量 hash 逐位相同，起始 episode 相同；replicate 间独立。进入学习后
两臂轨迹自然分叉。

| 项 | 值 |
| :--- | :--- |
| model seeds | 7000011 / 7000021 / 7000031 |
| worker banks | 6000011..16 / 6000021..26 / 6000031..36 |
| v53 selection bank | 530101（24 eps） |
| v53 report bank | 530102（40 eps） |
| v53 smoke bank | 530100（2 eps） |

v53 三个 bank 与 v48/v49/v50、旧 pilot bank 在**生成值**层面无交（见单测）。

## 2. 结束语义与目标定义

底层 `FishEscapeEnv` 在 `MAX_TIMESTEPS=500` 处把「所有鱼存活」也标为 `terminated`。
v50 wrapper 把存活到 500 步改写为 `truncated`，v53 treatment 保持为 terminal：

- **timeout_bootstrap**：`truncated=True`；SB3 collector 保存 `terminal_observation`
  并置 `TimeLimit.truncated`，对末步加 `γ·V(terminal_obs)`。这是 continuation 目标
  约定，**不是某个已证明正确的无限时域 MDP 的解**。
- **finite_terminal**：末步就是真正 terminal，无 bootstrap；存活末步仍拿 +0.7，
  此后的未来价值为 0。这是一个**不同的目标定义**，其合法性不因本轮负结果而改变。

`finite_terminal` 末步 target 是 +0.7、结束后的未来价值为 0，这是有限时域目标的
定义本身，**不能**据此说它是「有偏目标」「错误」或「抹去近末步保命激励」；死亡仍
−50，存活激励在两组中完全一致。

## 3. 结果（新 report bank 530102，40 paired episodes）

统计单位是 episode，不是 96 条鱼。先对每个 episode 求三个 replicate 的
`treatment − control`，再在 episode 内平均三组，得到长度 40 的向量，用 rng 530301
bootstrap 10,000 次。三 run std 只是这三个训练结果的描述性散布，不是训练随机性的
sampling CI；CI 条件于六个固定模型。

| 阶段 | control 均值 (std) | treatment 均值 (std) | 差值 | episode-paired 95% CI | treatment 胜出 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| selected | 0.92517361 (0.010434) | 0.88611111 (0.028936) | **−0.03906250** | [−0.04930556, −0.02951389] | 2/40 |
| final | 0.92352431 (0.008763) | 0.86701389 (0.032648) | **−0.05651042** | [−0.06935764, −0.04461806] | 1/40 |

per-replicate 差值（treatment − control）：selected −0.01796875 / −0.04557292 /
−0.05364583；final −0.02968750 / −0.08697917 / −0.05286458，三组全部同向。

规则锚点（同一 report bank）：`rule_flee_lead` 0.97239583、
`rule_safe_top` 0.91614583、`rule_hold` 0.75026042。control 的 selected 与 final
均值都**高于** `rule_safe_top`；treatment 均值低于它。这是本 bank 的均值排序，
两个 anchor 相差约 0.9 个百分点，**不据此宣称任何显著性**。

训练流累计焦点存活率（监控指标，非独立泛化曲线）：control 0.927 / 0.912 / 0.904，
treatment 0.859 / 0.854 / 0.857。

selection（530101，24 eps，tie → earliest）：control 选 u100 / u200 / u150，
treatment 选 u150 / u100 / u200。其中 rep1_control、rep2_treatment 在同一 run 内
selected==final（路径与真实 tensor hash 相同，允许 alias）。

## 4. 结论边界

**有界负结果**：在当前 stationary 训练配方、预算、三组配对模型与 530101/530102
两套 fresh bank 上，把 500 步存活语义从 timeout bootstrap 改为 finite terminal，
selected / final 的终局群体生存率分别低 3.91 / 5.65 个百分点。工程上继续保留
timeout bootstrap 作为当前基准。

必须遵守的边界：

- 负结果**条件于当前配方**：不证明 bootstrap 对原有限任务正确或无偏，不证明
  finite objective 无价值，也不证明已找到严格有限时域最优策略。
- **机制最多是未检验假设**。输入没有剩余时间特征，stationary critic 不能显式表示
  时间条件价值；目标定义与表达能力的相互作用是**一种可能解释**，本试验没有识别
  负结果原因，不能写成根源或已证实的因果机制。
- 局部可观测：环境本质为 POMDP，本轮 11 维无邻居观测仍是 stationary
  partial-observability baseline；不排除充分收敛或改变参数后 finite 语义有不同表现。
- 只有 3 个配对 replicate；三 run std 与 episode-paired CI 的解释范围如上。

## 5. 测试

- `tests/test_v53_semantics.py`（7）：horizon flag 差异；死亡优先级；连续 RNG /
  真实 autoreset 新鲜世界；同 seed 同动作世界轨迹一致；初始权重与种子配对；
  v53 banks 生成值无交；**真实 SB3 collector + RolloutBuffer** 下 finite 末步
  target == 0.700000（无 bootstrap），timeout 末步 target == 0.795791 =
  0.7 + γ·V（γ·V 0.095791）。
- `tests/test_v53_acceptance_gates.py`（15）：模型/path/seed 错配、缺 final、
  未知 run、重复臂名拒绝；analyzer 拒绝缺局、重复局、额外局、缺 final 且无 flag；
  并补两个 v50 缺口——heading bias 纳入 env 绑定，以及 `selected_is_final` 复用
  必须 flag 与 selected/final 真实 hash 同时一致。

`v53_evaluate.py --require-manifest` 时门禁生效（report orchestrator 确实传入；
它不是 evaluator 的无条件默认行为）。`selection_manifest.json` 默认拒绝覆盖，
但存在 `--allow-overwrite`；本轮为清除绝对路径确实使用过一次（见 §6）。

## 6. 数据溯源

本轮所有 raw 数值保持不覆盖。**selection 的 `created_utc`
`2026-10-09T11:33:52Z` 是初次 selection 映射的冻结时间，不是本 manifest 相对路径
版本的创建时间，也不是任何 reviewer 的审核时间。**

初次 selection / report 之后发生过两次有界的评测侧修正，与科学数字无关：

1. 首次 report 因 evaluator 的 rule 分支异常失败；补齐后第一次成功 report。
2. 初次 `selection_manifest.json` 内嵌了开发机的绝对 checkpoint 路径。为公开
   pre-flight，评估/选择工具改为写/读**仓库相对路径**，manifest 用
   `--reuse-existing --allow-overwrite` 重新生成（沿用原 `created_utc`，selection
   数字不重算），report 在相同 gate 下重跑一次得到当前 raw/summary。

当前 UTC 时间线（只读执行记录 + filesystem birthtime/mtime）：初次 selection
11:25–11:33:52；首次 report 尝试 11:33:55（gate 通过、rule 分支失败）；修 evaluator
后首次成功 report 11:35–11:43:13；相对路径修正 11:44:36–11:45:01；manifest 重生成
11:44:44 / 11:45:07–08；隐私修正后最终 report rerun 11:45:11–11:52:59；analysis
11:53:03。

因此公开 metadata 诚实记录「做过一次路径清理后重跑」：当前 report
raw/summary 是**路径修正后重跑**的产物，当前 manifest 是 repo-relative 版本；原
`created_utc` 与 selection 数字保持不变。**不能把当前 hash 或这份新 manifest 当作
初次 run 时就已有、或被独立审核过。**

`post_run_source_hashes.json` 是**事后**评测/分析工具快照，明确区分于历史运行时
训练来源；每个 run 的 `config.json` / `run_summary.json` 中的训练
`source_sha256`（v53_common/env/train + v50_common/env + fish_env）是运行时刻记录，
未以事后新 hash 替换。训练 common/env/train 与共享物理自运行以来未变。

模型 zip 只在本地用于核验真实 tensor 身份与更新计数，不进公开候选；不承诺位级复现。

## 7. 复现

从仓库根目录（复用既有 v48 venv）。**把 `--out-dir` / `--out-*` 指向新的
scratch 目录，不要覆盖已归档的 runs / results。**

```bash
VENV=experiments/v48/.venv/bin/python

# 测试
$VENV experiments/v53/tests/test_v53_semantics.py
$VENV experiments/v53/tests/test_v53_acceptance_gates.py

# 训练（treatment，3 个 replicate，各 614,400 步；示例 rep0）
$VENV experiments/v53/v53_train.py \
  --replicate 0 --termination-mode finite_terminal \
  --iterations 200 --num-envs 6 --n-steps 512 --batch-size 1024 \
  --n-epochs 10 --learning-rate 3e-4 --ent-coef 0.02 --gamma 0.99 \
  --gae-lambda 0.95 --clip-range 0.2 --net-arch 384,384 \
  --checkpoint-iters 50,100,150 \
  --out-dir experiments/v53/artifacts/scratch/runs/rep0_finite_terminal

# 选择（先冻结不可覆盖 manifest，早于任何 report）
# 注意：v53_select/report/analyze 按 v53_common 的固定目录读取六个已归档 run
# （control=v50 corrected survival_only，treatment=v53 finite_terminal），
# 不接受一个 --runs-dir 参数；因此它们复算的是归档 checkpoint 上的 selection/report。
$VENV experiments/v53/v53_select.py \
  --out-jsonl experiments/v53/artifacts/scratch/results/selection.jsonl \
  --manifest experiments/v53/artifacts/scratch/results/selection_manifest.json

# report（门控于 manifest）
$VENV experiments/v53/v53_report.py \
  --manifest experiments/v53/artifacts/scratch/results/selection_manifest.json \
  --out experiments/v53/artifacts/scratch/results/report.jsonl

# 汇总
$VENV experiments/v53/v53_analyze.py \
  --report-summary experiments/v53/artifacts/scratch/results/report.summary.json \
  --report-jsonl experiments/v53/artifacts/scratch/results/report.jsonl \
  --manifest experiments/v53/artifacts/scratch/results/selection_manifest.json \
  --out experiments/v53/artifacts/scratch/results/analysis.json
```

复现范围说明：只保证在本仓库代码与依赖快照下可重跑训练、选择、report 与汇总；
不承诺第三方在无重训情况下复原 PPO 权重执行。要复现当前归档数字，直接对归档
run 目录重跑 select/report/analyze；`v53_train.py` 的新 `--out-dir` 只写 scratch，
不覆盖归档，也不被 select 自动读取。

## 8. 文件清单（仓库相对）

- 代码：`experiments/v53/v53_{common,env,train,evaluate,select,report,analyze,verify,reuse_check}.py`
- 测试：`experiments/v53/tests/test_v53_{semantics,acceptance_gates}.py`
- 训练产物：`experiments/v53/artifacts/runs/rep{0,1,2}_finite_terminal/{config.json,run_summary.json,train_metrics.jsonl}`（checkpoints 不公开）
- 评测结果：`experiments/v53/artifacts/results/{selection.jsonl,selection.summary.json,selection_manifest.json,report.jsonl,report.summary.json,analysis.json,reuse_check.json,post_run_source_hashes.json}`
- control 数值资产：`experiments/v50/artifacts/corrected_streams/runs/rep{0,1,2}_survival_only/{config.json,run_summary.json,train_metrics.jsonl}`（**已有、不复制、不改**）
