# dev_v44

## 启动

- 2026-07-17 项目经理复盘,时隔上一次实质迭代(2025-11-12, v43)约八个月重新回到这个项目。
- 本轮不急于开新训练,而是先做一次**诊断性复盘 + opportunity sizing**:搞清楚指标到底卡在哪、过去几轮的工程量花在了哪、下一步真正的高杠杆动作是什么。
- 触发这一轮的核心判断:v41→v43 连续三轮迭代,绝大部分工程量都花在调 penalty gate 参数和 tail seed 调度上,却**零 stage 晋级**。这是一个"评判系统复杂度失控、真实指标却不动"的危险信号,值得停下来先看清楚再决定方向。

## 一、项目是什么(settings 回顾)

一个"AI 自主运营的实验室":人类当项目经理写 SOP 和基线,Codex CLI 按 `dev_vX.md` → 跑实验 → 记日志 → commit 的循环自主迭代。目标是训练一个策略,让一大群小鱼在高速捕食者面前保持高终局存活率。

### 环境设置(`fish_env.py`)

**物理世界**(硬编码常量):
- 圆形舞台,半径 `STAGE_RADIUS=10.0`;时间步 `dt=0.1`。
- **捕食者**:初速 `PREDATOR_INITIAL_VX=1.5`,带向下重力 `PREDATOR_GRAVITY=0.5`(注释"让大鱼更快"),撞墙反弹阻尼 `PREDATOR_BOUNCE_DAMPING=0.85`。所以它不是简单直线追击,而是一个会加速、会弹墙的重力球。
- **小鱼**:最大速度 `FISH_MAX_SPEED=2.0`,加速度 `FISH_ACCELERATION=1.0`,动作空间是 **5 个离散动作**(`spaces.Discrete(5)`,四方向 + 不动)。
- **观测**:每条鱼看到自己的位置/速度、捕食者相对信息;可选打开 7 维邻居特征(`include_neighbor_features`,群体感知),默认关闭。

**训练配置**(PPO via SB3,近几轮稳定用法):
- 128 并行环境 × 60 iteration,`n_steps=128, batch=1024, n_epochs=5, lr=2.5e-4`。
- **课程**:`15:9,20:10,25:9,25:8,...`(格式为 `fish数:predator速度档`,难度递增)。
- 一套非常精细的 **penalty gate 晋级机制** + **tail seed injection**(把历史 checkpoint 的困难样本注回训练缓冲)。

## 二、指标卡在哪(现状事实)

近三轮(v41→v43)的核心事实很清楚:

- **on-policy 存活率 ~0.85**;multi-eval(96 鱼)main 阶段 `avg_final` 在 **0.70–0.82**,最差样本 `min_final` **长期钉在 0.51 左右**,离成功阈值 0.8 差距明显。
- **真正的瓶颈不是策略,是那套 gate 判据自己把自己锁死了。** v41–v43 连续三轮,7/7 次 multi-eval 全部触发 `failure_hold`,stage 一直停在 0。原因是 `step_one_ratio`(0.022–0.028)和 `early_death_median`(~0.126)超过阈值,即使 `first_death_p10` 已经能到 200+。
- 换句话说:**过去至少 5-6 轮迭代的大部分工程量,都花在调 gate 参数和 tail injection 的调度上,而不是提升鱼的生存能力本身。** 这是典型的"工具/评判系统复杂度失控"信号。

参考数据(来自 dev_v42 的 multi-eval 表,96 fish × 40 ep):

| iter | stage | avg_final | min_final | death_p10 | step_one | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 8  | NE   | 0.869 | 0.708 | 76.9 | 42 | 0.0219 |
| 32 | main | 0.787 | 0.531 | 92.9 | 46 | 0.0240 |
| 40 | main | 0.771 | 0.510 | 64.0 | 53 | 0.0276 |
| 56 | main | 0.697 | 0.521 | 77.0 | 36 | 0.0187 |

## 三、下一步方向(opportunity sizing 的初步假设)

我的判断:**别再顺着 v41–v43 的方向调 gate 和 tail 参数了,那条路已进入收益递减的死胡同。** 应该往回退一步,分三个层次。这三条会在本轮的 debugging session 后被验证或修正。

1. **先诊断 0.51 那个最差样本到底是什么。** 现在所有讨论都停留在聚合指标上,但没人真正看过失败录像里鱼是怎么死的。v42 已把 tail 样本 mp4 存进 `artifacts/media/`。看几段 `min_final≈0.51` 的 episode:是鱼被逼到墙角?是捕食者反弹后的路径无解?还是初始布局就注定一批鱼必死?**这个观察会决定后面所有方向,而现在的迭代恰恰跳过了它。** —— 本轮 debugging session 的核心任务。

2. **质疑 penalty gate 这套机制本身是否值得保留。** 它现在是负债而非资产——三轮零晋级却消耗绝大部分调参精力。可以考虑直接拆掉,回到"纯 PPO + 课程 + 定期 multi-eval"的干净基线,把 `min_final` 和 `avg_final` 当唯一北极星指标。若去掉 gate 后指标不降反升,就证明它一直在帮倒忙。

3. **真要提升生存率,动作空间和感知比调度更值得投。** 现在鱼是 5 个离散动作,躲一个会弹墙加速的捕食者时很受限——最优逃逸往往需要精细转向。两个高杠杆改动:把动作换成连续/更细的方向控制;正式打开 `include_neighbor_features`(README 里一直挂着的 stretch goal),让鱼学会群体规避而非各自逃。这两个是能真正抬高天花板的改动,而 gate/tail 调参只是在现有天花板下反复擦地板。

## 四、Debugging session 结论(2026-07-17,已亲自复核)

对 v42 `dev_v42_stage_buffer_v2` 的 debug 产物(`penalty_stage_debug.jsonl`、`step_one_clusters.jsonl`、`step_one_worst_seeds.json`、`eval_multi_history.jsonl`、`pre_roll_stats.jsonl`)做了数据分析,并对两条承重结论亲自复核了源码与原始 JSONL。**结论推翻了第三节"先看录像找结构性死因"的部分假设**——真正的头号问题不在鱼,而在评判机制本身有个配置 bug。

### 最重要的发现:gate 因为一个配置阈值而永远无法晋级(已复核)

- v42 用 `--penalty_gate_failure_p10 95` 启动。`train.py:1013-1016` 里,只要 `first_death_p10 < 95` 就置 `failure=True`;而 `1087-1104` 的 if/elif 链里 **failure 在 success 之前判定**,一旦 failure 成立就直接 `failure_hold`,后面算好的 `success` 根本不会被采纳。
- 复核原始 `penalty_stage_debug.jsonl`:7 次 eval **全部** `failure_detected: True`,`rolling_p10_median` 依次为 79.9 / 76.95 / 73.45 / 70.5,**全部 < 95**;`failure_streak` 从 1 单调涨到 7。
- **也就是说,过去几轮辛苦调的那些 step_one / early_death success 判据全是死代码**:只要策略的 first-death p10 停在 60–90 区间,`failure_p10=95` 这道门就永远过不去。团队一直在调一个根本不会被执行到的分支。

### 死因拆解:结构性死亡很小,大头是策略在退化

- **step-one 结构性死亡确实存在但只占 ~2%**:鱼在半径 8 的圆盘内均匀 spawn(`fish_env.py:138`),捕食者 spawn 在中心(`:164`),capture 半径 0.7(`:555`)。spawn 在中心 0.7 内的鱼第一步必死,任何策略都救不了。但 `step_one_death_ratio` 只有 0.019–0.028,占总死亡的 6–17%。**这 ~2% 解释不了 0.51 的 min_final。**
- **大头是策略死亡,而且越训越差**:总死亡率从 iter8 的 13% 涨到 iter56 的 30%(存活 0.869→0.697)。死亡分布在整局(dying tail 的 p10≈63–93 步),不是集中在开局。去掉 ~2% 的结构 doom,剩下 11–28% 是有时间有空间逃却被抓的鱼。**这是策略问题,且策略在训练中回归变差。**
- **最差 seed 的共性**:捕食者 heading 落在 60–150° 带(top-6 cluster 全在此),且 pre-roll `speed_scale` 采样病态离散(单集 jitter 里出现过 76× 的值,见 `pre_roll_stats.jsonl`),制造出一批"慢速+居中"的捕食者,是 worst-seed 的主要来源。

### 对第三节假设的修正

- 假设 1(先看失败录像找结构死因)**部分被证伪**:结构性 spawn-doom 只有 ~2%,不是 0.51 floor 的成因。录像仍值得看,但焦点应是"策略为何越训越差",不是"初始布局是否注定必死"。
- 假设 2(质疑 penalty gate 是否该保留)**被强烈印证并升级**:gate 不只是负债,它有个让自己永不晋级的配置 bug。这是当前第一优先级。
- 假设 3(动作空间 + 群体感知才是天花板)**维持**,但要排在修 gate / 止住策略回归之后。

## 五、本轮 checklist

- [x] 落盘诊断分析(本文件)
- [x] Debugging session:定位死因 —— 结论是 gate 配置 bug + 策略回归,而非结构性 spawn-doom
- [x] 复核承重结论(gate 短路逻辑 + 原始 p10 数据)
- [ ] 与项目经理讨论后确定 v44 的实际改动

## 六、Factorial 对照结果(2026-07-17)—— **推翻了前面的核心假设**

跑了 tail-off/gate-on 的干净对照(`v44_clean_baseline_notail_nogate`,128 env × 60 iter),用**和 v42 完全相同的 multi-eval харness**(96 fish × 20 ep)对比。结论和第四节的 debugging 假设**相反**。

### 关键陷阱:on-policy 存活率是骗人的信号

我一开始被 on-policy 的 `sr` 骗了。看训练日志,tail-off 这跑存活率平稳在 ~73%(iter4=75.3%→iter60=72.5%,无回归),我据此以为"关掉 tail injection 就止住了退化"。**这个判断是错的。** on-policy sr 是在**训练分布**上测的,策略过拟合到这个分布,所以看着稳。真正的判决要看 held-out multi-eval:

| 跑法 | on-policy sr | multi-eval avg_final | multi-eval min_final | train/eval gap |
| --- | --- | --- | --- | --- |
| **v42(tail-on/gate-on)** | ~85% | **0.70–0.87** | 0.51–0.74 | ~15pt(泛化好) |
| **v44 clean(tail-off/gate-on)** | ~73% | **0.08–0.29** | 常态 0.000 | ~50pt(灾难性过拟合) |

### 结论:tail injection 不是病,是药

- **关掉 tail injection,held-out 泛化直接崩盘**:avg_final 从 v42 的 0.70–0.87 掉到 0.08–0.29,min_final 常态归零。策略在训练分布上看着好(73%),一到真 eval 就废。
- **v42 那个"0.87→0.70 的下降"根本不是退化**,而是策略被逐步注入的更硬样本推着走、同时保持了强 held-out 表现。tail seed injection 正是**强迫策略泛化到困难 predator 配置**的机制。
- **前面第四节把 on-policy 曲线的平稳当成好消息,是被训练分布过拟合骗了。** 这次对照的价值就在于:不做实验、只信 debugging 叙事,就会朝完全错误的方向改。

### 这对 min_final≈0.51 的重新理解

v42 的 min_final≈0.51 不是"策略差",而是**已经相当强的策略在最难 held-out 样本上的地板**。真正的问题不是"策略在退化"(它没有),而是:(a) 那道 `failure_p10=95` 的 gate 门槛策略够不到,让晋级机制空转;(b) 最难样本上还有 ~0.3 的提升空间,但这是"锦上添花"级别,不是"止血"级别。

## 七、修正后的方向(待验证,factorial 还差一格)

factorial 还缺 **tail-off/gate-off** 和 **tail-on/gate-off** 两格才完整,但主结论(tail 是药)已经稳了。修正后的候选方向:

1. **保留并强化 tail injection,而不是拆掉它。** 它是当前泛化的支柱。可以研究:更系统地覆盖 predator 配置空间(而非手工拼 stage spec),让注入的困难样本分布更均匀。
2. **修 gate 门槛让晋级机制真正生效。** `failure_p10=95` 策略够不到 → gate 空转。降到 ~60,或改成基于 held-out avg_final 的判据,让 density penalty phase 能真正推进。
3. **直接优化最难样本的 min_final(~0.51 地板)。** 针对 held-out 最差 predator 配置(慢速+居中、60–150° heading)做定向 hard-negative 训练。
4. **天花板改动**:5 离散动作 → 连续/更细转向;打开 `include_neighbor_features` 群体感知。
5. **可选**:环境侧加 spawn 最小间距,确定性消除 ~2% step-one 结构性必死。

**教训(已存档):承重结论必须靠 held-out eval,不能信 on-policy 训练指标——过拟合会让训练曲线看着很健康。**
