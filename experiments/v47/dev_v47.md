# dev_v47

## 目标(Direction 3:高密度课程,攻击 train/eval 密度 gap)

承接 v45/v46:健康 baseline(report set,best-checkpoint)= **0.869**(3 run 均值;final 均值 0.847)。剩余死亡集中在 96 鱼 held-out 场景,而课程只训到 25 鱼——**train(≤25 鱼)/eval(96 鱼)之间的密度 gap 是坏 run 泛化失败所在的轴,也可能是所有 run 的天花板**。v46 已确认 neighbor features(obs 18 维)和 384×384 网络早已启用,README 的"stretch goal"描述过时,天花板改动里唯一被数据直接支持的就是密度轴。

## 改动(纯配置,零代码)

课程尾段两个 25 鱼 phase 换成 48/96 鱼:

- baseline:`15:9,20:10,25:9,25:8,25:8,25:8,25:8`
- hidensity:`15:9,20:10,25:9,25:8,25:8,48:8,96:8`

其余参数与 ms_baseline_v42cfg 逐位一致(v42 命令 + v45 avg_final gate;经 repro_check 验证配置字段逐位吻合,详见 dev_v46)。已确认可行性:tail seed 只编码捕食者初速/角度,与鱼数解耦;gate 的 phase_limit 只封顶 density penalty 阶段,不阻塞课程鱼数推进。

## 运行

`multiseed_eval.py --label hidensity --seeds 3 --max-parallel 3`(seed 700000/701000/702000,注意 per v46:seed 不锚定轨迹,这只是 3 次独立抽样)。

## 判据(SOP 方法论)

训完对 3 个 run 跑 `checkpoint_sweep.py`(selection 20 eps / report 40 eps,与 v46 同一对测试集),比较:

- hidensity report-set **best-checkpoint 均值** vs baseline 0.869
- hidensity report-set **final 均值** vs baseline 0.847

超过 baseline + 噪声带(~0.02)才算真提升、才 merge 配置为新默认;否则如实记录负结果。

## 结果:负结果,不采纳为默认配置

3 run 训练全部正常完成(60 iter,高密度 phase 无崩溃,tail/gate 兼容性预判正确)。Report set(与 v46 同一对 selection/report 测试集)对比:

| 口径 | hidensity (3 run) | baseline (3 run) | Δ |
| --- | --- | --- | --- |
| final 均值 | 0.809 | 0.847 | **-0.038** |
| best-checkpoint 均值 | 0.855 | 0.869 | **-0.014** |

per-run(report set,final → best):700000 = 0.834→0.891(iter10);701000 = 0.819→0.819(final);702000 = 0.774→0.854(iter5)。数据:`artifacts/checkpoint_sweep_hidensity.json`、`artifacts/multiseed_hidensity.json`(driver 口径 0.810±0.018 vs baseline 0.848±0.016,同向)。

按 SOP 判据(须超过 baseline 0.869 + 噪声带):**两个口径都没超过,配置不 merge 为默认。** best 口径的 -0.014 在噪声带内(两臂各自 std ~0.02–0.04),final 口径的 -0.038 偏负。3v3 的检验力探测不到小效应,但至少可以说:高密度课程没有带来可检出的提升。

## 两个意外发现(比主结果更有价值)

1. **best checkpoint 集中在 iter5–10。** 两臂 7 个 run 里 4 个的最优 checkpoint 在 iter5 或 iter10(selection set 上 0.86–0.90),之后到 iter25 普遍回落到 0.84 以下——存在系统性的"早期泛化峰值 + 中期回落"模式。这既再次坐实 best-checkpoint 交付的价值(v46),也提示中期训练(iter10–25,tail 注入密集期)在损害 held-out 泛化,值得单独调查。
2. **高密度 phase 内的 within-run 改善信号。** iter40→60(高密度从 45 开始)selection-set 变化:hidensity 三 run 平均 **+0.032**(+0.003/+0.046/+0.047),baseline 同期平均 +0.009。within-run 对比部分绕开了 run 间抽签噪声,弱信号提示高密度训练在其生效区间内是正向的——只是 8+8 个 iter 太短,又被早期轨迹抽签淹没。若后续再试,应考虑高密度 phase 提前/加长,或从 tail 注入期就混入高密度 env。

## Learning / 本轮交付

1. **Direction 3 负结果**:课程尾段 48/96 鱼(3 run)未超过 baseline,不改默认配置。负结果照常入库,防止后人重复踩线。
2. 方法论再次验证:若只看 driver 口径单点(0.810 vs 0.848)会得出"高密度有害"的强结论,但 best-checkpoint 口径(-0.014,噪声内)和 within-run 信号(+0.032)说明真相更接近"中性、检验力不足"。**多口径交叉看,别用单一聚合数字下重结论。**
3. 全项目当前最优单策略:baseline seed700000 iter20 checkpoint,report set **0.899**(其次 hidensity seed700000 iter10 = 0.891)。
