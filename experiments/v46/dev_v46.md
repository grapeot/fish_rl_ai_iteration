# dev_v46

## 目标(Direction 2:减少坏 seed / 提升训练稳定性)

承接 v45:baseline 是双峰——健康 run avg_final ~0.85(3 seed 均值 0.848±0.016),但偶尔抽到坏 seed(v45 seed 450060 = 0.535)。目标是压低坏 seed 概率,把双峰变单峰高位,从而抬高多 seed 期望值。判据严格按 SOP:多 seed 均值须超过 baseline 0.848 + 噪声带才 merge。

## 已证伪的假设:早期高 LR

**假设**:`warm_cosine` 其实无 warmup(从 base_lr 开始衰减),早期高 LR 在高方差 rollout 上把坏 seed 踹进坏 basin。

**诊断(sub-agent 对比 bad v45=0.535 vs good seed700000=0.867 早期动态)证伪了它:**
- 两个 seed 在 iter1–8 的**所有训练侧信号统计上无法区分**:on-policy sr 都 84–86%;LR schedule 逐位相同;`approx_kl` 都极小无 spike;`clip_fraction` 两者**都恒为 0.0**(更新根本没碰 clip 边界,不存在大步长爆炸);entropy 平坦无坍缩;loss/EV 轨迹重合。
- **没有任何早期训练不稳定的指纹**。如果 LR 假设成立,坏 run 该在早期留下 KL spike / 非零 clip_fraction / entropy dip,但都没有。
- gap **纯粹是泛化**:坏 seed 在自己的 rollout 分布上训得和好 seed 一样好(84–86%),但在 held-out 上从第一次 eval(iter8)就已经落后(0.594 vs 0.800),之后还从 0.594 掉到 0.468 才稳在 ~0.51。**不是"训练崩了",是"从没泛化过"。**

**教训:差点基于一个被数据证伪的假设去加 warmup。先诊断救了一次。**

## 重新定位:泛化 gap 而非训练不稳

坏 seed 的问题是"收敛到一个过拟合自己 rollout 分布、不泛化到 held-out 的解",且这个差异在 iter8 前就定了(TB 每 iter 只记一点,更细看不到)。

**当前诊断(v46_diag_badseed_denseeval)**:用坏 seed 450060 + 密集早期 eval(`--multi_eval_interval 2`),看 iter 2/4/6/8 的 held-out avg_final,区分两种机制:
- **init 抽签**:iter2 就已经 ~0.5 → 初始权重+头几步决定一切 → 解法偏向 ensembling / seed 选择 / 更好初始化。
- **早期可学习分岔**:iter2 还 ~0.8、到 iter8 才掉下去 → 头 8 iter 的学习过程有关键窗口 → 可在该窗口干预。

## 诊断判定:早期分岔,不是 init 抽签(⚠️ 本节的"iter6 前健康"结论被后文 twin-eval 修正——那些读数来自另一条同 seed 轨迹)

dense-eval(坏 seed 450060,`--multi_eval_interval 2`)的 held-out avg_final:

| iter | avg_final | min_final |
| --- | --- | --- |
| 2 | 0.824 | 0.646 |
| 4 | 0.808 | 0.667 |
| 6 | 0.799 | 0.615 |

坏 seed 在 iter2–6 完全健康(与好 seed 同水平 ~0.80–0.82),塌方发生在 iter6 之后(v45 记录 iter8=0.594、iter16=0.468)。**好策略曾经存在,是被后续训练破坏的。**

进一步排查塌方窗口的调度事件(v45_avgfinal_gate 的 schedule_trace):iter6–8 之间**无任何调度变化**(density penalty iter10 才启动、entropy iter16、tail stage 切换 iter15)。塌方发生在静态调度期,且 on-policy sr 全程 84–86%。结论:**纯泛化漂移**——策略在 15 鱼 rollout 分布上持续变好/持平,但对 96 鱼 held-out 的泛化被训练更新逐步破坏。

(附注:episode_seeds 在同一 run 的不同 eval 之间随 RNG 推进而不同,只是跨 run 同 index 对齐;20 episodes 的抽样噪声 ~0.02–0.05,不影响 0.80→0.59 的塌方结论,但修正 dev_v45 里"固定共享测试集"的表述。)

## 干预设计:best-checkpoint 选择(零重训)

既然好策略在训练中途存在过、且 train.py 默认每 5 iter 存 checkpoint,干预就是**按 held-out avg_final 做 best-checkpoint 选择**,而不是交付 model_final:

- `checkpoint_sweep.py`:对 run 的全部 checkpoint 在固定 selection set(20 eps, rng 555001)上评测,选出 best;再把 best 与 final 放到**不相交的 report set**(40 eps, rng 555002)上出报告数字,避免"同一测试集既选又报"的选择过拟合。
- eval env 与训练期 multi-eval probe 一致(96 鱼、neighbor features、boost 固定 0.8 保证所有 checkpoint 面对同一测试分布)。

## Sweep 结果(4 run × 13 checkpoint,selection 20 eps / report 40 eps)

Report set(不相交 40 eps)上 final vs best-checkpoint:

| run | final | best (iter) | Δ |
| --- | --- | --- | --- |
| ms_baseline_seed700000 | 0.876 | **0.899** (20) | +0.023 |
| ms_baseline_seed701000 | 0.837 | **0.878** (5) | +0.042 |
| ms_baseline_seed702000 | 0.829 | **0.830** (40) | +0.001 |
| v45_avgfinal_gate(坏 run) | 0.485 | **0.586** (5) | +0.100 |

**健康 3-run 均值:final 0.847 → best 0.869;四个 run 无一变差。** 数据:`artifacts/checkpoint_sweep.json`。

## 关键自我纠错:上面"iter2–6 健康"的诊断读数来自另一条轨迹

sweep 里原坏 run 的 iter5 checkpoint 只有 0.605,与 dense-eval 诊断读到的 0.80+ 矛盾。twin-eval 定案:把**诊断 run 自存的 model_iter_5** 和**原 run 的 model_iter_5** 放到同一 selection set 上——

- diag_iter5:avg_final **0.815**
- orig_iter5:avg_final **0.605**

同 seed 450060、同配置、同 init(seed 固定 init 权重),两条轨迹在 iter5 已相差 0.21。**固定 seed 也无法复现训练**(CPU torch 浮点非确定性逐迭代放大)。这同时被 seed700000 的复现 run 印证(iter2 起 on-policy sr 即偏离,且该复现轨迹 iter8 读到 0.8625,配置字段与原 run 逐位一致——参数重建无误,差异纯来自非确定性)。

由此修正诊断结论:
1. **"坏 seed"不存在,只有"坏 run"。** 同 seed 重跑 450060 得到的是一条健康轨迹。分岔由训练期随机性造成,与 init 无关(init 由 seed 固定)——"不是 init 抽签"的结论反而被加强了。
2. **原坏 run 在 iter5(最早的 checkpoint)就已在坏 basin(0.605)**,"iter2–6 健康、iter6–8 塌方"的说法只对诊断那条(健康)轨迹成立,对原 run 不成立。分岔点在 iter5 之前,更早的形态因无 checkpoint 不可考。
3. **best-checkpoint 能白捡分,但救不回坏 run**:坏 run 全程 checkpoint 都在 0.45–0.60 带里,best 只到 0.586。要消灭坏 run 得靠早期检测+重启,但实测辨别有困难:seed702000 在 iter5–10 也只有 0.61–0.65,随后却恢复到 0.83——iter15 之前无法可靠区分"坏 run"和"慢热 run"。此路留作后续,不再往 gate 里加机制(本 repo 的历史教训就是判据复杂度失控)。

## Learning / 本轮交付

1. **Direction 2 交付**:`checkpoint_sweep.py`(selection/report 双集合防选择过拟合)+ SOP 新增方法论第 5、6 条(同 seed 非确定性;run 交付物=best checkpoint)。健康 run 均值 0.847→0.869,零训练成本。
2. **方法论升级**:seed 不锚定轨迹 → 一切对比都是 run 分布之间的对比;单条轨迹的"复现"不存在。
3. **下一步(Direction 3,v47 进行中)**:课程尾段 25 鱼 → 48/96 鱼(`15:9,20:10,25:9,25:8,25:8,48:8,96:8`),直接攻击 train(≤25 鱼)/eval(96 鱼)的密度 gap——这是坏 run 泛化失败所在的轴,也是全部 run 的天花板所在(neighbor features 和 384×384 网络早已启用,README 的 stretch goal 描述过时)。3 seed 已在跑,判据:report-set best-checkpoint 均值 vs baseline 0.869。
