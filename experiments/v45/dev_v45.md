# dev_v45

## 目标(Direction 1:修 gate 让晋级生效)

承接 v44 factorial 结论:tail injection 是泛化支柱、gate 因 `failure_p10=95` 空转。本轮修 gate,让它用 held-out `avg_final_survival_rate` 判晋级,而不是够不到的 step-count p10。

## 代码改动(基于 v43/train.py)

在 `PenaltyStageGate` 加两个开关:
- `--penalty_gate_success_avg_final`(>0 启用):held-out avg_final ≥ 阈值即 success。
- `--penalty_gate_failure_avg_final`(>0 启用):held-out avg_final < 阈值即 failure。
- 启用后覆盖旧的 step-count / p10 / min_final 判据(见 `handle_multi_eval` 的 `avg_final_gating` 分支)。debug payload 里输出 `avg_final_gating` / `avg_final` / 两个阈值,可观测。

**单元测试已过**(合成 eval 喂进 gate):avg 0.85/0.82→advance(stage 0→1→2),0.55→rollback,0.70→noop。逻辑正确。

## 运行:v45_avgfinal_gate(128 env × 60 iter,success=0.80 / failure=0.55)

命令与 v42 `stage_buffer_v2` 的 tail/curriculum/结构 gate 参数一致,仅:去掉 v42 的一堆 `--penalty_gate_success_*` step-count 阈值,换成两个 avg_final 阈值;seed 450060。

### 关键发现:gate 修对了,但暴露了一个更大的方法论问题——单 seed 不可信

gate 改动本身**验证成功**:debug 显示 `avg_gating=True`、阈值生效、按 avg_final 做真实决策。

但 held-out avg_final 却系统性低于 v42(**同 tail 配置、同 eval harness、同训练旋钮**):

| iter | v45 avg_final | v42 avg_final | v45 min_final | v42 min_final |
| --- | --- | --- | --- | --- |
| 8  | 0.594 | 0.869 | 0.344 | 0.708 |
| 16 | 0.468 | 0.843 | 0.292 | 0.740 |
| 24 | 0.509 | 0.823 | 0.208 | 0.646 |

**排查(逐一证伪,非猜测):**
- v43-vs-v42 的 train.py diff(77 行)**全是 gate `_by_stage` 参数**,不碰 reward/boost/density/entropy/PPO 超参 → 代码漂移不影响训练。
- v45 与 v42 每次 eval 的 `density_penalty_coef`(0.0/0.02/0.04)和 `escape_boost`(0.75/0.75/0.76)**逐位相同** → 我的 gate 改动没改训练动态(符合设计:两者早期都卡 stage0)。
- 训练旋钮既然逐位相同,~0.3 的 avg_final gap 只剩一个来源:**seed**。

**结论:配置完全相同、只换 seed,iter24 的 avg_final 就能从 0.82 掉到 0.51。这说明整个项目用单 seed 对比 run 是不可靠的——v40→v43 很可能有一大块"迭代"是在追 seed 噪声。** 这是比"gate 修没修好"更重要的发现。

**进一步定位方差在哪一层(已实证):** v42 与 v45 的 multi-eval `episode_seeds` **逐位相同**(同 `multi_eval_seed_base=411232` → 同 RNG 序列 → 同 40 个 predator 配置)。所以两者是在**完全相同的 held-out 测试集**上评估的 → 0.3 的 gap **纯粹来自训练出的策略不同(训练 seed 421842 vs 450060)**,不是 eval 抽样噪声。

**推论:eval harness 本身是健全的(固定、共享配置),不需要"加多 seed eval"。方差在训练层。正确的方法论修正是:每个候选配置跑多个训练 seed,比较 held-out avg_final 的分布(均值±方差),而不是拿单个训练 run 的单点去 merge。**

### 待补
- [ ] 等 v45 跑完看 iter40–60 是否回升(判断这个 seed 是否只是前期不利)。
- [ ] 下一步方法论修正:承重对比必须**多 seed**(≥3)取均值/方差,单 seed run 不能作为 merge 依据。

## 多 seed baseline 结果(baseline_v42cfg,3 seed,gate=avg_final)

`multiseed_eval.py --label baseline_v42cfg --seeds 3`(seed 700000/701000/702000),用固定共享 eval,取每个 run 最后 3 次 multi-eval 的 avg_final 均值:

| seed | final avg_final | gate 轨迹 |
| --- | --- | --- |
| 700000 | 0.867 | stage 0→1(iter40)→2(iter56) |
| 701000 | 0.847 | 同上 advance |
| 702000 | 0.829 | 同上 advance |
| **均值±std** | **0.848 ± 0.016** | 三个都 advance 到 stage 2 |

对照(单 seed,同配置):v42(旧 step-count gate)=0.735 且**下降**;v45(新 gate, seed 450060)=0.535。

### 两个决定性结论

1. **修好的 gate 真的会 advance 了。** 三个新 run 都出现 `gate=advance stage 0→1→2`——这是 **v41–v43 三轮永远卡在 stage 0** 的那个晋级,现在按 held-out avg_final 正常触发。gate fix 达成设计目标。

2. **gate 不是在制造差异,是在正确响应差异。** v45 的 0.535 不是 gate 的错:那个 seed 早期就弱(iter8=0.594),gate 正确地 `failure_hold` 不晋级它;而三个好 seed 早期强,gate 把它们 advance 到 stage 2 并稳在 0.83–0.87。**gate 拒绝晋级弱策略、晋级强策略,正是想要的行为。**

### 噪声带的真实形状(修正前面的说法)

前面基于 v42/v45 两点估的"方差 0.34 宽"是**不准确的**。真实情况是**双峰**:好 seed 紧密聚在 ~0.85(3 seed std 仅 0.016),偶有坏 seed(v45=0.535)早期没起来、gate 正确不晋级。所以:
- **可比的 baseline 均值 = 0.848 ± 0.016**(gate 能晋级的健康 run)。
- 但**任何单 seed 仍可能抽到 0.53 的坏 run**,所以单 seed 依旧不可作 merge 判据——必须多 seed 取均值。

## Learning / 下一步

1. **Direction 1 成功交付**:gate 用 held-out avg_final 判晋级,实测会 advance(修好了 v41–v43 的空转),健康 baseline = **0.848±0.016**。gate 代码 + `multiseed_eval.py` + SOP 方法论节一起 merge。
2. **方法论立规**:承重结论只看 held-out avg_final(不看 on-policy sr);merge 判据必须多 seed 均值超过 baseline 均值+噪声带。已写进 SOP。
3. **Direction 2/3** 都在此 baseline(0.848±0.016)上做,只有多 seed 均值显著超过才 merge。候选:(a) 针对早期弱 seed 的稳定性(减少 0.535 这类坏 run 的概率,如 warmup/lr schedule);(b) 直接优化最难 held-out 样本的 min_final;(c) 天花板改动(连续动作 / 群体感知)。
