# dev_v49: correct-identity PPO baseline (round 1 of 10)

## 1. 问题

v48 前置审查确认旧训练环境 `SingleFishEnv`（v45/v47 `train.py`）存在动作/奖励/观测身份错位：每步返回轮换鱼的单条观测、把一个动作广播到全群、返回全群平均奖励；一条 PPO rollout 记录因此混合了不同鱼的身份，GAE / bootstrap 跨鱼、跨死亡。评估侧用逐鱼推断（`policy_per_fish`），与训练口径不一致，同 checkpoint 下报告集差 18.49 个百分点。v48 将其记为执行口径差，不是修复训练的因果收益。

本轮（v49）目标是设计并**实际训练评估**一个语义正确、简单可信的 PPO baseline，完成第 1 轮。不改世界物理/动作/初始 spawn；不改旧版本；不套复杂 gate/tail 架构。

## 2. 世界物理解耦（不是严格等价）

`experiments/v48/tests/test_env_decomposability.py` 针对已安装的 `fish_env.py` 证明：

- 捕食者轨迹不依赖鱼的动作，也不依赖鱼的状态；
- 鱼与鱼之间无碰撞（死亡只在鱼-捕食者距离 < 0.7）。

由此可成立的结论限于**有限范围的物理解耦**：给定同一初始状态和同一焦点鱼动作序列，其他鱼如何动作不改变焦点鱼的物理轨迹与死亡，关闭邻居观测后同一轨迹上的焦点观测也不因其他鱼而变。这足以说明固定焦点方案为何能修复旧 wrapper 的身份错位。

但**不能**据此声称严格单智能体 MDP 或严格多鱼等价。11 维观测在捕食者不可见时隐藏其位置与速度，也不含 timestep，故即便 density=0 也不是完整 Markov 状态；状态转移仍依赖捕食者。density penalty（0.05）非零时奖励还读取其他存活鱼的邻域数量：训练时其他鱼 HOLD、评估时其他鱼执行策略，两种 rollout 共用同一 reward function 不等于奖励分布相同。按 95 个潜在邻居、`neighbor_average_count=6`、`density_target=0.4`、`coef=0.05`、`REWARD_SCALE=0.1`，单步密度罚上界约 0.0772，相对常见总存活奖励约 0.7 并非自动可忽略；它不直接改变存活动力学，但可能经训练目标改变学到的策略。这是需要准确陈述的局部观测与奖励上下文差异，不是可忽略的二阶项。

## 3. 方法

### 3.1 单焦点包装器

`experiments/v49/single_fish_env.py` 中的 `SingleFishControlEnv` 包装 `FishEscapeEnv`：

- 每局固定一条焦点鱼 `focal_id`，观测恒为该鱼自身观测；
- `step(action)` 只对焦点鱼施动作，其余存活鱼保持（不广播）；
- 奖励为焦点鱼**自身**的逐鱼奖励（沿用 env 原奖励函数）；
- `terminated=True` 当且仅当焦点鱼死亡（死亡惩罚只计一次，bootstrap 到 0）；`truncated=True` 当且仅当存活到 500 步上限。
- 观测/动作空间与评估器喂给策略的完全一致（11 维自观测 + 5 离散动作）。

死亡或截断后，包装器用新焦点鱼重置，形成"一鱼一 episode"的共享策略训练流。焦点 id 由 episode RNG 抽样，seed 固定则确定。

**500 步结束的约定**：基础世界把 500 步作为 `terminated`；wrapper 把存活到 500 步改为 `truncated`，SB3 用结束前 focal 的 `terminal_observation` 加 `gamma*V` bootstrap。这是把时间限当作采样截断的**训练目标设计**，不是对原生有限 500 步任务的唯一正确终止定义（存活到末步也可以 terminal、价值归零；剩余时间若要精确表达有限时域任务通常还需进入状态）。它是本 baseline 的显式约定，不能写成唯一正确的 termination 语义，也不能据此把后续收益都归给奖励。

### 3.1b 训练进度与指标记录边界

SB3 在 callback 的 `on_rollout_end` 保存，早于该轮的 `train()` 更新，故阶段标签 iN 实际对应 N−1 次已完成 PPO 更新。实际 checkpoint 元数据：i50/i100/i150/final 分别为 49/99/149/200 次更新（`num_timesteps` 153600/307200/460800/614400，`_n_updates` 490/990/1490/2000，n_epochs=10），三个 run 一致。每个 `train_metrics.jsonl` 只有 199 行（iteration=1…199，非 200）；第 n 行的 `train/*` 来自第 n 次更新，但 episodes/deaths/truncs 计数已含第 n+1 次 rollout；第 200 次更新的 loss/entropy/KL 未记录。由于 callback 早于 `dump_logs`，这些文件不含 `rollout/ep_rew_mean`、`rollout/ep_len_mean`、`time/total_timesteps`，公开可核验的是 loss、value loss、policy gradient loss、entropy、KL、clip、explained variance、learning rate、n_updates、wall time 与累计计数。详见 `artifacts/metadata_notes.json`。

### 3.2 训练

`experiments/v49/train.py`：stock SB3（`SubprocVecEnv` + `PPO`），无自定义策略、无 gate/tail。

| 项 | 值 |
| :--- | :--- |
| 算法 | PPO MlpPolicy，pi/vf 384×384 |
| num_envs（每 run） | 6（SubprocVecEnv，每进程 torch 1 线程） |
| n_steps / batch_size / n_epochs | 512 / 1024 / 10 |
| lr / gamma / gae_lambda / clip / ent_coef | 3e-4 / 0.99 / 0.95 / 0.2 / 0.02 |
| iterations / total steps | 200 / 614,400 |
| checkpoints | iter 50 / 100 / 150 / final |
| 训练 seed（模型/动作 RNG） | 4901001 / 4901002 / 4901003 |

三个 run 并行执行，各约 23 分钟 wall（约 440 env-steps/s），机器 32 核未过载（18 进程）。

**独立性限定**：三个 run 的模型/动作随机 seed 不同，但 SB3 `BaseVecEnv.seed` 按 `seed+rank` 分配，每 run 用 6 个环境，故 run1/2/3 的初始环境 RNG 流为 4901001–06 / 4901002–07 / 4901003–08，18 个 worker 只有 8 个不同环境 seed，相邻 run 两两共享 5/6。因此三-run std 是这三个 run 的描述性离散度，不是独立样本的泛化 CI；v50 应使用不重叠的环境 seed 段。

### 3.3 奖励与观测口径（明确 baseline 定义，非静默改动）

- 观测：**关闭邻居特征**（11 维局部观测），与评估一致；评估配置本身邻居奖励系数为 0。局部观测不构成严格 Markov 状态（捕食者不可见时隐藏其位置、速度，且无 timestep）。
- 奖励：保留 env 原逐鱼奖励，**含 density penalty（coef 0.05）不变**。该密度项依赖其他鱼（训练时它们保持，评估时执行策略），故焦点奖励的分布随训练/评估上下文不同，且无理论误差界；动力学、观测、终止不受其他鱼动作影响。这是局部观测 + 奖励上下文偏移的既定局限，不是可忽略的二阶项，也不构成严格多鱼等价。
- eval 分布（96 鱼、escape_boost 0.8、pre-roll/bias 配置）与 v48 相同，规则臂数字可直接对照。

### 3.4 种子与选择/报告流程

`experiments/v49/common.py` 冻结三组与 v48（481xxx）、历史（555xxx）互不相交的 RNG seed：selection=482101（20 局）、report=482102（40 局）。两组生成的 episode seed 实际无重复，selection/report 无交集，也均未命中旧银行生成的 episode seed。

**选择流程的诚实口径**：可见产物中 report 先完成（report.jsonl/summary mtime 约 00:09:55、logged wall 571.6s），selection 后完成（约 00:15:29、logged wall 254.3s）。没有独立的先行冻结选择 manifest 或 tie-break 记录，也没有自动 selection→report 管线。因此这是**事后按 selection 集每 run 均值 argmax 选出阶段、再在已生成的 report 上描述性汇总**，不能声称"查看 report 前已冻结选择"，也不能宣称本次"无选择过拟合"。全部 16 臂的 report 数据保留作探索性数据；final-only 均值提供一个无需阶段选择的描述性参照（见 §4.4）。

## 4. 结果

### 4.1 意义测试（训练前，全部通过）

`experiments/v48/.venv/bin/python experiments/v49/tests/test_v49_semantics.py`：

```
PASS test_fixed_focal_id_whole_episode
PASS test_other_fish_death_keeps_focal
PASS test_focal_death_terminates_once_with_penalty (reward=-50.00)
PASS test_truncation_at_horizon_bootstraps
PASS test_no_broadcast_only_focal_moves
PASS test_spaces_match_eval_distribution
PASS test_wrapper_policy_usable_under_eval_loop
PASS test_focal_reward_matches_world_reward_function
PASS test_obs_not_mean_of_alive
```

覆盖：固定 ID 映射、其他鱼死亡不换 ID、焦点死亡只奖罚一次且终止、时间限截断标志、动作独立不广播、train/eval 输入动作空间一致、奖励为自身值非均值。单环境一鱼一行，不引入死槽 padding（无需 mask）。

注意：`test_truncation_at_horizon_bootstraps` 只验证 wrapper 返回的 `truncated`/`terminated` 标志，**不是**对真实 value-target bootstrap 的完整检验（后者由 SB3 的 `TimeLimit.truncated` 分支实现）。`test_no_broadcast_only_focal_moves` 证明的是相对全 HOLD 的孪生世界只有 focal 状态产生差异；HOLD 是保持速度而非静止。九个测试在本次返修前由主线程重跑全过（含调用 `model.learn()` 的集成测试）。

### 4.2 训练流累计存活比例（焦点鱼；非单调、非独立测试曲线）

| iter | run1 | run2 | run3 |
| :--- | :--- | :--- | :--- |
| 10 | 0.756 | 0.732 | 0.738 |
| 50 | 0.842 | 0.809 | 0.807 |
| 100 | 0.844 | 0.828 | 0.800 |
| 150 | 0.842 | 0.825 | 0.807 |
| 199 | 0.854 | 0.848 | 0.812 |

表中数值是整个非平稳训练流上的**累计**存活比例（截断/(截断+死亡)，从训练开始累计），不是最新策略的独立测试曲线。三 run 从 ~0.74 整体升到 ~0.81–0.85，但过程非单调：198 次相邻累计比较中，run1/2/3 分别有 69/52/77 次下降。最终 entropy_loss 约 -1.4358 / -1.4372 / -1.4232（策略熵约 1.42–1.44，全程最低也未到 -1.0），说明策略动作分布发生变化，**不证明性能收敛或其因果来源**；有 final 低于中期 checkpoint 也说明性能曲线非单调。训练用随机采样动作，评估用确定性 argmax，两者不应等同。

### 4.3 40 局评估（report，482102；neighbor off）

| 臂 | 最终平均存活率 | std | s@1 | s@100 | s@500 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `rule_flee_lead` | 0.9771 | 0.020 | 0.9857 | 0.9846 | 0.9771 |
| `rule_safe_top` | 0.9302 | 0.034 | 0.9833 | 0.9302 | 0.9302 |
| run1 i150 | 0.9266 | 0.042 | 0.9833 | 0.9378 | 0.9266 |
| run2 i100 | 0.9146 | 0.056 | 0.9823 | 0.9253 | 0.9146 |
| run1 final | 0.9060 | 0.053 | 0.9831 | 0.9260 | 0.9060 |
| run3 i50 | 0.8846 | 0.057 | 0.9826 | 0.9253 | 0.8846 |
| run2 final | 0.8708 | 0.060 | 0.9823 | 0.9146 | 0.8708 |
| run1 i50 | 0.8682 | 0.070 | 0.9826 | 0.9229 | 0.8682 |
| run3 final | 0.8826 | 0.059 | 0.9823 | 0.9273 | 0.8826 |
| `untrained`（run1 初始化权重） | 0.7914 | 0.150 | 0.9828 | 0.8893 | 0.7914 |
| `rule_hold` | 0.7453 | 0.064 | 0.9826 | 0.9104 | 0.7453 |

三个训练 run 全部报告，未剔除。关键对比（配对，vs `rule_hold`，40 局，bootstrap 95% CI）：

- `rule_flee_lead` +0.2318 [+0.2138, +0.2503]，40/40 胜。
- `rule_safe_top` +0.1849 [+0.1638, +0.2068]，40/40 胜。
- 训练臂全部正向高于 `rule_hold`（最低 run3 i150 +0.0753 [+0.0497, +0.0982]）。
- `untrained` 0.7914，CI 跨 0（[-0.0018, +0.0909]）。这是**未拒绝零差异，不是等价证明**。

`untrained` 是 run1 同一初始化的**随机权重确定性 argmax** 策略，不是均匀随机策略：其平均动作分布（每局先按存活鱼-步归一化、再跨局等权平均）为 [0.055, 0.360, 0.433, 0.082, 0.071]，左/右转合计约 79%，远偏离均匀 20%/动作。直接 trained−untrained 配对差支撑"学到更有效行为"的有限结论（selected 三 run 均值 − untrained = +0.1172，CI [+0.0779, +0.1610]），但只有一个初始化对照，且并非每个阶段都显著（run2 i50 +0.0359，CI [−0.0091, +0.0859]；run3 i150 +0.0292，CI [−0.0193, +0.0820]）。本轮**没有** rule_random 臂，不能用 `untrained` 代替均匀随机对照。上述额外配对 CI 是从现有原始 report 记录后处理得到（固定模型下的 episode 抽样），不含训练随机性，不在已归档 `analysis.json` 中。

### 4.4 阶段选择的描述性汇总（非预注册选择）

事后按每 run 在 selection 集（482101，20 局）的均值 final survival argmax 选出阶段，再取该阶段在 report 集的数字：

| run | selection argmax 选中阶段 | report 集该阶段存活率 | report 集 final 存活率 |
| :--- | :--- | :--- | :--- |
| run1 | i150 | 0.9266 | 0.9060 |
| run2 | i100 | 0.9146 | 0.8708 |
| run3 | i50 | 0.8846 | 0.8826 |

选中阶段 report 均值 0.9086（三 run sample std 0.0216），final 均值 0.8865（std 0.0179）。两组恰好排名一致是可复核事实，但这**不恢复 report 从未被触碰的测试集地位，也不证明不存在选择偏差**。report 全阶段数字保留作探索性数据；若只需一个无需阶段选择的参照，用 final-only 均值。不要仅为复现相同数字而重评旧 40 局来"恢复"隔离性。

### 4.5 动作分布（report，每存活鱼-步均值）

| 臂 | 前进 | 左转 | 右转 | 减速 | 保持 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `rule_flee_lead` | 0.01 | 0.03 | 0.03 | 0.00 | 0.92 |
| `rule_hold` | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| `untrained` | 0.06 | 0.36 | 0.43 | 0.08 | 0.07 |
| run1 i150 | 0.00 | 0.07 | 0.03 | 0.77 | 0.13 |
| run2 i100 | 0.00 | 0.07 | 0.36 | 0.57 | 0.00 |

动作分布口径：先按每局的存活鱼-步归一化，再跨局等权平均（episode 等权，非全体 fish-step 合并加权）。学到策略以转向 + 减速为主体，与手写避让规则的动作构成不同；未训练策略是 run1 初始化的确定性 argmax，偏向左右转（合计约 79%），不是均匀随机。

## 5. 限制

1. **未收敛**：200 iter / 614k steps 是既定初始预算下的 learning baseline，不是收敛 SOTA。曲线非单调；不宣称收敛。
2. **低于手写规则**：训练策略 0.87–0.93 < `rule_flee_lead` 0.977，仍落后于已知手工避让启发式约 4–10 点。本轮只建立正确、可复现的训练口径，不追求击败规则，也未识别主要瓶颈（奖励、部分可观测、探索或优化）。
3. **训练/评估奖励上下文不同**：训练时其余鱼 HOLD，评估时其余鱼也由同一策略逐鱼控制。焦点鱼的物理轨迹与死亡不受其他鱼动作影响（物理解耦），但含 density penalty（0.05）时焦点奖励的分布不严格相同，且该密度项无理论误差界。这是局部观测 + 奖励上下文偏移的既定局限，不构成严格多鱼等价。若后续要考察拥挤行为，需改奖励或观测，属 v50+。
4. **单环境一鱼**：未使用多鱼 batch，故不涉及死槽 padding / gradient mask；这也意味着 throughput 受 env 限制（密度奖励 O(96²)，约 90% 时间）。
5. **数值范围与独立性**：所有数字限 report 集 482102 / selection 集 482101 的 40 / 20 局配对；三 run 共享部分环境 seed 流，其 std 是描述性离散度，不外推独立样本泛化 CI 或全局最优。
6. **复现层次**：公开数值资产使曲线、预算、耗时、训练结果与依赖版本可检查；但 zip 不公开，第三方无法在不重训的情况下复现 PPO 权重执行，也不能保证跨平台逐 bit 复现（依赖为宽泛下界，非精确锁文件）。

## 6. 复现

从仓库根目录：

```bash
# 环境（复用既有 v48 venv，未新建）
source experiments/v48/.venv/bin/activate   # Python 3.12.9, torch 2.13.0, sb3 2.9.0

# 意义测试
python experiments/v49/tests/test_v49_semantics.py

# 三个 run 训练（各自独立目录；checkpoint 保存于 iter 50/100/150 + final）
for s in 4901001 4901002 4901003; do
  python experiments/v49/train.py --seed $s --num-envs 6 --iterations 200 \
    --n-steps 512 --batch-size 1024 --checkpoint-iters 50,100,150 \
    --out-dir experiments/v49/artifacts/runs/seed$s
done

# report 集评估（482102，40 局）：规则 + untrained + 三 run 全阶段
# untrained 用字面量 "untrained" 作为路径；训练臂路径指向各 run 的 checkpoint
python experiments/v49/evaluate.py --seeds report --reference rule_hold --neighbor off \
  --workers 8 \
  --arm rule_hold --arm rule_flee_lead --arm rule_safe_top --arm untrained=untrained \
  --arm run1_i50=experiments/v49/artifacts/runs/seed4901001/checkpoints/model_iter_50.zip \
  --arm run1_i100=experiments/v49/artifacts/runs/seed4901001/checkpoints/model_iter_100.zip \
  --arm run1_i150=experiments/v49/artifacts/runs/seed4901001/checkpoints/model_iter_150.zip \
  --arm run1_final=experiments/v49/artifacts/runs/seed4901001/checkpoints/model_final.zip \
  --arm run2_i50=experiments/v49/artifacts/runs/seed4901002/checkpoints/model_iter_50.zip \
  --arm run2_i100=experiments/v49/artifacts/runs/seed4901002/checkpoints/model_iter_100.zip \
  --arm run2_i150=experiments/v49/artifacts/runs/seed4901002/checkpoints/model_iter_150.zip \
  --arm run2_final=experiments/v49/artifacts/runs/seed4901002/checkpoints/model_final.zip \
  --arm run3_i50=experiments/v49/artifacts/runs/seed4901003/checkpoints/model_iter_50.zip \
  --arm run3_i100=experiments/v49/artifacts/runs/seed4901003/checkpoints/model_iter_100.zip \
  --arm run3_i150=experiments/v49/artifacts/runs/seed4901003/checkpoints/model_iter_150.zip \
  --arm run3_final=experiments/v49/artifacts/runs/seed4901003/checkpoints/model_final.zip \
  --out experiments/v49/artifacts/results/report.jsonl

# selection 集评估（482101，20 局；同 12 个训练臂 + 3 规则臂，不含 untrained）
# 将上条命令的 --seeds report 改为 --seeds selection、--out 改为 selection.jsonl 即可。
# 选择规则：对每个 run 取该 run 各阶段 selection 均值 final survival 的 argmax。

# 汇总学习曲线与评估
python experiments/v49/analyze.py --runs-dir experiments/v49/artifacts/runs \
  --eval-summary experiments/v49/artifacts/results/report.summary.json \
  --out experiments/v49/artifacts/results/analysis.json
```

依赖版本随每个 run 的 `config.json` 与 `run_summary.json` 保存（含 `python`/`platform`/包版本快照）；训练指标逐 iteration 写 `train_metrics.jsonl`。`config.json` 只保存 argparse 字段，不含可执行绝对路径或原始完整 argv；完整 predator heading/speed bias 以 `experiments/v49/common.py` 为准。checkpoint（*.zip）默认不入库。上述命令重跑得到的是新流程的产物；它不改变已归档旧结果的解释边界，也不"洗白"旧流程的选择时序。

## 7. 下一步

v50 的承重问题：本轮奖励与最终生存率**可能不对齐，需等预算验证**。活鱼的存活奖励 +2 经 `REWARD_SCALE=0.1` 后为 +0.2；视野外距离奖励 +5 缩放后为 +0.5（常见基准合计 +0.7/步），死亡 −50 另在缩放后加入，gamma=0.99 又折扣远期收益，故不能简单用 500×2 推断 −50 太小或 +2 主导。需要判断当前训练策略落后手写规则的主要瓶颈是否在奖励与生存目标对齐，还是部分可观测、探索或优化；并用**不动物理**的等预算多 run 对比验证。**不把世界后段视为无风险**：v48 反事实记录 100 步后 flee_lead 继续控制 vs 改 HOLD 的配对差 +0.2141（CI [+0.1971, +0.2310]），本方 `rule_hold` s@100=0.9104 → s@500=0.7453 也不支持后段无风险；后段仍需必要控制，safe_top 样本内的安全不可外推到所有策略与轨迹。v50 另需为非重叠环境 seed 段、以及"查看确认集前冻结 checkpoint 选择规则"做前置约定。
