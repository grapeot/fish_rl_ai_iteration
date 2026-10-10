# Fish RL Iterations

> Reinforcement-learning fish survival, managed as an AI-operated laboratory.

📊 **历史结果与方法论教训:[grapeot.github.io/fish_rl_ai_iteration](https://grapeot.github.io/fish_rl_ai_iteration/)**（v44–v47 冲刺结论，0.899 为当时报告集上的单策略成绩）

## 2026-10：世界设定审查与十轮迭代（v49–v58）

保留原有局部避险世界，修正训练身份对应并采用存活/死亡奖励；初速随机化训练未达到新场景确认中的替换门槛。手写避让规则在最终同场景比较中仍领先 PPO，后续重点是诊断学到的威胁响应。[十轮结果与当前基线](experiments/iteration_review_v49_v58.md)包含逐轮证据、负结果和复现边界；当前训练入口见 [v50](experiments/v50/dev_v50.md)，历史页面分数不与新场景分数直接排序。

### 同场景前后对照与视频

在同一批 64 个新场景中，每局 96 条鱼运行 500 步，旧模型平均存活率 **91.11%**，新方案三模型均值 **93.72%**，手写规则 **96.97%**。新旧差为 **2.61 个百分点**，约每局多活 2.5 条鱼；这里统计最终活鱼比例，不是通关胜率。

▶ [观看 50 秒三栏对比视频](https://grapeot.github.io/fish_rl_ai_iteration/assets/before_after_202610.mp4) · [基准定义与比较边界](demos/before_after_202610/benchmark_explainer.md)。视频是预设首个场景的单局过程，不能代替 64 局均值。

## 为什么存在
- **项目目标**：训练一套策略，让大量小鱼在高速捕食者面前依旧保持高存活率。
- **工作方式**：人类扮演项目经理，搭建 SOP、工具与基线；自动化代理（Codex CLI 等）按照 SOP 自主循环，记录 `dev_vX.md`、运行实验、写日志、提交至 GitHub。
- **成功判据**：在困难配置（捕食者速度快、小鱼多、动作受限）下依旧能保持高终局存活率，并且任何时间点都能通过仓库重现最近一次实验。

## 架构速览
```
fish_rl/
├── fish_env.py                    # 通用环境定义
├── experiments/
│   └── v2/
│        ├── train.py             # 该 iteration 的训练脚本
│        ├── dev_v2.md            # 工作文档（计划/结果/下一步）
│        └── artifacts/
│             ├── checkpoints/    # SB3 模型与 stats.pkl、曲线
│             ├── logs/           # 训练日志（txt）
│             ├── tb_logs/        # TensorBoard events
│             ├── plots/          # PNG/SVG 等静态图（training_curve 等，纳入 git）
│             └── media/          # mp4/gif（500 帧以内，纳入 git 以远程查看）
├── scripts/run_codex_iterations.sh # 自动迭代脚本
├── SOP.md                        # 操作手册
├── codex_usage.md                # Codex CLI 指南
├── requirements.txt
└── venv/                         # uv venv venv 创建的环境
```

未来的新版本按 `experiments/v3/`, `experiments/v4/`……依次追加，历史 artifacts 只读。

## 循环式工作流
1. 阅读上一轮 `experiments/v{X}/dev_v{X}.md` 的 learning/plan，开启 `dev_v{X+1}.md` 草稿。
2. （可选）小规模 sanity run，确认旧基线仍可复现。
3. 更新计划、添加日志/metrics，必要时修改 `experiments/v{X+1}/train.py`。
4. 在 32 核 / 512 GB 机器上运行 64~128 并行环境的大规模训练，所有输出写入 `artifacts/`。
5. 生成曲线/媒体并在 `dev_v{X+1}.md` 中引用，记录命令、指标、路径、下一步计划。
6. `git add` + `git commit` + `git push origin master`，确保远端始终可追溯。

详尽步骤见 [SOP.md](./SOP.md)。

## 历史 v2 运行示例

以下保留早期版本命令；恢复当前实验请使用上方 v50 入口及对应版本的复现说明。
```bash
# 1) 安装依赖
uv venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
uv pip install -r requirements.txt

# 2) 运行历史 v2 迭代
python experiments/v2/train.py --total_iterations 100 --num_envs 128 --num_fish 25 \
  > experiments/v2/artifacts/logs/train_v2_iter100.log

# 3) 查看曲线 / TensorBoard
python visualize.py --stats experiments/v2/artifacts/checkpoints/training_stats.pkl
tensorboard --logdir experiments/v2/artifacts/tb_logs --port 6006
```
日志、模型、plot、media 会自动写到 `experiments/v2/artifacts/`。若需要录制逃逸视频，可利用 `watch.py` / `visualize.py` 输出 mp4 并放入 `media/`。

## 自动化迭代
- 执行 `scripts/run_codex_iterations.sh 2 3 --model gpt-5-codex` 可让 Codex CLI 读取 SOP/上一轮文档，生成新的 `experiments/v3/`、跑实验、写日志并提醒提交。
- 该脚本会在提示中强制遵守 `venv` 约定、要求 ≥64 并行环境、并在结束阶段执行 `git status`/commit/push`。运行日志保存在 `codex_runs/`（可用 `CODEX_RUN_LOG_DIR` 覆盖）。

## 感知拓展（已完成,非 Stretch Goal）
- 群体感知早已启用并成为默认配置:`include_neighbor_features=True`(obs 18 维 = 11 基础 + 7 邻居特征,`neighbor_radius=3.0`,平均最近 6 邻居),网络为 384×384 MLP。
- (v46 勘误:本节旧文案称其为"长期目标",导致后续 session 误以为未实装。以 `fish_env.py` 与 train.py argparse 默认值为准。)

## 贡献指南
- 所有代码/文档改动必须附带 `experiments/vX/dev_vX.md` 的相应记录。
- artifacts 目录中的二进制文件不入库（由 `.gitignore` 排除），但其生成脚本和路径必须写进文档。
- 若引入新依赖，请更新 `requirements.txt` 并在 README 中说明用途。

## 历史状态（2026-07，v47）
- 最优单策略 held-out 终局存活率 **0.899**(96 鱼,独立 report set);健康 run baseline **0.848±0.016**。
- 评估方法论(held-out only、多 run 分布对比、best-checkpoint 交付)见 `SOP.md` 方法论节,结论综述见上方 Pages 链接。
- 开放问题:中期训练(iter10–25)侵蚀 held-out 泛化的机制;高密度课程的 within-run 正信号(v47)。
