# Fish RL Iterations

> Reinforcement-learning fish survival, managed as an AI-operated laboratory.

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

## 人类快速上手
```bash
# 1) 安装依赖
uv venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
uv pip install -r requirements.txt

# 2) 运行当前迭代（建议 64+ 环境）
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

## 贡献指南
- 所有代码/文档改动必须附带 `experiments/vX/dev_vX.md` 的相应记录。
- artifacts 目录中的二进制文件不入库（由 `.gitignore` 排除），但其生成脚本和路径必须写进文档。
- 若引入新依赖，请更新 `requirements.txt` 并在 README 中说明用途。

## 下一步
- v2 已经添加 reward scaling 与改进的 logging，存活率仍在 ~70% 徘徊。
- v3 计划：引入单鱼 VecEnv、扩大并行度、实现 deterministic evaluator，并继续按照 SOP 推进。
