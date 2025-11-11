# Codex-Usage

## 流程概述
- 按 `SOP.md` 循环推进：每个版本位于 `experiments/vX/`，包含 `train.py`、`dev_vX.md` 以及 `artifacts/`（checkpoints / logs / tb_logs / plots）。
- 目标是让新的 `dev_vX.md` 记录计划、实验与结论，`train.py`/artifacts 可复现实验。
- 本机硬件：16 物理核 / 128 GB 内存，足够支撑 64~128 个并行环境；缺省超参数请至少使用 `--num_envs 64`。

## 运行前准备
1. **定位仓库**：`codex ... -C /path/to/fish_rl` 或先 `cd` 进仓库。
2. **虚拟环境约定**：始终检查根目录的 `venv`，若不存在则 `uv venv venv` 创建；存在则 `source venv/bin/activate`。安装依赖使用 `uv pip install ...`。
3. **确认资源**：若需要更大并行度，可把 `NUM_ENVS` 或 `--num_envs` 提升至 128；注意同步更新文档与日志。
4. **风险评估**：`--dangerously-bypass-approvals-and-sandbox` 会关闭所有交互防护，仅在隔离环境或已经备份时使用。

## 使用 `scripts/run_codex_iterations.sh`
该脚本可自动对指定版本区间运行 Codex Exec 流程：

```bash
scripts/run_codex_iterations.sh 2 3 --model gpt-5-codex --json
```

脚本行为：
- 为目标版本创建 `experiments/vX/` 骨架及 `artifacts/` 子目录；
- 生成提示，要求 Codex 阅读 `SOP.md` 与上一版 `experiments/v{X-1}/dev_v{X-1}.md`，然后在 `experiments/vX/` 内记录新的 dev 文档、训练脚本与输出；
- 强制遵守 `venv` 约定、利用 ≥64 个并行环境，并在结束时输出 `git status` 摘要；
- 所有 Exec 日志写到 `codex_runs/`（可通过 `CODEX_RUN_LOG_DIR` 覆盖）。

如需单轮手动运行，可设置 `START_VER=END_VER`。可以通过环境变量覆盖 `CODEX_BIN`、`CODEX_APPROVAL_FLAGS`（默认 `--dangerously-bypass-approvals-and-sandbox`）或 `EXPERIMENTS_ROOT`。

## 单次 Exec Demo
若只需快速验证 Codex 能否遵守 `venv` 约定，可使用：

```bash
codex exec --dangerously-bypass-approvals-and-sandbox \\
  "Create hello.py that prints 'Hello, World!', make sure to create/activate venv if needed, then run it."
```

预期行为：
1. 检查 `venv`，如无则 `uv venv venv` 并激活；
2. 写入 `hello.py`；
3. 通过 `source venv/bin/activate && python hello.py` 运行并输出 `Hello, World!`。

## 最佳实践
1. **控制作用范围**：把复杂实验放在新的 `experiments/vX/`，历史 artifacts 保持只读；演示脚本可放在临时文件中并在提交前清理。
2. **留痕**：保留 Codex 命令与输出（脚本默认写入 `codex_runs/`），便于审计与回溯。
3. **Git 审查**：在 Exec 前后运行 `git status`、`git diff`，必要时使用 `git add -p` 选择性提交。
4. **清晰提示**：把需求拆成明确步骤（例如“更新 dev_v3.md 并跑 100 iteration”），避免一次性让模型做太多无关工作。
5. **适时退出危险模式**：完成自动化实验后可改用默认审批模式或 `--full-auto`，防止误操作。

## 进一步扩展
- 使用 `--json` 或 `--event-log` 获取更结构化的 Exec 输出，便于机器解析；
- 若需要让 Codex 长时间迭代，可在提示中引用 `scripts/run_codex_iterations.sh` 的要求，或直接调用该脚本。
