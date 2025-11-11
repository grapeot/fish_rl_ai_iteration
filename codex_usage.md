# Codex-Usage

## 场景概述
Codex CLI 是一个可以直接在本地仓库中读写文件、运行命令的智能体。为了复现“ExecEC”式的全自动体验，我们可以使用 `codex exec` 搭配 `--dangerously-bypass-approvals-and-sandbox`，让模型在一个非交互会话里自行规划与执行。本文以 `fish_rl` 仓库为例，记录一次使用该模式生成并运行 Python Hello World 的流程，并总结若干最佳实践。

## 运行前准备
- **确认工作目录**：使用 `-C` 或先切到目标仓库，避免 Codex 无法找到 Git 背景。
- **虚拟环境约定**：按照项目规范，先检查当前目录是否存在 `venv` 文件夹；若不存在，用 `uv venv venv` 创建；若已存在，则 `source venv/bin/activate` 激活后再进行 Python 操作；依赖安装请使用 `uv pip install ...`。
- **评估风险**：`--dangerously-bypass-approvals-and-sandbox` 会关闭所有命令确认，并给予模型完整的磁盘与网络权限，仅在外部环境已隔离（例如一次性 dev 容器）时才使用。

## Hello World 演示
下面的命令通过 Exec 模式一次性让 Codex 写入并运行脚本：

```bash
codex exec --dangerously-bypass-approvals-and-sandbox "Create a Python script hello.py that prints 'Hello, World!' and run it."
```

执行日志要点：
1. Codex 自动列出目录并确认 `venv` 不存在，于是运行 `uv venv venv` 创建虚拟环境。
2. 生成 `hello.py`，内容为 `print('Hello, World!')`。
3. 使用 `source venv/bin/activate && python hello.py` 在虚拟环境中运行脚本，输出 `Hello, World!`。

该流程验证了 Codex 在危险模式下会主动遵守仓库约定（如虚拟环境检查）并完成任务。

## 最佳实践
1. **最小化作用范围**：为演示类任务准备独立分支或一次性文件（如 `hello.py`），结束后按需清理，避免污染主干。
2. **留存日志**：危险模式不会弹出审批，建议将命令行输出或 Codex session id（如本次的 `019a7177-31f1-7980-8641-8b3ee1763e47`）记录在文档或 issue 中，方便审计。
3. **搭配 Git 审查**：在执行前/后使用 `git status`、`git diff` 自查，必要时通过 `codex apply` 或手动 `git add -p` 控制变更。
4. **限制指令粒度**：即使在危险模式，也应把需求描述清晰且聚焦（例如“创建 Hello World 并运行”），防止模型产生过多副作用。
5. **切换模式**：完成实验后再次运行 `codex` 时可以改用 `--full-auto` 或默认审批模式，避免无意间继续在危险模式下工作。

## 后续扩展
- 参考 `codex --help` 或 `codex exec --help` 探索更多旗标，例如 `--json` 输出事件流、`-m` 切换模型等。
- 如需复现“ExecEC”风格的完全自主流程，可继续把更复杂的任务写入同类命令，再把输出整理进 SOP/培训文档。
