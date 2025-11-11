#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <start_version_number> <end_version_number> [codex_extra_args...]" >&2
  echo "Example: $0 3 15 --model gpt-5-codex" >&2
  exit 1
fi

START_VER=$1
END_VER=$2
shift 2

if (( START_VER < 1 )); then
  echo "start_version_number must be >= 1" >&2
  exit 1
fi

if (( END_VER < START_VER )); then
  echo "end_version_number must be >= start_version_number" >&2
  exit 1
fi

WORKDIR=${FISH_RL_WORKDIR:-$(pwd)}
LOG_DIR=${CODEX_RUN_LOG_DIR:-codex_runs}
EXPERIMENTS_ROOT=${EXPERIMENTS_ROOT:-experiments}
mkdir -p "$LOG_DIR"

CODEX_BIN=${CODEX_BIN:-codex}
CODEX_APPROVAL_FLAGS=${CODEX_APPROVAL_FLAGS:---dangerously-bypass-approvals-and-sandbox}

for (( VERSION=START_VER; VERSION<=END_VER; VERSION++ )); do
  PREV=$(( VERSION - 1 ))
  CUR_EXP_DIR="${EXPERIMENTS_ROOT}/v${VERSION}"
  PREV_EXP_DIR="${EXPERIMENTS_ROOT}/v${PREV}"
  DEV_FILE="${CUR_EXP_DIR}/dev_v${VERSION}.md"
  PREV_FILE="${PREV_EXP_DIR}/dev_v${PREV}.md"
  RUN_ID="v${PREV}-to-v${VERSION}-$(date +%Y%m%d-%H%M%S)"
  LOG_FILE="$LOG_DIR/${RUN_ID}.log"

  mkdir -p "${CUR_EXP_DIR}/artifacts/checkpoints" \
           "${CUR_EXP_DIR}/artifacts/logs" \
           "${CUR_EXP_DIR}/artifacts/tb_logs" \
           "${CUR_EXP_DIR}/artifacts/plots"

  PROMPT=$(cat <<EOF
你现在扮演 Codex CLI，负责按照 `SOP.md` 的工作流，把 Fish RL 项目从 dev_v${PREV} 推进到 dev_v${VERSION}。请遵守以下硬性要求：
1. 在任何 Python 操作前，检查仓库根目录是否存在 `venv`，若无则使用 `uv venv venv` 创建，若有则 `source venv/bin/activate` 并使用 `uv pip install` 安装依赖。
2. 阅读 `SOP.md` 与 ${PREV_FILE}，总结上一轮的 Learning/计划，写入新的 ${DEV_FILE} 的“启动”“观察”“实验计划”部分。
3. 规划并执行必要的训练/可视化脚本（通常位于 `experiments/v*/train.py`）。所有实验输出（日志、TensorBoard、曲线、检查点）写入 `${CUR_EXP_DIR}/artifacts/`。
4. 建议充分利用本机 16 核 / 128GB 资源：默认把 `--num_envs` 设为 ≥64，除非文档明确要求更小规模。
5. 完成后，在 ${DEV_FILE} 记录：运行命令、关键指标、输出路径、下一步计划；必要时复制训练曲线到 `${CUR_EXP_DIR}/artifacts/plots/`。
6. 保持仓库整洁：可复制上一版本的 `train.py` 做基础，但不要删除历史数据；运行结束前给出 `git status` 摘要。

目标：让 dev_v${VERSION} 的文档、代码、日志完整可交接。完成后输出“v${VERSION} done”。
EOF
  )

  echo "[INFO] Running iteration v${PREV} -> v${VERSION} (log: ${LOG_FILE})"
  printf '%s\n' "$PROMPT" | \
    "$CODEX_BIN" exec $CODEX_APPROVAL_FLAGS -C "$WORKDIR" "$@" | tee "$LOG_FILE"

done
