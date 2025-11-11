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
CODEX_JSON_FLAGS=${CODEX_JSON_FLAGS:---json}

APPROVAL_ARGS=()
JSON_ARGS=()

if [[ -n "${CODEX_APPROVAL_FLAGS}" ]]; then
  # shellcheck disable=SC2206
  APPROVAL_ARGS=(${CODEX_APPROVAL_FLAGS})
fi

if [[ -n "${CODEX_JSON_FLAGS}" ]]; then
  # shellcheck disable=SC2206
  JSON_ARGS=(${CODEX_JSON_FLAGS})
fi

EXTRA_ARGS=("$@")

for (( VERSION=START_VER; VERSION<=END_VER; VERSION++ )); do
  PREV=$(( VERSION - 1 ))
  CUR_EXP_DIR="${EXPERIMENTS_ROOT}/v${VERSION}"
  PREV_EXP_DIR="${EXPERIMENTS_ROOT}/v${PREV}"
  DEV_FILE="${CUR_EXP_DIR}/dev_v${VERSION}.md"
  PREV_FILE="${PREV_EXP_DIR}/dev_v${PREV}.md"
  RUN_ID="v${PREV}-to-v${VERSION}-$(date +%Y%m%d-%H%M%S)"
  LOG_FILE="$LOG_DIR/${RUN_ID}.jsonl"

  mkdir -p "${CUR_EXP_DIR}/artifacts/checkpoints" \
           "${CUR_EXP_DIR}/artifacts/logs" \
           "${CUR_EXP_DIR}/artifacts/tb_logs" \
           "${CUR_EXP_DIR}/artifacts/plots"

  PROMPT=$(cat <<'PROMPT_EOF'
按照 `SOP.md` 的循环流程，把 Fish RL 项目从 dev_vPREV_PLACEHOLDER 推进到 dev_vVERSION_PLACEHOLDER。硬性要求：
1. 在任何 Python 操作前，检查仓库根目录是否存在 `venv`，若无则使用 `uv venv venv` 创建，若有则 `source venv/bin/activate` 并使用 `uv pip install` 安装依赖。
2. 阅读 `SOP.md` 与 PREV_FILE_PLACEHOLDER，总结上一轮的 Learning/计划，写入新的 DEV_FILE_PLACEHOLDER 的"启动""观察""实验计划"部分。
3. 规划并执行必要的训练/可视化脚本（通常位于 `experiments/v*/train.py`）。所有实验输出（日志、TensorBoard、曲线、检查点、媒体）写入 `CUR_EXP_DIR_PLACEHOLDER/artifacts/`。
4. 本机为 32 核 CPU / 512GB RAM，默认把 `--num_envs` 设为 ≥64（推荐 128），除非任务明确要求更低并行度。
5. 完成后，在 DEV_FILE_PLACEHOLDER 记录：运行命令、关键指标、输出路径、媒体（mp4/gif）位置、下一步计划；必要时复制训练曲线/视频到 `CUR_EXP_DIR_PLACEHOLDER/artifacts/plots|media/`。
6. 保持仓库整洁：可复制上一版本的 `train.py` 做基础，但不要删除历史数据；运行结束前给出 `git status` 摘要，并将所有文本/脚本/plot 更新提交 (`git commit`) 且推送到 `origin`。

目标：让 dev_vVERSION_PLACEHOLDER 的文档、代码、日志完整可交接。完成后输出"vVERSION_PLACEHOLDER done"。
PROMPT_EOF
  )
  # Substitute variables in PROMPT
  PROMPT=${PROMPT//PREV_PLACEHOLDER/$PREV}
  PROMPT=${PROMPT//VERSION_PLACEHOLDER/$VERSION}
  PROMPT=${PROMPT//PREV_FILE_PLACEHOLDER/$PREV_FILE}
  PROMPT=${PROMPT//DEV_FILE_PLACEHOLDER/$DEV_FILE}
  PROMPT=${PROMPT//CUR_EXP_DIR_PLACEHOLDER/$CUR_EXP_DIR}

  echo "[INFO] Running iteration v${PREV} -> v${VERSION} (log: ${LOG_FILE})"
  # Build command array, handling empty EXTRA_ARGS
  CMD_ARGS=("${APPROVAL_ARGS[@]}" "${JSON_ARGS[@]}" -C "$WORKDIR")
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD_ARGS+=("${EXTRA_ARGS[@]}")
  fi
  printf '%s\n' "$PROMPT" | \
    "$CODEX_BIN" exec "${CMD_ARGS[@]}" \
    | tee "$LOG_FILE"

done
