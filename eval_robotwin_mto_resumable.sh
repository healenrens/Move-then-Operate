#!/usr/bin/env bash
set -euo pipefail

MTO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
read -r -a task_configs <<< "${TASK_CONFIGS:-demo_clean}"
selection=()
if [[ -n "${TASK_LIST:-}" ]]; then
  selection=(--task-list "$TASK_LIST")
elif [[ -n "${TASK_MANIFEST:-}" ]]; then
  selection=(--manifest "$TASK_MANIFEST")
elif [[ -n "${TASK_CATALOG:-}" ]]; then
  selection=(--task-catalog "$TASK_CATALOG")
fi

export PYTHONPATH="$MTO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${MTO_PY:?Set MTO_PY to the MTO Python executable}" -m mto.eval_robotwin \
  --mto-root "$MTO_DIR" \
  --robotwin-root "${ROBOTWIN_ROOT:?Set ROBOTWIN_ROOT}" \
  --robotwin-py "${ROBOTWIN_PY:?Set ROBOTWIN_PY to the RoboTwin Python executable}" \
  --params-path "${PARAMS_PATH:?Set PARAMS_PATH to the checkpoint params directory}" \
  --run-config "${RUN_CONFIG:?Set RUN_CONFIG to resolved_config.json}" \
  --move-norm-stats-path "${MOVE_NORM_STATS_PATH:?Set separate move statistics}" \
  --operate-norm-stats-path "${OPERATE_NORM_STATS_PATH:?Set separate operate statistics}" \
  --tokenizer-path "${TOKENIZER_PATH:?Set TOKENIZER_PATH}" \
  --run-id "${RUN_ID:?Set a run ID; reuse it only to resume the same evaluation}" \
  --output-root "${EVAL_ROOT:-$MTO_DIR/eval_results}" \
  --task-configs "${task_configs[@]}" \
  --test-num "${TEST_NUM:-100}" \
  --seed "${SEED:-0}" \
  --instruction-type "${INSTRUCTION_TYPE:-unseen}" \
  --exec-steps "${EXEC_STEPS:-30}" \
  --model-gpus "${MODEL_GPUS:-0}" \
  --sim-gpus "${SIM_GPUS:-0}" \
  --server-port "${SERVER_PORT_BASE:-9700}" \
  "${selection[@]}" "$@"
