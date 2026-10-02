#!/usr/bin/env bash
set -euo pipefail

MTO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$MTO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
# The experiment's wandb_id.txt owns resume identity.
unset WANDB_RUN_ID
read -r -a splits <<< "${SPLIT_NAMES:-demo_clean}"
wandb_args=(--no-wandb-enabled)
if [[ "${WANDB_ENABLED:-true}" == true ]]; then
  wandb_args=(--wandb-enabled)
fi
args=(
  --data-format lerobot_v3 --data-root "${DATA_ROOT:?Set DATA_ROOT}" \
  --labels-root "${LABELS_ROOT:?Set LABELS_ROOT}" --split-names "${splits[@]}" \
  --checkpoint-base-dir "${CHECKPOINT_BASE_DIR:?Set CHECKPOINT_BASE_DIR}" \
  --exp-name "${EXP_NAME:?Set EXP_NAME or use the full/LoRA wrapper}" \
  --project-name "${WANDB_PROJECT:-move-then-operate}" \
  --init-params "${PI0_PARAMS:?Set PI0_PARAMS}" --init-source pi0_base \
  --tokenizer-path "${TOKENIZER_PATH:?Set TOKENIZER_PATH}" \
  --training-mode "${TRAINING_MODE:-full}" --action-horizon 30 --model-action-dim 32 \
  --batch-size 256 --num-steps 100000 --num-workers "${NUM_WORKERS:-8}" \
  --lr-warmup 1000 --lr 5e-5 --lr-final 1e-5 --ema-decay 0.99 \
  --normalize-method zscore --long-phase-ratio 0.5 \
  --move-norm-stats-path "${MOVE_NORM_STATS_PATH:?Set MOVE_NORM_STATS_PATH}" \
  --operate-norm-stats-path "${OPERATE_NORM_STATS_PATH:?Set OPERATE_NORM_STATS_PATH}" \
  --save-interval 10000 --keep-period 10000 --fsdp-devices 8 --seed 42 --resume \
  "${wandb_args[@]}"
)
if [[ -n "${TASK_NAMES:-}" ]]; then
  read -r -a tasks <<< "$TASK_NAMES"
  args+=(--task-names "${tasks[@]}")
fi
exec "${MTO_PY:-$MTO_DIR/.venv/bin/python}" -m mto.train "${args[@]}" "$@"
