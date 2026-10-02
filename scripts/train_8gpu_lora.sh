#!/usr/bin/env bash
set -euo pipefail
export TRAINING_MODE=lora
export EXP_NAME="${EXP_NAME:-mto_lora_h30_b256_100k_dim32zero}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_8gpu.sh" "$@"
