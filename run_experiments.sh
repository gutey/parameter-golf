#!/usr/bin/env bash
# run_experiments.sh — Launch helpers for Parameter Golf experiments
# Works on single GPU (A4000, H100) or multi-GPU (8xH100 via torchrun).
#
# Usage:
#   # Single GPU, enhanced transformer (default settings for H100):
#   bash run_experiments.sh enhanced
#
#   # Single GPU, weight sharing (4 unique layers cycled 16x):
#   NUM_UNIQUE_LAYERS=4 NUM_LAYERS=16 bash run_experiments.sh shared
#
#   # Single GPU, mLSTM:
#   bash run_experiments.sh mlstm
#
#   # 8xH100 distributed, enhanced transformer:
#   bash run_experiments.sh enhanced 8
#
#   # A4000 quick test (reduced batch, shorter training):
#   bash run_experiments.sh enhanced 1 a4000
#
set -euo pipefail

SCRIPT="${1:-enhanced}"
NUM_GPUS="${2:-1}"
PROFILE="${3:-h100}"

# Map script names to files
case "$SCRIPT" in
    baseline)   TRAIN_SCRIPT="train_gpt.py" ;;
    enhanced)   TRAIN_SCRIPT="train_gpt_enhanced.py" ;;
    shared)     TRAIN_SCRIPT="train_gpt_shared.py" ;;
    mlstm)      TRAIN_SCRIPT="train_gpt_mlstm.py" ;;
    mamba)      TRAIN_SCRIPT="train_gpt_mamba.py" ;;
    ttt)        TRAIN_SCRIPT="train_gpt_ttt.py" ;;
    *)          TRAIN_SCRIPT="$SCRIPT" ;;
esac

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "ERROR: Script $TRAIN_SCRIPT not found"
    exit 1
fi

# Profile-specific defaults (can be overridden by env vars)
if [[ "$PROFILE" == "a4000" ]]; then
    export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-131072}"
    export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
    export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-1200}"
    export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}"
    export WARMDOWN_ITERS="${WARMDOWN_ITERS:-200}"
    echo "=== A4000 profile: batch=${TRAIN_BATCH_TOKENS}, seq=${TRAIN_SEQ_LEN}, wall=${MAX_WALLCLOCK_SECONDS}s ==="
elif [[ "$PROFILE" == "h100" ]]; then
    export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-786432}"
    export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
    export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
    # Reduce grad accumulation on single GPUs for larger microbatches (better H100 utilization)
    if [[ "$NUM_GPUS" -eq 1 ]]; then
        export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
    fi
    echo "=== H100 profile: batch=${TRAIN_BATCH_TOKENS}, seq=${TRAIN_SEQ_LEN}, wall=${MAX_WALLCLOCK_SECONDS}s, accum=${GRAD_ACCUM_STEPS:-auto} ==="
fi

echo "=== Running $TRAIN_SCRIPT on ${NUM_GPUS} GPU(s) ==="
echo "=== Key env: NUM_LAYERS=${NUM_LAYERS:-default} NUM_UNIQUE_LAYERS=${NUM_UNIQUE_LAYERS:-default} MODEL_DIM=${MODEL_DIM:-default} ==="

if [[ "$NUM_GPUS" -gt 1 ]]; then
    torchrun --nproc_per_node="$NUM_GPUS" "$TRAIN_SCRIPT"
else
    python "$TRAIN_SCRIPT"
fi
