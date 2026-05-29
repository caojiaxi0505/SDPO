#!/usr/bin/env bash
set -euo pipefail

# Evaluate GRPO checkpoint on BFCL v4 single-turn.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_DIR=${RUN_DIR:-/cfs_turbo/jiaxicao/ckpt/ucsdpo_runs/grpo-tooluse-qwen3-8b-h20-260527}
export STEP=${STEP:-630}
export EVAL_ROOT=${EVAL_ROOT:-/cfs_turbo/jiaxicao/evals/bfcl/grpo}

export BFCL_MODEL=${BFCL_MODEL:-Qwen/Qwen3-8B}
export TEST_CATEGORY=${TEST_CATEGORY:-single_turn}
export TEMPERATURE=${TEMPERATURE:-0.0}
export NUM_THREADS=${NUM_THREADS:-16}
export PORT=${PORT:-8000}
export TP_SIZE=${TP_SIZE:-4}
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-18944}

bash "${HERE}/_run_one_bfcl.sh"
