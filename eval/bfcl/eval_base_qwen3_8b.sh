#!/usr/bin/env bash
set -euo pipefail

# Evaluate base Qwen3-8B on BFCL v4 single-turn.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH_OVERRIDE=${MODEL_PATH_OVERRIDE:-/cfs_turbo/jiaxicao/ckpt/hf/Qwen3-8B}
export STEP=${STEP:-base}
export EVAL_ROOT=${EVAL_ROOT:-/cfs_turbo/jiaxicao/evals/bfcl/base-qwen3-8b}

export BFCL_MODEL=${BFCL_MODEL:-Qwen/Qwen3-8B}
export TEST_CATEGORY=${TEST_CATEGORY:-single_turn}
export TEMPERATURE=${TEMPERATURE:-0.0}
export NUM_THREADS=${NUM_THREADS:-16}
export PORT=${PORT:-8000}
export TP_SIZE=${TP_SIZE:-4}
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-18944}

bash "${HERE}/_run_one_bfcl.sh"
