#!/usr/bin/env bash
set -euo pipefail

# Serve a local HF checkpoint with an OpenAI-compatible vLLM endpoint.

MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to the HF-format checkpoint directory}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-ucsdpo-qwen3-8b}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
TP_SIZE=${TP_SIZE:-4}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-18944}
DTYPE=${DTYPE:-bfloat16}
EXTRA_VLLM_ARGS=${EXTRA_VLLM_ARGS:-}

echo "[serve] MODEL_PATH=${MODEL_PATH}"
echo "[serve] SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "[serve] HOST=${HOST} PORT=${PORT} TP_SIZE=${TP_SIZE} MAX_MODEL_LEN=${MAX_MODEL_LEN}"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --dtype "${DTYPE}" \
  ${EXTRA_VLLM_ARGS}
