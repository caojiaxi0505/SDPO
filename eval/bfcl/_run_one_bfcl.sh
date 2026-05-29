#!/usr/bin/env bash
set -euo pipefail

# Shared BFCL v4 single-turn evaluator.
# It serves one local model with vLLM, asks BFCL to generate through the local
# OSS/OpenAI-compatible endpoint, then runs BFCL evaluate.
#
# Required for trained verl checkpoints:
#   RUN_DIR=/path/to/run-dir
#   STEP=630
#   EVAL_ROOT=/path/to/eval-output
#
# Required for a flat HF model:
#   MODEL_PATH_OVERRIDE=/path/to/hf-model
#   STEP=base
#   EVAL_ROOT=/path/to/eval-output

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BFCL_MODEL=${BFCL_MODEL:-Qwen/Qwen3-8B}
EVAL_ROOT=${EVAL_ROOT:?Set EVAL_ROOT}
MODEL_PATH_OVERRIDE=${MODEL_PATH_OVERRIDE:-}
if [[ -n "${MODEL_PATH_OVERRIDE}" ]]; then
  RUN_DIR=${RUN_DIR:-"${MODEL_PATH_OVERRIDE}"}
  STEP=${STEP:-base}
else
  RUN_DIR=${RUN_DIR:?Set RUN_DIR to the run trainer.default_local_dir (or set MODEL_PATH_OVERRIDE)}
  STEP=${STEP:?Set STEP to the checkpoint step to evaluate (or set MODEL_PATH_OVERRIDE)}
fi

TEST_CATEGORY=${TEST_CATEGORY:-single_turn}
TEMPERATURE=${TEMPERATURE:-0.0}
NUM_THREADS=${NUM_THREADS:-16}
BFCL_CMD=${BFCL_CMD:-bfcl}
FORCE_MERGE=${FORCE_MERGE:-0}

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
TP_SIZE=${TP_SIZE:-4}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-18944}
DTYPE=${DTYPE:-bfloat16}
SERVE_WAIT_SECS=${SERVE_WAIT_SECS:-900}

ACTOR_CKPT="${RUN_DIR}/global_step_${STEP}/actor"
HF_DIR="${ACTOR_CKPT}/huggingface"
MERGED_DIR="${EVAL_ROOT}/merged_hf/step_${STEP}"
RESULT_DIR="${EVAL_ROOT}/result"
SCORE_DIR="${EVAL_ROOT}/score"
LOG_DIR="${EVAL_ROOT}/logs"
SERVE_LOG="${LOG_DIR}/vllm_step${STEP}.log"
BASE_URL="http://${HOST}:${PORT}/v1"

mkdir -p "${EVAL_ROOT}" "${RESULT_DIR}" "${SCORE_DIR}" "${LOG_DIR}"

echo "=================================================================="
echo "[eval] run        : ${RUN_DIR}"
echo "[eval] step       : ${STEP}"
echo "[eval] bfcl model : ${BFCL_MODEL} (local_inference handler, prompt mode)"
echo "[eval] category   : ${TEST_CATEGORY} (temp=${TEMPERATURE})"
echo "[eval] endpoint   : ${BASE_URL}"
echo "[eval] eval root  : ${EVAL_ROOT}"
echo "=================================================================="

if ! command -v "${BFCL_CMD}" >/dev/null 2>&1; then
  echo "[eval] ERROR: BFCL command not found: ${BFCL_CMD}" >&2
  exit 1
fi

if curl -sf "${BASE_URL}/models" >/dev/null 2>&1; then
  echo "[eval] ERROR: ${BASE_URL}/models is already reachable before starting this run." >&2
  echo "[eval] Kill the stale vLLM server or set PORT to a free port." >&2
  exit 1
fi

has_hf_weights() {
  local d="$1"
  [[ -d "$d" ]] || return 1
  compgen -G "${d}/*.safetensors" >/dev/null 2>&1 && return 0
  compgen -G "${d}/pytorch_model*.bin" >/dev/null 2>&1 && return 0
  compgen -G "${d}/model*.bin" >/dev/null 2>&1 && return 0
  return 1
}

if [[ -n "${MODEL_PATH_OVERRIDE}" ]]; then
  if ! has_hf_weights "${MODEL_PATH_OVERRIDE}"; then
    echo "[eval] ERROR: MODEL_PATH_OVERRIDE has no weight files: ${MODEL_PATH_OVERRIDE}" >&2
    exit 1
  fi
  MODEL_PATH="${MODEL_PATH_OVERRIDE}"
elif [[ "${FORCE_MERGE}" == "1" ]]; then
  [[ -d "${ACTOR_CKPT}" ]] || { echo "[eval] ERROR: missing actor checkpoint: ${ACTOR_CKPT}" >&2; exit 1; }
  rm -rf "${MERGED_DIR}"
  ACTOR_CKPT="${ACTOR_CKPT}" TARGET_DIR="${MERGED_DIR}" bash "${HERE}/merge_fsdp_actor_to_hf.sh"
  MODEL_PATH="${MERGED_DIR}"
elif has_hf_weights "${HF_DIR}"; then
  MODEL_PATH="${HF_DIR}"
elif has_hf_weights "${MERGED_DIR}"; then
  MODEL_PATH="${MERGED_DIR}"
else
  [[ -d "${ACTOR_CKPT}" ]] || { echo "[eval] ERROR: missing actor checkpoint: ${ACTOR_CKPT}" >&2; exit 1; }
  ACTOR_CKPT="${ACTOR_CKPT}" TARGET_DIR="${MERGED_DIR}" bash "${HERE}/merge_fsdp_actor_to_hf.sh"
  MODEL_PATH="${MERGED_DIR}"
fi
has_hf_weights "${MODEL_PATH}" || { echo "[eval] ERROR: no HF weights in ${MODEL_PATH}" >&2; exit 1; }
echo "[eval] model path : ${MODEL_PATH}"

echo "[eval] starting vLLM (log: ${SERVE_LOG})"
if command -v setsid >/dev/null 2>&1; then
  MODEL_PATH="${MODEL_PATH}" \
  SERVED_MODEL_NAME="${BFCL_MODEL}" \
  HOST="${HOST}" PORT="${PORT}" TP_SIZE="${TP_SIZE}" \
  GPU_MEM_UTIL="${GPU_MEM_UTIL}" MAX_MODEL_LEN="${MAX_MODEL_LEN}" DTYPE="${DTYPE}" \
    setsid bash "${HERE}/serve_vllm.sh" >"${SERVE_LOG}" 2>&1 &
  SERVE_PGID=$!
else
  MODEL_PATH="${MODEL_PATH}" \
  SERVED_MODEL_NAME="${BFCL_MODEL}" \
  HOST="${HOST}" PORT="${PORT}" TP_SIZE="${TP_SIZE}" \
  GPU_MEM_UTIL="${GPU_MEM_UTIL}" MAX_MODEL_LEN="${MAX_MODEL_LEN}" DTYPE="${DTYPE}" \
    bash "${HERE}/serve_vllm.sh" >"${SERVE_LOG}" 2>&1 &
  SERVE_PGID=""
fi
SERVE_PID=$!

cleanup() {
  echo "[eval] stopping vLLM server (pid ${SERVE_PID})"
  if [[ -n "${SERVE_PGID}" ]]; then
    kill -- "-${SERVE_PGID}" 2>/dev/null || true
    sleep 2
    kill -9 -- "-${SERVE_PGID}" 2>/dev/null || true
  else
    pkill -TERM -P "${SERVE_PID}" 2>/dev/null || true
    kill "${SERVE_PID}" 2>/dev/null || true
    sleep 2
    pkill -KILL -P "${SERVE_PID}" 2>/dev/null || true
    kill -9 "${SERVE_PID}" 2>/dev/null || true
  fi
  wait "${SERVE_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "[eval] waiting for ${BASE_URL}/models (up to ${SERVE_WAIT_SECS}s)"
deadline=$((SECONDS + SERVE_WAIT_SECS))
while true; do
  if ! kill -0 "${SERVE_PID}" 2>/dev/null; then
    echo "[eval] ERROR: vLLM exited early. Tail of ${SERVE_LOG}:" >&2
    tail -n 40 "${SERVE_LOG}" >&2 || true
    exit 1
  fi
  if curl -sf "${BASE_URL}/models" >/dev/null 2>&1; then
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "[eval] ERROR: server not ready within ${SERVE_WAIT_SECS}s" >&2
    tail -n 40 "${SERVE_LOG}" >&2 || true
    exit 1
  fi
  sleep 5
done

python - <<PY
import json
import sys
from urllib import request
base_url = "${BASE_URL}".rstrip("/")
expected = "${BFCL_MODEL}"
try:
    with request.urlopen(base_url + "/models", timeout=10) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
except Exception as exc:
    print(f"[eval] ERROR: failed to query {base_url}/models: {exc}", file=sys.stderr)
    sys.exit(1)
ids = [item.get("id") for item in payload.get("data", [])]
print(f"[eval] endpoint models: {ids}")
if expected not in ids:
    print(f"[eval] ERROR: expected served model id {expected!r}, got {ids}", file=sys.stderr)
    sys.exit(1)
PY

export REMOTE_OPENAI_BASE_URL="${BASE_URL}"
export REMOTE_OPENAI_TOKENIZER_PATH="${MODEL_PATH}"
export LOCAL_SERVER_ENDPOINT="${HOST}"
export LOCAL_SERVER_PORT="${PORT}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

GEN_LOG="${LOG_DIR}/generate_step${STEP}.log"

echo "[eval] bfcl generate -> ${RESULT_DIR} (log: ${GEN_LOG})"
"${BFCL_CMD}" generate \
  --model "${BFCL_MODEL}" \
  --test-category "${TEST_CATEGORY}" \
  --skip-server-setup \
  --temperature "${TEMPERATURE}" \
  --num-threads "${NUM_THREADS}" \
  --result-dir "${RESULT_DIR}" \
  -o 2>&1 | tee "${GEN_LOG}"

EVAL_LOG="${LOG_DIR}/evaluate_step${STEP}.log"
echo "[eval] bfcl evaluate -> ${SCORE_DIR} (log: ${EVAL_LOG})"
"${BFCL_CMD}" evaluate \
  --model "${BFCL_MODEL}" \
  --test-category "${TEST_CATEGORY}" \
  --result-dir "${RESULT_DIR}" \
  --score-dir "${SCORE_DIR}" \
  --partial-eval 2>&1 | tee "${EVAL_LOG}"

echo "[eval] scores (from ${SCORE_DIR}/*.csv):"
{
  for csv in data_overall.csv data_non_live.csv data_live.csv; do
    f="${SCORE_DIR}/${csv}"
    [[ -f "$f" ]] || continue
    echo "===== ${csv} ====="
    cat "$f"
    echo
  done
} | tee "${EVAL_ROOT}/summary_step${STEP}.txt"

echo "[eval] DONE. results: ${RESULT_DIR}  scores: ${SCORE_DIR}"
echo "[eval] summary: ${EVAL_ROOT}/summary_step${STEP}.txt"
