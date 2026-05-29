#!/usr/bin/env bash
set -euo pipefail

# Convert a verl FSDP actor checkpoint into a HuggingFace model directory.

ACTOR_CKPT=${ACTOR_CKPT:?Set ACTOR_CKPT to a verl global_step_*/actor checkpoint directory}
TARGET_DIR=${TARGET_DIR:?Set TARGET_DIR to the output HuggingFace model directory}
EXTRA_MERGE_ARGS=${EXTRA_MERGE_ARGS:-}

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${ACTOR_CKPT}" \
  --target_dir "${TARGET_DIR}" \
  ${EXTRA_MERGE_ARGS}
