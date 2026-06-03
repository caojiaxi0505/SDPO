#!/bin/bash
# 顺序启动 3 个方法 × 2 个种子（seed1/seed2）共 6 个训练脚本。
# 每个脚本都独占 8 张卡（CUDA_VISIBLE_DEVICES=0-7），因此只能串行运行，不能并行。
# 每个子脚本自带 tee 日志；本脚本只负责依次调度并打印进度与汇总。

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(
  "grpo-toolace-qwen3-8b-h200-epoch1-seed1-260602.sh"
  "grpo-toolace-qwen3-8b-h200-epoch1-seed2-260602.sh"
  "sdpo-toolace-qwen3-8b-h200-epoch1-seed1-260602.sh"
  "sdpo-toolace-qwen3-8b-h200-epoch1-seed2-260602.sh"
  "grpo-ucsdpoaux-toolace-qwen3-8b-h200-epoch1-seed1-emajf-vllm-260602.sh"
  "grpo-ucsdpoaux-toolace-qwen3-8b-h200-epoch1-seed2-emajf-vllm-260602.sh"
)

declare -a STATUS

total=${#SCRIPTS[@]}
for i in "${!SCRIPTS[@]}"; do
  name="${SCRIPTS[$i]}"
  path="${SCRIPT_DIR}/${name}"
  idx=$((i + 1))
  echo "============================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] (${idx}/${total}) START: ${name}"
  echo "============================================================"
  if [ ! -f "${path}" ]; then
    echo "ERROR: script not found: ${path}"
    STATUS[$i]="MISSING"
    continue
  fi
  start=$(date +%s)
  bash "${path}"
  rc=$?
  end=$(date +%s)
  dur=$((end - start))
  if [ ${rc} -eq 0 ]; then
    STATUS[$i]="OK (${dur}s)"
  else
    STATUS[$i]="FAILED rc=${rc} (${dur}s)"
  fi
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] (${idx}/${total}) END: ${name} -> ${STATUS[$i]}"
done

echo "============================================================"
echo "SUMMARY"
echo "============================================================"
for i in "${!SCRIPTS[@]}"; do
  printf '%-70s %s\n' "${SCRIPTS[$i]}" "${STATUS[$i]}"
done
