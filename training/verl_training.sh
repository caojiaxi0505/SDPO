#!/bin/bash
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export PYTHONBUFFERED=1
# export RAY_DEBUG=1
ulimit -c 0

export WANDB_ENTITY="sample-efficient-rlvr" # team
export EXPERIMENT=${1:-"experiment"}
CONFIG_NAME=${2:-"ppo_trainer"}
export TASK=${3:-"datasets/ttcs/lasgroup_verifiable-corpus_math-ai_math500_1000"}

# removes the first three arguments from the command line
if [ "$#" -ge 3 ]; then
    shift 3
else
    echo "Usage: $0 <experiment_name> <config_name> <data_path>"
    echo "Example: $0 test ppo_trainer datasets/ttcs/lasgroup_verifiable-corpus_math-ai_math500_1000"
    exit 1
fi

echo "Experiment: $EXPERIMENT"
echo "Config: $CONFIG_NAME"
echo "Task: $TASK"
echo "Arguments: $@"

has_override() {
    local key="$1"
    shift
    for arg in "$@"; do
        case "$arg" in
            ${key}=*) return 0 ;;
        esac
    done
    return 1
}

case "$TASK" in
    datasets/apigen|datasets/toolace|*/datasets/apigen|*/datasets/toolace)
        if [[ "$TASK" = /* ]]; then
            DATASET_DIR="$TASK"
        else
            DATASET_DIR="${PWD}/${TASK}"
        fi
        if [ ! -f "${DATASET_DIR}/train.json" ] || [ ! -f "${DATASET_DIR}/test.json" ]; then
            python data/convert_function_calling_datasets.py --datasets "$(basename "$TASK")" --base-dir "$(dirname "$DATASET_DIR")"
        fi
        if ! has_override "data.train_files" "$@"; then
            set -- "$@" "data.train_files=[\"${DATASET_DIR}/train.json\"]"
        fi
        if ! has_override "data.val_files" "$@"; then
            set -- "$@" "data.val_files=[\"${DATASET_DIR}/test.json\"]"
        fi
        ;;
    datasets/openr1_math|datasets/taco|datasets/scienceqa|*/datasets/openr1_math|*/datasets/taco|*/datasets/scienceqa)
        if [[ "$TASK" = /* ]]; then
            DATASET_DIR="$TASK"
        else
            DATASET_DIR="${PWD}/${TASK}"
        fi
        if [ ! -f "${DATASET_DIR}/train.json" ] || [ ! -f "${DATASET_DIR}/test.json" ]; then
            python data/convert_rlvr_datasets.py --datasets "$(basename "$TASK")" --base-dir "$(dirname "$DATASET_DIR")"
        fi
        if ! has_override "data.train_files" "$@"; then
            set -- "$@" "data.train_files=[\"${DATASET_DIR}/train.json\"]"
        fi
        if ! has_override "data.val_files" "$@"; then
            set -- "$@" "data.val_files=[\"${DATASET_DIR}/test.json\"]"
        fi
        ;;
esac

python -m verl.trainer.main_ppo --config-name $CONFIG_NAME "$@"
