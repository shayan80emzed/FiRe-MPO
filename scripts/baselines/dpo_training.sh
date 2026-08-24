#!/bin/bash
# Sequence-level DPO baseline (Table 1).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

DATA_PATH="${DATA_PATH:-/data/shayan/med-align/preference_dataset/Qwen3-VL-4B-Instruct/slake/greedy/dpo_D2.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/shayan/med-align/models/qwen3-4B-slake-dpo}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
SEED="${SEED:-42}"

case "$(basename "${BASE_MODEL_PATH%/}")" in
  HuatuoGPT-Vision-7B|HuatuoGPT-Vision-7B-Qwen2.5VL)
    PROCESSOR_NAME="${PROCESSOR_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
    ;;
  *)
    PROCESSOR_NAME="${PROCESSOR_NAME:-$BASE_MODEL_PATH}"
    ;;
esac

echo "BASE_MODEL_PATH=$BASE_MODEL_PATH"
echo "PROCESSOR_NAME=$PROCESSOR_NAME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

mkdir -p logs

if [[ -f dpo_env/bin/activate ]]; then
  # shellcheck disable=SC1091
  source dpo_env/bin/activate
  echo "Activated venv: $VIRTUAL_ENV"
fi

export LEARNING_RATE
export PER_DEVICE_TRAIN_BATCH_SIZE
export WANDB_MODE="${WANDB_MODE:-offline}"

PYTHONUNBUFFERED=1 accelerate launch --mixed_precision bf16 src/train/train_dpo.py \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --base_model_path "$BASE_MODEL_PATH" \
  --processor_name "$PROCESSOR_NAME" \
  --learning_rate "$LEARNING_RATE" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --seed "$SEED"

echo "DPO training completed."
