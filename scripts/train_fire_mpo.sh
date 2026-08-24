#!/bin/bash
# Train FiRe-MPO with paper hyperparameters (Table 6).
# Usage:
#   ./scripts/train_fire_mpo.sh
#   CONFIG=configs/ablations/wo_visual.yaml ./scripts/train_fire_mpo.sh
#   DATA_PATH=... OUTPUT_DIR=... BASE_MODEL_PATH=... ./scripts/train_fire_mpo.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONFIG="${CONFIG:-configs/fire_mpo_default.yaml}"
DATA_PATH="${DATA_PATH:-/data/shayan/med-align/preference_dataset/Qwen3-VL-4B-Instruct/slake/greedy/rrpo_with_medsam3_rejected.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/shayan/med-align/models/qwen3-4B-fire-mpo-slake}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"

LEARNING_RATE="${LEARNING_RATE:-1e-6}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"

# Paper defaults (overridden by CONFIG / explicit env)
ALPHA="${ALPHA:-0.01}"
GAMMA="${GAMMA:-0.1}"
LAMBDA="${LAMBDA:-0.5}"
SEED="${SEED:-42}"

case "$(basename "${BASE_MODEL_PATH%/}")" in
  HuatuoGPT-Vision-7B|HuatuoGPT-Vision-7B-Qwen2.5VL)
    PROCESSOR_NAME="${PROCESSOR_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
    ;;
  *)
    PROCESSOR_NAME="${PROCESSOR_NAME:-$BASE_MODEL_PATH}"
    ;;
esac

echo "CONFIG=$CONFIG"
echo "BASE_MODEL_PATH=$BASE_MODEL_PATH"
echo "PROCESSOR_NAME=$PROCESSOR_NAME"
echo "DATA_PATH=$DATA_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "α=$ALPHA γ=$GAMMA λ=$LAMBDA"

mkdir -p logs
JOB_START_SEC=$(date +%s)
echo "Wall-clock start: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"

if [[ -f dpo_env/bin/activate ]]; then
  # shellcheck disable=SC1091
  source dpo_env/bin/activate
  echo "Activated venv: $VIRTUAL_ENV"
fi

export LEARNING_RATE
export PER_DEVICE_TRAIN_BATCH_SIZE
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE="${WANDB_MODE:-offline}"

EXTRA=""
# Ablation visual terms (optional)
if [[ -n "${ALPHA_V1:-}" ]]; then EXTRA="$EXTRA --alpha_v1 $ALPHA_V1"; fi
if [[ -n "${ALPHA_V2:-}" ]]; then EXTRA="$EXTRA --alpha_v2 $ALPHA_V2"; fi
if [[ -n "${MAX_STEPS:-}" ]]; then EXTRA="$EXTRA --max_steps $MAX_STEPS"; fi
if [[ -n "${GRAD_CKPT:-}" ]]; then EXTRA="$EXTRA --gradient_checkpointing $GRAD_CKPT"; fi
if [[ -n "${NUM_WORKERS:-}" ]]; then EXTRA="$EXTRA --dataloader_num_workers $NUM_WORKERS"; fi

echo "Starting FiRe-MPO training..."
PYTHONUNBUFFERED=1 accelerate launch --mixed_precision bf16 -m fire_mpo.train \
  --config "$CONFIG" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --base_model_path "$BASE_MODEL_PATH" \
  --processor_name "$PROCESSOR_NAME" \
  --learning_rate "$LEARNING_RATE" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --alpha "$ALPHA" \
  --gamma "$GAMMA" \
  --lambda_ "$LAMBDA" \
  --seed "$SEED" \
  $EXTRA

JOB_END_SEC=$(date +%s)
JOB_ELAPSED=$((JOB_END_SEC - JOB_START_SEC))
echo "Wall-clock end: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "Total wall time: ${JOB_ELAPSED}s"
echo "Done."
