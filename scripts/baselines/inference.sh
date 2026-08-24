#!/bin/bash
# Local / non-Slurm GPU run: uses physical GPU 0 only. Same logic as the Slurm batch script; no #SBATCH.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Record start time in seconds since epoch
JOB_START_TIME=$(date +%s)

# BASE_MODEL_PATH="FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL"
# All of these are env-overridable so a sweep can drive this script; the values here
# are the defaults used for one-off manual runs.
BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
# ON_REASONING=true
SPLIT="${SPLIT:-test}"
DATASET_NAME="${DATASET_NAME:-slake}"
PEFT_MODEL_PATH="${PEFT_MODEL_PATH:-/data/shayan/med-align/models/qwen3-4B-slake-mask-dpo/checkpoint-610}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-4}"
# WITH_AUGMENTATION=true
# AUGMENTED_IMAGES_DIR=/home/emzed/projects/aip-dolatab6/emzed/med-align/augmented_images

if [ -z "$BASE_MODEL_PATH" ]; then
    echo "Error: set BASE_MODEL_PATH in scripts/inference.sh (or export before running)." >&2
    exit 1
fi

# Edit BASE_MODEL_PATH above; processor matches it after realpath unless basename is HuatuoGPT-Vision-7B.
if [ -d "$BASE_MODEL_PATH" ] || [ -f "$BASE_MODEL_PATH" ]; then
    BASE_MODEL_PATH=$(realpath "$BASE_MODEL_PATH")
fi
case "$(basename "${BASE_MODEL_PATH%/}")" in
    HuatuoGPT-Vision-7B)
        PROCESSOR_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
        ;;
    *)
        PROCESSOR_NAME="$BASE_MODEL_PATH"
        ;;
esac

MODEL_NAME=$(basename "$BASE_MODEL_PATH")
OUTPUT_DIR="${OUTPUT_DIR:-./experiments/${DATASET_NAME}_${MODEL_NAME}/}"

# export BNB_CUDA_VERSION=126

mkdir -p logs

# Optional: Apache Arrow module on clusters that use Environment Modules
if command -v module >/dev/null 2>&1; then
    module load arrow/21.0.0 2>/dev/null || true
fi

source dpo_env/bin/activate

echo "=========================================="
echo "Model Inference Job"
echo "=========================================="
echo "PID: $$ (local; no Slurm)"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Parameters:"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  Base Model Path: $BASE_MODEL_PATH"
echo "  Processor Name: $PROCESSOR_NAME"
echo "  Dataset Name: $DATASET_NAME"
echo "  Split: $SPLIT"
echo "  Batch Size: $BATCH_SIZE"
echo "  Max New Tokens: $MAX_NEW_TOKENS"
echo "  Num Return Sequences: $NUM_RETURN_SEQUENCES"
echo "  On Reasoning: ${ON_REASONING:-}"
echo "  With Augmentation: ${WITH_AUGMENTATION:-}"
if [ "${WITH_AUGMENTATION:-}" = "true" ]; then
    echo "  Augmented Images Dir: ${AUGMENTED_IMAGES_DIR:-}"
fi
echo "  Output Dir: $OUTPUT_DIR"
if [ -n "${PEFT_MODEL_PATH:-}" ]; then
    echo "  PEFT Model Path: $PEFT_MODEL_PATH"
fi
echo "=========================================="

echo "GPU Info:"
nvidia-smi

echo ""
if [ "${WITH_AUGMENTATION:-}" = "true" ]; then
    echo "Starting model inference with augmentation..."
    echo "Processing: original, blur_medsam, noise_medsam, mask_medsam"
else
    echo "Starting model inference..."
fi

CMD_ARGS=(
    "--base_model_path" "$BASE_MODEL_PATH"
    "--processor_name" "$PROCESSOR_NAME"
    "--dataset_name" "$DATASET_NAME"
    "--split" "$SPLIT"
    "--batch_size" "$BATCH_SIZE"
    "--max_new_tokens" "$MAX_NEW_TOKENS"
    "--num_return_sequences" "$NUM_RETURN_SEQUENCES"
    "--output_dir" "$OUTPUT_DIR"
)

if [ -n "${PEFT_MODEL_PATH:-}" ]; then
    CMD_ARGS+=("--peft_model_path" "$PEFT_MODEL_PATH")
fi

# if [ "${ON_REASONING:-}" = "true" ]; then
#     CMD_ARGS+=("--on_reasoning")
# fi

if [ "${WITH_AUGMENTATION:-}" = "true" ]; then
    CMD_ARGS+=("--with_augmentation" "--augmented_images_dir" "${AUGMENTED_IMAGES_DIR:-}")
fi

set +e
accelerate launch utils/inference/qwen25_inference.py "${CMD_ARGS[@]}"
INFER_EXIT=$?
set -e

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $INFER_EXIT"
JOB_END_TIME=$(date +%s)
JOB_RUNTIME=$((JOB_END_TIME - JOB_START_TIME))
printf "Job runtime: %02d:%02d:%02d (hh:mm:ss)\n" $((JOB_RUNTIME/3600)) $(((JOB_RUNTIME%3600)/60)) $((JOB_RUNTIME%60))
