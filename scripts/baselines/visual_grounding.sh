#!/bin/bash
# Local GPU run: select a single physical GPU via CUDA_VISIBLE_DEVICES.
# NOTE: CUDA_VISIBLE_DEVICES remaps device indices inside the process,
# so the selected GPU is always cuda:0 from the program's perspective.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=0

JOB_START_TIME=$(date +%s)

# -------------------------------
# User-configurable parameters
# -------------------------------
# BASE_MODEL_PATH="FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL"
BASE_MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"
# Optional; leave unset or empty for base model only
PEFT_MODEL_PATH="/data/shayan/med-align/models/qwen3-4B-slake-dpo/checkpoint-613"
# PEFT_MODEL_PATH=

SUBSET=loc
# SAMPLE_ID=
BATCH_SIZE=32
# MAX_DATASET_SAMPLES=
ATTN_IMPLEMENTATION=eager
# Always cuda:0 here; the physical GPU is selected via CUDA_VISIBLE_DEVICES above.
DEVICE=cuda:0

CSV_PATH=

mkdir -p logs

source dpo_env/bin/activate

# dpo_env / shell may set negin's HF caches; point at shayan's hub for this job.
export HF_HOME=/data/shayan/huggingface
export HF_HUB_CACHE=/data/shayan/huggingface/hub
unset TRANSFORMERS_CACHE

echo "=========================================="
echo "Visual Grounding"
echo "=========================================="
echo "PID: $$"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Parameters:"
echo "  Base Model Path: $BASE_MODEL_PATH"
if [ -n "$PEFT_MODEL_PATH" ]; then
    echo "  PEFT Model Path: $PEFT_MODEL_PATH"
fi
echo "  Subset: ${SUBSET:-<both>}"
echo "  Sample ID: ${SAMPLE_ID:-<dataset-mean mode>}"
echo "  Batch Size: $BATCH_SIZE"
echo "  Max Dataset Samples: ${MAX_DATASET_SAMPLES:-<all>}"
echo "  Attention Implementation: $ATTN_IMPLEMENTATION"
echo "  Device: $DEVICE"
echo "  CSV Path: ${CSV_PATH:-<auto default>}"
echo "=========================================="

echo "GPU Info:"
nvidia-smi

CMD_ARGS=(
    "--base_model_path" "$BASE_MODEL_PATH"
    "--batch_size" "$BATCH_SIZE"
    "--attn_implementation" "$ATTN_IMPLEMENTATION"
    "--device" "$DEVICE"
)

if [ -n "$PEFT_MODEL_PATH" ]; then
    CMD_ARGS+=("--peft_model_path" "$PEFT_MODEL_PATH")
fi

if [ -n "$SUBSET" ]; then
    CMD_ARGS+=("--subset" "$SUBSET")
fi

if [ -n "${SAMPLE_ID:-}" ]; then
    CMD_ARGS+=("--sample_id" "$SAMPLE_ID")
fi

if [ -n "${MAX_DATASET_SAMPLES:-}" ]; then
    CMD_ARGS+=("--max_dataset_samples" "$MAX_DATASET_SAMPLES")
fi

if [ -n "$CSV_PATH" ]; then
    CMD_ARGS+=("--csv_path" "$CSV_PATH")
fi

echo ""
echo "Starting visual grounding evaluation..."
python utils/eval/visual_grounding.py "${CMD_ARGS[@]}"
RUN_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Completed at: $(date)"
echo "Exit code: $RUN_EXIT_CODE"
JOB_END_TIME=$(date +%s)
JOB_RUNTIME=$((JOB_END_TIME - JOB_START_TIME))
printf "Runtime: %02d:%02d:%02d (hh:mm:ss)\n" $((JOB_RUNTIME/3600)) $(((JOB_RUNTIME%3600)/60)) $((JOB_RUNTIME%60))
echo "=========================================="

exit $RUN_EXIT_CODE
