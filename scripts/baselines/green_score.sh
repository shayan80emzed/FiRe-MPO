#!/bin/bash
# Local GPU run: uses physical GPU 0 only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=0

JOB_START_TIME=$(date +%s)

# Input CSV (inference output with answer + output columns)
CSV_PATH="./experiments/iu_xray_Qwen3-VL-4B-Instruct/qwen3-4b-iu-xray-dpo-2_1_test.csv"

# Base directory for GREEN reports; each run writes experiments/green_reports/<csv_stem>/
OUTPUT_DIR=./experiments/green_reports
MODEL_NAME=StanfordAIMI/GREEN-radllama2-7b

mkdir -p logs

source dpo_env/bin/activate

# export PYTHONPATH="${PWD}/GREEN:${PYTHONPATH}"

echo "=========================================="
echo "GREEN Score"
echo "=========================================="
echo "PID: $$"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo "CSV: $CSV_PATH"
echo "Output base: $OUTPUT_DIR"
echo "Model: $MODEL_NAME"
echo "=========================================="

echo "GPU Info:"
nvidia-smi

python utils/eval/green_score_eval.py \
    --csv "$CSV_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME"

echo ""
echo "Completed at: $(date)"
echo "Exit code: $?"
JOB_END_TIME=$(date +%s)
JOB_RUNTIME=$((JOB_END_TIME - JOB_START_TIME))
printf "Runtime: %02d:%02d:%02d (hh:mm:ss)\n" $((JOB_RUNTIME/3600)) $(( (JOB_RUNTIME%3600)/60)) $((JOB_RUNTIME%60))
