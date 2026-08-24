#!/bin/bash
# Model evaluation via OpenAI API (CPU; no GPU).
# Table 1 VQA accuracy (GPT-4o-mini judge).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

INFERENCE_CSV_PATH="${INFERENCE_CSV_PATH:-./experiments/slake_Qwen3-VL-4B-Instruct/qwen3-4b-slake-mask-dpo_4_test.csv}"
TEXT_MODEL="${TEXT_MODEL:-gpt-4o-mini}"
DELAY="${DELAY:-0}"
OUTPUT_DIR="$PROJECT_ROOT/experiments/evaluation_outputs/"
REPORTS_DIR="$PROJECT_ROOT/experiments/evaluation_reports/"

echo "=========================================="
echo "Model Evaluation (CPU / API)"
echo "=========================================="
echo "Inference CSV Path: $INFERENCE_CSV_PATH"
echo "Text Model: $TEXT_MODEL"
echo "API Delay: $DELAY seconds"
echo "Output Directory: $OUTPUT_DIR"
echo "Reports Directory: $REPORTS_DIR"
echo "=========================================="

mkdir -p "$OUTPUT_DIR"
mkdir -p "$REPORTS_DIR"

if [[ -f dpo_env/bin/activate ]]; then
  # shellcheck disable=SC1091
  source dpo_env/bin/activate
fi

PARENT_DIR=$(basename "$(dirname "$INFERENCE_CSV_PATH")")
BASE_NAME=$(basename "$INFERENCE_CSV_PATH" .csv)
INPUT_FILENAME="${PARENT_DIR}/${BASE_NAME}"
EVALUATED_CSV_PATH="${OUTPUT_DIR%/}/${INPUT_FILENAME}_evaluated.csv"
mkdir -p "$(dirname "$EVALUATED_CSV_PATH")"

REPORT_OUTPUT_DIR="${REPORTS_DIR%/}/${PARENT_DIR}"
mkdir -p "$REPORT_OUTPUT_DIR"

echo "Starting correctness evaluation..."

python utils/correctness_evaluator.py \
  --input_csv "$INFERENCE_CSV_PATH" \
  --output_csv "$EVALUATED_CSV_PATH" \
  --model "$TEXT_MODEL" \
  --delay "$DELAY"

echo "Evaluation complete: $EVALUATED_CSV_PATH"
echo "Reports: $REPORT_OUTPUT_DIR"
