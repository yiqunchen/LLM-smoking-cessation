#!/bin/bash

set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: bash run_generic_dt7030_pipeline.sh <model> [provider] [max_concurrent]"
    echo "Example: bash run_generic_dt7030_pipeline.sh gpt-5 openai 10"
    exit 1
fi

MODEL="$1"
PROVIDER="${2:-openai}"
MAX_CONCURRENT="${3:-10}"

MODEL_SAFE=$(echo "$MODEL" | tr '/' '_' | tr ':' '_')
CANONICAL_SPLITS_DIR="data_splits/canonical"
OUTPUT_DIR="results_manuscript_${MODEL_SAFE}"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
LOG_DIR="$OUTPUT_DIR/logs"

mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR" "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/generic_dt7030_${TIMESTAMP}.log"

run_eval() {
    local prompt_config="$1"
    local output_name="$2"
    local checkpoint_name="$3"
    echo "[$(date +'%F %T')] Running $prompt_config -> $output_name" | tee -a "$LOG_FILE"
    uv run python analysis-script/main_eval.py \
        --mode text-only \
        --model "$MODEL" \
        --provider "$PROVIDER" \
        --prompt-config "$prompt_config" \
        --data-file "$CANONICAL_SPLITS_DIR/test_digital_twin_7030.json" \
        --max-concurrent "$MAX_CONCURRENT" \
        --checkpoint-interval 50 \
        --output-file "$OUTPUT_DIR/$output_name" \
        --checkpoint-file "$CHECKPOINT_DIR/$checkpoint_name" \
        2>&1 | tee -a "$LOG_FILE"
}

run_eval "zero-shot" "generic_llm_1_zero_shot_dt7030.json" "checkpoint_generic_llm_1_dt7030.json"
run_eval "zero-shot-feature-select" "generic_llm_2_zero_shot_select_dt7030.json" "checkpoint_generic_llm_2_dt7030.json"
run_eval "few-shot" "generic_llm_3_few_shot_dt7030.json" "checkpoint_generic_llm_3_dt7030.json"
run_eval "few-shot-feature-select" "generic_llm_4_few_shot_select_dt7030.json" "checkpoint_generic_llm_4_dt7030.json"

echo "[$(date +'%F %T')] Completed generic PP 70/30 rerun for $MODEL" | tee -a "$LOG_FILE"
