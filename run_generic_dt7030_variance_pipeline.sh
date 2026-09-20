#!/bin/bash

set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: bash run_generic_dt7030_variance_pipeline.sh <model> [provider] [data_file] [domain] [reps] [max_concurrent]"
    echo "Example: bash run_generic_dt7030_variance_pipeline.sh gpt-5 openai revision/figures/dt7030_common_subset_content.json content 3 10"
    exit 1
fi

MODEL="$1"
PROVIDER="${2:-openai}"
DATA_FILE="${3:-revision/figures/dt7030_common_subset_content.json}"
DOMAIN="${4:-content}"
REPS="${5:-3}"
MAX_CONCURRENT="${6:-10}"

MODEL_SAFE=$(echo "$MODEL" | tr '/' '_' | tr ':' '_')
OUTPUT_DIR="results_manuscript_${MODEL_SAFE}"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
LOG_DIR="$OUTPUT_DIR/logs"

mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR" "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/generic_dt7030_variance_${DOMAIN}_${TIMESTAMP}.log"

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
        --data-file "$DATA_FILE" \
        --max-concurrent "$MAX_CONCURRENT" \
        --checkpoint-interval 50 \
        --output-file "$OUTPUT_DIR/$output_name" \
        --checkpoint-file "$CHECKPOINT_DIR/$checkpoint_name" \
        2>&1 | tee -a "$LOG_FILE"
}

for rep in $(seq 1 "$REPS"); do
    echo "[$(date +'%F %T')] Starting repetition $rep/$REPS on $DOMAIN subset ($DATA_FILE)" | tee -a "$LOG_FILE"
    run_eval "zero-shot" "generic_llm_1_zero_shot_dt7030_${DOMAIN}_rep${rep}.json" "checkpoint_generic_llm_1_dt7030_${DOMAIN}_rep${rep}.json"
    run_eval "zero-shot-feature-select" "generic_llm_2_zero_shot_select_dt7030_${DOMAIN}_rep${rep}.json" "checkpoint_generic_llm_2_dt7030_${DOMAIN}_rep${rep}.json"
    run_eval "few-shot" "generic_llm_3_few_shot_dt7030_${DOMAIN}_rep${rep}.json" "checkpoint_generic_llm_3_dt7030_${DOMAIN}_rep${rep}.json"
    run_eval "few-shot-feature-select" "generic_llm_4_few_shot_select_dt7030_${DOMAIN}_rep${rep}.json" "checkpoint_generic_llm_4_dt7030_${DOMAIN}_rep${rep}.json"
done

echo "[$(date +'%F %T')] Completed variance reruns for $MODEL on $DOMAIN" | tee -a "$LOG_FILE"
