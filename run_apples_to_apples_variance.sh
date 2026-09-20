#!/bin/bash

set -euo pipefail

DOMAIN="${1:-content}"
MODE="${2:-watch-and-launch}"
LOG_DIR="revision/figures"
LOG_FILE="$LOG_DIR/apples_to_apples_variance_${DOMAIN}.log"
SUBSET_FILE="$LOG_DIR/dt7030_common_subset_${DOMAIN}.json"

mkdir -p "$LOG_DIR"

if [ "$MODE" = "watch-and-launch" ]; then
    echo "[watch] Waiting for dt7030 generic outputs..." | tee -a "$LOG_FILE"
    uv run python analysis-script/apples_to_apples_variance.py subset --domain "$DOMAIN" --wait --save "$SUBSET_FILE" | tee -a "$LOG_FILE"
    echo "[watch] Shared subset ready at $SUBSET_FILE" | tee -a "$LOG_FILE"
    echo "[watch] Launching 3-repetition variance jobs for each model in tmux..." | tee -a "$LOG_FILE"

    for model in gpt-4o-mini gpt-5 deepseek_deepseek-r1-0528 x-ai_grok-4-fast gemini-2.5-pro; do
        model_safe=$(echo "$model" | tr '/' '_' | tr ':' '_')
        case "$model" in
            gpt-4o-mini|gpt-5) provider="openai" ;;
            deepseek_deepseek-r1-0528|x-ai_grok-4-fast) provider="openrouter" ;;
            gemini-2.5-pro) provider="gemini" ;;
        esac
        session="dt7030-var-${DOMAIN}-${model_safe}"
        tmux new-session -d -s "$session" "bash run_generic_dt7030_variance_pipeline.sh '$model' '$provider' '$SUBSET_FILE' '$DOMAIN' 3 10"
        echo "[watch] Started $session" | tee -a "$LOG_FILE"
    done
    echo "[watch] Variance jobs launched. Use the summarize mode once they finish." | tee -a "$LOG_FILE"
elif [ "$MODE" = "summary" ]; then
    shift 2 || true
    PATTERN="${1:-results_manuscript_*/generic_llm_*_dt7030_${DOMAIN}_rep*.json}"
    uv run python analysis-script/apples_to_apples_variance.py summarize --pattern "$PATTERN" --domain "$DOMAIN" | tee -a "$LOG_FILE"
else
    echo "Unknown mode: $MODE" >&2
    exit 1
fi
