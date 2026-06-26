#!/bin/bash
################################################################################
# FULL END-TO-END MANUSCRIPT PIPELINE
# 
# This script runs all experiments with proper checkpointing, error handling,
# and progress tracking for reproducible results.
#
# Usage:
#   bash run_full_manuscript_pipeline.sh <model_name> [<provider>]
#
# Examples:
#   bash run_full_manuscript_pipeline.sh gpt-4o-mini
#   bash run_full_manuscript_pipeline.sh gpt-4o
#   bash run_full_manuscript_pipeline.sh deepseek/deepseek-r1 openrouter
#   bash run_full_manuscript_pipeline.sh x-ai/grok-2-1212 openrouter
#   bash run_full_manuscript_pipeline.sh gemini-2.0-flash-exp gemini
#
# Features:
# - ✅ Checkpointing: Resume from interruption
# - ✅ API robustness: Exponential backoff on errors
# - ✅ Progress bars: tqdm shows completion status
# - ✅ Canonical splits: All methods use same 70/30 split
################################################################################

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Parse arguments
if [ -z "$1" ]; then
    echo "❌ Error: Model name is required!"
    echo ""
    echo "Usage: bash run_full_manuscript_pipeline.sh <model_name> [<provider>]"
    echo ""
    echo "Examples:"
    echo "  bash run_full_manuscript_pipeline.sh gpt-4o-mini"
    echo "  bash run_full_manuscript_pipeline.sh gpt-4o"
    echo "  bash run_full_manuscript_pipeline.sh deepseek/deepseek-r1 openrouter"
    echo "  bash run_full_manuscript_pipeline.sh x-ai/grok-2-1212 openrouter"
    echo "  bash run_full_manuscript_pipeline.sh gemini-2.0-flash-exp gemini"
    exit 1
fi

MODEL="$1"
PROVIDER="${2:-openai}"  # Default to openai

# Sanitize model name for filenames (replace / with _)
MODEL_SAFE=$(echo "$MODEL" | tr '/' '_' | tr ':' '_')

# Configuration
CANONICAL_SPLITS_DIR="data_splits/canonical"
OUTPUT_DIR="results_manuscript_${MODEL_SAFE}"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
COMPARISON_DIR="$OUTPUT_DIR/comparisons"
LOG_DIR="$OUTPUT_DIR/logs"

MAX_CONCURRENT=20  # Adjust based on your API rate limits
CHECKPOINT_INTERVAL=50  # Save every 50 items

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$COMPARISON_DIR"
mkdir -p "$LOG_DIR"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/pipeline_${TIMESTAMP}.log"

echo "================================================================================"
echo "FULL MANUSCRIPT PIPELINE - REPRODUCIBLE RESULTS"
echo "================================================================================"
echo "Model: $MODEL"
echo "Provider: $PROVIDER"
echo "Timestamp: $TIMESTAMP"
echo "Output directory: $OUTPUT_DIR"
echo "Canonical splits: $CANONICAL_SPLITS_DIR"
echo "Max concurrent API calls: $MAX_CONCURRENT"
echo "Checkpoint interval: every $CHECKPOINT_INTERVAL items"
echo "Log file: $MAIN_LOG"
echo ""
echo "This pipeline will:"
echo "  1. Generate canonical train/test splits (70/30 default)"
echo "  2. Run Generic LLM methods (6 configurations)"
echo "  3. Run PP methods (7 configurations)"
echo "  4. Generate Traditional ML comparisons"
echo "  5. Analyze all results with comprehensive metrics"
echo ""
echo "Press Ctrl+C at any time to interrupt - progress will be saved!"
echo "================================================================================"
echo ""

# Log function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

# Run command with logging and error handling
run_with_logging() {
    local cmd="$1"
    local description="$2"
    
    log "▶ $description"
    log "   Command: $cmd"
    
    if eval "$cmd" 2>&1 | tee -a "$MAIN_LOG"; then
        log "✓ $description - COMPLETED"
        return 0
    else
        log "✗ $description - FAILED"
        return 1
    fi
}

################################################################################
# STEP 1: Generate Canonical Splits
################################################################################

log ""
log "=========================================="
log "STEP 1: Generate Canonical Splits"
log "=========================================="

if [ -f "$CANONICAL_SPLITS_DIR/test_participant_7030.json" ]; then
    log "✓ Canonical splits already exist. Skipping..."
else
    run_with_logging \
        "python analysis-script/create_canonical_splits.py" \
        "Creating canonical train/test splits"
fi

################################################################################
# STEP 2: Run Generic LLM Methods
################################################################################

log ""
log "=========================================="
log "STEP 2: Run Generic LLM Methods (70/30 split)"
log "=========================================="

# Method 1: Zero-shot all features
run_with_logging \
    "python analysis-script/main_eval.py \
        --mode text-only \
        --model $MODEL --provider $PROVIDER \
        --provider $PROVIDER \
        --prompt-config zero-shot \
        --data-file $CANONICAL_SPLITS_DIR/test_participant_7030.json \
        --max-concurrent $MAX_CONCURRENT \
        --checkpoint-interval $CHECKPOINT_INTERVAL \
        --output-file $OUTPUT_DIR/generic_llm_1_zero_shot.json \
        --checkpoint-file $CHECKPOINT_DIR/checkpoint_generic_llm_1.json" \
    "Generic LLM 1: Zero-shot (all features)"

# Method 2: Zero-shot selected features
run_with_logging \
    "python analysis-script/main_eval.py \
        --mode text-only \
        --model $MODEL --provider $PROVIDER \
        --prompt-config zero-shot-feature-select \
        --data-file $CANONICAL_SPLITS_DIR/test_participant_7030.json \
        --max-concurrent $MAX_CONCURRENT \
        --checkpoint-interval $CHECKPOINT_INTERVAL \
        --output-file $OUTPUT_DIR/generic_llm_2_zero_shot_select.json \
        --checkpoint-file $CHECKPOINT_DIR/checkpoint_generic_llm_2.json" \
    "Generic LLM 2: Zero-shot (selected features)"

# Method 3: Few-shot all features
run_with_logging \
    "python analysis-script/main_eval.py \
        --mode text-only \
        --model $MODEL --provider $PROVIDER \
        --prompt-config few-shot \
        --data-file $CANONICAL_SPLITS_DIR/test_participant_7030.json \
        --max-concurrent $MAX_CONCURRENT \
        --checkpoint-interval $CHECKPOINT_INTERVAL \
        --output-file $OUTPUT_DIR/generic_llm_3_few_shot.json \
        --checkpoint-file $CHECKPOINT_DIR/checkpoint_generic_llm_3.json" \
    "Generic LLM 3: Few-shot (all features)"

# Method 4: Few-shot selected features
run_with_logging \
    "python analysis-script/main_eval.py \
        --mode text-only \
        --model $MODEL --provider $PROVIDER \
        --prompt-config few-shot-feature-select \
        --data-file $CANONICAL_SPLITS_DIR/test_participant_7030.json \
        --max-concurrent $MAX_CONCURRENT \
        --checkpoint-interval $CHECKPOINT_INTERVAL \
        --output-file $OUTPUT_DIR/generic_llm_4_few_shot_select.json \
        --checkpoint-file $CHECKPOINT_DIR/checkpoint_generic_llm_4.json" \
    "Generic LLM 4: Few-shot (selected features)"

# Method 5: Continuous rating (natural language profile)
run_with_logging \
    "python analysis-script/main_eval.py \
        --mode text-only \
        --model $MODEL --provider $PROVIDER \
        --prompt-config zero-shot-natural-lang \
        --data-file $CANONICAL_SPLITS_DIR/test_participant_7030.json \
        --max-concurrent $MAX_CONCURRENT \
        --checkpoint-interval $CHECKPOINT_INTERVAL \
        --output-file $OUTPUT_DIR/generic_llm_5_continuous.json \
        --checkpoint-file $CHECKPOINT_DIR/checkpoint_generic_llm_5.json" \
    "Generic LLM 5: Continuous rating (natural language)"

# Method 6: Continuous rating with probabilities
run_with_logging \
    "python analysis-script/main_eval.py \
        --mode text-only \
        --model $MODEL --provider $PROVIDER \
        --prompt-config zero-shot-prob \
        --data-file $CANONICAL_SPLITS_DIR/test_participant_7030.json \
        --max-concurrent $MAX_CONCURRENT \
        --checkpoint-interval $CHECKPOINT_INTERVAL \
        --output-file $OUTPUT_DIR/generic_llm_6_continuous_select.json \
        --checkpoint-file $CHECKPOINT_DIR/checkpoint_generic_llm_6.json" \
    "Generic LLM 6: Continuous rating (with probabilities)"

################################################################################
# STEP 3: Run PP Methods
################################################################################

log ""
log "=========================================="
log "STEP 3: Run PP Methods"
log "=========================================="

# Method 1: Full features (70/30 split - default)
run_with_logging \
    "python analysis-script/main_eval.py \
        --mode text-only \
        --model $MODEL --provider $PROVIDER \
        --prompt-config digital-twin \
        --data-file $CANONICAL_SPLITS_DIR/test_digital_twin_7030.json \
        --train-file $CANONICAL_SPLITS_DIR/train_digital_twin_7030.json \
        --max-concurrent $MAX_CONCURRENT \
        --checkpoint-interval $CHECKPOINT_INTERVAL \
        --output-file $OUTPUT_DIR/digital_twin_1_full_7030.json \
        --checkpoint-file $CHECKPOINT_DIR/checkpoint_digital_twin_1_7030.json" \
    "PP 1: Full features (70/30)"

# Method 2: Selected features (70/30 split)
run_with_logging \
    "python analysis-script/main_eval.py \
        --mode text-only \
        --model $MODEL --provider $PROVIDER \
        --prompt-config digital-twin-select \
        --data-file $CANONICAL_SPLITS_DIR/test_digital_twin_7030.json \
        --train-file $CANONICAL_SPLITS_DIR/train_digital_twin_7030.json \
        --max-concurrent $MAX_CONCURRENT \
        --checkpoint-interval $CHECKPOINT_INTERVAL \
        --output-file $OUTPUT_DIR/digital_twin_2_select_7030.json \
        --checkpoint-file $CHECKPOINT_DIR/checkpoint_digital_twin_2_7030.json" \
    "PP 2: Selected features (70/30)"

# Method 3: With feedback (70/30 split)
run_with_logging \
    "python analysis-script/main_eval.py \
        --mode text-only \
        --model $MODEL --provider $PROVIDER \
        --prompt-config digital-twin-feedback \
        --data-file $CANONICAL_SPLITS_DIR/test_digital_twin_7030.json \
        --train-file $CANONICAL_SPLITS_DIR/train_digital_twin_7030.json \
        --max-concurrent $MAX_CONCURRENT \
        --checkpoint-interval $CHECKPOINT_INTERVAL \
        --output-file $OUTPUT_DIR/digital_twin_3_feedback_7030.json \
        --checkpoint-file $CHECKPOINT_DIR/checkpoint_digital_twin_3_7030.json" \
    "PP 3: With feedback (70/30)"

# Method 4: CBT/ACT-informed (multiple splits)
for split_name in "1090" "3070" "7030" "9010"; do
    run_with_logging \
        "python analysis-script/main_eval.py \
            --mode text-only \
            --model $MODEL --provider $PROVIDER \
            --prompt-config digital-twin-cbtact \
            --data-file $CANONICAL_SPLITS_DIR/test_digital_twin_${split_name}.json \
            --train-file $CANONICAL_SPLITS_DIR/train_digital_twin_${split_name}.json \
            --max-concurrent $MAX_CONCURRENT \
            --checkpoint-interval $CHECKPOINT_INTERVAL \
            --output-file $OUTPUT_DIR/digital_twin_4_cbtact_${split_name}.json \
            --checkpoint-file $CHECKPOINT_DIR/checkpoint_digital_twin_4_${split_name}.json" \
        "PP 4: CBT/ACT-informed (${split_name})"
done

################################################################################
# STEP 4: Traditional ML Comparison
################################################################################

log ""
log "=========================================="
log "STEP 4: Traditional ML Comparison"
log "=========================================="

# Compare with best LLM model (e.g., zero-shot natural language)
if [ -f "$OUTPUT_DIR/generic_llm_5_continuous.json" ]; then
    run_with_logging \
        "python analysis-script/compare_llm_vs_individual.py \
            --data-path $CANONICAL_SPLITS_DIR/test_participant_7030.json \
            --llm-results-path $OUTPUT_DIR/generic_llm_5_continuous.json \
            --output-dir $COMPARISON_DIR/traditional_ml_vs_llm" \
        "Traditional ML comparison (Random Forest vs LLM)"
else
    log "⚠️  LLM results not found. Skipping traditional ML comparison."
fi

################################################################################
# STEP 5: Analyze All Results
################################################################################

log ""
log "=========================================="
log "STEP 5: Analyze All Results"
log "=========================================="

run_with_logging \
    "python analysis-script/analyze_manuscript_results.py" \
    "Analyzing all results with comprehensive metrics"

################################################################################
# COMPLETION
################################################################################

log ""
log "================================================================================"
log "✅ PIPELINE COMPLETE!"
log "================================================================================"
log "Results saved to: $OUTPUT_DIR/"
log "  - Individual JSON files for each method"
log "  - summary_table.csv: Full results table"
log "  - summary_table.md: Markdown version"
log "  - summary_table_pivot.csv: Pivot summary"
log "  - method_comparison_all_metrics.png: Visualization"
log ""
log "Traditional ML comparison: $COMPARISON_DIR/traditional_ml_vs_llm/"
log "  - summary.json: Accuracy metrics"
log "  - aligned_predictions.csv: Per-sample predictions"
log "  - accuracy_comparison.png: Visual comparison"
log ""
log "Logs saved to: $MAIN_LOG"
log "================================================================================"

# Print quick summary
echo ""
echo "📊 Quick Summary:"
echo "  Total methods run: $(ls -1 $OUTPUT_DIR/*.json 2>/dev/null | wc -l | tr -d ' ')"
echo "  Checkpoints saved: $(ls -1 $CHECKPOINT_DIR/*.json 2>/dev/null | wc -l | tr -d ' ')"
echo "  Log file: $MAIN_LOG"
echo ""
echo "To view results:"
echo "  cat $OUTPUT_DIR/summary_table.md"
echo ""
echo "To compare methods:"
echo "  open $OUTPUT_DIR/method_comparison_all_metrics.png"
