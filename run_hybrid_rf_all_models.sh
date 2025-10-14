#!/bin/bash

################################################################################
# Hybrid RF + Digital Twin Evaluation
# Runs Digital Twin (70/30) with Random Forest predictions as additional context
# Tests ALL 5 models: GPT-4o-mini, GPT-5, DeepSeek-R1, Grok-4-Fast, Gemini-2.5-Pro
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "HYBRID RF + DIGITAL TWIN EVALUATION (ALL MODELS)"
echo "================================================================================"
echo ""
echo "This script will run Digital Twin (70/30 split) with Random Forest predictions"
echo "as additional context for ALL 5 models."
echo ""
echo "Models to test:"
echo "  1. GPT-4o-mini (OpenAI)"
echo "  2. GPT-5 (OpenAI)"
echo "  3. DeepSeek-R1 (OpenRouter)"
echo "  4. Grok-4-Fast (OpenRouter)"
echo "  5. Gemini-2.5-Pro (Gemini)"
echo ""
echo "================================================================================"

# Source environment (removed conda activation - will use system python)
# source ~/.bash_profile
# conda activate research

# Paths (MUST match all other methods - participant 70/30 split!)
TRAIN_FILE="data_splits/canonical/train_participant_7030.json"
TEST_FILE="data_splits/canonical/test_participant_7030.json"
MAX_CONCURRENT=5

# Check if RF predictions exist
if [ ! -f "results_manuscript_hybrid_rf_grok4/rf_predictions_all_features.json" ]; then
    echo "❌ ERROR: RF predictions not found!"
    echo "Please run: python analysis-script/run_hybrid_rf_digital_twin.py"
    exit 1
fi

echo "✓ Found RF predictions: results_manuscript_hybrid_rf_grok4/rf_predictions_all_features.json"
echo ""

# Array of models to test
declare -a MODELS=(
    "gpt-4o-mini:openai"
    "gpt-5:openai"
    "deepseek/deepseek-r1-0528:openrouter"
    "x-ai/grok-4-fast:openrouter"
    "gemini-2.5-pro:gemini"
)

# Run evaluation for each model
for model_spec in "${MODELS[@]}"; do
    IFS=':' read -r model provider <<< "$model_spec"
    
    # Create safe filename (must match run_full_manuscript_pipeline.sh convention)
    model_safe=$(echo "$model" | tr '/' '_' | tr ':' '_')
    
    OUTPUT_DIR="results_manuscript_hybrid_${model_safe}"
    OUTPUT_FILE="${OUTPUT_DIR}/evaluation_results_${model_safe}_text-only_hybrid-rf-digital-twin.json"
    CHECKPOINT_FILE="${OUTPUT_DIR}/temp_checkpoint/checkpoint_results_${model_safe}_text-only_hybrid-rf-digital-twin.json"
    
    echo "--------------------------------------------------------------------------------"
    echo "MODEL: $model (Provider: $provider)"
    echo "--------------------------------------------------------------------------------"
    echo "Output: $OUTPUT_FILE"
    echo ""
    
    # Check if already completed
    if [ -f "$OUTPUT_FILE" ]; then
        entry_count=$(jq '. | length' "$OUTPUT_FILE" 2>/dev/null || echo "0")
        if [ "$entry_count" -ge 320 ]; then
            echo "✓ Already completed with $entry_count entries. Skipping..."
            echo ""
            continue
        else
            echo "⚠️  Incomplete ($entry_count entries). Resuming..."
        fi
    fi
    
    # Run evaluation (text-only for efficiency, vision mode not needed for hybrid)
    python analysis-script/main_eval.py \
        --mode text-only \
        --model "$model" \
        --provider "$provider" \
        --prompt-config hybrid-rf-digital-twin \
        --data-file "$TEST_FILE" \
        --train-file "$TRAIN_FILE" \
        --max-concurrent "$MAX_CONCURRENT" \
        --output-file "$OUTPUT_FILE" \
        --checkpoint-file "$CHECKPOINT_FILE"
    
    echo ""
    echo "✓ Completed: $model"
    echo ""
done

echo "================================================================================"
echo "✓ HYBRID EVALUATION COMPLETE FOR ALL MODELS!"
echo "================================================================================"
echo ""
echo "Results saved to:"
for model_spec in "${MODELS[@]}"; do
    IFS=':' read -r model provider <<< "$model_spec"
    model_safe=$(echo "$model" | sed 's/\//-/g')
    echo "  - results_manuscript_hybrid_${model_safe}/"
done
echo ""
echo "Next steps:"
echo "  1. Run analysis: python analysis-script/analyze_manuscript_results.py"
echo "  2. Generate figures: python analysis-script/create_comprehensive_figures.py"
echo ""

