#!/bin/bash
# Example script showing how to run traditional ML benchmarks and LLM comparison
# using the canonical 70/30 split

set -e

# Configuration
CANONICAL_SPLITS_DIR="data_splits/canonical"
OUTPUT_DIR="results_manuscript/comparisons"
LLM_RESULTS_FILE="results_manuscript/generic_llm_1_zero_shot.json"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "RUNNING TRADITIONAL ML + LLM COMPARISON"
echo "=========================================="
echo "Using canonical 70/30 split"
echo "Data: $CANONICAL_SPLITS_DIR/test_participant_7030.json"
echo "LLM Results: $LLM_RESULTS_FILE"
echo "Output: $OUTPUT_DIR"
echo ""

# Run comparison
# This will:
# 1. Build traditional ML (Random Forest) on participant characteristics using cross-validation
# 2. Compare with LLM predictions
# 3. Generate ensemble methods (voting, stacking)
# 4. Calculate correlation metrics
# 5. Create visualizations

python analysis-script/compare_llm_vs_individual.py \
  --data-path "$CANONICAL_SPLITS_DIR/test_participant_7030.json" \
  --llm-results-path "$LLM_RESULTS_FILE" \
  --output-dir "$OUTPUT_DIR/traditional_ml_vs_llm"

echo ""
echo "✓ Comparison complete!"
echo ""
echo "Results saved to: $OUTPUT_DIR/traditional_ml_vs_llm/"
echo "  - summary.json: Accuracy metrics for individual, LLM, and ensemble methods"
echo "  - aligned_predictions.csv: Per-sample predictions from all methods"
echo "  - accuracy_comparison.png: Visual comparison of methods"
echo "  - scatter plots: Individual vs LLM predictions by domain"
echo ""
echo "To run with message embeddings (joint model), add: --run-joint-model"
echo "(Requires message_embeddings.pkl to be generated first)"

