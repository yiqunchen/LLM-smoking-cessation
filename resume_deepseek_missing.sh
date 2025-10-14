#!/bin/bash
################################################################################
# Resume Missing DeepSeek-R1 Experiments
# 
# This script resumes the 3 missing DeepSeek-R1 experiments that have
# checkpoints but no final result files.
################################################################################

set -e

MODEL="deepseek/deepseek-r1-0528"
MODEL_SAFE="deepseek_deepseek-r1-0528"
PROVIDER="openrouter"
OUTPUT_DIR="results_manuscript_${MODEL_SAFE}"

# Source environment
source ~/.bash_profile
conda activate research

echo "================================================================================"
echo "RESUMING MISSING DEEPSEEK-R1 EXPERIMENTS"
echo "================================================================================"
echo ""
echo "Model: $MODEL"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Method 3: Few-shot (all features)
echo "[1/3] Generic LLM 3: Few-shot + all features"
python analysis-script/main_eval.py \
  --mode text-only \
  --model "$MODEL" \
  --provider "$PROVIDER" \
  --prompt-config few-shot \
  --data-file data_splits/canonical/test_participant_7030.json \
  --train-file data_splits/canonical/train_participant_7030.json \
  --max-concurrent 5 \
  --output-file "$OUTPUT_DIR/generic_llm_3_few_shot.json" \
  --checkpoint-file "$OUTPUT_DIR/checkpoints/checkpoint_generic_llm_3.json"

echo ""

# Method 4: Few-shot (selected features)
echo "[2/3] Generic LLM 4: Few-shot + selected features"
python analysis-script/main_eval.py \
  --mode text-only \
  --model "$MODEL" \
  --provider "$PROVIDER" \
  --prompt-config few-shot-feature-select \
  --data-file data_splits/canonical/test_participant_7030.json \
  --train-file data_splits/canonical/train_participant_7030.json \
  --max-concurrent 5 \
  --output-file "$OUTPUT_DIR/generic_llm_4_few_shot_select.json" \
  --checkpoint-file "$OUTPUT_DIR/checkpoints/checkpoint_generic_llm_4.json"

echo ""

# Digital Twin 4: CBT/ACT (70/30)
echo "[3/3] Digital Twin 4: CBT/ACT-informed (7030)"
python analysis-script/main_eval.py \
  --mode text-only \
  --model "$MODEL" \
  --provider "$PROVIDER" \
  --prompt-config digital-twin-cbtact \
  --data-file data_splits/canonical/test_digital_twin_7030.json \
  --train-file data_splits/canonical/train_digital_twin_7030.json \
  --max-concurrent 5 \
  --output-file "$OUTPUT_DIR/digital_twin_4_cbtact_7030.json" \
  --checkpoint-file "$OUTPUT_DIR/checkpoints/checkpoint_digital_twin_4_7030.json"

echo ""
echo "================================================================================"
echo "✅ COMPLETED DEEPSEEK-R1 MISSING EXPERIMENTS"
echo "================================================================================"
echo ""
echo "Generated files:"
echo "  - $OUTPUT_DIR/generic_llm_3_few_shot.json"
echo "  - $OUTPUT_DIR/generic_llm_4_few_shot_select.json"
echo "  - $OUTPUT_DIR/digital_twin_4_cbtact_7030.json"
echo ""
echo "Now regenerate figures:"
echo "  python analysis-script/create_comprehensive_figures.py"

