#!/bin/bash
#
# Master Evaluation Script for Manuscript Methods
# 
# This script runs ALL methods described in wip-manuscript.md Section 2.2
# using the canonical data splits to ensure fair apples-to-apples comparison.
#
# Usage: bash run_manuscript_evaluations.sh [--dry-run] [--methods METHOD_LIST]
#
# Options:
#   --dry-run        Print commands without executing
#   --methods        Comma-separated list of methods to run (e.g., "generic_llm,digital_twin")
#                    Available: generic_llm, hybrid, digital_twin, benchmarks, all (default)
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONDA_ENV="research"
OUTPUT_DIR="results_manuscript"
CANONICAL_SPLITS_DIR="data_splits/canonical"
DEFAULT_MODEL="gpt-4o-mini"
MAX_CONCURRENT=10

# Parse arguments
DRY_RUN=false
METHODS="all"

while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --methods)
      METHODS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Function to run command or print (if dry-run)
run_cmd() {
  if [ "$DRY_RUN" = true ]; then
    echo -e "${BLUE}[DRY-RUN]${NC} $@"
  else
    echo -e "${GREEN}[RUNNING]${NC} $@"
    eval "$@"
  fi
}

# Setup
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Manuscript Evaluation Script${NC}"
echo -e "${YELLOW}========================================${NC}"
echo "Conda environment: $CONDA_ENV"
echo "Output directory: $OUTPUT_DIR"
echo "Methods to run: $METHODS"
echo "Dry run: $DRY_RUN"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Activate conda environment
echo -e "${BLUE}Activating conda environment...${NC}"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

#############################################################################
# 2.2.1 BENCHMARK MODELS
#############################################################################

if [[ "$METHODS" == "all" || "$METHODS" == *"benchmarks"* ]]; then
  echo ""
  echo -e "${YELLOW}========================================${NC}"
  echo -e "${YELLOW}2.2.1 BENCHMARK MODELS${NC}"
  echo -e "${YELLOW}========================================${NC}"
  echo -e "${BLUE}Using canonical participant split (70/30)${NC}"
  echo -e "${BLUE}Traditional ML uses existing compare_llm_vs_individual.py${NC}"
  echo -e "${BLUE}Results will be generated alongside LLM comparisons${NC}"
  echo ""
  
  echo -e "${GREEN}Note: Traditional ML benchmarks (Random Forest on individual characteristics)${NC}"
  echo -e "${GREEN}      are automatically generated when running compare_llm_vs_individual.py${NC}"
  echo -e "${GREEN}      with the canonical 70/30 split.${NC}"
  echo ""
  echo -e "${GREEN}To generate traditional ML benchmarks, run:${NC}"
  echo -e "${GREEN}  python analysis-script/compare_llm_vs_individual.py \\${NC}"
  echo -e "${GREEN}    --data-path $CANONICAL_SPLITS_DIR/test_participant_7030.json \\${NC}"
  echo -e "${GREEN}    --llm-results-path <your_llm_results.json> \\${NC}"
  echo -e "${GREEN}    --output-dir $OUTPUT_DIR/traditional_ml_comparison${NC}"
fi

#############################################################################
# 2.2.2 GENERIC LLM MODELS
#############################################################################

if [[ "$METHODS" == "all" || "$METHODS" == *"generic_llm"* ]]; then
  echo ""
  echo -e "${YELLOW}========================================${NC}"
  echo -e "${YELLOW}2.2.2 GENERIC LLM MODELS${NC}"
  echo -e "${YELLOW}========================================${NC}"
  echo -e "${BLUE}Using canonical participant split (70/30)${NC}"
  echo ""
  
  # Method 1: Zero-shot + all features
  echo -e "${GREEN}Method 1: Zero-shot + all features${NC}"
  run_cmd "python analysis-script/main_eval.py \
    --mode text-only \
    --model $DEFAULT_MODEL \
    --prompt-config zero-shot \
    --data-file $CANONICAL_SPLITS_DIR/test_participant_7030.json \
    --max-concurrent $MAX_CONCURRENT \
    --output-file $OUTPUT_DIR/generic_llm_1_zero_shot_all.json \
    --checkpoint-file $OUTPUT_DIR/checkpoint_generic_llm_1.json"
  
  # Method 2: Zero-shot + selected features
  echo -e "${GREEN}Method 2: Zero-shot + selected features${NC}"
  run_cmd "python analysis-script/main_eval.py \
    --mode text-only \
    --model $DEFAULT_MODEL \
    --prompt-config zero-shot-feature-select \
    --data-file $CANONICAL_SPLITS_DIR/test_participant_7030.json \
    --max-concurrent $MAX_CONCURRENT \
    --output-file $OUTPUT_DIR/generic_llm_2_zero_shot_select.json \
    --checkpoint-file $OUTPUT_DIR/checkpoint_generic_llm_2.json"
  
  # Method 3: Few-shot + all features
  echo -e "${GREEN}Method 3: Few-shot + all features${NC}"
  run_cmd "python analysis-script/main_eval.py \
    --mode text-only \
    --model $DEFAULT_MODEL \
    --prompt-config few-shot \
    --data-file $CANONICAL_SPLITS_DIR/test_participant_7030.json \
    --max-concurrent $MAX_CONCURRENT \
    --output-file $OUTPUT_DIR/generic_llm_3_few_shot_all.json \
    --checkpoint-file $OUTPUT_DIR/checkpoint_generic_llm_3.json"
  
  # Method 4: Few-shot + selected features
  echo -e "${GREEN}Method 4: Few-shot + selected features${NC}"
  run_cmd "python analysis-script/main_eval.py \
    --mode text-only \
    --model $DEFAULT_MODEL \
    --prompt-config few-shot-feature-select \
    --data-file $CANONICAL_SPLITS_DIR/test_participant_7030.json \
    --max-concurrent $MAX_CONCURRENT \
    --output-file $OUTPUT_DIR/generic_llm_4_few_shot_select.json \
    --checkpoint-file $OUTPUT_DIR/checkpoint_generic_llm_4.json"
  
  # Method 5: Continuous rating + all features (for calibration)
  echo -e "${GREEN}Method 5: Continuous rating + all features${NC}"
  run_cmd "python analysis-script/main_eval.py \
    --mode text-only \
    --model $DEFAULT_MODEL \
    --prompt-config zero-shot-natural-lang \
    --data-file $CANONICAL_SPLITS_DIR/test_participant_7030.json \
    --max-concurrent $MAX_CONCURRENT \
    --output-file $OUTPUT_DIR/generic_llm_5_continuous_all.json \
    --checkpoint-file $OUTPUT_DIR/checkpoint_generic_llm_5.json"
  
  # Method 6: Continuous rating + selected features
  echo -e "${GREEN}Method 6: Continuous rating + selected features${NC}"
  run_cmd "python analysis-script/main_eval.py \
    --mode text-only \
    --model $DEFAULT_MODEL \
    --prompt-config zero-shot-prob \
    --data-file $CANONICAL_SPLITS_DIR/test_participant_7030.json \
    --max-concurrent $MAX_CONCURRENT \
    --output-file $OUTPUT_DIR/generic_llm_6_continuous_select.json \
    --checkpoint-file $OUTPUT_DIR/checkpoint_generic_llm_6.json"
fi

#############################################################################
# 2.2.3 HYBRID ML-LLM MODELS
#############################################################################

if [[ "$METHODS" == "all" || "$METHODS" == *"hybrid"* ]]; then
  echo ""
  echo -e "${YELLOW}========================================${NC}"
  echo -e "${YELLOW}2.2.3 HYBRID ML-LLM MODELS${NC}"
  echo -e "${YELLOW}========================================${NC}"
  echo -e "${BLUE}Using canonical participant split (70/30)${NC}"
  echo ""
  
  echo -e "${BLUE}Note: Hybrid ML-LLM models need separate implementation${NC}"
  echo -e "${BLUE}      Will use compare_llm_vs_individual.py with --run-joint-model flag${NC}"
fi

#############################################################################
# 2.2.4 DIGITAL TWIN MODELS
#############################################################################

if [[ "$METHODS" == "all" || "$METHODS" == *"digital_twin"* ]]; then
  echo ""
  echo -e "${YELLOW}========================================${NC}"
  echo -e "${YELLOW}2.2.4 DIGITAL TWIN MODELS${NC}"
  echo -e "${YELLOW}========================================${NC}"
  echo -e "${BLUE}Using canonical digital twin splits (message-based within participant)${NC}"
  echo ""
  
  # Note: Digital twin prompts need to be updated to load from split files
  # rather than from a single Excel file
  
  # Method 1: Full-feature (70/30 - DEFAULT)
  echo -e "${GREEN}Method 1: Full-feature (70/30 split)${NC}"
  run_cmd "python analysis-script/main_eval.py \
    --mode text-only \
    --model $DEFAULT_MODEL \
    --prompt-config digital-twin \
    --data-file $CANONICAL_SPLITS_DIR/test_digital_twin_7030.json \
    --train-file $CANONICAL_SPLITS_DIR/train_digital_twin_7030.json \
    --max-concurrent $MAX_CONCURRENT \
    --output-file $OUTPUT_DIR/digital_twin_1_full_7030.json \
    --checkpoint-file $OUTPUT_DIR/checkpoint_digital_twin_1_7030.json"

  # Method 2: Selected-feature (70/30)
  echo -e "${GREEN}Method 2: Selected-feature (70/30 split)${NC}"
  run_cmd "python analysis-script/main_eval.py \
    --mode text-only \
    --model $DEFAULT_MODEL \
    --prompt-config digital-twin-select \
    --data-file $CANONICAL_SPLITS_DIR/test_digital_twin_7030.json \
    --train-file $CANONICAL_SPLITS_DIR/train_digital_twin_7030.json \
    --max-concurrent $MAX_CONCURRENT \
    --output-file $OUTPUT_DIR/digital_twin_2_select_7030.json \
    --checkpoint-file $OUTPUT_DIR/checkpoint_digital_twin_2_7030.json"

  # Method 3: Full-feature + feedback (70/30)
  echo -e "${GREEN}Method 3: Full-feature + feedback (70/30 split)${NC}"
  run_cmd "python analysis-script/main_eval.py \
    --mode text-only \
    --model $DEFAULT_MODEL \
    --prompt-config digital-twin-feedback \
    --data-file $CANONICAL_SPLITS_DIR/test_digital_twin_7030.json \
    --train-file $CANONICAL_SPLITS_DIR/train_digital_twin_7030.json \
    --max-concurrent $MAX_CONCURRENT \
    --output-file $OUTPUT_DIR/digital_twin_3_feedback_7030.json \
    --checkpoint-file $OUTPUT_DIR/checkpoint_digital_twin_3_7030.json"

  # Method 4: Domain-informed CBT/ACT across splits (DEFAULT: 70/30)
  for split_name in "1090" "3070" "7030" "9010"; do
    echo -e "${GREEN}Method 4: CBT/ACT-informed ($split_name split)${NC}"
    run_cmd "python analysis-script/main_eval.py \
      --mode text-only \
      --model $DEFAULT_MODEL \
      --prompt-config digital-twin-cbtact \
      --data-file $CANONICAL_SPLITS_DIR/test_digital_twin_${split_name}.json \
      --train-file $CANONICAL_SPLITS_DIR/train_digital_twin_${split_name}.json \
      --max-concurrent $MAX_CONCURRENT \
      --output-file $OUTPUT_DIR/digital_twin_4_cbtact_${split_name}.json \
      --checkpoint-file $OUTPUT_DIR/checkpoint_digital_twin_4_${split_name}.json"
  done
fi

#############################################################################
# COMPLETION
#############################################################################

echo ""
echo -e "${YELLOW}========================================${NC}"
echo -e "${GREEN}✓ Evaluation script complete!${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Next steps:"
echo "1. Run analysis on results: python analysis-script/analyze_manuscript_results.py"
echo "2. Generate comparison table: python analysis-script/generate_results_table.py"
echo "3. Create visualizations: python analysis-script/plot_manuscript_results.py"

