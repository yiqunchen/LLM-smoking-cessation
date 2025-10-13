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
  
  echo -e "${BLUE}Note: Benchmark models (random guess, regression) need separate implementation${NC}"
  echo -e "${BLUE}      These will be added in a follow-up script.${NC}"
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
    --max-concurrent $MAX_CONCURRENT \
    --output-file $OUTPUT_DIR/generic_llm_1_zero_shot_all.json \
    --checkpoint-file $OUTPUT_DIR/checkpoint_generic_llm_1.json"
  
  # Method 2: Zero-shot + selected features
  echo -e "${GREEN}Method 2: Zero-shot + selected features${NC}"
  run_cmd "python analysis-script/main_eval.py \
    --mode text-only \
    --model $DEFAULT_MODEL \
    --prompt-config zero-shot-feature-select \
    --max-concurrent $MAX_CONCURRENT \
    --output-file $OUTPUT_DIR/generic_llm_2_zero_shot_select.json \
    --checkpoint-file $OUTPUT_DIR/checkpoint_generic_llm_2.json"
  
  # Method 3: Few-shot + all features
  echo -e "${GREEN}Method 3: Few-shot + all features${NC}"
  run_cmd "python analysis-script/main_eval.py \
    --mode text-only \
    --model $DEFAULT_MODEL \
    --prompt-config few-shot \
    --max-concurrent $MAX_CONCURRENT \
    --output-file $OUTPUT_DIR/generic_llm_3_few_shot_all.json \
    --checkpoint-file $OUTPUT_DIR/checkpoint_generic_llm_3.json"
  
  # Method 4: Few-shot + selected features
  echo -e "${GREEN}Method 4: Few-shot + selected features${NC}"
  run_cmd "python analysis-script/main_eval.py \
    --mode text-only \
    --model $DEFAULT_MODEL \
    --prompt-config few-shot-feature-select \
    --max-concurrent $MAX_CONCURRENT \
    --output-file $OUTPUT_DIR/generic_llm_4_few_shot_select.json \
    --checkpoint-file $OUTPUT_DIR/checkpoint_generic_llm_4.json"
  
  # Method 5: Continuous rating + all features (for calibration)
  echo -e "${GREEN}Method 5: Continuous rating + all features${NC}"
  run_cmd "python analysis-script/main_eval.py \
    --mode text-only \
    --model $DEFAULT_MODEL \
    --prompt-config zero-shot-natural-lang \
    --max-concurrent $MAX_CONCURRENT \
    --output-file $OUTPUT_DIR/generic_llm_5_continuous_all.json \
    --checkpoint-file $OUTPUT_DIR/checkpoint_generic_llm_5.json"
  
  # Method 6: Continuous rating + selected features
  echo -e "${GREEN}Method 6: Continuous rating + selected features${NC}"
  run_cmd "python analysis-script/main_eval.py \
    --mode text-only \
    --model $DEFAULT_MODEL \
    --prompt-config zero-shot-prob \
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
  
  echo -e "${RED}WARNING: Digital twin methods need script updates to use canonical splits!${NC}"
  echo -e "${BLUE}Current implementation uses data/digitalTwin_msg.xlsx${NC}"
  echo -e "${BLUE}Needs update to use data_splits/canonical/train_digital_twin_*.json${NC}"
  
  # Placeholder commands (will work after script update)
  # Method 1: Full-feature (50/50)
  echo -e "${GREEN}Method 1: Full-feature (50/50 split)${NC}"
  echo -e "${BLUE}  [PLACEHOLDER - needs implementation update]${NC}"
  
  # Method 2: Selected-feature (50/50)
  echo -e "${GREEN}Method 2: Selected-feature (50/50 split)${NC}"
  echo -e "${BLUE}  [PLACEHOLDER - needs implementation update]${NC}"
  
  # Method 3: Full-feature + feedback (50/50)
  echo -e "${GREEN}Method 3: Full-feature + feedback (50/50 split)${NC}"
  echo -e "${BLUE}  [PLACEHOLDER - needs implementation update]${NC}"
  
  # Method 4: Domain-informed CBT/ACT
  for split_name in "5050" "7030" "9010"; do
    echo -e "${GREEN}Method 4: CBT/ACT-informed ($split_name split)${NC}"
    echo -e "${BLUE}  [PLACEHOLDER - needs implementation update]${NC}"
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

