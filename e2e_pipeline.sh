#!/bin/bash
################################################################################
# END-TO-END PIPELINE - RUN ALL 5 MODELS
#
# This script runs the full manuscript pipeline for all 5 models sequentially.
# Each model runs 13 configurations + Traditional ML + comprehensive analysis.
#
# Usage:
#   bash e2e_pipeline.sh
#
# Requirements:
#   - source ~/.bash_profile (to load API keys)
#   - conda activate research
#   - OPENAI_API_KEY, OPENROUTER_API_KEY, GEMINI_API_KEY set in bash_profile
################################################################################

set -e

# Load environment
source ~/.bash_profile
conda activate research

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "================================================================================"
echo "END-TO-END PIPELINE - ALL 5 MODELS"
echo "================================================================================"
echo "This will run the full pipeline for:"
echo "  1. gpt-4o-mini (OpenAI)"
echo "  2. gpt-5 (OpenAI)"
echo "  3. deepseek/deepseek-r1 (OpenRouter)"
echo "  4. x-ai/grok-4-fast (OpenRouter)"
echo "  5. gemini-2.5-pro (Gemini)"
echo ""
echo "Each model runs 13 configs + Traditional ML + 8 metrics analysis"
echo "Estimated total time: 2-4 hours (depending on API speeds)"
echo "================================================================================"
echo ""

# Timestamp
START_TIME=$(date +%s)

# Model 1: GPT-4o-mini
echo -e "${GREEN}[1/5] Running GPT-4o-mini...${NC}"
bash run_full_manuscript_pipeline.sh gpt-4o-mini openai
echo ""

# Model 2: GPT-5
echo -e "${GREEN}[2/5] Running GPT-5...${NC}"
bash run_full_manuscript_pipeline.sh gpt-5 openai
echo ""

# Model 3: DeepSeek-R1
echo -e "${GREEN}[3/5] Running DeepSeek-R1...${NC}"
bash run_full_manuscript_pipeline.sh deepseek/deepseek-r1-0528 openrouter
echo ""

# Model 4: Grok-4-Fast
echo -e "${GREEN}[4/5] Running Grok-4-Fast...${NC}"
bash run_full_manuscript_pipeline.sh x-ai/grok-4-fast openrouter
echo ""

# Model 5: Gemini 2.5 Pro
echo -e "${GREEN}[5/5] Running Gemini 2.5 Pro...${NC}"
bash run_full_manuscript_pipeline.sh gemini-2.5-pro gemini
echo ""

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo "================================================================================"
echo "✅ ALL 5 MODELS COMPLETED!"
echo "================================================================================"
echo "Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "Results saved to:"
echo "  - results_manuscript_gpt-4o-mini/"
echo "  - results_manuscript_gpt-5/"
echo "  - results_manuscript_deepseek_deepseek-r1/"
echo "  - results_manuscript_x-ai_grok-4-fast/"
echo "  - results_manuscript_gemini-2.5-pro/"
echo ""
echo "Each directory contains:"
echo "  • 13 JSON result files (6 Generic LLM + 7 Digital Twin)"
echo "  • summary_table.csv/md (all metrics)"
echo "  • method_comparison_all_metrics.png (visualization)"
echo "  • checkpoints/ (for resuming)"
echo "  • comparisons/ (Traditional ML vs LLM)"
echo "  • logs/ (full execution logs)"
echo "================================================================================"

