#!/bin/bash
################################################################################
# CLEANUP AND RESUME SCRIPT
#
# This script:
# 1. Identifies files with errors/missing predictions
# 2. Removes those files and their checkpoints
# 3. Allows the pipeline to resume and reprocess only the problematic items
#
# Usage: bash cleanup_and_resume.sh [model_name]
#   bash cleanup_and_resume.sh                    # Check all models
#   bash cleanup_and_resume.sh x-ai_grok-4-fast   # Clean specific model
################################################################################

set -e

MODEL_FILTER="${1:-}"

echo "================================================================================"
echo "CLEANUP AND RESUME - CHECKING FOR ERRORS"
echo "================================================================================"
echo ""

if [ -n "$MODEL_FILTER" ]; then
    echo "Checking model: $MODEL_FILTER"
    RESULT_DIRS="results_manuscript_${MODEL_FILTER}"
else
    echo "Checking all models..."
    RESULT_DIRS=$(ls -d results_manuscript_* 2>/dev/null || echo "")
fi

if [ -z "$RESULT_DIRS" ]; then
    echo "No result directories found!"
    exit 1
fi

FILES_TO_CLEAN=()

for result_dir in $RESULT_DIRS; do
    if [ ! -d "$result_dir" ]; then
        continue
    fi
    
    echo ""
    echo "Checking: $result_dir"
    echo "----------------------------------------"
    
    for json_file in "$result_dir"/*.json; do
        if [ ! -f "$json_file" ]; then
            continue
        fi
        
        # Skip metadata/summary files
        if [[ "$json_file" == *"summary"* ]] || [[ "$json_file" == *"metadata"* ]]; then
            continue
        fi
        
        # Check for ERROR in predictions using grep
        error_count=$(grep -o '"predicted_content": "ERROR"' "$json_file" 2>/dev/null | wc -l | tr -d ' ')
        error_count=$((error_count + $(grep -o '"predicted_design": "ERROR"' "$json_file" 2>/dev/null | wc -l | tr -d ' ')))
        error_count=$((error_count + $(grep -o '"predicted_coping": "ERROR"' "$json_file" 2>/dev/null | wc -l | tr -d ' ')))
        error_count=$((error_count + $(grep -o '"predicted_quitting": "ERROR"' "$json_file" 2>/dev/null | wc -l | tr -d ' ')))
        
        if [ "$error_count" -gt 0 ]; then
            echo "  ⚠️  Found $error_count errors in: $(basename $json_file)"
            FILES_TO_CLEAN+=("$json_file")
        fi
    done
done

echo ""
echo "================================================================================"

if [ ${#FILES_TO_CLEAN[@]} -eq 0 ]; then
    echo "✅ NO ERRORS FOUND! All results are clean."
    echo "================================================================================"
    exit 0
fi

echo "FOUND ${#FILES_TO_CLEAN[@]} FILES WITH ERRORS"
echo "================================================================================"
echo ""

for file in "${FILES_TO_CLEAN[@]}"; do
    echo "  $file"
done

echo ""
echo "================================================================================"
echo "CLEANUP OPTIONS:"
echo "================================================================================"
echo ""
echo "1. Delete ONLY the result files (keep checkpoints for faster resume):"
echo "   This will reprocess only the items with errors."
echo ""

for file in "${FILES_TO_CLEAN[@]}"; do
    echo "   rm \"$file\""
done

echo ""
echo "2. Delete BOTH result files AND checkpoints (full rerun for those configs):"
echo "   This will reprocess all items for these configurations."
echo ""

for file in "${FILES_TO_CLEAN[@]}"; do
    # Find corresponding checkpoint
    basename=$(basename "$file")
    dirname=$(dirname "$file")
    checkpoint="${dirname}/checkpoints/checkpoint_${basename}"
    
    echo "   rm \"$file\""
    if [ -f "$checkpoint" ]; then
        echo "   rm \"$checkpoint\""
    fi
done

echo ""
echo "================================================================================"
echo "AUTOMATIC CLEANUP (Option 1 - Delete only result files):"
echo "================================================================================"
echo ""
read -p "Proceed with automatic cleanup? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Cleaning up..."
    for file in "${FILES_TO_CLEAN[@]}"; do
        rm "$file"
        echo "  ✓ Deleted: $file"
    done
    echo ""
    echo "✅ Cleanup complete!"
    echo ""
    echo "To resume, run:"
    echo "  bash e2e_pipeline.sh"
    echo ""
    echo "Or for a specific model:"
    echo "  bash run_full_manuscript_pipeline.sh <model> <provider>"
else
    echo ""
    echo "Cleanup cancelled. No files were deleted."
fi

echo "================================================================================"

