#!/usr/bin/env bash
# Coordinate verification and plotting after independently launched model
# workers complete.  It deliberately does not start duplicate workers: each
# runner is interruption-safe and should be resumed explicitly with its model-
# appropriate concurrency if a provider has a transient connection failure.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

while tmux has-session -t reviewer-pp-dt10 2>/dev/null; do
  echo "$(date -u +%FT%TZ) Waiting for Grok 4.3 run to finish." >> logs/reviewer_ablations_all_models.log
  sleep 60
done

.venv/bin/python analysis-script/verify_reviewer_ablation_completion.py \
  --results-dir results_reviewer_ablations_x-ai_grok-4.3 \
  >> logs/reviewer_ablations_all_models.log 2>&1

for session in \
  reviewer-gpt4omini reviewer-gpt5 reviewer-gemini \
  reviewer-deepseek-pp reviewer-deepseek-full \
  reviewer-deepseek-ratings reviewer-deepseek-history; do
  while tmux has-session -t "$session" 2>/dev/null; do
    echo "$(date -u +%FT%TZ) Waiting for independently started ${session}." >> logs/reviewer_ablations_all_models.log
    sleep 60
  done
done

for directory in \
  results_reviewer_ablations_openai_gpt-4o-mini \
  results_reviewer_ablations_openai_gpt-5 \
  results_reviewer_ablations_deepseek-r1 \
  results_reviewer_ablations_google_gemini-2.5-pro; do
  .venv/bin/python analysis-script/verify_reviewer_ablation_completion.py \
    --results-dir "$directory" \
    >> logs/reviewer_ablations_all_models.log 2>&1
done

.venv/bin/python analysis-script/plot_dt10_reviewer_ablations_all_models.py \
  >> logs/reviewer_ablations_all_models.log 2>&1

.venv/bin/python analysis-script/audit_reviewer_ablation_jsons.py \
  >> logs/reviewer_ablations_all_models.log 2>&1

# The bootstrap script validates SHA-256 input hashes and reuses the four
# completed-model caches.  At this point it computes DeepSeek-R1 only, then
# refreshes the combined CI tables and Word-ready reviewer document.
.venv/bin/python analysis-script/bootstrap_dt10_reviewer_ablations.py \
  --n-bootstrap 2000 \
  >> logs/reviewer_ablations_all_models.log 2>&1
python3 analysis-script/create_reviewer_bootstrap_word_doc.py \
  >> logs/reviewer_ablations_all_models.log 2>&1
