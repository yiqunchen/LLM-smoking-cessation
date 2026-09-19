#!/usr/bin/env bash
# Queue the four non-Grok model families after the active Grok 4.3 run passes
# its canonical completeness check.  The jobs run in parallel, one request at
# a time per model (four total), below the prior failing six-request level.
# Re-running this script is safe: the Python runner resumes only missing rows.
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

run_model() {
  local model="$1"
  local directory="$2"
  local log_name="logs/$(basename "$directory").log"
  echo "$(date -u +%FT%TZ) Starting ${model}." >> "$log_name"
  .venv/bin/python analysis-script/run_dt10_reviewer_ablations_grok.py \
    --model "$model" \
    --output-dir "$directory" \
    --max-concurrent 1 \
    --checkpoint-interval 25 \
    >> "$log_name" 2>&1
  .venv/bin/python analysis-script/verify_reviewer_ablation_completion.py \
    --results-dir "$directory" \
    >> "$log_name" 2>&1
  echo "$(date -u +%FT%TZ) Finished ${model}." >> "$log_name"
}

run_model "openai/gpt-4o-mini" "results_reviewer_ablations_openai_gpt-4o-mini" & pid_4omini=$!
run_model "openai/gpt-5" "results_reviewer_ablations_openai_gpt-5" & pid_gpt5=$!
run_model "deepseek/deepseek-r1-0528" "results_reviewer_ablations_deepseek-r1" & pid_deepseek=$!
run_model "google/gemini-2.5-pro" "results_reviewer_ablations_google_gemini-2.5-pro" & pid_gemini=$!

for pid in "$pid_4omini" "$pid_gpt5" "$pid_deepseek" "$pid_gemini"; do
  wait "$pid"
done

.venv/bin/python analysis-script/plot_dt10_reviewer_ablations_all_models.py \
  >> logs/reviewer_ablations_all_models.log 2>&1
