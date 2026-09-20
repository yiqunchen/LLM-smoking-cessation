#!/usr/bin/env bash
# Launch (or resume) the Reviewer 3 prompt ablations for every model and
# condition on the canonical dt10-k7 split.  One tmux session per
# (model, condition); the runner checkpoints, so re-running this script only
# fills in rows that are still pending.  Completed conditions are skipped.
#
#   bash scripts/run_reviewer_ablations.sh            # all five models
#   bash scripts/run_reviewer_ablations.sh gpt5 grok  # subset
#   CONCURRENCY=10 bash scripts/run_reviewer_ablations.sh
#
# Requires OPENROUTER_API_KEY in the environment and tmux on PATH.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

CONCURRENCY="${CONCURRENCY:-30}"
PY=".venv/bin/python analysis-script/run_dt10_reviewer_ablations_grok.py"
CONDITIONS=(pp_cbtact full_pp_no_cbtact history_ratings_only history_text_only)

# name | OpenRouter model | output dir | model-specific runner flags
# (reasoning models need a larger output budget; Gemini 2.5 Pro is uncapped
# because its hidden thinking otherwise exhausts max_tokens and returns empty content)
MODELS=(
  "gpt4o|openai/gpt-4o-mini|results_reviewer_ablations_openai_gpt-4o-mini|--max-output-tokens 300"
  "gpt5|openai/gpt-5|results_reviewer_ablations_openai_gpt-5|--max-output-tokens 1000 --reasoning-effort low"
  "deep|deepseek/deepseek-r1-0528|results_reviewer_ablations_deepseek-r1|--max-output-tokens 2000 --reasoning-effort low"
  "grok|x-ai/grok-4.3|results_reviewer_ablations_x-ai_grok-4.3|--max-output-tokens 300"
  "gemini|google/gemini-2.5-pro|results_reviewer_ablations_google_gemini-2.5-pro|--max-output-tokens 0"
)

wanted=("$@")
selected() { [ ${#wanted[@]} -eq 0 ] && return 0; for w in "${wanted[@]}"; do [ "$w" = "$1" ] && return 0; done; return 1; }

complete_rows() { # file -> number of saved rows (0 if missing)
  [ -f "$1" ] || { echo 0; return; }
  .venv/bin/python -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$1"
}

for spec in "${MODELS[@]}"; do
  IFS='|' read -r name model outdir flags <<<"$spec"
  selected "$name" || continue
  mkdir -p "$outdir"
  # Write the manifest once up front so parallel workers never race on it.
  $PY --model "$model" --output-dir "$outdir" --validate-only >/dev/null
  for cond in "${CONDITIONS[@]}"; do
    session="rerun-$name-$cond"
    if [ "$(complete_rows "$outdir/${cond}_dt10_k7.json")" -ge 898 ]; then
      echo "skip   $session (complete)"; continue
    fi
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "running $session (already live)"; continue
    fi
    tmux new-session -d -s "$session" \
      "cd '$PWD' && $PY --model $model --output-dir $outdir --conditions $cond \
         --max-concurrent $CONCURRENCY --checkpoint-interval 5 $flags \
         > logs/$session.log 2>&1"
    echo "start  $session"
  done
done
echo; echo "Monitor: tmux ls | grep rerun-   |   tail -c 300 logs/rerun-*.log"
