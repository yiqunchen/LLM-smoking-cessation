#!/usr/bin/env bash
# One real request per model (canonical dt10-k7 row 0, pp_cbtact condition) to
# confirm every OpenRouter route answers with a parseable three-outcome JSON.
# Writes nothing (logs/smoke_test is only a required, gitignored placeholder). Costs a few cents. Requires OPENROUTER_API_KEY.
#
#   bash scripts/smoke_test_apis.sh                       # pp_cbtact only
#   CONDITIONS="pp_cbtact history_text_only" bash scripts/smoke_test_apis.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python analysis/run_prompt_ablations.py"
CONDITIONS="${CONDITIONS:-pp_cbtact}"
MODELS=(
  "openai/gpt-4o-mini|--max-output-tokens 300"
  "openai/gpt-5|--max-output-tokens 1000 --reasoning-effort low"
  "deepseek/deepseek-r1-0528|--max-output-tokens 2000 --reasoning-effort low"
  "x-ai/grok-4.3|--max-output-tokens 300"
  "google/gemini-2.5-pro|--max-output-tokens 0"
)
failed=0
for spec in "${MODELS[@]}"; do
  IFS='|' read -r model flags <<<"$spec"
  echo "=== $model ==="
  # shellcheck disable=SC2086
  if ! $PY --model "$model" --output-dir logs/smoke_test --conditions $CONDITIONS --preflight-only $flags 2>&1 | grep -vE '^Canonical split'; then
    failed=$((failed + 1))
  fi
done
echo
if [ "$failed" -eq 0 ]; then echo "SMOKE TEST PASSED: all ${#MODELS[@]} models answered."; else echo "SMOKE TEST FAILED: $failed model(s) did not answer." >&2; exit 1; fi
