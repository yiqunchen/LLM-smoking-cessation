#!/usr/bin/env bash
# Verify, audit, bootstrap, plot, and package the Reviewer 3 prompt ablations.
# Refuses to proceed unless every model has all four conditions x 898 rows.
#
#   bash scripts/build_prompt_ablation_package.sh
#
# Outputs
#   figures/prompt_ablations/            figures + CSV tables
#   reports/prompt_ablation_results.docx
#   reports/prompt_ablation_package.zip
#   ~/Downloads/Reviewer_3_prompt_ablation_package_<UTC stamp>.zip   (off-repo copy)
#   ~/Downloads/reviewer_ablation_raw_results_<UTC stamp>.zip        (raw JSON backup)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

for d in results/prompt_ablations/gpt-4o-mini results/prompt_ablations/gpt-5 \
         results/prompt_ablations/deepseek-r1 results/prompt_ablations/grok-4.3 \
         results/prompt_ablations/gemini-2.5-pro; do
  $PY analysis/verify_prompt_ablations.py --results-dir "$d"
done
$PY analysis/audit_prompt_ablations.py
$PY analysis/bootstrap_prompt_ablations.py "$@"
$PY analysis/plot_prompt_ablations.py
$PY analysis/report_prompt_ablations_docx.py

# Guard: the shipped package must be three-outcome only.
if grep -rl 'predicted_design\|ground_truth_design' results/prompt_ablations/*/ figures/prompt_ablations/ 2>/dev/null | grep -q .; then
  echo "ERROR: Design-outcome fields found in ablation outputs" >&2; exit 1
fi

( cd revision && rm -f prompt_ablation_package.zip && \
  zip -q -r prompt_ablation_package.zip \
      prompt_ablation_results.docx figures/reviewer_ablations_dt10 \
      -x 'figures/reviewer_ablations_dt10/bootstrap_cache_dt10/*' && unzip -tq prompt_ablation_package.zip )

stamp=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p ~/Downloads
cp reports/prompt_ablation_package.zip ~/Downloads/Reviewer_3_prompt_ablation_package_${stamp}.zip
zip -q -r ~/Downloads/reviewer_ablation_raw_results_${stamp}.zip results/prompt_ablations/*/ -x '*.tmp' '*_errors.log'
echo "Package: reports/prompt_ablation_package.zip"
echo "Copies:  ~/Downloads/Reviewer_3_prompt_ablation_package_${stamp}.zip"
echo "         ~/Downloads/reviewer_ablation_raw_results_${stamp}.zip"
