#!/usr/bin/env bash
# Verify, audit, bootstrap, plot, and package the Reviewer 3 prompt ablations.
# Refuses to proceed unless every model has all four conditions x 898 rows.
#
#   bash scripts/build_reviewer_ablation_package.sh
#
# Outputs
#   revision/figures/reviewer_ablations_dt10/            figures + CSV tables
#   revision/Reviewer_3_prompt_ablation_bootstrap_results.docx
#   revision/Reviewer_3_prompt_ablation_package.zip
#   ~/Downloads/Reviewer_3_prompt_ablation_package_<UTC stamp>.zip   (off-repo copy)
#   ~/Downloads/reviewer_ablation_raw_results_<UTC stamp>.zip        (raw JSON backup)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

for d in results_reviewer_ablations_openai_gpt-4o-mini results_reviewer_ablations_openai_gpt-5 \
         results_reviewer_ablations_deepseek-r1 results_reviewer_ablations_x-ai_grok-4.3 \
         results_reviewer_ablations_google_gemini-2.5-pro; do
  $PY analysis-script/verify_reviewer_ablation_completion.py --results-dir "$d"
done
$PY analysis-script/audit_reviewer_ablation_jsons.py
$PY analysis-script/bootstrap_dt10_reviewer_ablations.py "$@"
$PY analysis-script/plot_dt10_reviewer_ablations_all_models.py
$PY analysis-script/create_reviewer_bootstrap_word_doc.py

# Guard: the shipped package must be three-outcome only.
if grep -rl 'predicted_design\|ground_truth_design' results_reviewer_ablations_*/ revision/figures/reviewer_ablations_dt10/ 2>/dev/null | grep -q .; then
  echo "ERROR: Design-outcome fields found in ablation outputs" >&2; exit 1
fi

( cd revision && rm -f Reviewer_3_prompt_ablation_package.zip && \
  zip -q -r Reviewer_3_prompt_ablation_package.zip \
      Reviewer_3_prompt_ablation_bootstrap_results.docx figures/reviewer_ablations_dt10 \
      -x 'figures/reviewer_ablations_dt10/bootstrap_cache_dt10/*' && unzip -tq Reviewer_3_prompt_ablation_package.zip )

stamp=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p ~/Downloads
cp revision/Reviewer_3_prompt_ablation_package.zip ~/Downloads/Reviewer_3_prompt_ablation_package_${stamp}.zip
zip -q -r ~/Downloads/reviewer_ablation_raw_results_${stamp}.zip results_reviewer_ablations_*/ -x '*.tmp' '*_errors.log'
echo "Package: revision/Reviewer_3_prompt_ablation_package.zip"
echo "Copies:  ~/Downloads/Reviewer_3_prompt_ablation_package_${stamp}.zip"
echo "         ~/Downloads/reviewer_ablation_raw_results_${stamp}.zip"
