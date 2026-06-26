# Ablation Status

No observed `message-only`, `profile-only`, or `history-only` ablation outputs exist in this repo yet.

Available observed data:
- `full-PP` reference rows in `ablation_results.csv`
- Saved PP prompt variants already in the repo: `digital_twin_1_full_7030.json`, `digital_twin_2_select_7030.json`, `digital_twin_3_feedback_7030.json`, and `digital_twin_4_cbtact_7030.json` (coverage varies by model; these are not the same as the requested `message-only` / `profile-only` / `history-only` ablations)
- Prompt constructors in `analysis-script/ablation_study.py`
- Canonical PP splits in `data_splits/canonical/`

Under the no-mock-data policy, `ablation_study.png` and `ablation_study.pdf` are intentionally omitted until real API-run ablation outputs are available.
