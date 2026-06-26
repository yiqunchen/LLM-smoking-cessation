# Supplementary Refresh Audit

Date refreshed: 2026-06-03

## Refreshed for Completeness

- `accuracy_confidence_intervals.csv` / `.png` / `.pdf` / `.md`
  - Source: `analysis-script/accuracy_ci_summary.py`
  - Uses cleaned canonical PP 70/30 rows with known duplicate test items removed.

- `class_distribution.csv`
- `class_distribution_by_domain.png` / `.pdf`
- `qwk_results.csv`
  - Source: `analysis-script/class_imbalance_analysis.py`
  - Uses cleaned canonical PP 70/30 rows with known duplicate test items removed.

- `pairwise_significance_tests.csv`
- `pairwise_significance_heatmap.png` / `.pdf`
  - Source: `analysis-script/pairwise_significance_tests.py`
  - Refreshed after applying the same known duplicate-item filter before model alignment.
  - The pairwise tests use 303 shared cleaned PP rows across all five LLMs.

- `learning_curve_accuracy.png` / `.pdf`
- `learning_curve_directional_accuracy.png` / `.pdf`
- `learning_curve_directional_macro_f1.png` / `.pdf`
- `learning_curve_kappa.png` / `.pdf`
- `learning_curve_spearman_rho.png` / `.pdf`
  - Source: `create_publication_figures.create_learning_curves("figures/supplementary")`
  - These are legacy PP 10/30/70/90 prompted-LLM learning curves, not the corrected RF/demographics benchmark.
  - DeepSeek-R1 is not shown in these multi-split curves because only its 70/30 PP result file is present locally.
  - The stale combined learning-curve export is omitted from the ready supplementary folder; use these per-metric 300-DPI files.

## Already Current From 2026-06-01

- `ai_vs_ml_strict_snapshot.png` / `.pdf`
- `ai_vs_ml_strict_scaling.png` / `.pdf`
- `ordinal_qwk_spearman_snapshot.png` / `.pdf`
- `confusion_matrices_k7.png` / `.pdf` / `.csv`
- `message_selection_gain.png` / `.pdf`
- `message_selection_methods_k7.csv`
- progress-summary tables and markdown files

No placeholder or fabricated rows were generated.
