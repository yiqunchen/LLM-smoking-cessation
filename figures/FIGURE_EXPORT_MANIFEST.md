# Figure Export Manifest

Date: 2026-06-03

This folder now has clean per-figure subfolders for Word assembly. The original
source files were left in place; these are organized copies.

## Main Figures

### Figure 1

Folder: `figures/figure1/`

Use one of:

- `llm-message-paper-figure1.pdf`
- `llm-message-paper-figure1.png`
- `llm-message-paper-figure1.svg`

Status: updated repo Figure 1. The current Word draft still embeds the older
Figure 1, so replace it manually in Word.

### Figure 2

Folder: `figures/figure2/`

Updated panels:

- `figure2_main_top3_vertical_300dpi.png` / `.pdf` (main Figure 2:
  Overall Accuracy, Macro-F1, Weighted Kappa stacked in one column)
- `figure2_appendix_other3_vertical_300dpi.png` / `.pdf` (appendix/supplement:
  Cohen's Kappa, Directional Accuracy, Directional Macro-F1, Kendall's tau stacked in one column)
- `bars_all_methods_accuracy.png` / `.pdf`
- `bars_all_methods_f1.png` / `.pdf`
- `bars_all_methods_qwk.png` / `.pdf`
- `bars_all_methods_kappa.png` / `.pdf`
- `bars_all_methods_directional_accuracy.png` / `.pdf`
- `bars_all_methods_directional_macro_f1.png` / `.pdf`
- `bars_all_methods_kendall_tau.png` / `.pdf`

Source audit files:

- `bars_all_methods_manifest.md`
- `bars_all_methods_source_audit.csv`
- `bars_all_methods_source_table.csv`

Status: updated and ready to replace the old Figure 2 panels in Word. Use
`figure2_main_top3_vertical_300dpi.png` for the main manuscript. Use
`figure2_appendix_other3_vertical_300dpi.png` for the supplementary metrics in the
appendix/supplement. The supervised baseline lines now use true demographics
only and the revision-added `Demographics + History + Message Embedding`
comparator, with the supervised RF rerun aligned to the v1 RF setting
(`n_estimators=100`, `random_state=42`).

### Figure 3

Folder: `figures/figure3/`

Updated panels:

- `figure3_assembled_vertical_300dpi.png` / `.pdf`
- `figure3_assembled_300dpi.png` / `.pdf`
- `figure3_score_distributions_content.png` / `.pdf`
- `figure3_score_distributions_coping.png` / `.pdf`
- `figure3_score_distributions_quitting.png` / `.pdf`

Status: current. Use `figure3_assembled_vertical_300dpi.png` for the main
manuscript. Each domain panel now has four method rows: zero-shot (select),
Supervised, few-shot (select), and PP. The Supervised row combines RF
(demographics) and LR (demographics + history + message embeddings). Gray bars
show the observed human-rating distribution; colored grouped bars show model or
supervised predictions. These panels should be interpreted as
class-imbalance/calibration context, not as evidence that wider score
distributions prove stronger personalization.

### Figure 4

Folder: `figures/figure4/`

Original top-K figure files:

- `llm_selection_quality.png` / `.pdf` / `.csv`
- `top_k_agreement_line.png` / `.pdf`
- `top_k_agreement.csv`

Supporting strict method-level benchmark:

- `supporting_message_selection_quality_dt10.png`
- `supporting_message_selection_quality_dt10.pdf`
- `supporting_message_selection_gain_dt10.png`
- `supporting_message_selection_gain_dt10.pdf`
- `supporting_message_selection_methods_k7.csv`

Status: original Figure 4 slot can be preserved. The LLM-only top-K CSV/plot was
refreshed on 2026-06-01 from the manuscript result JSONs; the supporting
benchmark should be used for the revised supervised-vs-LLM message-selection
comparison. The supporting table uses dt10 `k_train=7`, the fixed
`Demographics + History + Message Embedding` RF feature block, 122 messages per
domain, and only two methods: Supervised RF and LLM-PP. Anchor hybrids are
excluded. The LLM-only top-K curve and the supervised dt10 support use different
message pools, so keep them as separate panels/artifacts unless every method is
recomputed on a single shared message pool.

## Supplementary / Revision Diagnostics

Folder: `figures/supplementary/`

Includes learning curves, accuracy CIs, class distributions, QWK/confusion
diagnostics, pairwise significance, and strict AI-vs-ML snapshots. The
progress-summary learning curves, strict AI-vs-ML snapshots, message-selection
tables, and supporting diagnostics were refreshed on 2026-06-01 after the
corrected RF/demographics rerun. The remaining supplementary diagnostics
that do not depend on the corrected RF/demographics pipeline were refreshed for
completeness: accuracy CIs, class distributions, QWK summary, pairwise
significance, and legacy one-metric PP learning curves.
