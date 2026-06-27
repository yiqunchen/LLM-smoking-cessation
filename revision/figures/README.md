# Revision Results

Generated on 2026-06-04 06:35:33.

All figures in this directory are saved as 400 dpi PNG plus matching PDF.
Current model-comparison artifacts use the cleaned canonical PP 70/30 source unless a section explicitly says it is descriptive full-data context or a separate sensitivity analysis.
Missing split-specific methods are omitted rather than backfilled from another split.

## Class Imbalance and QWK

- Cleaned PP 70/30 test-set positive skew remains clear: ratings 4-5 account for Content 73.9%, Coping 63.2%, Quitting 66.1%. All QWK rows use `digital_twin_7030` and the same source policy as `../../figures/bars_all_methods_source_table.csv`.
- Best quadratic weighted kappa by domain:
| Domain | Model | Method | QWK | Accuracy |
| --- | --- | --- | --- | --- |
| Content | GPT-5 | Hybrid RF+PP | 0.370 | 0.474 |
| Coping | Grok-4-Fast | Hybrid RF+PP | 0.455 | 0.420 |
| Quitting | Grok-4-Fast | Hybrid RF+PP | 0.541 | 0.492 |
- Raw subgroup rating histograms are descriptive full PP train+test context, not a held-out model-comparison metric.
- Artifacts: [class_distribution.csv](class_distribution.csv), [class_distribution_by_domain.png](class_distribution_by_domain.png), [class_distribution_by_domain.pdf](class_distribution_by_domain.pdf), [all_rating_subgroup_histograms.png](all_rating_subgroup_histograms.png), [all_rating_subgroup_histograms.pdf](all_rating_subgroup_histograms.pdf), [qwk_results.csv](qwk_results.csv)

## Cost Analysis

- Cheapest per-participant estimate: GPT-4o-mini / Zero-shot (select) at $0.0021.
- Most expensive per-participant estimate: GPT-5 / Few-shot (all) at $0.0615.
- Highest total-study cost combinations:
| Model | Method | Cost_Per_Participant | Total_Cost |
| --- | --- | --- | --- |
| Gemini-2.5-Pro | Few-shot (all) | 0.0615 | 1.685 |
| GPT-5 | Few-shot (all) | 0.0615 | 1.685 |
| Gemini-2.5-Pro | Few-shot (select) | 0.0380 | 1.042 |
| GPT-5 | Few-shot (select) | 0.0380 | 1.042 |
| Gemini-2.5-Pro | Hybrid RF+PP | 0.0326 | 0.893 |
- Artifacts: [cost_latency_analysis.csv](cost_latency_analysis.csv), [cost_per_participant.png](cost_per_participant.png), [cost_per_participant.pdf](cost_per_participant.pdf)

## Hybrid LR vs RF

- Held-out test comparison between demographic-only Logistic Regression and Random Forest:
| Domain | Winner | Accuracy | F1 | QWK |
| --- | --- | --- | --- | --- |
| Content | Logistic Regression | 0.358 | 0.222 | 0.136 |
| Coping | Logistic Regression | 0.314 | 0.295 | 0.199 |
| Quitting | Random Forest | 0.420 | 0.257 | 0.129 |
- Artifacts: [hybrid_lr_vs_rf_comparison.csv](hybrid_lr_vs_rf_comparison.csv), [hybrid_lr_vs_rf.png](hybrid_lr_vs_rf.png), [hybrid_lr_vs_rf.pdf](hybrid_lr_vs_rf.pdf)

## Label Leakage

- Leakage checks:
| Check_Type | N_Overlapping | Total_Checked | Pct_Overlap | Risk_Level |
| --- | --- | --- | --- | --- |
| Few-shot exemplar overlap | 0 | 2 | 0.00 | None |
| PP exact text overlap | 16 | 319 | 5.02 | Medium |
| PP TF-IDF cosine similarity | 16 | 301 | 5.32 | Medium |
| Cross-split message text overlap | 104 | 107 | 97.20 | Low |
- Artifacts: [label_leakage_analysis.csv](label_leakage_analysis.csv), [profile_test_similarity_distribution.png](profile_test_similarity_distribution.png), [profile_test_similarity_distribution.pdf](profile_test_similarity_distribution.pdf)

## Pairwise Significance

- Pairwise tests completed: 120 total comparisons, 23 significant after Bonferroni correction.
- Significant comparisons with the smallest corrected p-values:
| Domain | Metric | Model_A | Model_B | Diff | p_corrected |
| --- | --- | --- | --- | --- | --- |
| Coping | Directional Accuracy | GPT-4o-mini | Grok-4-Fast | -0.172 | 0.0000 |
| Quitting | F1 | GPT-4o-mini | Gemini-2.5-Pro | -0.196 | 0.0000 |
| Coping | F1 | GPT-4o-mini | Gemini-2.5-Pro | -0.198 | 0.0000 |
| Quitting | F1 | GPT-4o-mini | Grok-4-Fast | -0.187 | 0.0000 |
| Coping | Kappa | GPT-4o-mini | Grok-4-Fast | -0.155 | 0.0000 |
| Coping | Directional Accuracy | GPT-4o-mini | Gemini-2.5-Pro | -0.142 | 0.0000 |
| Coping | Accuracy | GPT-4o-mini | Grok-4-Fast | -0.139 | 0.0000 |
| Coping | F1 | GPT-4o-mini | Grok-4-Fast | -0.172 | 0.0000 |
| Coping | F1 | GPT-4o-mini | DeepSeek-R1 | -0.157 | 0.0000 |
| Quitting | F1 | GPT-4o-mini | GPT-5 | -0.176 | 0.0000 |
- Artifacts: [pairwise_significance_tests.csv](pairwise_significance_tests.csv), [pairwise_significance_heatmap.png](pairwise_significance_heatmap.png), [pairwise_significance_heatmap.pdf](pairwise_significance_heatmap.pdf)

## Accuracy Confidence Intervals

- Revision-folder accuracy CI artifact generated with 2000 bootstrap resamples per row on `digital_twin_7030_cleaned`.
- Highest-accuracy row within each domain/method bucket:
| Domain | Method | Model | Accuracy | Accuracy_CI_Lower | Accuracy_CI_Upper |
| --- | --- | --- | --- | --- | --- |
| Content | Few-shot (all) | GPT-5 | 0.402 | 0.346 | 0.454 |
| Content | Hybrid RF+PP | GPT-5 | 0.474 | 0.418 | 0.529 |
| Content | PP | Grok-4-Fast | 0.487 | 0.428 | 0.546 |
| Content | Zero-shot (all) | GPT-5 | 0.402 | 0.350 | 0.458 |
| Coping | Few-shot (all) | Grok-4-Fast | 0.322 | 0.274 | 0.371 |
| Coping | Hybrid RF+PP | GPT-5 | 0.427 | 0.371 | 0.482 |
| Coping | PP | Grok-4-Fast | 0.446 | 0.388 | 0.502 |
| Coping | Zero-shot (all) | Grok-4-Fast | 0.313 | 0.261 | 0.365 |
| Quitting | Few-shot (all) | Grok-4-Fast | 0.355 | 0.303 | 0.414 |
| Quitting | Hybrid RF+PP | Grok-4-Fast | 0.492 | 0.440 | 0.547 |
| Quitting | PP | Gemini-2.5-Pro | 0.450 | 0.394 | 0.505 |
| Quitting | Zero-shot (all) | GPT-4o-mini | 0.309 | 0.261 | 0.362 |
- Artifacts: [accuracy_confidence_intervals.csv](accuracy_confidence_intervals.csv), [accuracy_confidence_intervals.png](accuracy_confidence_intervals.png), [accuracy_confidence_intervals.pdf](accuracy_confidence_intervals.pdf), [accuracy_confidence_intervals.md](accuracy_confidence_intervals.md)

## CBT vs ACT

- Mean accuracy across selected model/method combinations by therapy family:
| Domain | ACT | CBT | ACT_minus_CBT |
| --- | --- | --- | --- |
| Content | 0.383 | 0.407 | -0.024 |
| Coping | 0.339 | 0.344 | -0.005 |
| Quitting | 0.367 | 0.326 | 0.040 |
- Artifacts: [cbt_act_comparison.csv](cbt_act_comparison.csv), [cbt_vs_act_performance.png](cbt_vs_act_performance.png), [cbt_vs_act_performance.pdf](cbt_vs_act_performance.pdf)
- The supervised panel in the CBT/ACT figure is regenerated from
  [history_supervised_predictions.csv](history_supervised_predictions.csv), not
  from participant-split text baselines. This keeps the supervised source
  aligned with Figure 2's cleaned `digital_twin_7030` baselines.
- Source tables for the supervised CBT/ACT panel are
  `cbt_act_nonllm_accuracy_summary.csv` and
  `cbt_act_nonllm_accuracy_significance.csv`.

## Selection Quality

- Selection-quality Figure 4 artifact refreshed with uncertainty for random, human oracle, and LLM curves (2000 bootstrap resamples; 108 messages per domain).
- Preview of oracle vs random summary rows:
| Domain | K | Human Oracle (Human Rating) | Human Oracle SE | Random Mean | Random SE |
| --- | --- | --- | --- | --- | --- |
| Content | 5 | 5.000 | 0.013 | 3.955 | 0.320 |
| Content | 10 | 5.000 | 0.057 | 3.951 | 0.224 |
| Content | 15 | 4.889 | 0.082 | 3.950 | 0.186 |
| Content | 20 | 4.792 | 0.075 | 3.947 | 0.161 |
| Content | 25 | 4.733 | 0.065 | 3.945 | 0.141 |
| Coping | 5 | 5.000 | 0.036 | 3.715 | 0.349 |
| Coping | 10 | 4.933 | 0.089 | 3.714 | 0.249 |
| Coping | 15 | 4.800 | 0.087 | 3.710 | 0.200 |
| Coping | 20 | 4.725 | 0.075 | 3.709 | 0.173 |
- Artifacts: [llm_selection_quality.csv](llm_selection_quality.csv)
- Strict supporting method-level selection benchmark uses the fixed
  `Demographics + History + Message Embedding` feature block across all
  domains on dt10 `k_train=7` rows:
  [message_selection_quality_dt10.png](message_selection_quality_dt10.png),
  [message_selection_quality_dt10.pdf](message_selection_quality_dt10.pdf),
  [message_selection_gain.png](message_selection_gain.png),
  [message_selection_gain.pdf](message_selection_gain.pdf), and
  [message_selection_methods_k7.csv](message_selection_methods_k7.csv).
- The LLM-only Figure 4 curve uses a 108-message digital-twin 70/30 pool and is
  context only. The supervised method-level support uses a separate 122-message
  dt10 pool. Do not combine these curves in one panel unless all methods are
  recomputed on the same message pool.

## Demographic Subgroups

- Demographic subgroup analysis generated 120 rows; 15 entries are flagged as n < 20.
- Accuracy gaps across subgroup families:
| Domain | Subgroup_Type | Gap |
| --- | --- | --- |
| Content | Race | 0.167 |
| Coping | Race | 0.244 |
| Quitting | Race | 0.269 |
| Content | Gender | 0.354 |
| Coping | Gender | 0.354 |
| Quitting | Gender | 0.667 |
| Content | Age | 0.145 |
| Coping | Age | 0.153 |
| Quitting | Age | 0.210 |
- Artifacts: [demographic_subgroup_results.csv](demographic_subgroup_results.csv), [demographic_subgroup_race.png](demographic_subgroup_race.png), [demographic_subgroup_race.pdf](demographic_subgroup_race.pdf), [demographic_subgroup_gender.png](demographic_subgroup_gender.png), [demographic_subgroup_gender.pdf](demographic_subgroup_gender.pdf)

## Text Baselines

- Missing text baseline outputs.

## Spearman Rank Summary

- The old standalone Spearman rank artifacts were removed because they mixed participant-split and PP-split rows.
- Use the current strict diagnostics instead: [progress_summary/ordinal_qwk_spearman_snapshot.png](progress_summary/ordinal_qwk_spearman_snapshot.png), [progress_summary/table_spearman_rho.csv](progress_summary/table_spearman_rho.csv), and [progress_summary/table_spearman_rho.md](progress_summary/table_spearman_rho.md).

## Figure Redesign

- Redesign source table saved with 105 rows.
- Generated 7 generic-method figures, 7 personalized-method figures, and 3 enlarged Figure 3 domain panels.
- Artifacts: [figure_redesign_results.csv](figure_redesign_results.csv)
- Generic method figures: [bars_generic_methods_accuracy.png](bars_generic_methods_accuracy.png), [bars_generic_methods_directional_accuracy.png](bars_generic_methods_directional_accuracy.png), [bars_generic_methods_directional_macro_f1.png](bars_generic_methods_directional_macro_f1.png), [bars_generic_methods_f1.png](bars_generic_methods_f1.png), [bars_generic_methods_kappa.png](bars_generic_methods_kappa.png), [bars_generic_methods_qwk.png](bars_generic_methods_qwk.png), [bars_generic_methods_spearman_rho.png](bars_generic_methods_spearman_rho.png)
- Personalized method figures: [bars_personalized_methods_accuracy.png](bars_personalized_methods_accuracy.png), [bars_personalized_methods_directional_accuracy.png](bars_personalized_methods_directional_accuracy.png), [bars_personalized_methods_directional_macro_f1.png](bars_personalized_methods_directional_macro_f1.png), [bars_personalized_methods_f1.png](bars_personalized_methods_f1.png), [bars_personalized_methods_kappa.png](bars_personalized_methods_kappa.png), [bars_personalized_methods_qwk.png](bars_personalized_methods_qwk.png), [bars_personalized_methods_spearman_rho.png](bars_personalized_methods_spearman_rho.png)
- Large Figure 3 panels: [figure3_score_distributions_content.png](figure3_score_distributions_content.png), [figure3_score_distributions_coping.png](figure3_score_distributions_coping.png), [figure3_score_distributions_quitting.png](figure3_score_distributions_quitting.png)

## Ablation Status

- Ablation status:
| Data_Source | Rows |
| --- | --- |
| observed | 9 |
- No observed `message-only`, `profile-only`, or `history-only` outputs exist in this repo, so no no-mock ablation comparison figure is produced.
- The observed artifact here is only the full PP reference rows saved in `ablation_results.csv`.
- Status note: [ablation_status.md](ablation_status.md)
- Artifacts: [ablation_results.csv](ablation_results.csv)
