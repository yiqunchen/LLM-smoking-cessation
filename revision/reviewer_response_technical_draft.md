# Technical Response Draft

Updated: 2026-05-21

This file tracks reviewer requests against the current revision artifacts. It
is a working draft for the response letter, not final `.docx` prose.

## Core Update

The newest strict shared-row analyses revise the central technical claim. The
paper should not say that LLM-PP methods outperform supervised baselines.
It should say that historical participant ratings are the strongest observed
personalization signal in the current benchmark, supervised RF is the strongest
aggregate classifier (QWK 0.527), and LLM-PP is a competitive, complementary
persona-conditioned signal that is most competitive at low history / cold start
and wins clearly on coping/quitting message selection.

Primary source:

- [figures/progress_summary/clean_results.md](figures/progress_summary/clean_results.md)
- [figures/progress_summary/progress_summary_all.md](figures/progress_summary/progress_summary_all.md)
- [figures/progress_summary/test_set_audit.md](figures/progress_summary/test_set_audit.md)

Strict `k_train=7` result to use in the response letter (dt10 within-participant
split, Demographics feature block, identical 898 held-out rows for both methods):

| Endpoint | Feature block | N | Supervised RF | LLM-PP |
|---|---:|---:|---:|---:|
| Accuracy | Demographics | 898 | 0.474 | 0.462 |
| Macro-F1 | Demographics | 898 | 0.403 | 0.370 |
| QWK | Demographics | 898 | 0.527 | 0.488 |
| Within-participant Spearman rho | Demographics | 898 | NA (undefined) | ~0.06 |

Supervised RF assigns one value per participant and so cannot rank a
participant's messages; its within-participant Spearman is undefined. LLM-PP's
within-participant Spearman is small but defined (~0.06).

## Historical-Ratings Implementation Map

Core response-letter sentence:

> The revised analyses identify prior participant ratings / response history
> as the strongest observed personalization signal for PME prediction in this
> benchmark. As more history was available, the supervised RF baseline improved
> and was difficult to outperform on aggregate metrics. We therefore reframed
> the manuscript around response-history-informed personalized PME prediction,
> with LLM-PP contributing a complementary persona-conditioned signal that is
> most competitive at low history and clearly best for coping/quitting message
> selection.

Use this wording instead of the absolute sentence "historical ratings are the
most important element" unless a clean history-only/profile-only/message-only
ablation is added.

Places that must reflect this revised message:

1. Title/keywords: prefer "response history" or "rating history" in the title
   if acceptable; otherwise add it to the keywords.
2. Abstract Methods: identify prior participant ratings as the history context
   used for personalization.
3. Abstract Results: include the `k_train=1,3,7` learning-curve evidence
   alongside the strict `k_train=7` RF-vs-LLM-PP table.
4. Abstract Discussion/Conclusion: state that response-history supervised
   baselines captured substantial PME signal; avoid LLM-dominance language.
5. Introduction final paragraph: make the first contribution the shared-row
   test of how much participant history supports personalized PME prediction.
6. Methods 2.2 split: define history ratings as the personalization substrate
   and held-out ratings as evaluation only.
7. Methods supervised RF: describe the Demographics-block RF as the primary
   supervised comparator, and state the same-participant caveat — because dt10
   is within-participant (all 301 participants appear in both train and test),
   the near-unique demographic vector acts as a participant fingerprint, so RF
   behaves as a strong participant-calibrated baseline rather than evidence that
   demographics predict PME.
8. Methods LLM-PP: specify that LLM-PP uses the same prior-rating history as
   context, so it is compared against supervised history-calibrated scoring on
   identical held-out rows.
9. Results 3.1: present strict RF-vs-LLM-PP performance without implying LLM
   superiority on aggregate metrics.
10. Results 3.2 / Supplementary Figure 1: make history-length sensitivity the
    central evidence for personalization.
11. Figure 1: show prior ratings/history as the visible personalization input.
12. Figure 2 caption: label the benchmark as a within-participant comparison of
    supervised RF vs LLM-PP on the same shared held-out rows.
13. Figure 4 and supporting selection gain figure: include supervised RF as a
    response-history-based message-selection comparator.
14. Discussion first paragraph: lead with response-history signal and the
    revised role of LLMs as complementary.
15. Limitations: state that the history result is observational and does not
    prove a universal causal ordering of input importance.
16. Reviewer responses AE-1, AE-3/R4-1, R2-1, R3-9, R3-12, and R4-2: use the
    same history-first framing, with the two-method (RF vs LLM-PP) comparison
    and the same-participant caveat.

## Coverage Snapshot

### Answered By Current Artifacts

- AE-2 / R1 / R3-3: accuracy CIs and pairwise LLM significance tests.
- AE-4 / R3-5: leakage audit and partition clarification.
- R2 / R3 / R4: stronger supervised baselines and strict shared-row
  RF/LLM/hybrid benchmark.
- R2-1: sensitivity to number of history ratings via `k_train=1,3,7`
  learning curves.
- R2-4: demographic subgroup analysis.
- R3-6: class imbalance, QWK, and confusion matrices.
- R3-7: cost estimates. Latency remains unmeasured.
- R3-12: redesigned readable figure suite.
- R3-12c / R4 minor: message-selection uncertainty and supervised RF included
  in message selection.
- R4-2: CBT vs ACT comparison.
- R4-3: LR vs RF comparison.

### Partial / Needs Careful Wording

- AE-3 / R4-1: QWK, confusion matrices, and within-participant Spearman help
  address rating-style concerns, but there is still no direct z-score rerun.
- R1 / R3-4: prompt sensitivity is partially informed by saved variants, but
  there is no complete factorial prompt-sensitivity or repeated-run variance
  artifact.
- R3-8: image handling is technically clear from the scripts, but still needs
  an explicit final prose statement.
- R3-9: no clean observed message-only/profile-only/history-only ablation is
  available; do not invent one.
- R3-11: a consolidated reproducibility table is still needed.
- R3 recommender-system baselines: discussed as future work; no new CF/neural
  recommender artifact exists.

## Current Figure Package

Preserve the original written/published figure numbering. The strict
shared-row analyses update the captions, Results text, and supporting
appendix/reviewer figures, but they should not replace the original Figure
1-4 architecture.

Main-paper figures to cite:

- Figure 1, study design schematic:
  [../figures/llm-message-paper-figure1.svg](../figures/llm-message-paper-figure1.svg),
  [../figures/llm-message-paper-figure1.pdf](../figures/llm-message-paper-figure1.pdf)
- Figure 2, model-performance comparison across domains:
  [../figures/bars_all_methods_accuracy.png](../figures/bars_all_methods_accuracy.png),
  [../figures/bars_all_methods_f1.png](../figures/bars_all_methods_f1.png),
  [../figures/bars_all_methods_directional_accuracy.png](../figures/bars_all_methods_directional_accuracy.png),
  [../figures/bars_all_methods_directional_macro_f1.png](../figures/bars_all_methods_directional_macro_f1.png),
  [../figures/bars_all_methods_kappa.png](../figures/bars_all_methods_kappa.png).
  The rebuilt source table and audit are
  [../figures/bars_all_methods_source_table.csv](../figures/bars_all_methods_source_table.csv)
  and
  [../figures/bars_all_methods_source_audit.csv](../figures/bars_all_methods_source_audit.csv).
  All plotted rows use the cleaned canonical PP 70/30 source; original
  LR/RF demographics baselines are retained and LR/RF `Demographics + History + Message Embedding`
  baselines are added as matched 70/30 reference lines.
- Figure 3, predicted-score distributions:
  [../figures/figure3_score_distributions_content.png](../figures/figure3_score_distributions_content.png),
  [../figures/figure3_score_distributions_coping.png](../figures/figure3_score_distributions_coping.png),
  [../figures/figure3_score_distributions_quitting.png](../figures/figure3_score_distributions_quitting.png)
- Figure 4, top-K message-selection quality:
  [../figures/llm_selection_quality.png](../figures/llm_selection_quality.png),
  [../figures/top_k_agreement_line.png](../figures/top_k_agreement_line.png)
- Supporting supervised/LLM message-selection benchmark:
  [figures/progress_summary/message_selection_gain.png](figures/progress_summary/message_selection_gain.png),
  [figures/progress_summary/message_selection_methods_k7.csv](figures/progress_summary/message_selection_methods_k7.csv)
- Appendix / Supplementary Figure 1, history-length learning curves:
  [../figures/learning_curve_accuracy.png](../figures/learning_curve_accuracy.png),
  [../figures/learning_curve_directional_macro_f1.png](../figures/learning_curve_directional_macro_f1.png),
  [../figures/learning_curve_kappa.png](../figures/learning_curve_kappa.png),
  [../figures/learning_curve_spearman_rho.png](../figures/learning_curve_spearman_rho.png)

Original proof audit:

- Source checked: [revision/amiajnl-2026-019201_Proof_hi.pdf](amiajnl-2026-019201_Proof_hi.pdf).
- Found original main figures: Figure 1, Figure 2, Figure 3, and Figure 4.
- Found one original supplementary figure in Appendix A3: Supplementary Figure 1
  for history-length learning curves. The proof caption line incorrectly labels
  it "Figure 4"; treat it as Supplementary Figure 1 in the revision.
- I did not find any additional original manuscript figure captions beyond
  these five figure slots. Appendix A1 is prompt templates; Appendix A2 is
  bootstrap CI tables, not additional figures.
- Revision-only supporting figures should stay appendix/reviewer support unless
  the manuscript figure plan is intentionally expanded.

Strict shared-row reviewer-supporting figures, not replacement main figures:

- [figures/progress_summary/ai_vs_ml_strict_snapshot.png](figures/progress_summary/ai_vs_ml_strict_snapshot.png)
- [figures/progress_summary/learning_curves.png](figures/progress_summary/learning_curves.png)
- [figures/progress_summary/ai_vs_ml_strict_scaling.png](figures/progress_summary/ai_vs_ml_strict_scaling.png)
- [figures/progress_summary/ordinal_qwk_spearman_snapshot.png](figures/progress_summary/ordinal_qwk_spearman_snapshot.png)
- [figures/progress_summary/confusion_matrices_k7.png](figures/progress_summary/confusion_matrices_k7.png)
- [figures/progress_summary/message_selection_gain.png](figures/progress_summary/message_selection_gain.png)

Supplemental reviewer figures:

- [figures/accuracy_confidence_intervals.png](figures/accuracy_confidence_intervals.png)
- [figures/pairwise_significance_heatmap.png](figures/pairwise_significance_heatmap.png)
- [figures/class_distribution_by_domain.png](figures/class_distribution_by_domain.png)
- [figures/all_rating_subgroup_histograms.png](figures/all_rating_subgroup_histograms.png)
- [figures/profile_test_similarity_distribution.png](figures/profile_test_similarity_distribution.png)
- [figures/partition_schematic.png](figures/partition_schematic.png)
- [figures/partition_matrix_schematic.png](figures/partition_matrix_schematic.png)
- [figures/all_baselines_comparison.png](figures/all_baselines_comparison.png)
- [figures/baseline_feature_ablation.png](figures/baseline_feature_ablation.png)
- [figures/history_supervised_best.png](figures/history_supervised_best.png)
- [figures/hybrid_lr_vs_rf.png](figures/hybrid_lr_vs_rf.png)
- [figures/cost_per_participant.png](figures/cost_per_participant.png)
- [figures/demographic_subgroup_race.png](figures/demographic_subgroup_race.png)
- [figures/demographic_subgroup_gender.png](figures/demographic_subgroup_gender.png)
- [figures/cbt_vs_act_performance.png](figures/cbt_vs_act_performance.png)
- [figures/progress_summary/message_selection_gain.png](figures/progress_summary/message_selection_gain.png)
  as the strict RF-vs-LLM supporting selection benchmark.

## Associate Editor

### AE-1: Digital-twin terminology

Status: `answered by prose revision`

Draft response:

We agree that the original wording overstated the digital-twin analogy. We
have revised the title, abstract, introduction, methods, and figure captions to
use more precise terminology: persona-conditioned LLM prediction and
history-augmented prediction. Where we retain the phrase "digital-twin
prompting," we define it narrowly as shorthand for persona-conditioned,
history-augmented LLM prediction of PME ratings. We do not claim to model
longitudinal smoking behavior, physiological dynamics, or causal treatment
response.

### AE-2: Missing hypothesis testing / significance

Status: `answered`

Draft response:

We added formal uncertainty analyses for the LLM comparisons. The revision now
reports 95% bootstrap confidence intervals for accuracy with 2,000 resamples
per model-method-domain row, and paired model-comparison tests across 10 LLM
pairs, 3 domains, and 4 metrics. The saved file includes both Bonferroni and
BH-adjusted columns; using the conservative Bonferroni flag, 23/120 tests
remain significant, concentrated in Coping (13) and Quitting (9), with fewer
in Content (1). We revised the Results to emphasize effect estimates and
uncertainty rather than unsupported absolute model rankings.

Evidence:

- Total tests: `120`
- Bonferroni-significant tests: `23`
- Significant by domain: Content `1`, Coping `13`, Quitting `9`
- Significant by metric: Accuracy `4`, Directional Accuracy `5`, F1 `9`,
  Kappa `5`
- Accuracy CI rows: `60`
- Accuracy CI source: cleaned canonical PP 70/30; known duplicate
  test items removed from every method
- Bootstrap resamples per accuracy-CI row: `2000`
- Artifacts:
  [figures/pairwise_significance_tests.csv](figures/pairwise_significance_tests.csv),
  [figures/pairwise_significance_heatmap.png](figures/pairwise_significance_heatmap.png),
  [figures/accuracy_confidence_intervals.csv](figures/accuracy_confidence_intervals.csv),
  [figures/accuracy_confidence_intervals.png](figures/accuracy_confidence_intervals.png)

### AE-3 / R4-1: Are models learning rating style?

Status: `partial`

Draft response:

We agree that participant-level rating tendency is a serious alternative
explanation. In the revised analysis, participant history is included in the
supervised benchmark rather than treated as an LLM-only advantage. We also
added QWK and row-normalized confusion matrices to evaluate ordinal agreement
under class imbalance, and within-participant Spearman as a secondary
rank-sensitive diagnostic. These analyses show that supervised RF captures
substantial aggregate signal and is the strongest aggregate classifier, while
LLM-PP is competitive and, unlike RF, can rank a participant's messages.
Critically, because dt10 is within-participant (all 301 participants appear in
both train and test), the near-unique demographic vector acts as a participant
fingerprint, so RF behaves as a strong participant-calibrated baseline rather
than evidence that demographics predict PME. A trivial "predict each
participant's most-frequent prior rating" baseline reproduces RF accuracy
(Content .53, Coping .45, Quitting .47); on a strict unseen-participant split RF
falls to ~0.35. These analyses do not fully replace a direct z-score rerun,
which we now identify as a remaining limitation.

Current strict ordinal/rank values at `k_train=7`, Demographics block (per
domain, same held-out rows for both methods):

| Domain | Method | Accuracy | QWK | Spearman rho |
|---|---|---:|---:|---:|
| Content | Supervised RF | 0.523 | 0.449 | NA (undefined) |
| Content | LLM-PP | 0.499 | 0.402 | ~0.06 |
| Coping | Supervised RF | 0.450 | 0.543 | NA (undefined) |
| Coping | LLM-PP | 0.431 | 0.503 | ~0.06 |
| Quitting | Supervised RF | 0.450 | 0.589 | NA (undefined) |
| Quitting | LLM-PP | 0.457 | 0.559 | ~0.06 |

Note: Content RF wins on both accuracy and QWK; Coping RF wins on both;
Quitting is a ~tie — LLM-PP edges accuracy (0.457 vs 0.450) while RF leads QWK
(0.589 vs 0.559). The mean across domains is RF accuracy 0.474 / QWK 0.527 vs
LLM-PP accuracy 0.462 / QWK 0.488.

Evidence:

- [figures/progress_summary/ordinal_qwk_spearman_snapshot.png](figures/progress_summary/ordinal_qwk_spearman_snapshot.png)
- [figures/progress_summary/confusion_matrices_k7.png](figures/progress_summary/confusion_matrices_k7.png)
- [figures/progress_summary/confusion_metrics_k7.csv](figures/progress_summary/confusion_metrics_k7.csv)

### AE-4 / R3-5: Potential label leakage

Status: `answered`

Draft response:

We added a dedicated leakage audit covering few-shot exemplar overlap, exact
history/test overlap within the PP split, TF-IDF similarity between
profile and held-out messages, and expected cross-split message reuse under
participant splitting. Few-shot exemplars had zero overlap with participant
test rows. The within-participant audit identified 16/319 near-duplicate or
exact-overlap items; these were removed from the headline evaluation, changing
accuracy/F1 by no more than 0.005. Cross-split message-text overlap is high by
design because different participants rate the same message library; this is
not leakage of a participant's held-out label.

Evidence:

- Few-shot exemplar overlap: `0/2` (`0.00%`)
- Digital-twin exact text overlap: `16/319` (`5.02%`)
- Digital-twin TF-IDF high-similarity count: `16/301` (`5.32%`)
- Similarity distribution: mean cosine `0.209`, median `0.163`
- Cross-split message overlap: `104/107` (`97.20%`), expected by design
- Artifacts:
  [figures/label_leakage_analysis.csv](figures/label_leakage_analysis.csv),
  [figures/profile_test_similarity_distribution.png](figures/profile_test_similarity_distribution.png),
  [figures/partition_schematic.png](figures/partition_schematic.png),
  [figures/partition_matrix_schematic.png](figures/partition_matrix_schematic.png)

## Reviewer 2

### R2-1: Sensitivity versus number of history ratings

Status: `answered`

Draft response:

We added a history-length sensitivity analysis for `k_train=1,3,7` on the dt10
within-participant split (Demographics block). Additional participant history
improves both methods. Mean accuracy for supervised RF increases from 0.420 to
0.448 to 0.474, while LLM-PP increases from 0.417 to 0.438 to 0.462. Mean QWK
also increases: supervised RF from 0.418 to 0.447 to 0.527, and LLM-PP from
0.417 to 0.457 to 0.488. LLM-PP ties RF on QWK at `k=1` and slightly exceeds it
at `k=3` (0.457 vs 0.447), with RF pulling ahead at `k=7`; the LLM is thus most
competitive at low history / cold start. We note that each `k_train` value uses
its own shared test-row intersection, so the curves are best read as within-k
method comparisons and scaling evidence. These results make response history the
strongest observed personalization signal in the revised benchmark, while not
proving a universal causal ordering of input importance.

Evidence:

- [figures/progress_summary/learning_curves.png](figures/progress_summary/learning_curves.png)
- [figures/progress_summary/table_accuracy.md](figures/progress_summary/table_accuracy.md)
- [figures/progress_summary/table_qwk.md](figures/progress_summary/table_qwk.md)
- [figures/progress_summary/sweep_by_k_train.md](figures/progress_summary/sweep_by_k_train.md)

### R2-2 / R3-1 / R4-2: Stronger supervised baselines

Status: `answered`

Draft response:

We substantially strengthened the baseline suite and rebuilt the headline
analysis as a strict shared-row comparison of supervised RF vs LLM-PP on the
dt10 within-participant split (Demographics block). This changed the
interpretation: supervised RF is the strongest aggregate classifier, and LLM-PP
is competitive and complementary rather than dominant. At `k_train=7`, mean
accuracy is 0.474 for supervised RF and 0.462 for LLM-PP; mean macro-F1 is 0.403
vs 0.370; and mean QWK is 0.527 vs 0.488.

Evidence:

- [figures/progress_summary/ai_vs_ml_strict_snapshot.png](figures/progress_summary/ai_vs_ml_strict_snapshot.png)
- [figures/progress_summary/ai_vs_ml_strict_fixed_feature.csv](figures/progress_summary/ai_vs_ml_strict_fixed_feature.csv)
- [figures/progress_summary/ai_vs_ml_fixed_feature_mean_by_k.csv](figures/progress_summary/ai_vs_ml_fixed_feature_mean_by_k.csv)
- Context-only older baseline figures:
  [figures/all_baselines_comparison.png](figures/all_baselines_comparison.png),
  [figures/baseline_feature_ablation.png](figures/baseline_feature_ablation.png),
  [figures/history_supervised_best.png](figures/history_supervised_best.png)

### R2-4: Demographic subgroup analysis

Status: `answered`

Draft response:

We added subgroup analyses across race/ethnicity, gender, and age using the
PP 70/30 source and explicitly flag small cells. The table contains
120 subgroup rows, of which 15 are flagged as `n < 20`; we therefore interpret
subgroup differences cautiously.

Evidence:

- Total subgroup rows: `120`
- Small-cell flags: `15`
- Largest non-flagged race gap example: `GPT-5 / Quitting`, Black `0.474` vs
  Other `0.256` (`0.218`)
- Largest non-flagged gender gap example: `DeepSeek-R1 / Quitting`, Female
  `0.451` vs Male `0.370` (`0.082`)
- Largest non-flagged age gap example: `DeepSeek-R1 / Quitting`, age `25-30`
  `0.446` vs age `18-24` `0.277` (`0.169`)
- Artifacts:
  [figures/demographic_subgroup_results.csv](figures/demographic_subgroup_results.csv),
  [figures/demographic_subgroup_race.png](figures/demographic_subgroup_race.png),
  [figures/demographic_subgroup_gender.png](figures/demographic_subgroup_gender.png)

## Reviewer 3

### R3-4 / R1: Prompt sensitivity and generation settings

Status: `partial`

Draft response:

The revision clarifies the prompt templates and canonical-prompt comparison
design. The repo contains observed prompt-variant outputs for full-feature,
selected-feature, full-plus-feedback, and CBT/ACT variants, but there is no
complete factorial prompt-sensitivity or repeated-run variance artifact.
Therefore the response should not claim that stochasticity and prompt
sensitivity are fully characterized. The correct response is that a canonical
prompt was used for interpretability, observed variants are reported where
available, and systematic prompt sensitivity remains a limitation.

Evidence:

- Observed-only prompt variants: [figures/ablation_results.csv](figures/ablation_results.csv)
- Status note: [figures/ablation_status.md](figures/ablation_status.md)
- Prompt templates: [../docs/prompt_templates.md](../docs/prompt_templates.md)

### R3-6: Class imbalance and QWK

Status: `answered`

Draft response:

We added an explicit class-distribution analysis and now report QWK and
confusion matrices. Held-out labels are skewed toward ratings 4-5, so accuracy
alone can reward modal-class predictions. In the latest strict `k_train=7`
comparison, supervised RF has the highest mean QWK (0.527), with LLM-PP
competitive behind it (0.488). The confusion matrices make clear where each
method over- or under-predicts ordinal categories.

Evidence:

- dt10 held-out test ratings 4-5 share, Content: `68.2%`
- dt10 held-out test ratings 4-5 share, Coping: `61.6%`
- dt10 held-out test ratings 4-5 share, Quitting: `61.7%`
- Latest strict mean QWK, `k_train=7` (Demographics block): supervised RF
  `0.527`, LLM-PP `0.488`
- Artifacts:
  [figures/class_distribution.csv](figures/class_distribution.csv),
  [figures/class_distribution_by_domain.png](figures/class_distribution_by_domain.png),
  [figures/progress_summary/ordinal_qwk_spearman_snapshot.png](figures/progress_summary/ordinal_qwk_spearman_snapshot.png),
  [figures/progress_summary/confusion_matrices_k7.png](figures/progress_summary/confusion_matrices_k7.png),
  [figures/qwk_results.csv](figures/qwk_results.csv)

### R3-7: Cost / latency / scalability

Status: `partial`

Draft response:

We added a token-based cost analysis for all model/method combinations, which
addresses API-spend scalability. The cheapest saved estimate is GPT-4o-mini
zero-shot select at $0.002135 per participant. The most expensive saved
configurations are GPT-5 and Gemini-2.5-Pro few-shot all at $0.061513 per
participant. GPT-5/Gemini PP prompting is $0.030612 per participant
and Hybrid RF+PP is $0.032600 per participant. The saved artifact does not
contain empirical latency measurements, so latency should be stated as a
remaining deployment-specific limitation.

Evidence:

- Cheapest per-participant estimate: `GPT-4o-mini / Zero-shot (select)` =
  `$0.002135`
- Most expensive per-participant estimate: `GPT-5 / Few-shot (all)` and
  `Gemini-2.5-Pro / Few-shot (all)` = `$0.061513`
- Highest saved total-study estimate: `$1.6854`
- Example personalized estimate: `GPT-5 / Digital-Twin-Based Prompting` =
  `$0.003061` per call, `$0.030612` per participant, `$0.8388` total
- Artifacts:
  [figures/cost_latency_analysis.csv](figures/cost_latency_analysis.csv),
  [figures/cost_per_participant.png](figures/cost_per_participant.png)

### R3-8: Ambiguous image handling

Status: `answered technically; prose still needed`

Draft response:

The response letter and Methods should explicitly state what inputs were
included in the evaluated model calls. Although prompt templates retain legacy
language asking the model to describe any provided image, the manuscript
experiments were run with `--mode text-only`; `main_eval.py` only attaches
image payloads in the `mode == 'vision'` branch. Therefore, the primary
benchmark should be described as a text/profile/history prediction task rather
than a multimodal image-interpretation task. The design domain should remain
outside the primary results because it depends most directly on visual
presentation.

Evidence:

- Prompt templates: [../docs/prompt_templates.md](../docs/prompt_templates.md)
- Evaluation code: [../analysis-script/main_eval.py](../analysis-script/main_eval.py)
- Pipeline scripts call `main_eval.py` with `--mode text-only`:
  [../run_full_manuscript_pipeline.sh](../run_full_manuscript_pipeline.sh),
  [../run_generic_dt7030_pipeline.sh](../run_generic_dt7030_pipeline.sh),
  [../run_generic_dt7030_variance_pipeline.sh](../run_generic_dt7030_variance_pipeline.sh)

### R3-9 / R4-1: Component ablation

Status: `partial`

Draft response:

We do not claim message-only, profile-only, or history-only ablation results
because those exact observed runs are not present in the current revision
artifacts. The repo contains real saved prompt-variant outputs for
full-feature, selected-feature, full-plus-feedback, and CBT/ACT variants, which
partially inform prompt sensitivity but do not cleanly isolate profile versus
history versus message-only effects. The response should explicitly say that
these ablations remain future work.

Evidence:

- [figures/ablation_results.csv](figures/ablation_results.csv)
- [figures/ablation_status.md](figures/ablation_status.md)

### R3-11: Reproducibility details

Status: `partial`

Current state:

The repo exposes prompt templates, split files, evaluation scripts, and raw
response artifacts, but the response package still lacks a compact
reproducibility table listing model ID, provider/endpoint, call date range,
prompt file, split file, and generation parameter policy.

Recommended action:

Add the compact appendix table before final `.docx` assembly. Use the
past-tense response below only after that table exists.

We added a reproducibility manifest describing exact model identifiers, prompt
templates, split files, and saved raw outputs. Because commercial API model
snapshots can change, the raw response JSONs and prediction tables are the
archival record of the evaluated outputs. A compact appendix table should be
included with the final response package.

### R3-12: Figure readability

Status: `answered`

Draft response:

We redesigned the figures while preserving the original manuscript structure.
Figure 1 remains the study-design schematic, with terminology revised to avoid
overclaiming LLM-PP superiority. Figure 2 remains the cross-domain
model-performance comparison, regenerated from one cleaned canonical
PP 70/30 source and with matched LR/RF `Demographics + History + Message Embedding`
reference lines added. Captions and Results text should state that missing
split-specific methods were omitted rather than backfilled from another split.
Figure 3 remains the
predicted-score distribution figure, with enlarged panels and clearer
class-distribution labeling. Figure 4 remains the top-K message-selection
figure, now with uncertainty for the LLM curves and supervised RF included in
the supporting selection analysis. Dense all-method grids and strict
shared-row diagnostics are retained as appendix/reviewer-supporting figures.

Evidence:

- Main Figure 1-4 files:
  [../figures/llm-message-paper-figure1.svg](../figures/llm-message-paper-figure1.svg),
  [../figures/bars_all_methods_accuracy.png](../figures/bars_all_methods_accuracy.png),
  [../figures/figure3_score_distributions_content.png](../figures/figure3_score_distributions_content.png),
  [../figures/llm_selection_quality.png](../figures/llm_selection_quality.png)
- Reviewer-supporting strict diagnostics:
  stale generic/personalized redesign grids were deleted so they are not
  mistaken for replacement main figures. Current supporting diagnostics are
  [figures/progress_summary/ai_vs_ml_strict_snapshot.png](figures/progress_summary/ai_vs_ml_strict_snapshot.png),
  [figures/progress_summary/ordinal_qwk_spearman_snapshot.png](figures/progress_summary/ordinal_qwk_spearman_snapshot.png),
  [figures/progress_summary/confusion_matrices_k7.png](figures/progress_summary/confusion_matrices_k7.png)

### R3-12c / R4 minor 3: Figure 4 uncertainty bands and oracle trend

Status: `answered`

Draft response:

We refreshed the original top-K selection figure rather than replacing it.
Figure 4 now displays bootstrap variability for random, human oracle, and LLM
curves. In addition, a strict supporting method-level selection analysis
gives supervised learning the same message-selection role as the LLM methods:
supervised RF predicts a numeric PME rating for each candidate message,
messages are ranked by the predicted score, and the top-K selected messages
are evaluated by their observed human ratings. This benchmark includes
supervised RF and LLM-PP on the same held-out rows and
reports gain over random for `K=5,10,25`. The gain-over-random figure now
uses shaded 95% confidence bands computed from message-level standard errors
(SD divided by sqrt(n)). The
oracle trend declines as K increases because the oracle must include
progressively lower-rated messages as the selected set widens.

Supervised message-selection definition:

For the supervised-learning message-selection benchmark, the trained RF model
is used as a scoring rule. For each domain, candidate messages are assigned the
model-predicted numeric PME rating on held-out rows; predictions are averaged
at the message level; messages are ranked by this predicted score; and the top
K messages are selected. The evaluation metric is the mean observed human
rating of the selected messages, reported as gain over random selection with
the human-oracle upper bound. Shaded bands for gain over random are 95%
normal-approximation confidence intervals based on message-level standard
errors (SD divided by sqrt(n)). The same held-out rows and message pool are
used for supervised RF and LLM-PP. To avoid domain-wise feature-set
cherry-picking, all domains use the fixed `Demo+History+Embedding` supervised
feature block.

Method-level message-selection gains at `k_train=7`. RF selects best for Content
(message quality); LLM-PP selects far better for Coping and Quitting
(helpfulness), the deployment-relevant domains:

| Domain | Feature set | Method | K=5 | K=10 |
|---|---|---|---:|---:|
| Content | Demographics + History + Message Embedding | Supervised RF | 0.389 | 0.150 |
| Content | Demographics + History + Message Embedding | LLM-PP | 0.184 | 0.204 |
| Coping | Demographics + History + Message Embedding | Supervised RF | 0.245 | 0.075 |
| Coping | Demographics + History + Message Embedding | LLM-PP | 0.570 | 0.466 |
| Quitting | Demographics + History + Message Embedding | Supervised RF | 0.123 | 0.059 |
| Quitting | Demographics + History + Message Embedding | LLM-PP | 0.540 | 0.521 |

Evidence:

- Main Figure 4:
  [../figures/llm_selection_quality.csv](../figures/llm_selection_quality.csv),
  [../figures/llm_selection_quality.png](../figures/llm_selection_quality.png)
- Strict supporting method-level figure:
  [figures/progress_summary/message_selection_gain.png](figures/progress_summary/message_selection_gain.png)
- Method-level source:
  [figures/progress_summary/message_selection_methods_k7.csv](figures/progress_summary/message_selection_methods_k7.csv)

### R3-13: Recommender-system baselines

Status: `missing`

Current state:

The current revision adds stronger supervised and history-summary baselines,
but not full collaborative-filtering or neural recommender-system baselines.
The response should not imply these were implemented. It can state that simple
history/participant-mean signals were added and that matrix-factorization or
neural CF baselines are future work.

## Reviewer 4

### R4-1: Rating preferences as confound

Status: `partial`

Use the AE-3 response. Important correction: do not use the old claim that
LLM-PP Spearman is high while supervised baselines collapse to zero. The
latest strict result is more modest and tie-sensitive.

### R4-2: Supervised baseline using response history

Status: `answered`

Draft response:

We added a supervised RF baseline using the demographic feature block (which, in
a within-participant split, calibrates to each participant) and compared it with
LLM-PP on identical shared held-out rows. This was a central reason for revising
the paper's claim. At `k_train=7`, supervised RF has the highest mean accuracy
(0.474), macro-F1 (0.403), and QWK (0.527); LLM-PP is competitive behind it
(0.462 / 0.370 / 0.488). The revised manuscript frames the supervised baseline
as a strong participant-calibrated benchmark, not a weak foil, and as the
clearest observed personalization signal in the current analysis — while noting
that this reflects same-participant calibration, not demographic generalization.

Evidence:

- [figures/progress_summary/ai_vs_ml_strict_snapshot.png](figures/progress_summary/ai_vs_ml_strict_snapshot.png)
- [figures/progress_summary/table_accuracy.md](figures/progress_summary/table_accuracy.md)
- [figures/progress_summary/table_qwk.md](figures/progress_summary/table_qwk.md)

### R4-3: Why RF rather than LR in the hybrid setting?

Status: `answered`

Draft response:

We added a direct LR-vs-RF comparison on the same held-out test split. LR
performs better for Content and Coping, while RF performs better for Quitting.
We therefore avoid a blanket claim that RF is uniformly superior and report
LR/RF as a supervised-baseline sensitivity analysis.

Evidence:

- Content: LR accuracy `0.358`, QWK `0.136`; RF accuracy `0.350`, QWK `0.077`
- Coping: LR accuracy `0.314`, QWK `0.199`; RF accuracy `0.296`, QWK `0.069`
- Quitting: RF accuracy `0.420`, QWK `0.129`; LR accuracy `0.332`, QWK `0.110`
- Artifacts:
  [figures/hybrid_lr_vs_rf_comparison.csv](figures/hybrid_lr_vs_rf_comparison.csv),
  [figures/hybrid_lr_vs_rf.png](figures/hybrid_lr_vs_rf.png)

### R4-4: CBT vs ACT performance

Status: `answered`

Draft response:

We added a CBT-vs-ACT analysis using message-type labels from the metadata and
reran bootstrap intervals with 2,000 resamples. Mean accuracy is slightly
higher for CBT in Content and Coping, and slightly higher for ACT in Quitting;
the strongest individual rows are domain-specific rather than supporting a
universal CBT or ACT advantage.

Evidence:

- Mean accuracy, Content: ACT `0.383` vs CBT `0.409`
- Mean accuracy, Coping: ACT `0.338` vs CBT `0.347`
- Mean accuracy, Quitting: ACT `0.367` vs CBT `0.330`
- Strongest Content row: `Grok-4-Fast Hybrid RF+PP / CBT` = `0.500` accuracy,
  CI `[0.425, 0.575]`, `N=174`
- Strongest Coping row: `Grok-4-Fast PP / CBT` = `0.471` accuracy,
  CI `[0.397, 0.546]`, `N=174`
- Strongest Quitting row: `Grok-4-Fast Hybrid RF+PP / ACT` = `0.523`
  accuracy, CI `[0.443, 0.604]`, `N=149`
- Artifacts:
  [figures/cbt_act_comparison.csv](figures/cbt_act_comparison.csv),
  [figures/cbt_vs_act_performance.png](figures/cbt_vs_act_performance.png)

## Do Not Claim

- Do not claim LLMs broadly outperform supervised ML.
- Do not claim a 10+ percentage-point LLM improvement over supervised RF in
  the main result.
- Do not mix generic zero-/few-shot LLM dt10 results with the strict
  shared-row AI-vs-ML table.
- Do not claim completed z-score, message-only, profile-only, or history-only
  ablations.
- Do not claim empirical latency results.
- Do not claim recommender-system baselines were run.
- Do not claim historical ratings are causally or universally the most
  important element unless a clean history-only/profile-only/message-only
  ablation is added; use "strongest observed personalization signal in this
  benchmark."
