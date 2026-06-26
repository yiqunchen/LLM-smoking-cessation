# Remaining Reviewer-Response Updates

Updated: 2026-05-21

This file is the paste-ready working map for unfinished or stale sections of
the response letter. The main change from the earlier draft is substantive:
the response letter should no longer claim that LLM-PP methods outperform
supervised ML. The newest strict shared-row results support a benchmark story:
historical participant ratings are the strongest observed personalization
signal in the current benchmark, supervised RF is the strongest aggregate
classifier (QWK 0.527), and LLM-PP is a competitive, complementary
persona-conditioned signal that is most useful at low history / cold start and
for coping/quitting message selection.

Do not reuse older sentences claiming 10+ percentage-point LLM gains over
supervised baselines unless they are explicitly labeled as an older LLM-only
prompt-family comparison on a different subset.

## Source Discipline To Use Everywhere

Primary results should come from:

- `figures/progress_summary/clean_results.md`
- `figures/progress_summary/ai_vs_ml_strict_fixed_feature.csv`
- `figures/progress_summary/ai_vs_ml_fixed_feature_mean_by_k.csv`
- `figures/progress_summary/table_accuracy.md`
- `figures/progress_summary/table_macro_f1.md`
- `figures/progress_summary/table_qwk.md`
- `figures/progress_summary/table_spearman_rho.md`

Main strict `k_train=7` result to use in all high-level responses (single
`Demographics` feature block; dt10 within-participant split; same 898 held-out
rows for both methods):

| Endpoint | Feature block | N | Supervised RF | LLM-PP |
|---|---:|---:|---:|---:|
| Accuracy | Demographics | 898 | 0.474 | 0.462 |
| Macro-F1 | Demographics | 898 | 0.403 | 0.370 |
| QWK | Demographics | 898 | 0.527 | 0.488 |
| Within-participant Spearman rho | Demographics | 898 | NA | 0.058 |

Note: RF's within-participant Spearman is undefined (NA) because, being
message-blind, it assigns one value per participant and cannot rank a
participant's messages; LLM-PP's is small but defined (~0.06).

Key prose:

> We agree that the original framing placed too much weight on comparisons
> that did not align supervised and LLM methods on shared held-out rows. We
> therefore rebuilt the main analysis around a strict two-method comparison
> (supervised RF vs persona-conditioned LLM prediction, LLM-PP) and added
> ordinal metrics, confusion matrices, learning curves, and supervised
> message-selection baselines. These analyses revised the central claim: LLM-PP
> did not dominate supervised learning on aggregate PME classification.
> Instead, supervised RF was the strongest aggregate classifier (QWK 0.527),
> with LLM-PP competitive but slightly behind, and the clearest evidence for
> personalization came from the availability of prior participant ratings.
> Additional history improved the supervised baseline most, while LLM-based
> persona scoring remained competitive at low history and complementary for
> message-selection workflows.

## Historical-Ratings Core Message To Implement

Use this as the revised through-line:

> Historical participant ratings are the strongest observed personalization
> signal in this benchmark. More prior ratings improve the supervised baseline
> most, and history-aware supervised RF is difficult to outperform on aggregate
> metrics. LLM-PP should be framed as a competitive, complementary
> persona-conditioned scorer that is most useful at low history / cold start and
> for coping/quitting message selection, not as the main source of aggregate
> predictive advantage.

Do not write "historical ratings are the most important element" as a causal
or universal claim unless a clean history-only/profile-only/message-only
ablation is added. In the current revision, write "strongest observed
personalization signal in this dataset/benchmark."

Implement this message in these places:

1. Title and keywords: add "response history" or "rating history" if the title
   can absorb it cleanly; at minimum add it to keywords.
2. Abstract Objective and Methods: state that the study tests whether prior
   participant ratings can support personalized PME prediction and selection.
3. Abstract Results: lead with the strict supervised RF result (QWK 0.527) and
   the `k_train=1,3,7` history-length improvement, not LLM superiority.
4. Abstract Discussion and Conclusion: say strong supervised models using
   response history captured substantial PME signal; LLM-PP is complementary.
5. Introduction: reframe the gap as whether a small number of prior ratings
   can personalize PME prediction, and whether LLM scores add beyond that
   history-based baseline.
6. Methods split description: define the within-participant history/test split
   as the central personalization design; the held-out labels are never used
   as history.
7. Supervised Methods: make RF with participant history summaries the primary
   comparator rather than a weak baseline.
8. LLM Methods: say LLM-PP receives the same prior-rating history as persona
   context, and is compared directly with the history-calibrated supervised RF
   baseline.
9. Results 3.1: report the strict shared-row supervised RF vs LLM-PP comparison
   without claiming LLM dominance.
10. Results 3.2: make this the central historical-ratings evidence section:
    on the single Demographics block, supervised RF accuracy increased from
    0.420 to 0.448 to 0.474 across `k_train=1,3,7`, and RF QWK from 0.418 to
    0.447 to 0.527; LLM-PP rose in parallel (accuracy 0.417/0.438/0.462, QWK
    0.417/0.457/0.488), tying RF on QWK at k=1 and slightly exceeding it at k=3.
11. Figure 1: include prior ratings/history as the visible personalization
    substrate in the schematic.
12. Figure 2: caption the bars as history-aware supervised RF vs LLM-PP
    comparisons on shared held-out rows.
13. Figure 4 and supporting selection benchmark: define supervised RF message
    selection as a response-history-based ranking comparator.
14. Supplementary Figure 1 / Appendix A3: caption the learning curves as
    history-length sensitivity and evidence that rating history is useful.
15. Discussion first paragraph: make historical ratings, not LLM-PP
    superiority, the first interpretive point.
16. Limitations: acknowledge that the history-size analysis shows an observed
    benchmark pattern, not a causal proof that history is universally the most
    important input.
17. Reviewer-response cross-references: use this message in AE #1, AE #3/R4 #1,
    R2 #1, R3 #9, R3 #12, and R4 #2.

## Figure And Artifact Checklist

Preserve the original written/published figure architecture. The strict
shared-row results should update the captions, Results text, and selected
supporting panels, but they should not replace the original Figure 1-4
sequence with a new progress-summary sequence.

Original main figure slots:

- Figure 1, study design schematic:
  `../figures/llm-message-paper-figure1.svg`,
  `../figures/llm-message-paper-figure1.pdf`
- Figure 2, model-performance comparison across rating domains:
  `../figures/bars_all_methods_accuracy.png`,
  `../figures/bars_all_methods_f1.png`,
  `../figures/bars_all_methods_directional_accuracy.png`,
  `../figures/bars_all_methods_directional_macro_f1.png`,
  `../figures/bars_all_methods_kappa.png`.
  Rebuilt from one cleaned canonical PP 70/30 source; source rows are
  documented in `../figures/bars_all_methods_source_table.csv`,
  `../figures/bars_all_methods_source_audit.csv`, and
  `../figures/bars_all_methods_manifest.md`. Original demographics LR/RF
  baselines are retained and LR/RF `Demographics + History + Message Embedding` baselines are added
  as matched 70/30 reference lines.
- Figure 3, predicted-score distributions:
  `../figures/figure3_score_distributions_content.png`,
  `../figures/figure3_score_distributions_coping.png`,
  `../figures/figure3_score_distributions_quitting.png`
- Figure 4, top-K message-selection quality:
  `../figures/llm_selection_quality.png`,
  `../figures/top_k_agreement_line.png`
- Supporting supervised/LLM selection benchmark:
  `figures/progress_summary/message_selection_gain.png`,
  `figures/progress_summary/message_selection_methods_k7.csv`
- Appendix / Supplementary Figure 1, history-length learning curves:
  `../figures/learning_curve_accuracy.png`,
  `../figures/learning_curve_directional_macro_f1.png`,
  `../figures/learning_curve_kappa.png`,
  `../figures/learning_curve_spearman_rho.png`

Original proof audit:

- Source checked: `revision/amiajnl-2026-019201_Proof_hi.pdf`.
- Found original main figures: Figure 1, Figure 2, Figure 3, and Figure 4.
- Found one original supplementary figure in Appendix A3: Supplementary Figure 1
  for history-length learning curves. The proof caption line incorrectly says
  "Figure 4"; treat it as Supplementary Figure 1 in the revision.
- Did not find any additional original manuscript figure captions beyond these
  five figure slots. Appendix A1 is prompt templates; Appendix A2 is bootstrap
  CI tables, not additional figures.
- Revision-only supporting figures should therefore stay appendix/reviewer
  support unless the manuscript figure plan is intentionally expanded.

Strict shared-row reviewer-supporting artifacts, not replacement main figures:

- `figures/progress_summary/ai_vs_ml_strict_snapshot.png`
- `figures/progress_summary/learning_curves.png`
- `figures/progress_summary/ai_vs_ml_strict_scaling.png`
- `figures/progress_summary/ordinal_qwk_spearman_snapshot.png`
- `figures/progress_summary/confusion_matrices_k7.png`
- `figures/progress_summary/message_selection_gain.png`

Supplemental / reviewer-specific figures:

- Accuracy CIs: `figures/accuracy_confidence_intervals.png`
- Pairwise LLM significance: `figures/pairwise_significance_heatmap.png`
- Leakage / similarity audit: `figures/profile_test_similarity_distribution.png`
- Data partition schematic: `figures/partition_schematic.png`,
  `figures/partition_matrix_schematic.png`
- Stronger baseline comparisons: `figures/all_baselines_comparison.png`,
  `figures/baseline_feature_ablation.png`,
  `figures/history_supervised_best.png`
- LR vs RF: `figures/hybrid_lr_vs_rf.png`
- Cost: `figures/cost_per_participant.png`
- Demographic subgroups: `figures/demographic_subgroup_race.png`,
  `figures/demographic_subgroup_gender.png`
- CBT vs ACT: `figures/cbt_vs_act_performance.png`
- Removed outdated redesign grids:
  the generic/personalized method bar grids were deleted because they conflicted
  with the original Figure 1-4 architecture requested for this revision.

## AE #1 / R3 #2 - Digital-twin terminology

Current state: the old response correctly conceded the terminology issue, but
it should be updated to match the new framing and title.

Recommended action: remove "digital twin" from the title and abstract. Use
"persona-conditioned LLM prediction", "history-augmented prediction", or
"LLM-PP" in main claims. If "digital-twin prompting" remains, define it as a
limited shorthand only.

Draft response:

> We agree that our original wording overstated the digital-twin analogy. We
> have revised the title, abstract, introduction, methods, and figure captions
> to use more precise terminology: "persona-conditioned LLM prediction" and
> "history-augmented prediction." Where we retain the phrase "digital-twin
> prompting," we define it narrowly as shorthand for persona-conditioned,
> history-augmented LLM prediction of PME ratings. We do not claim to model
> longitudinal smoking behavior, physiological dynamics, or causal treatment
> response. This terminology revision also aligns with the updated benchmark
> results, which show complementary LLM-PP value rather than broad
> LLM superiority.

## AE #2 / R1 / R3 #3 - Uncertainty and hypothesis testing

Current state: answered by current artifacts, but older prose used placeholder
figure labels and less conservative correction wording. The saved artifact
includes both Bonferroni and BH columns; the conservative headline is
Bonferroni.

Recommended action: use this response for AE #2, cross-reference it from R1
and R3 #3.

Draft response:

> We agree that the original manuscript needed more explicit uncertainty
> quantification and should not rank LLMs as if all point-estimate differences
> were statistically resolved. We added per-method 95% bootstrap confidence
> intervals for accuracy across model, method, and domain on the cleaned
> canonical PP 70/30 source
> (`accuracy_confidence_intervals.csv`; 2,000 bootstrap resamples per row) and
> paired model-comparison tests across 10 LLM pairs, 3 domains, and 4 metrics
> (`pairwise_significance_tests.csv`; 120 tests). Using the conservative
> Bonferroni flag in the saved artifact, 23/120 comparisons remain significant,
> concentrated in coping (13 tests) and quitting (9 tests), with fewer in
> content (1 test). We revised the Results to emphasize effect estimates,
> uncertainty, and shared-row benchmark comparisons rather than unsupported
> absolute model rankings.

Figures / artifacts:

- `figures/accuracy_confidence_intervals.png`
- `figures/accuracy_confidence_intervals.csv`
- `figures/pairwise_significance_heatmap.png`
- `figures/pairwise_significance_tests.csv`

## AE #4 / R3 #5 - Data partition and label leakage

Current state: ready to paste from `data_partition_leakage_overview.md`.

Recommended action: paste the full four-point response and include the
partition schematic.

Draft response:

> Thank you for the opportunity to clarify the study design. We use a
> within-participant 70/30 split: each participant's 10 ratings are divided
> into 7 history items used as personalization context and 3 held-out items
> used only for evaluation. We added a dedicated leakage audit covering four
> issues: programmatic prompt assembly from explicit holdout IDs, few-shot
> exemplar overlap, within-participant near-duplicate history/test items, and
> expected cross-split message reuse under participant-level splitting. The
> few-shot exemplar audit found 0/2 overlapping exemplar pairs. The
> within-participant audit identified 16/319 near-duplicate or exact-overlap
> items; these were removed from the headline evaluation, changing accuracy/F1
> by no more than 0.005. Cross-split message-text overlap was high by design
> because the same message library was rated by different participants; this is
> not label leakage because no participant's held-out rating is used as a
> prompt input or training target for that participant.

Figures / artifacts:

- `figures/label_leakage_analysis.csv`
- `figures/profile_test_similarity_distribution.png`
- `figures/partition_schematic.png`
- `figures/partition_matrix_schematic.png`

## AE #5 - Sociodemographic table

Current state: still needs manuscript-table insertion if not already done.

Recommended action: add Table 1 with participant demographics and
smoking-related variables. The response can be short. Use the past-tense
response below only after the table is actually inserted.

Draft response:

> We agree this improves interpretability and generalizability assessment. We
> added Table 1 summarizing participant sociodemographic, smoking-related, and
> psychological-flexibility characteristics, including age, sex, race/ethnicity,
> household income, education, time-to-first-cigarette, cigarettes per day,
> motivation to quit, and AAQ-II score. Continuous variables are reported as
> mean/SD and categorical variables as counts/percentages.

## R1 - Confidence intervals on accuracy

Current state: answered by AE #2 artifacts.

Recommended action: cross-reference AE #2 and avoid saying that CIs establish
LLM dominance.

Draft response:

> We agree. As described in our response to AE #2, we added 95% bootstrap
> confidence intervals for accuracy using 2,000 resamples per model-method-
> domain row. We also added paired model-comparison tests and revised the
> Results to avoid overinterpreting overlapping point estimates. The revised
> Results text and figure captions now emphasize the strict shared-row
> supervised RF vs LLM-PP comparison rather than a simple LLM
> ranking, while preserving the original Figure 1-4 structure.

## R1 - Prompt-centric write-up

Current state: response prose still needed.

Draft response:

> We thank the reviewer for highlighting the role of prompting. We added a
> Discussion paragraph noting that between-model differences may reflect
> prompt-model interactions rather than model capability alone. We retain a
> canonical prompt per configuration to keep cross-model comparisons
> interpretable, and we now describe prompt sensitivity as a methodological
> boundary. We also distinguish the strict shared-row benchmark from older
> prompt-family comparisons so that prompt effects are not confused with
> supervised-vs-LLM performance differences.

## R2 #3 / R3 #1 / R4 #2 - Stronger baselines and central claim

Current state: strategically the most important response. The older draft is
now stale because it still describes outdated method names and split-specific runs
as the main comparison. Use the strict two-method result.

Recommended action: replace the old baseline response wholesale.

Draft response:

> We agree that the original baseline suite was not strong enough to support a
> broad LLM-superiority claim. In response, we rebuilt the main analysis around
> a strict shared-row benchmark comparing supervised RF and persona-conditioned
> LLM prediction (LLM-PP) on the same held-out rows within a single fixed
> feature block (Demographics). This materially changed the interpretation. At
> `k_train=7` (N = 898), supervised RF was the strongest aggregate classifier:
> mean accuracy across Content, Coping, and Quitting was 0.474 for supervised RF
> versus 0.462 for LLM-PP, macro-F1 0.403 versus 0.370, and QWK 0.527 versus
> 0.488. LLM-PP was competitive but did not beat RF on any aggregate metric at
> the headline block; its strengths emerged at low history and in
> coping/quitting message selection (see below).
>
> We therefore revised the abstract, Results, and Discussion. The manuscript no
> longer claims that persona-conditioned LLMs uniformly outperform supervised
> baselines. Instead, it shows that prior participant ratings / response
> history are the strongest observed personalization signal, strong supervised
> RF captures substantial aggregate PME signal, and LLM-PP provides
> complementary value at low history / cold start and for practical
> coping/quitting message selection. We also added learning curves across
> `k_train=1,3,7`, confusion matrices, QWK/Spearman diagnostics, and
> message-selection gain over random with supervised RF included as a comparator.
>
> We caution that, because dt10 is a within-participant split (all 301
> participants appear in both train and test), the supervised RF is best read as
> a strong participant-calibrated reference rather than as evidence that
> demographics predict PME: a trivial "predict each participant's most-frequent
> prior rating" rule reproduces RF accuracy (Content 0.53, Coping 0.45,
> Quitting 0.47), and on a strict unseen-participant split RF falls to ~0.35.

Figures / artifacts:

- Main Figure 2 remains the original model-performance comparison:
  `../figures/bars_all_methods_accuracy.png`,
  `../figures/bars_all_methods_f1.png`,
  `../figures/bars_all_methods_directional_accuracy.png`,
  `../figures/bars_all_methods_directional_macro_f1.png`,
  `../figures/bars_all_methods_kappa.png`.
  These panels now use only the cleaned canonical PP 70/30 source;
  missing split-specific methods are omitted rather than backfilled.
- Main Figure 4 remains the original top-K selection analysis:
  `../figures/llm_selection_quality.png`
- Strict shared-row supporting artifacts:
  `figures/progress_summary/ai_vs_ml_strict_snapshot.png`,
  `figures/progress_summary/learning_curves.png`,
  `figures/progress_summary/ordinal_qwk_spearman_snapshot.png`,
  `figures/progress_summary/confusion_matrices_k7.png`,
  `figures/progress_summary/message_selection_gain.png`

## R2 #1 - Sensitivity to number of history ratings

Current state: now answered by the original Appendix learning-curve slot,
with the progress-summary learning curves retained as a strict shared-row
audit.

Draft response:

> We added a history-length sensitivity analysis for `k_train=1,3,7` using the
> shared-row ensemble files on the single Demographics feature block. Across
> these values, additional participant history improves the supervised baseline.
> Mean accuracy for supervised RF increases from 0.420 to 0.448 to 0.474, and
> QWK from 0.418 to 0.447 to 0.527. LLM-PP rises in parallel (accuracy
> 0.417/0.438/0.462, QWK 0.417/0.457/0.488); it ties RF on QWK at k=1 and
> slightly exceeds it at k=3 (0.457 vs 0.447), with RF pulling ahead at k=7. We
> report these as learning curves and note that each `k_train` value uses its
> own shared test-row intersection, so the curves should be read as within-k
> method comparisons and scaling evidence rather than as a single fixed test
> cohort. These results make response history the strongest observed
> personalization signal in the revised benchmark, while not proving a universal
> causal ordering of input importance.

Figure:

- Main appendix figure family:
  `../figures/learning_curve_accuracy.png`,
  `../figures/learning_curve_directional_macro_f1.png`,
  `../figures/learning_curve_kappa.png`,
  `../figures/learning_curve_spearman_rho.png`
- Strict shared-row audit: `figures/progress_summary/learning_curves.png`

## R2 #4 - Demographic subgroup analysis

Current state: answered.

Draft response:

> We added subgroup analyses across race/ethnicity, gender, and age using the
> PP 70/30 source and explicitly flag small cells. The saved table
> contains 120 subgroup rows, with 15 flagged as `n < 20`, so subgroup
> differences are interpreted cautiously rather than as definitive fairness
> claims. We added subgroup figures and a paragraph in the Discussion noting
> that larger samples are needed for stable subgroup-specific conclusions.

Figures / artifacts:

- `figures/demographic_subgroup_results.csv`
- `figures/demographic_subgroup_race.png`
- `figures/demographic_subgroup_gender.png`

## R3 #4b/c - Generation parameters, stochasticity, and prompt sensitivity

Current state: partially answered. Do not claim a full repeated-run study
unless it is in a saved artifact.

Draft response:

> We clarified the API model identifiers and prompt templates and added a
> prompt-sensitivity limitation. All primary comparisons used a canonical
> prompt per configuration so that method comparisons remained interpretable.
> The repository contains observed prompt-variant outputs for full-feature,
> selected-feature, full-plus-feedback, and CBT/ACT variants, but we do not
> claim a full factorial prompt-sensitivity or repeated-run variance study.
> We now state this limitation directly and identify systematic prompt
> sensitivity and stochastic repeated-run evaluation as future work.

## R3 #6 - Class imbalance and QWK

Current state: answered; update to latest strict result.

Draft response:

> We agree that accuracy alone is insufficient for imbalanced ordinal ratings.
> We added class-distribution histograms, macro-F1, QWK, and row-normalized
> confusion matrices. Held-out ratings are skewed toward classes 4-5 (Content
> 68.2%, Coping 61.6%, Quitting 61.7% on the dt10 held-out test
> set). In the latest strict `k_train=7`
> comparison, supervised RF achieved the highest mean QWK across domains (0.527),
> with LLM-PP competitive but slightly lower (0.488). We therefore use QWK and
> confusion matrices as primary ordinal diagnostics and treat within-participant
> Spearman as a secondary, tie-sensitive rank diagnostic.

Figures / artifacts:

- `figures/class_distribution_by_domain.png`
- `figures/progress_summary/ordinal_qwk_spearman_snapshot.png`
- `figures/progress_summary/confusion_matrices_k7.png`
- `figures/qwk_results.csv`

## R3 #7 - Cost and latency

Current state: cost answered; empirical latency not measured.

Draft response:

> We added a token-based cost analysis for all evaluated model/method
> combinations. In the saved artifact, the cheapest per-participant estimate is
> GPT-4o-mini zero-shot select at $0.002135, while the most expensive evaluated
> configurations are GPT-5 and Gemini-2.5-Pro few-shot all at $0.061513 per
> participant. GPT-5/Gemini PP prompting is estimated at $0.030612
> per participant. We did not
> measure empirical latency in this study; because the intended use case is
> offline message pretesting rather than real-time intervention delivery, we
> now state latency as a deployment-specific limitation.

Figure / artifact:

- `figures/cost_per_participant.png`
- `figures/cost_latency_analysis.csv`

## R3 #8 - Image handling

Current state: now technically clear from the scripts, but the manuscript and
response letter still need an explicit sentence.

Draft response:

> We clarified image handling in the Methods. Although the prompt templates
> retain legacy language asking the model to describe any provided image, the
> manuscript experiments were run with `--mode text-only`; `main_eval.py` only
> attaches image payloads when `--mode vision` is selected. Thus, the primary
> benchmark evaluated text/profile/history inputs rather than multimodal image
> interpretation. We now state this explicitly and exclude the design domain
> from the primary results because it depends most directly on visual
> presentation.

Evidence: `../run_full_manuscript_pipeline.sh`,
`../run_generic_dt7030_pipeline.sh`, and
`../run_generic_dt7030_variance_pipeline.sh` call
`../analysis-script/main_eval.py` with `--mode text-only`; image payloads are
appended only inside the `mode == 'vision'` branch.

## R3 #9 / R4 #1 - Rating tendency, z-scoring, and ablations

Current state: partial. The latest strict analysis gives better diagnostics,
but not a true z-score rerun and not clean message-only/profile-only/history-
only ablations.

Draft response:

> We agree that participant-level rating tendency is an important alternative
> explanation. We addressed it in three ways. First, the main revised benchmark
> includes supervised RF models using demographic and history features, so
> rating-history signal is no longer treated as an LLM-only advantage. Second,
> we added QWK and confusion matrices to distinguish ordinal agreement from
> modal-class accuracy. Third, we added within-participant Spearman as a
> secondary rank diagnostic, which is less sensitive to simple within-person
> rating-scale shifts but remains tie-sensitive with only a small number of
> held-out messages per participant. These analyses reduce but do not fully
> eliminate the rating-style concern. We also note that, because dt10 is a
> within-participant split, supervised RF on demographics functions largely as a
> participant-calibrated "predict this person's typical rating" baseline rather
> than as demographic prediction. We acknowledge that a direct z-scored-rating
> rerun and clean message-only/profile-only/history-only ablations remain
> important future work.

Do not claim a completed history-only ablation unless a real observed artifact
exists. The current observed-only ablation artifact does not support that
claim.

Artifacts:

- `figures/ablation_status.md`
- `figures/ablation_results.csv`
- `figures/progress_summary/ordinal_qwk_spearman_snapshot.png`

## R3 #11 - Reproducibility

Current state: partial. Model identifiers and prompts exist in code/docs, but
there is not yet a consolidated reproducibility table.

Recommended action: add the compact appendix table first. Use the response
below only after that table is inserted.

> We added a reproducibility note listing the exact model identifiers used in
> the evaluation code, the prompt templates, the split files, and the saved raw
> outputs. We also acknowledge the limitation that commercial API model
> snapshots can change over time; for that reason, we retain raw response JSONs
> and generated prediction tables as the reproducible record of model outputs.

Action: before final submission, add a compact Appendix table with model ID,
provider/endpoint, call date range, prompt file, split file, and generation
parameter policy.

## R3 #12 - Figure readability

Current state: answered, but preserve the original figure numbering and
describe the redesign as an update to the written/published Figure 1-4 slots,
not a replacement by the progress-summary plots.

Draft response:

> We substantially revised the original figure suite while preserving the
> manuscript's figure structure. Figure 1 remains the study-design schematic,
> but its wording now avoids overclaiming "digital twin" superiority. Figure 2
> remains the cross-domain model-performance comparison, with captions and
> Results text updated to reflect the strict supervised RF vs LLM-PP benchmark
> rather than an LLM-superiority claim. Figure 3 remains the predicted-score
> distribution figure, with enlarged panels and clearer class-distribution
> labeling. Figure 4 remains the top-K message-selection figure, now with
> uncertainty shown for the LLM curves and supervised RF included in the
> supporting selection analysis. Dense full-grid and strict shared-row
> diagnostics are retained as appendix/reviewer-supporting figures.

Main figures:

- Figure 1: `../figures/llm-message-paper-figure1.svg`
- Figure 2: `../figures/bars_all_methods_accuracy.png`,
  `../figures/bars_all_methods_f1.png`,
  `../figures/bars_all_methods_directional_accuracy.png`,
  `../figures/bars_all_methods_directional_macro_f1.png`,
  `../figures/bars_all_methods_kappa.png`
  with source audit `../figures/bars_all_methods_source_audit.csv`
- Figure 3: `../figures/figure3_score_distributions_content.png`,
  `../figures/figure3_score_distributions_coping.png`,
  `../figures/figure3_score_distributions_quitting.png`
- Figure 4: `../figures/llm_selection_quality.png`,
  `../figures/top_k_agreement_line.png`

Deleted stale redesign grids:

- The generic/personalized method bar grids were removed from
  `revision/figures` so they are not mistaken for replacement main figures.
- The score-distribution panels are kept only in the original manuscript
  location: `../figures/figure3_score_distributions_*.png`.

## R3 #12c / R4 minor - Figure 4 uncertainty and oracle trend

Current state: answered by two complementary artifacts. Keep the original
top-K selection figure as Figure 4; use the strict method-level
message-selection figure as supporting reviewer evidence rather than replacing
Figure 4.

Draft response:

> We revised the original top-K message-selection figure to address the
> reviewer's concern directly. Figure 4 still shows top-K selection quality,
> but now displays uncertainty for the LLM-based curves rather than only for
> the random baseline. We also added a strict shared-row supporting analysis
> that gives supervised learning the same message-selection role as the LLM
> methods: supervised RF predicts a numeric PME rating for each candidate
> message, messages are ranked by the predicted score, and the top-K selected
> messages are evaluated by their observed human ratings. This benchmark
> includes supervised RF alongside LLM-PP on the
> same held-out rows and message pools. The gain-over-random plot now displays
> shaded 95% confidence bands computed from message-level standard errors
> (SD divided by sqrt(n)). The
> human-oracle curve decreases as K increases because the oracle must include
> progressively lower-rated messages as the selected set becomes larger.

Supervised message-selection definition to add to Methods / caption:

> For the supervised-learning message-selection benchmark, we used the trained
> supervised RF model as a scoring rule. For each domain, candidate messages
> were assigned the model-predicted numeric PME rating on the held-out rows,
> predictions were averaged at the message level, and messages were ranked by
> this predicted score. We then selected the top K messages and reported the
> mean observed human rating of those selected messages, together with gain
> over random selection and the human-oracle upper bound. Shaded bands for
> gain over random are 95% normal-approximation confidence intervals based on
> message-level standard errors (SD divided by sqrt(n)). The same held-out
> rows and message pool were used for supervised RF and LLM-PP. To avoid
> domain-wise feature-set cherry-picking, all domains use the fixed
> `Demographics + History + Message Embedding` supervised feature block.

Figures:

- Main Figure 4: `../figures/llm_selection_quality.png`
- Supporting selection benchmark:
  `figures/progress_summary/message_selection_gain.png`

## R3 #13 - Related literature and recommender-system baselines

Current state: related-literature prose mostly exists; recommender-system
baselines are still not implemented as revision artifacts.

Draft tightening:

> We expanded the related-work discussion to distinguish supervised prediction,
> recommender-system approaches, and persona-conditioned LLM scoring. The
> revised supervised baselines capture some simple recommender-style signals,
> including participant mean/history summaries, but we agree that full
> collaborative-filtering and neural recommender baselines would be valuable
> next benchmarks. We now state this explicitly as future work rather than
> implying that the current supervised baselines exhaust the recommender-system
> design space.

## R4 #1 - Rating preferences as a confound

Current state: update with the same cautious rating-tendency language used for
R3 #9. Remove the old high-Spearman LLM-PP claim.

Draft response:

> We agree that individual rating tendency is a real source of signal and a
> possible alternative explanation. We now treat this directly rather than
> dismissing it. In the revised benchmark, supervised RF models using
> demographics and history summaries are included as primary comparators, so
> participant-level rating tendency is part of the benchmark rather than an
> LLM-only advantage. We also added QWK, confusion matrices, and
> within-participant Spearman to separate aggregate modal-class accuracy,
> ordinal agreement, and within-person ranking. The rank signal is modest and
> tie-sensitive, so we do not claim that Spearman alone proves strong
> personalization. We also note that supervised RF is message-blind and so its
> within-participant Spearman is undefined, whereas LLM-PP can rank a
> participant's messages (small but defined Spearman, ~0.06). The revised
> interpretation is that strong supervised models capture much of the aggregate
> PME signal, while LLM-PP can provide complementary ordinal and selection
> information, especially at low history and for coping/quitting message
> selection.

## R4 #2 - Supervised baseline using response history

Current state: answered by strict RF / history-feature analyses, but the old
draft's specific Quitting comparison is stale.

Draft response:

> We thank the reviewer for this suggestion. We added supervised RF baselines
> that use participant history summaries and then compared those baselines with
> LLM-PP on shared held-out rows. This was a central reason
> for revising the paper's claim. At `k_train=7` (N = 898, single Demographics
> block), supervised RF was the strongest aggregate classifier, with mean
> accuracy 0.474, macro-F1 0.403, and QWK 0.527, versus LLM-PP at 0.462, 0.370,
> and 0.488. Thus, response-history supervised baselines are not weak foils; they
> are strong benchmarks and provide the clearest observed personalization
> signal in the revised analysis. We revised the manuscript accordingly and now
> frame the LLM-PP contribution as complementary rather than uniformly superior.
> We also caution that, because all 301 participants appear in both train and
> test, this supervised RF is a strong participant-calibrated reference rather
> than evidence that demographics predict PME.

## R4 #3 - RF vs LR for hybrid baseline

Current state: answered by direct LR-vs-RF comparison.

Draft response:

> We agree that the choice of RF rather than LR should be justified
> empirically. We added a direct LR-vs-RF comparison using the same held-out
> test split. LR performed better for Content (accuracy 0.358, QWK 0.136 vs RF
> accuracy 0.350, QWK 0.077) and Coping (accuracy 0.314, QWK 0.199 vs RF
> accuracy 0.296, QWK 0.069), while RF performed better for Quitting (accuracy
> 0.420, QWK 0.129 vs LR accuracy 0.332, QWK 0.110). We therefore avoid a
> blanket claim that RF is uniformly stronger than LR and report the LR/RF
> comparison as a supervised-baseline sensitivity analysis.

Figures / artifacts:

- `figures/hybrid_lr_vs_rf.png`
- `figures/hybrid_lr_vs_rf_comparison.csv`

## R4 #4 - CBT vs ACT

Current state: answered.

Draft response:

> We added a CBT-vs-ACT analysis using the message-type labels available in the
> metadata and reran bootstrap intervals with 2,000 resamples. Mean accuracy
> was slightly higher for CBT in Content (0.409 vs 0.383 for ACT) and Coping
> (0.347 vs 0.338), while ACT was slightly higher for Quitting (0.367 vs
> 0.330). The strongest individual rows were domain-specific rather than
> showing a universal CBT or ACT advantage, so the revised manuscript presents
> this as exploratory subgroup evidence.

Figures / artifacts:

- `figures/cbt_vs_act_performance.png`
- `figures/cbt_act_comparison.csv`

## Priority Before Submission

1. Replace all broad LLM-superiority and 10+ point improvement claims.
2. Make historical participant ratings / response history the explicit
   through-line for personalized PME prediction.
3. Use the strict supervised RF vs LLM-PP snapshot (single Demographics block,
   dt10, N = 898) as the main response to baseline concerns.
4. Preserve the original Figure 1-4 structure: study schematic, model
   performance comparison, score distributions, and top-K selection quality.
5. Use the strict snapshot, learning curves, ordinal/confusion diagnostics,
   and RF-including message-selection gain as supporting reviewer/appendix
   artifacts, not replacement main figures.
6. Add the reproducibility table and explicit image-handling statement before
   final `.docx` assembly.
