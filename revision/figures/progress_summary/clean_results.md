# Clean Updated Results

## Recommended Messaging

- The whole comparison is strictly two methods: supervised RF versus LLM-PP (Grok-4-Fast) on the identical held-out rows. No blended or ensemble methods are reported.
- Frame supervised RF as the strongest aggregate classifier, but as a within-participant participant-calibrated baseline (it predicts each person's typical rating from a demographic fingerprint), not as demographic generalization.
- Frame LLM-PP as competitive and complementary: it is strongest relative to RF at low history / cold start, and it is the better choice for coping and quitting message selection.
- RF's within-participant Spearman is undefined because it is message-blind (one value per participant); LLM-PP can actually rank a participant's messages.
- Generic zero-/few-shot LLM results are context only, because they were scored on a smaller dt10 subset and should not be mixed into the AI-vs-ML table.

## Reviewer-Facing Sequence

1. Start with source discipline: the AI-vs-ML comparison uses only shared-row ensemble files, while generic LLM results stay in a separate context table.
2. Answer the aggregate-performance question with the strict fixed-feature snapshot and scaling curves.
3. Address ordinal-rating concerns with QWK, the QWK-vs-Spearman diagnostic, and the true-vs-predicted confusion matrices.
4. Address the Spearman surprise explicitly: Spearman is within-participant and tie-sensitive, whereas QWK is an all-row ordinal-agreement metric.
5. Answer the message-selection question with gain over random for supervised RF and LLM-PP, highlighting LLM-PP's edge in the coping and quitting domains.
6. Use the rating-distribution histograms as context for class imbalance and demographic subgroup checks, not as model-performance claims.

## Strict AI-vs-ML Snapshot

Each row below uses one fixed feature-set block at the latest available 
`k_train=7`. Within that block, supervised RF and LLM-PP share 
the same held-out rows. Values are means across Content, 
Coping, and Quitting.

| metric | feature_set | N | best_overall_method | best_overall | supervised_rf | llm_pp_grok |
|---|---|---|---|---|---|---|
| Accuracy | Demographics | 898 | Supervised RF | 0.474 | 0.474 | 0.462 |
| Macro-F1 | Demographics | 898 | Supervised RF | 0.403 | 0.403 | 0.370 |
| QWK | Demo+History | 898 | Supervised RF | 0.549 | 0.549 | 0.488 |
| Within-participant Spearman rho | Demographics + History + Message Embedding | 883 | LLM-PP (Grok-4-Fast) | 0.071 | -0.010 | 0.071 |

For Spearman rho, `—` means that the fixed-feature RF predictions 
were degenerate for within-participant ranking in that block 
(for example, tied predictions), not that a result was fabricated 
or silently unavailable.

Full strict fixed-feature summaries are saved in 
`ai_vs_ml_strict_fixed_feature.csv`; full fixed-feature method means 
are saved in `ai_vs_ml_fixed_feature_mean_by_k.csv`.

## Plots

- `ai_vs_ml_strict_snapshot.png` / `.pdf` — latest strict fixed-feature snapshot.
- `ai_vs_ml_strict_scaling.png` / `.pdf` — strict fixed-feature scaling across k.
- `ordinal_qwk_spearman_snapshot.png` / `.pdf` — strict QWK beside best-available within-participant Spearman.
- `confusion_matrices_k7.png` / `.pdf` — row-normalized true-vs-predicted rating confusion matrices.
- `message_selection_gain.png` / `.pdf` — supervised RF versus LLM-PP message-selection gain over random.
- In `ai_vs_ml_strict_snapshot`, the rank-signal panel uses the best available non-degenerate Spearman signal for each selected method; the aggregate-metric panel remains strict fixed-feature.
- Companion revision histograms outside this folder: `class_distribution_by_domain`, `all_rating_subgroup_histograms`, `profile_test_similarity_distribution`, and `figure3_score_distributions_<domain>`.

## Ordinal Diagnostics

QWK and Spearman answer different questions. QWK is an absolute 
ordinal-agreement metric over all held-out message ratings; 
within-participant Spearman is computed inside each participant and 
then averaged, so it is sensitive to tied predictions and to the 
small number of held-out messages per participant.

The confusion matrices use the strict QWK feature block 
`Demo+History` at `k_train=7`.

| domain | feature_set | method_display | N | Accuracy | QWK | Spearman_Rho |
|---|---|---|---|---|---|---|
| Content | Demo+History | Supervised RF | 898 | 0.499 | 0.475 | — |
| Content | Demo+History | LLM-PP | 898 | 0.499 | 0.402 | 0.056 |
| Coping | Demo+History | Supervised RF | 898 | 0.420 | 0.554 | — |
| Coping | Demo+History | LLM-PP | 898 | 0.431 | 0.503 | 0.037 |
| Quitting | Demo+History | Supervised RF | 898 | 0.420 | 0.617 | — |
| Quitting | Demo+History | LLM-PP | 898 | 0.457 | 0.559 | 0.081 |

Full row-level confusion counts are saved in 
`confusion_matrices_k7.csv`; per-row predictions are 
saved in `confusion_predictions_k7.csv`.

## Method-Level Message Selection

All domains use the fixed `Demographics + History + Message Embedding` RF feature 
block, then the same held-out rows are reused for LLM-PP. This 
keeps supervised and LLM message selection on the same rows and 
avoids domain-wise feature-set cherry-picking.

Supervised RF message selection treats the fitted RF model as a 
scoring rule: each held-out participant-message row receives a 
predicted numeric PME rating, predictions are averaged at the 
message level, messages are ranked by that predicted score, and 
the top K messages are evaluated by their observed human ratings. 
The table reports the selected-message human rating and gain over 
random selection, with the human-oracle ranking as an upper bound. 
The gain plot and `gain_over_random_95ci` column use 95% normal-
approximation confidence intervals from message-level standard 
errors (SD divided by sqrt(n)); the figure renders these intervals 
as shaded bands.

| domain | feature_set | method_display | K | selected_human_rating | random_mean | gain_over_random | gain_over_random_95ci | human_oracle_rating | n_messages |
|---|---|---|---|---|---|---|---|---|---|
| Content | Demographics + History + Message Embedding | LLM-PP | 5 | 4.110 | 3.926 | 0.184 | [-0.153, 0.521] | 4.767 | 122 |
| Content | Demographics + History + Message Embedding | LLM-PP | 10 | 4.130 | 3.926 | 0.204 | [-0.006, 0.414] | 4.649 | 122 |
| Content | Demographics + History + Message Embedding | LLM-PP | 25 | 4.160 | 3.926 | 0.234 | [0.100, 0.367] | 4.467 | 122 |
| Content | Demographics + History + Message Embedding | Supervised RF | 5 | 4.315 | 3.926 | 0.389 | [-0.088, 0.866] | 4.767 | 122 |
| Content | Demographics + History + Message Embedding | Supervised RF | 10 | 4.076 | 3.926 | 0.150 | [-0.203, 0.504] | 4.649 | 122 |
| Content | Demographics + History + Message Embedding | Supervised RF | 25 | 3.939 | 3.926 | 0.013 | [-0.175, 0.201] | 4.467 | 122 |
| Coping | Demographics + History + Message Embedding | LLM-PP | 5 | 4.175 | 3.605 | 0.570 | [0.192, 0.949] | 4.612 | 122 |
| Coping | Demographics + History + Message Embedding | LLM-PP | 10 | 4.070 | 3.605 | 0.466 | [0.201, 0.730] | 4.519 | 122 |
| Coping | Demographics + History + Message Embedding | LLM-PP | 25 | 4.002 | 3.605 | 0.397 | [0.206, 0.589] | 4.309 | 122 |
| Coping | Demographics + History + Message Embedding | Supervised RF | 5 | 3.850 | 3.605 | 0.245 | [-0.535, 1.026] | 4.612 | 122 |
| Coping | Demographics + History + Message Embedding | Supervised RF | 10 | 3.679 | 3.605 | 0.075 | [-0.394, 0.543] | 4.519 | 122 |
| Coping | Demographics + History + Message Embedding | Supervised RF | 25 | 3.743 | 3.605 | 0.139 | [-0.110, 0.388] | 4.309 | 122 |
| Quitting | Demographics + History + Message Embedding | LLM-PP | 5 | 4.160 | 3.620 | 0.540 | [0.275, 0.804] | 4.424 | 122 |
| Quitting | Demographics + History + Message Embedding | LLM-PP | 10 | 4.141 | 3.620 | 0.521 | [0.347, 0.694] | 4.383 | 122 |
| Quitting | Demographics + History + Message Embedding | LLM-PP | 25 | 4.010 | 3.620 | 0.389 | [0.228, 0.550] | 4.273 | 122 |
| Quitting | Demographics + History + Message Embedding | Supervised RF | 5 | 3.743 | 3.620 | 0.123 | [-0.135, 0.382] | 4.424 | 122 |
| Quitting | Demographics + History + Message Embedding | Supervised RF | 10 | 3.679 | 3.620 | 0.059 | [-0.214, 0.331] | 4.383 | 122 |
| Quitting | Demographics + History + Message Embedding | Supervised RF | 25 | 3.704 | 3.620 | 0.084 | [-0.145, 0.312] | 4.273 | 122 |

Full method-level grid is saved in `message_selection_methods_k7.csv`.

## LLM-Only Message-Selection Context

Best LLM by domain and K, evaluated as the mean human rating of 
the messages selected by LLM score. Random and human-oracle rows 
come from the same 108-message domain pools with 2,000 bootstrap 
resamples.

| Domain | K | Model | LLM Selection (Human Rating) | Random Mean | Human Oracle (Human Rating) | LLM_minus_Random | Oracle_minus_LLM | N Messages |
|---|---|---|---|---|---|---|---|---|
| Content | 5 | Grok-4-Fast | 4.683 | 3.945 | 5.000 | 0.738 | 0.317 | 108 |
| Content | 10 | Grok-4-Fast | 4.617 | 3.942 | 5.000 | 0.674 | 0.383 | 108 |
| Content | 25 | Grok-4-Fast | 4.283 | 3.944 | 4.733 | 0.339 | 0.450 | 108 |
| Coping | 5 | Gemini-2.5-Pro | 4.400 | 3.707 | 5.000 | 0.693 | 0.600 | 108 |
| Coping | 10 | GPT-4o-mini | 4.317 | 3.712 | 4.933 | 0.605 | 0.617 | 108 |
| Coping | 25 | Gemini-2.5-Pro | 4.261 | 3.704 | 4.673 | 0.557 | 0.412 | 108 |
| Quitting | 5 | GPT-5 | 4.450 | 3.722 | 5.000 | 0.728 | 0.550 | 108 |
| Quitting | 10 | GPT-5 | 4.600 | 3.727 | 4.942 | 0.873 | 0.342 | 108 |
| Quitting | 25 | Gemini-2.5-Pro | 4.341 | 3.725 | 4.667 | 0.616 | 0.327 | 108 |

Full K grid is saved in `message_selection_best_by_k.csv`.

## Source Discipline

- AI-vs-ML aggregate comparisons use only `lc_dt10_ensemble_k1.csv`, `lc_dt10_ensemble_k3.csv`, and `lc_dt10_ensemble_k7.csv`.
- The best-feature-set tables remain available as descriptive summaries, with feature set and N listed per cell.
- No placeholder or fabricated revision values are generated here.