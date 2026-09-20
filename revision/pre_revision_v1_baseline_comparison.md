# Pre-Revision/V1 Baseline Comparison

Date checked: 2026-06-01

## Bottom Line

The corrected Figure 2 supervised baselines are **not directly comparable** to
the pre-revision/v1 supervised baseline numbers, because the v1 analyses used
different splits and different feature definitions.

The apparent discrepancy is expected once those differences are separated.

## V1 Publication-Figure Baseline

Source:

- `analysis-script/create_publication_figures.py`
- tracked output: `figures/all_results_with_baselines.csv`

V1 split:

- `train_participant_7030.json`
- `test_participant_7030.json`
- N = 274 per domain

V1 feature extractor was labeled participant/demographic, but it was not true
demographics only. It used:

- age
- gender
- race
- education
- income
- smoking status
- quit motivation
- cigarettes per day

V1 supervised RF results from the tracked table:

| Domain | RF accuracy | RF kappa | RF macro-F1 | N |
|---|---:|---:|---:|---:|
| Content | 0.358 | 0.055 | 0.215 | 274 |
| Coping | 0.303 | 0.012 | 0.182 | 274 |
| Quitting | 0.376 | 0.113 | 0.249 | 274 |

These are much lower than the corrected current Figure 2 baselines because this
was an unseen-participant split, not the digital-twin held-out-message split.

## Older Archive "Participant Only" Result

Source:

- `archive_results/COMPREHENSIVE_RESULTS.md`

Archive split:

- message-based 50/50 split
- includes all four dimensions, including design

Archive feature set:

- `Participant Only`
- described as demographics, smoking history, and psychological measures
- approximately 40 participant features

Archive headline:

| Model | Feature set | Mean accuracy | Mean kappa | Mean Spearman r |
|---|---|---:|---:|---:|
| Random Forest | Participant Only | 0.495 | 0.279 | 0.436 |

This archive result is broadly consistent with the older broad-metadata
"Demographics" runs, but it is **not** demographics only.

## Current Corrected Figure 2

Source:

- `revision/figures/history_supervised_baselines.csv`
- `figures/figure2/bars_all_methods_source_table.csv`

Current split:

- canonical digital-twin 70/30
- duplicate-filtered test rows
- N = 306/307 per domain

Current corrected demographics-only features:

- age
- gender
- race/ethnicity
- Hispanic/Latino status
- sexual orientation
- education
- household income

Current corrected RF demographics-only results:

| Domain | RF accuracy | RF QWK | RF macro-F1 | N |
|---|---:|---:|---:|---:|
| Content | 0.484 | 0.348 | 0.330 | 306 |
| Coping | 0.459 | 0.357 | 0.389 | 307 |
| Quitting | 0.469 | 0.466 | 0.387 | 307 |

Current corrected revision-added LR result:

| Domain | LR demographics + history + embedding accuracy | QWK |
|---|---:|---:|
| Content | 0.484 | 0.325 |
| Coping | 0.433 | 0.333 |
| Quitting | 0.502 | 0.480 |

Current corrected revision-added RF result:

| Domain | RF demographics + history + embedding accuracy | QWK |
|---|---:|---:|
| Content | 0.363 | 0.042 |
| Coping | 0.355 | 0.065 |
| Quitting | 0.349 | 0.073 |

## Why They Differ

1. **Different split.** V1 publication figures used participant 70/30; current
   Figure 2 uses digital-twin 70/30. The digital-twin split is a held-out-message
   setting with participant history available, so it is easier than a strict
   unseen-participant split.

2. **Different feature definition.** V1 did not use true demographics only; it
   included smoking status, quit motivation, and cigarettes per day. The older
   archive `Participant Only` result was broader still, including smoking history
   and psychological measures.

3. **Different sample size.** V1 publication rows used N = 274; current Figure 2
   uses N = 306/307 after digital-twin duplicate filtering.

4. **RF setup now aligned.** The revised supervised RF rerun now uses the v1
   RF setting: `RandomForestClassifier(n_estimators=100, random_state=42)`.
   Remaining differences are therefore split, feature definition, and sample
   size.

5. **Same-participant proxy signal.** The current PP 70/30 split is
   within-participant. Even the corrected demographics-only RF line can benefit
   from participant-proxy information because many demographic vectors uniquely
   identify a test participant who is also present in training. This is not
   explicit history leakage, but it means the current Demographics RF line
   should be described as a same-participant supervised reference rather than
   an independent-participant demographic-generalization result.

## Current RF-vs-LLM Interpretation

In the current Figure 2 source table, Demographics RF is close to the best
LLM/PP/hybrid configuration on aggregate metrics:

| Domain | Metric | RF Demographics | Best LLM/PP/hybrid | Difference |
|---|---|---:|---:|---:|
| Content | Accuracy | 0.484 | 0.487 | +0.003 |
| Content | Macro-F1 | 0.330 | 0.341 | +0.010 |
| Content | QWK | 0.348 | 0.370 | +0.022 |
| Coping | Accuracy | 0.459 | 0.446 | -0.013 |
| Coping | Macro-F1 | 0.389 | 0.391 | +0.003 |
| Coping | QWK | 0.357 | 0.455 | +0.098 |
| Quitting | Accuracy | 0.469 | 0.492 | +0.023 |
| Quitting | Macro-F1 | 0.387 | 0.433 | +0.047 |
| Quitting | QWK | 0.466 | 0.541 | +0.075 |

Thus the corrected revision should not claim that LLM-PP broadly outperforms
supervised RF. A more defensible interpretation is that RF is a strong
same-participant aggregate predictor, while LLM/PP/hybrid methods provide
competitive or stronger ordinal agreement in several domains and remain useful
for message-selection analyses.

## Recommended Manuscript Handling

Do not say the corrected current demographics-only row exactly reproduces the
pre-revision/v1 baseline. It does not.

Use wording like:

> We reran the supervised baselines under the cleaned digital-twin 70/30 split
> using a stricter demographics-only feature definition. This differs from the
> earlier participant-only/broad-metadata baseline, which included smoking and
> psychosocial variables and used a different split.
