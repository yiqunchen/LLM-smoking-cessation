# START HERE: Revision Edit Checklist

Date: 2026-06-02 (numbers/scope updated 2026-06-19)

> **SUPERSEDED-IN-PART BANNER (2026-06-19).** The author has since locked three
> decisions that override parts of this checklist:
> 1. **Anchor methods (RF-anchor / LLM-anchor) are DROPPED entirely** — the
>    comparison is now just **two methods: supervised RF vs LLM-PP**. Ignore every
>    anchor checklist item below; they have been struck/rewritten in place.
> 2. **One split for everything: `dt10`** (within-participant; 301 participants;
>    **N = 898** held-out rows at `k_train = 7`).
> 3. **One supervised feature block for the headline: `Demographics`** (7
>    attributes only). Headline QWK is **RF 0.527 / LLM-PP 0.488** (single
>    Demographics block), NOT the old 0.547/0.555 (which mixed the Avg-History block).
>
> The canonical source of truth for all numbers is now
> `revision/SPLIT_DEFINITION_AND_UNIFIED_RESULTS.md`,
> `revision/why_new_numbers_higher_reviewer_explanation.md`, and
> `revision/reviewer_clarity_summary.md`. Where this checklist disagrees, those win.

Use this file as the single implementation checklist for assembling the
revised manuscript and reviewer response. The detailed copy-paste prose is in
`main_paper_docx_concrete_edits.md`; this file tells you exactly what to change,
where to put it, and which updated artifacts/numbers to cite.

For the shortest final readiness summary, see
`MANUSCRIPT_REWRITE_READY.md`.

## Source Policy To Preserve

- Main model-comparison claims use the single `dt10` within-participant split
  (301 participants; N = 898 held-out rows at `k_train = 7`) with the single
  `Demographics` supervised feature block, unless a section explicitly says it is
  descriptive context or a separate sensitivity analysis.
- The comparison is **two methods only: supervised RF vs LLM-PP**. Anchor methods
  are dropped.
- Missing split-specific methods are omitted rather than backfilled from a
  different split.
- Within-participant evaluation means all 301 participants appear in BOTH train
  and test (by design). Frame supervised RF as a strong **participant-calibrated**
  reference, not as evidence that demographics predict PME.
- Historical participant ratings / response history should be described as the
  strongest observed personalization signal in this benchmark, not as a causal
  or universal input-importance claim.
- Preserve the original Figure 1-4 manuscript architecture.

## Core Message

Use this message consistently in Abstract, Results, Discussion, Conclusion, and
figure captions:

> Prior participant ratings / response history were the strongest observed
> personalization signal for PME prediction in this benchmark. In strict
> shared-row (within-participant, N = 898) comparisons, response-history-informed
> supervised RF was the strongest aggregate classifier (QWK 0.527); LLM-PP was
> competitive, especially at low history / cold start and for coping/quitting
> message selection. The methods are complementary; neither uniformly dominates.

## Immediate Deletes

Delete or replace every main-text claim that says or implies:

- LLM-PP methods outperform supervised baselines by 10+ percentage
  points.
- Personalized LLM methods are the best aggregate classifiers.
- Macro-F1 exceeded supervised baselines by 8-12 points.
- Wider predicted-score distributions prove better individual simulation.
- Historical ratings are causally or universally the most important element.

Additionally, DELETE these specific legacy figures/claims wherever they are
asserted as current results (they do not reproduce from current data):

- "+12 percentage points over zero/few-shot" and "+13 percentage points over
  supervised".
- "accuracies 0.49 / 0.45 / 0.49 for LLM-PP".
- High LLM-PP within-participant Spearman (~0.27-0.43); the defined value is ~0.06.
- Old QWK 0.547 / 0.555 as the headline (those mixed the Avg-History block); the
  single-Demographics-block QWK is RF 0.527 / LLM-PP 0.488.
- Any reference to RF-anchor or LLM-anchor (anchor methods are dropped entirely).

## Main Manuscript Edits

### 1. Title And Keywords

Replace title with:

> Response-History-Informed Smoking-Cessation Message Evaluation

Replace keywords with:

> Large language models (LLMs); supervised learning; response history; rating
> history; hybrid models; perceived message effectiveness (PME); smoking
> cessation; personalized interventions

### 2. Abstract

Replace the full abstract with the block in
`main_paper_docx_concrete_edits.md` under:

`## Abstract` -> `Replace The Entire Abstract With This`

Key numbers that must appear (dt10, Demographics block, N = 898):

| Metric at `k_train=7` | Supervised RF | LLM-PP |
|---|---:|---:|
| Mean accuracy | 0.474 | 0.462 |
| Macro-F1 | 0.403 | 0.370 |
| QWK | 0.527 | 0.488 |
| Within-participant Spearman rho | NA (undefined) | 0.058 |

(RF's within-participant Spearman is undefined because it is message-blind and
assigns one tied value per participant; LLM-PP's is small but defined, ~0.06.)

### 3. Introduction

Replace the personalized-prompt motivation paragraph and final Introduction paragraph
using the blocks in `main_paper_docx_concrete_edits.md` under:

- `Replace The Personalized-Prompt Motivation Paragraph`
- `Replace The Final Introduction Paragraph`

Make sure the Introduction frames the gap as:

> Whether prior participant ratings can personalize PME prediction and whether
> persona-conditioned LLM scores add predictive or message-selection value
> beyond response-history-informed supervised baselines.

### 4. Methods

Apply all Methods replacements from `main_paper_docx_concrete_edits.md`:

- Replace Section 2.2 opening.
- Rename Section 2.2.1 to `Supervised random forest benchmark`.
- Replace Section 2.2.1 text.
- Add the caveat at the start of Section 2.2.2.
- Rename Section 2.2.3 to `Persona-conditioned LLM method`.
- Replace Section 2.2.3 text (describe LLM-PP only; remove any hybrid/anchor method).
- Replace Section 2.3 evaluation metrics.
- Replace Section 2.4 top-K message selection.

Important Methods details to preserve:

- Single split: `dt10` within-participant; 301 participants; N = 898 held-out at
  `k_train = 7`. History/test design uses `k_train = 1, 3, 7`.
- LLM-PP uses participant profile plus available prior rating history.
- The supervised headline uses the single `Demographics` feature block (7
  attributes: age, gender, race/ethnicity, Hispanic/Latino, sexual orientation,
  education, household income -> ~46 one-hot columns; excludes smoking/quit/
  psychosocial features).
- Message-selection benchmark uses fixed `Demographics + History + Message
  Embedding` across all domains.
- Generic LLM prompt-family results are contextual, not the primary AI-vs-ML
  benchmark, and cover a smaller subset (N ~ 87 at k=7), reported separately.

### 5. Results

Replace the opening Results paragraph and Sections 3.1-3.5 using
`main_paper_docx_concrete_edits.md`.

Required section structure:

1. `3.1 Strict shared-row comparison of supervised RF and LLM-PP`
2. `3.2 Prediction improved with additional participant history`
3. `3.3 Ordinal agreement and within-participant rank signal`
4. `3.4 Rating distributions and confusion matrices`
5. `3.5 Message-selection gain over random`

For Section 3.1, the headline (dt10, Demographics, N = 898, k=7) is RF acc 0.474 /
macro-F1 0.403 / QWK 0.527 vs LLM-PP acc 0.462 / macro-F1 0.370 / QWK 0.488. RF was
the strongest aggregate classifier; LLM-PP close behind. By domain (k=7): Content
RF best (acc .523 / QWK .449 vs LLM-PP .499 / .402); Coping RF wins aggregate
(.450 / .543 vs .431 / .503); Quitting ~tie (LLM-PP acc .457 > RF .450; RF QWK
.589 > .559).

For Section 3.2, learning curve (Demographics, k=1/3/7): RF acc 0.420/0.448/0.474,
RF QWK 0.418/0.447/0.527; LLM-PP acc 0.417/0.438/0.462, LLM-PP QWK 0.417/0.457/0.488.
LLM-PP ties RF on QWK at k=1 and slightly exceeds at k=3 (0.457 vs 0.447); RF pulls
ahead at k=7. RF's edge grows with history (its advantage is history-memorization);
LLM-PP is most competitive at low history / cold start, the realistic deployment regime.

For Section 3.3, RF's within-participant Spearman is undefined (message-blind, tied
predictions per participant); LLM-PP's is small but defined (~0.06).

Updated class-distribution sentence for Section 3.4 / reviewer response (dt10
held-out, % ratings in categories 4-5):

> On the dt10 held-out test set, ratings 4-5 accounted for 68.2%
> of Content ratings, 61.6% of Coping ratings, and 61.7% of Quitting ratings.

Updated message-selection numbers for Section 3.5 (Demographics + History +
Embedding block):

| Domain | Fixed feature block | Method with largest gain at K=5 | Gain |
|---|---|---|---:|
| Content | Demographics + History + Message Embedding | Supervised RF | 0.389 |
| Coping | Demographics + History + Message Embedding | LLM-PP | 0.570 |
| Quitting | Demographics + History + Message Embedding | LLM-PP | 0.540 |

Reading: Content message selection favors RF (0.389 vs LLM-PP 0.184); Coping and
Quitting message selection favor LLM-PP (Coping 0.570 vs RF 0.245; Quitting 0.540
vs RF 0.123).

Also note:

> At K=10 for Quitting, LLM-PP was highest with gain 0.521.

### 6. Discussion

Replace these Discussion paragraphs using
`main_paper_docx_concrete_edits.md`:

- First Discussion paragraph.
- Personalized-prompt opportunity paragraph.
- Population-level trends paragraph.
- Directional accuracy paragraph.
- Score-dispersion paragraph.
- Limitations paragraph.

Discussion must not claim LLM superiority. It should say:

> LLM-based personalization is best framed as a complementary scoring and
> selection component integrated with strong supervised benchmarks.

### 7. Conclusion

Replace the entire Conclusion with the block in
`main_paper_docx_concrete_edits.md` under:

`## Conclusion` -> `Replace The Entire Conclusion`

## Figure And Caption Edits

### Figure 1

Use:

- `figures/llm-message-paper-figure1.svg`
- `figures/llm-message-paper-figure1.pdf`

Caption must say within-participant history ratings are the personalization
substrate.

### Figure 2

Use the top-three vertical assembled file for the main manuscript:

- `figures/figure2/figure2_main_top3_vertical_300dpi.png`

This contains Accuracy, Macro-F1, and QWK stacked in one column. Put the other
three metrics in the appendix/supplement:

- `figures/figure2/figure2_appendix_other3_vertical_300dpi.png`

Individual panel files remain available in:

- `figures/figure2/bars_all_methods_accuracy.png`
- `figures/figure2/bars_all_methods_f1.png`
- `figures/figure2/bars_all_methods_qwk.png`
- `figures/figure2/bars_all_methods_kappa.png`
- `figures/figure2/bars_all_methods_directional_accuracy.png`
- `figures/figure2/bars_all_methods_directional_macro_f1.png`

Caption must say:

- Bars use the `dt10` within-participant split (N = 898 at k=7), comparing the
  two methods supervised RF and LLM-PP.
- Original LR/RF demographics baselines are retained.
- LR/RF `Demographics + History + Message Embedding` reference lines are
  added.
- Missing split-specific methods are omitted rather than backfilled.

Source audit:

- `figures/bars_all_methods_source_table.csv`
- `figures/bars_all_methods_source_audit.csv`
- `figures/bars_all_methods_manifest.md`

### Figure 3

Use the vertical assembled manuscript file:

- `figures/figure3/figure3_assembled_vertical_300dpi.png`

Individual domain panels remain available in:

- `figures/figure3/figure3_score_distributions_content.png`
- `figures/figure3/figure3_score_distributions_coping.png`
- `figures/figure3/figure3_score_distributions_quitting.png`

Caption must frame these as class-imbalance/calibration context, not proof that
any model simulates individual behavior better. The updated panels now show a
vertical stack of human ratings, the PP LLM (LLM-PP) rows, and LR/RF
`Demographics + History + Message Embedding` supervised rows. (No hybrid/anchor
rows — anchor methods are dropped.)

### Figure 4

Use:

- `figures/llm_selection_quality.png`
- `figures/top_k_agreement_line.png`

Supporting strict selection benchmark:

- `revision/figures/message_selection_gain.png`
- `revision/figures/message_selection_methods_k7.csv`

Caption/text must say:

- Original Figure 4 remains top-K LLM selection quality.
- Supporting benchmark includes the two methods supervised RF and LLM-PP.
- Supporting benchmark uses fixed `Demographics + History + Message
  Embedding` across domains.
- Bands are 95% normal-approximation intervals using message-level SE
  (SD / sqrt(n)).

## Reviewer Response Edits

### AE / Reviewer: Uncertainty

Use this source policy:

> Accuracy CIs use the `dt10` within-participant source (N = 898 at k=7), with
> known duplicate test items removed from every method.

Artifacts:

- `revision/figures/accuracy_confidence_intervals.csv`
- `revision/figures/accuracy_confidence_intervals.png`
- `revision/figures/pairwise_significance_tests.csv`
- `revision/figures/pairwise_significance_heatmap.png`

### Reviewer: Class Imbalance / QWK

Use latest dt10 held-out percentages (ratings in categories 4-5):

- Content ratings 4-5: 68.2%
- Coping ratings 4-5: 61.6%
- Quitting ratings 4-5: 61.7%

Artifacts:

- `revision/figures/class_distribution.csv`
- `revision/figures/class_distribution_by_domain.png`
- `revision/figures/qwk_results.csv`
- `revision/figures/progress_summary/confusion_matrices_k7.png`
- `revision/figures/progress_summary/ordinal_qwk_spearman_snapshot.png`

### Reviewer: Figure 2 / All-Model Consistency

State:

> We rebuilt the original Figure 2 bar family on the `dt10` within-participant
> split, comparing the two methods (supervised RF and LLM-PP). Rows missing
> split-specific files were omitted rather than backfilled from another split.
> The source table and audit are provided.

Artifacts:

- `figures/bars_all_methods_source_table.csv`
- `figures/bars_all_methods_source_audit.csv`
- `figures/bars_all_methods_manifest.md`

### Reviewer: Figure 4 / Message Selection

State:

> We preserved the original Figure 4 top-K selection structure and added a
> strict supporting method-level benchmark. The supporting benchmark uses one
> fixed `Demographics + History + Message Embedding` feature block across Content, Coping, and Quitting, so
> the domain panels no longer reflect domain-wise feature-set cherry-picking.

Artifacts:

- `revision/figures/message_selection_gain.png`
- `revision/figures/message_selection_methods_k7.csv`

## Appendix / Supplement Edits

### Appendix A2

Replace LLM-superiority language with:

> The bootstrap confidence intervals summarize uncertainty for the LLM-only
> prompt-family analyses. These analyses contextualize the effect of
> history-augmented prompting, but the main revised AI-vs-ML claims are based
> on strict shared-row comparisons between supervised RF and LLM-PP.

### Appendix A3 / Supplementary Figure 1

Use the learning-curve message:

> These sensitivity analyses suggest that a small number of prior ratings can
> improve model performance, particularly for the supervised RF method (whose
> edge grows with history). LLM-PP is most competitive at low history / cold
> start. These results support response history as the strongest observed
> personalization signal in the revised benchmark.

## Final QA Before Resubmission

- [ ] Abstract no longer says LLM-PP methods outperform supervised
  baselines.
- [ ] Every main-text "digital twin" use is either replaced or explicitly
  defined as persona-conditioned, history-augmented LLM prediction.
- [ ] Methods state the split/source policy for primary comparisons (dt10,
  Demographics block, two methods: RF vs LLM-PP).
- [ ] No anchor (RF-anchor / LLM-anchor) methods remain anywhere.
- [ ] Results include strict RF / LLM-PP numbers (QWK 0.527 / 0.488).
- [ ] Results include history-length sensitivity.
- [ ] Results class-distribution percentages match dt10 held-out:
  68.2%, 61.6%, 61.7%.
- [ ] Message-selection text says fixed `Demographics + History + Message Embedding`.
- [ ] Figure 2 caption references the dt10 split and source audit.
- [ ] Figure 3 caption avoids using distribution width as personalization
  proof.
- [ ] Figure 4 caption preserves the original figure slot and points to the
  supervised supporting benchmark.
- [ ] Limitations state that causal/universal feature importance is not proven.
- [ ] Reviewer response cites `bars_all_methods_source_table.csv`,
  `qwk_results.csv`, `accuracy_confidence_intervals.csv`, and
  `message_selection_methods_k7.csv` where relevant.
