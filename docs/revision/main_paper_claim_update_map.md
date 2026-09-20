# Main Paper Claim Update Map

Date: 2026-06-02 (numbers superseded 2026-06-19)

> **SUPERSEDED / UNIFIED ON dt10 (2026-06-19).** This doc has been updated to the
> locked single-split, two-method analysis. The comparison is now **supervised RF
> vs LLM-PP only** (anchor methods dropped). All headline numbers use the **dt10**
> within-participant split (301 participants; N = 898 held-out rows at k_train = 7)
> and the **Demographics** supervised feature block. Source of truth:
> `SPLIT_DEFINITION_AND_UNIFIED_RESULTS.md`,
> `why_new_numbers_higher_reviewer_explanation.md`,
> `reviewer_clarity_summary.md`.

## Direct Answer

No: the broad "10+ percentage point" improvement claim should not remain in
the main paper.

The original proof says:

> LLM-based digital twins outperformed zero-/few-shot (+12 percentage points
> on average) LLMs and supervised baselines (+13 percentage points).

That statement does not hold under the unified within-participant comparison.
At `k_train=7` (dt10, Demographics block, N = 898), the current main numbers are:

| Metric | Supervised RF | LLM-PP |
|---|---:|---:|
| Accuracy | 0.474 | 0.462 |
| Macro-F1 | 0.403 | 0.370 |
| QWK | 0.527 | 0.488 |
| Within-participant Spearman rho | undefined* | 0.058 |

\* RF is message-blind on the Demographics block: it assigns one prediction per
participant, so within-participant ranking (and thus Spearman) is undefined. LLM-PP
produces a small but defined within-participant correlation (~0.06).

So the main-paper statement should be:

> In unified within-participant comparisons, supervised RF was the strongest
> aggregate classifier (QWK 0.527), and LLM-PP was competitive — especially at
> low history (cold start) and for coping/quitting message selection — rather than
> a uniform improvement over supervised learning. Across history-length analyses,
> additional prior participant ratings improved supervised RF, making response
> history the strongest observed personalization signal in this benchmark; LLM-PP
> was most competitive when history was scarce.

**Same-participant caveat (state wherever RF/demographics is called strong):** dt10
is within-participant by design — all 301 participants appear in both train and
test. A trivial "predict each participant's most-frequent prior rating" baseline
reproduces RF accuracy (Content .53, Coping .45, Quitting .47), and on a strict
unseen-participant split RF falls to ~0.35. So supervised RF is a strong
**participant-calibrated** reference, **not** evidence that demographics predict PME.

## Revised Core Message: Historical Ratings Drive Personalization

Use this through-line across the manuscript:

> Personalized PME prediction in this dataset was driven most clearly by
> historical participant ratings. More prior ratings improved supervised RF, and
> participant-calibrated supervised RF was difficult to outperform on aggregate
> classification. Persona-conditioned LLMs (LLM-PP) should therefore be framed
> as complementary scoring and selection tools — strongest at low history and for
> coping/quitting message selection — not as uniformly superior digital twins.

Boundary: do not claim a causal or universal input-importance hierarchy unless
a clean history-only/profile-only/message-only ablation is run. The safe claim
is "strongest observed personalization signal in the current benchmark."

## What Can Still Be Said

These statements are still safe:

- Historical participant ratings / response history are the strongest observed
  personalization signal in the current benchmark.
- Performance improved as more prior ratings were available for supervised RF.
- LLM-PP performs better than generic zero-/few-shot prompting in the older
  LLM-only prompt-family comparison, but those results should be presented as
  contextual or supplemental because they are not the strict shared-row
  AI-vs-ML table.
- Supervised RF is the strongest aggregate classifier (QWK 0.527), but only as a
  participant-calibrated reference — it reproduces a per-participant modal-rating
  baseline, not demographic prediction.
- LLM-PP is competitive with RF and ties/edges it on QWK at low history
  (k=1 tie; k=3 LLM-PP 0.457 vs RF 0.447); RF pulls ahead at k=7.
- LLM-PP wins message selection in the helpfulness domains: gain@K=5 of 0.570
  (coping) and 0.540 (quitting) vs RF 0.245 / 0.123; RF wins content selection
  (0.389 vs 0.184).
- The revised analyses support a calibrated, complementary personalized-message
  selection story (RF for content/aggregate, LLM-PP for coping/quitting selection
  and cold start), not a simple LLM-superiority story.

These statements are not safe:

- LLM digital twins outperform supervised baselines by more than 10 percentage
  points.
- LLMs are the best aggregate classifiers.
- Digital twins capture individual behavior better than supervised learning in
  general.
- Historical ratings are universally or causally the most important element
  without a clean history-only/profile-only/message-only ablation.
- The old zero-/few-shot comparison and current RF comparison are
  apples-to-apples.

## Required Main-Paper Edits

### Abstract Results

Replace the old 10+ point sentence with:

> In a unified within-participant comparison at `k_train=7` (dt10 split,
> Demographics feature block, N = 898 held-out ratings), supervised RF was the
> strongest aggregate classifier and LLM-PP was competitive. Mean accuracy across
> Content, Coping, and Quitting was 0.474 for supervised RF and 0.462 for LLM-PP;
> macro-F1 was 0.403 and 0.370. Supervised RF had the higher ordinal agreement by
> QWK (0.527 vs 0.488 for LLM-PP). Because all participants appear in both train
> and test, supervised RF functions as a participant-calibrated reference rather
> than demographic prediction. Performance improved as more prior participant
> ratings were available, identifying response history as the strongest observed
> personalization signal; LLM-PP was most competitive at low history. In
> message-selection analyses, both methods selected messages above random, with
> LLM-PP best for coping and quitting.

### Abstract Discussion

Replace:

> integrating personal profiles with LLMs captures person-specific differences
> in PME and outperforms supervised learning

with:

> prior participant ratings provided the strongest observed personalization
> signal for PME prediction; a participant-calibrated supervised model captured
> substantial aggregate PME signal, while persona-conditioned LLM (LLM-PP) scores
> provided complementary value at low history and for coping/quitting message
> selection.

### Conclusion

Replace:

> LLM-based digital twin models show potential for predicting PME

with:

> A participant-calibrated supervised RF remains a strong benchmark for
> aggregate PME prediction, while persona-conditioned LLM (LLM-PP) methods may
> support cold-start scoring and coping/quitting message-selection workflows.

## Results Section Rewrite

The first Results paragraph should explicitly acknowledge the changed
interpretation:

> The unified within-participant comparison revised the interpretation of the
> original analysis. Rather than showing uniform LLM superiority, the updated
> results showed that supervised RF was the strongest aggregate classifier, with
> LLM-PP competitive behind it. At `k_train=7` (dt10, Demographics block,
> N = 898), mean accuracy across Content, Coping, and Quitting was 0.474 for
> supervised RF and 0.462 for LLM-PP, and macro-F1 was 0.403 and 0.370. Supervised
> RF achieved the higher QWK (0.527 vs 0.488). Because the split is
> within-participant — every participant appears in both train and test — this
> supervised performance reflects participant calibration rather than demographic
> prediction; a trivial per-participant modal-rating baseline reproduces it, and
> on an unseen-participant split RF accuracy falls to ~0.35. History-length
> analyses showed that additional prior participant ratings improved supervised RF
> (QWK 0.418 → 0.447 → 0.527 across k = 1/3/7), making response history the
> clearest observed personalization signal; LLM-PP tied RF on QWK at k = 1 and
> slightly exceeded it at k = 3 (0.457 vs 0.447), with RF pulling ahead at k = 7.

## Discussion Rewrite

The first Discussion paragraph should say:

> In response to reviewer concerns, we strengthened the benchmark by adding a
> unified within-participant comparison, stronger supervised RF baselines, ordinal
> metrics, confusion matrices, and message-selection analyses that include
> supervised methods. These analyses revise the central interpretation:
> LLM-PP did not dominate supervised learning on aggregate PME classification.
> Instead, supervised RF was the strongest aggregate performer — though, because
> the evaluation is within-participant, this reflects a strong participant-calibrated
> reference rather than demographic prediction. The clearest personalization
> signal was the availability of prior participant ratings: as more history was
> available, supervised RF improved. LLM-PP remained useful as a complementary
> persona-conditioned scoring approach, particularly at low history (cold start)
> and for coping/quitting message selection.

## If A 10+ Point Claim Is Mentioned At All

Do not put it in the abstract, main Results headline, or Discussion thesis.
If retained, it must be limited to a caveated supplemental sentence such as:

> In the LLM-only prompt-family comparison, history-augmented prompting
> improved over generic zero-/few-shot prompting; however, these results are
> not used as the primary AI-vs-ML comparison because the revised main analysis
> uses a unified within-participant comparison of supervised RF and LLM-PP.

Even this caveat should only be used if the exact LLM-only table is retained
and clearly labeled.

## Recommended Main-Paper Figure Logic

Preserve the original Figure 1-4 manuscript architecture, but revise captions
and supporting panels so the message is history-first:

1. Figure 1: study schematic should show prior ratings / response history as
   the personalization substrate.
2. Figure 2: model-performance bars should be captioned as a two-method
   comparison (supervised RF vs LLM-PP) on the unified dt10 within-participant
   split (Demographics block, N = 898 at k_train=7), with Content/Coping/Quitting
   shown on Accuracy / Macro-F1 / QWK and the same held-out rows for both methods.
   Do not present anchor/hybrid methods, and do not backfill missing methods from a
   different split. Generic zero-/few-shot bars, if shown, must be a separate,
   clearly labeled context panel (N ≈ 87, not comparable to the 898-row panel).
3. Figure 3: score distributions should be framed as class-imbalance and
   calibration context, not proof of LLM-PP superiority.
4. Figure 4: top-K message selection should include the supervised RF
   response-history comparator in the text/caption or supporting panel.
5. Supplementary Figure 1 / Appendix A3: learning curves across
   `k_train=1,3,7` should carry the explicit history-length sensitivity claim.

This makes the paper publishable because the main claim matches the strongest
current evidence.
