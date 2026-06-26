> SUPERSEDED IN PART (2026-06-19): anchors dropped; see SPLIT_DEFINITION_AND_UNIFIED_RESULTS.md and reviewer_clarity_summary.md for the authoritative numbers.

# Latest Revision Narrative Reframe

Date: 2026-06-02

## Bottom Line

The revised manuscript should no longer be framed as "LLM-based digital
twins outperform supervised ML." The newest apples-to-apples analyses show
that a strong supervised random forest is the strongest aggregate classifier
(QWK 0.527), while LLM-PP is competitive, especially at low history and for
coping/quitting message selection.

The revised central claim should be:

> Strong supervised models capture much of the aggregate PME signal, but
> persona-conditioned LLM scores provide complementary value for
> within-participant ranking and practical message selection. The comparison
> is a calibrated, two-method benchmark between supervised RF and LLM-PP.
> Caveat: because the dt10 split is within-participant (all 301 participants
> appear in both train and test), supervised RF functions as a strong
> participant-calibrated baseline, not as evidence that demographics predict
> PME.

This is a substantial narrative change from the original proof, which stated
that LLM personalization outperformed zero-/few-shot LLMs and supervised
baselines by large margins. That claim is not supported by the current strict
shared-row results.

## Most Updated Results To Lead With

Primary comparison: strict shared-row results at `k_train=7`, on the `dt10`
within-participant split, using the same N = 898 held-out rows for both
methods and the single `Demographics` feature block. Values below are means
across Content, Coping, and Quitting.

| Endpoint | Feature block | N | Best method | Supervised RF | LLM-PP |
|---|---:|---:|---|---:|---:|
| Accuracy | Demographics | 898 | Supervised RF | 0.474 | 0.462 |
| Macro-F1 | Demographics | 898 | Supervised RF | 0.403 | 0.370 |
| QWK | Demographics | 898 | Supervised RF | 0.527 | 0.488 |
| Within-participant Spearman rho | Demographics | 898 | LLM-PP (RF undefined) | NA | 0.058 |

Interpretation:

- Supervised RF is the strongest aggregate classifier on all three metrics
  (accuracy 0.474, macro-F1 0.403, QWK 0.527); LLM-PP is close behind but does
  not beat it on any of them at the headline block.
- Caveat (state wherever RF/demographics looks strong): the `dt10` split is
  within-participant — all 301 participants appear in both train and test. The
  7 demographic attributes nearly uniquely identify each participant, so RF
  effectively predicts each person's modal prior rating. A trivial
  "predict each participant's most-frequent prior rating" baseline reproduces
  RF accuracy, and on a strict unseen-participant split RF falls to ~0.35. So
  supervised RF is a strong participant-calibrated reference, NOT evidence that
  demographics predict PME.
- Within-participant Spearman is modest and tie-sensitive. RF's
  within-participant Spearman is undefined because it assigns one tied value
  per participant and cannot rank that participant's messages; this should be
  explained directly rather than treated as model failure. LLM-PP, by contrast,
  produces a small but defined within-participant rank signal (~0.06).
- LLM-PP is not the top aggregate classifier in the strict table. It should
  be framed as a complementary persona-conditioned signal and message-ranking
  method, most competitive at low history / cold start.

## Learning-Curve Gist

Across `k_train = 1, 3, 7` (all on `dt10`, Demographics block), more
participant history improves both methods. The key learning-curve values are:

| Metric | Method | k=1 | k=3 | k=7 |
|---|---|---:|---:|---:|
| Accuracy | Supervised RF | 0.420 | 0.448 | 0.474 |
| Accuracy | LLM-PP | 0.417 | 0.438 | 0.462 |
| QWK | Supervised RF | 0.418 | 0.447 | 0.527 |
| QWK | LLM-PP | 0.417 | 0.457 | 0.488 |

Revision interpretation:

- The curves support "history helps," not "LLMs dominate."
- LLM-PP ties RF on QWK at k=1 (0.417 vs 0.418) and slightly exceeds it at
  k=3 (0.457 vs 0.447); RF pulls ahead at k=7 (0.527 vs 0.488). RF's edge thus
  grows with history — consistent with RF's advantage being
  participant-history memorization — while LLM-PP is most competitive at low
  history / cold start, the regime most realistic for deployment.
- LLM-PP improves with more history but remains at or just below RF on
  aggregate accuracy across all k.

## Message-Selection Results

The cleanest message-selection figure should compare the two methods:
Supervised RF and LLM-PP. This directly answers the reviewer concern that
supervised methods should be included in the selection task.

At `k_train=7` (Demographics + History + Message Embedding block), both methods
select messages above random on average. Selected gains over random at K=5 are:

| Domain | Feature set | Supervised RF | LLM-PP | Best |
|---|---|---:|---:|---|
| Content | Demographics + History + Message Embedding | 0.389 | 0.184 | RF |
| Coping | Demographics + History + Message Embedding | 0.245 | 0.570 | LLM-PP |
| Quitting | Demographics + History + Message Embedding | 0.123 | 0.540 | LLM-PP |

(At K=10, LLM-PP's Quitting gain remains high at 0.521.)

Revision interpretation:

- Message selection is where the practical value of the LLM is clearest. For
  Content (message quality), RF selects better messages; for Coping and Quitting
  (helpfulness), LLM-PP is the clear winner (gain ≈0.54-0.57 vs RF ≈0.12-0.25).
- The result is not that one method always selects better messages. The methods
  are complementary: RF leads on Content, LLM-PP leads on the helpfulness
  domains.
- The old LLM-only message-selection table can stay as context only, because
  it uses a different 108-message pool and should not be mixed into the main
  apples-to-apples method comparison.

## Confusion Matrix And Ordinal Explanation

The confusion matrices and QWK/Spearman diagnostic should be included because
they answer a real reviewer concern: accuracy alone is unstable under ordinal
class imbalance.

Reviewer-facing explanation:

- QWK measures ordinal agreement over all held-out ratings.
- Spearman rho is calculated within participant and then averaged, so it is
  sensitive to tied predictions and to the limited number of held-out messages
  per participant.
- RF can have strong QWK but undefined within-participant Spearman if it makes
  tied predictions within the same participant (as it does on the Demographics
  block, where it assigns one value per participant).
- Therefore QWK and confusion matrices should be the primary ordinal evidence;
  Spearman should be reported as a secondary rank diagnostic.

Important strict `k_train=7`, Demographics-block by-domain values:

| Domain | Method | Accuracy | QWK |
|---|---|---:|---:|
| Content | Supervised RF | 0.523 | 0.449 |
| Content | LLM-PP | 0.499 | 0.402 |
| Coping | Supervised RF | 0.450 | 0.543 |
| Coping | LLM-PP | 0.431 | 0.503 |
| Quitting | Supervised RF | 0.450 | 0.589 |
| Quitting | LLM-PP | 0.457 | 0.559 |

By domain (k=7): Content RF best on both accuracy and QWK; Coping RF wins
aggregate on both; Quitting is ~a tie (LLM-PP edges accuracy 0.457 vs 0.450,
RF edges QWK 0.589 vs 0.559). Class imbalance (dt10 held-out, % ratings in
categories 4-5): Content 68.2%, Coping 61.6%, Quitting 61.7%.

## What Must Change In The Manuscript

### Title And Terminology

Current title language around LLM-based personalization is too strong unless
the manuscript explicitly defines the term as a limited predictive analogue.
Short title direction:

> Response-History-Informed Smoking-Cessation Message Evaluation

Use "persona-conditioned LLM prompting," "history-augmented LLM prediction,"
or "LLM-PP" in place of broad "digital twin" claims. If "digital twin" remains,
define it narrowly and acknowledge that it is not a mechanistic or longitudinal
behavioral twin.

### Abstract

Remove or replace:

- "LLM-based personalization outperformed zero-/few-shot LLMs and supervised
  baselines."
- "+12 percentage points" and "+13 percentage points" superiority claims.
- Any statement implying LLM methods are the aggregate winner.

Replacement direction:

> In strict shared-row comparisons at `k_train=7` on the within-participant
> dt10 split (N = 898, Demographics block), supervised RF was the strongest
> aggregate classifier (accuracy 0.474, macro-F1 0.403, QWK 0.527); LLM-PP was
> competitive (accuracy 0.462, macro-F1 0.370, QWK 0.488). Because the split is
> within-participant, supervised RF functions as a strong participant-calibrated
> baseline rather than evidence that demographics predict PME. Within-participant
> rank signal was modest; RF's was undefined (tied per-participant predictions)
> while LLM-PP's was small but defined. In message-selection analyses, both
> methods selected messages above random, with LLM-PP the clear winner for the
> coping and quitting helpfulness domains.

Conclusion direction:

> Strong supervised baselines explain much of the aggregate PME signal under
> within-participant evaluation, while persona-conditioned LLM scores provide
> complementary value for within-participant ranking and message selection,
> especially at low history and for the helpfulness domains.

### Introduction

The introduction should pivot from "LLMs can replace traditional ML for
personalization" to a benchmark question:

- How much signal is captured by a supervised demographic model under
  within-participant evaluation (where demographics act as a participant
  identifier)?
- Does persona-conditioned LLM scoring add value when compared on the same
  held-out rows — for aggregate prediction, within-participant ranking, and
  practical message selection?

This is a stronger reviewer-facing frame because it acknowledges that RF is a
serious participant-calibrated baseline rather than a weak foil.

### Methods

Required updates:

- Describe the two main methods explicitly: Supervised RF (Demographics block:
  7 attributes — age, gender, race/ethnicity, Hispanic/Latino, sexual
  orientation, education, household income → ~46 one-hot columns) and LLM-PP
  (persona-conditioned LLM prompting).
- State the split explicitly: dt10, within-participant, 301 participants, each
  rates 10 messages, k of 10 used as history and (10−k) held out; primary k=7
  gives N = 898 held-out rows; all 301 participants appear in both train and
  test by design.
- State that generic zero-/few-shot LLM results are not mixed into the primary
  AI-vs-ML table because they come from a smaller/different `dt10` subset
  (N ≈ 87 at k=7).
- Add QWK as an ordinal metric and explain why it complements accuracy and
  macro-F1.
- Explain within-participant Spearman as a secondary rank diagnostic and note
  how ties can make it undefined.
- Clarify that message-selection analyses use gain over random and include
  supervised RF.
- De-emphasize or rename "directional accuracy" unless it is precisely defined
  as coarse ordinal direction.

### Results

Suggested sequence:

1. Rating distributions and class imbalance histograms.
2. Strict shared-row AI-vs-ML snapshot.
3. Learning curves across `k_train=1,3,7`.
4. QWK and confusion matrix diagnostics.
5. Within-participant Spearman explanation.
6. Message-selection gain over random.
7. Generic zero-/few-shot LLM results as context only.

Do not begin Results with a statement that personalized LLMs have the best
performance. The first result should say that supervised RF is the strongest
aggregate classifier (with the participant-calibrated-baseline caveat), while
LLM-PP remains competitive but not dominant on aggregate prediction.

### Discussion

Replace the old discussion thesis with:

- The original expectation that LLM-persona methods would dominate supervised
  baselines was not supported under stricter shared-row benchmarking.
- Under within-participant evaluation, a participant-calibrated supervised RF
  is hard to beat for aggregate PME classification — but this reflects
  participant identification (demographics fingerprinting each person), not
  demographic generalization. On an unseen-participant split RF falls to ~0.35.
- Practical value remains in message selection: even when exact rating
  prediction is difficult, the models can rank/select messages above random,
  and LLM-PP selects markedly better messages than RF for the coping and
  quitting helpfulness domains.
- LLM-PP is most competitive at low history / cold start, the regime most
  relevant to real deployments.
- Results concern perceived message effectiveness, not confirmed smoking
  behavior change.

### Limitations

Add or strengthen:

- PME is a proximal perceptual endpoint, not cessation behavior.
- Evaluation is within-participant; supervised RF results reflect
  participant calibration, not generalization to new individuals. The
  unseen-participant contrast (RF ~0.35) should be reported alongside.
- Generic LLM comparisons use a separate subset (N ≈ 87) and should not be
  interpreted as strict apples-to-apples evidence against the RF-vs-LLM-PP
  comparison.
- Spearman rank signal is modest and sensitive to ties.
- The current message-selection analysis is not a full recommender-system
  benchmark.
- The study does not prove longitudinal behavioral-simulation fidelity.
- Image/multimodal handling and prompt sensitivity should be described
  clearly if prompts use message screenshots or non-textual context.

## Claims To Delete Or Replace

| Old claim | Why it must change | Replacement |
|---|---|---|
| LLM-PP methods outperform supervised baselines by large margins. | Strict shared-row results show supervised RF leads aggregate metrics. | RF is the aggregate leader (QWK 0.527); LLM-PP is complementary, especially at low history and for coping/quitting selection. |
| "+12 percentage points over zero/few-shot" / "+13 percentage points over supervised". | Unsupported by current data; delete entirely. | Report only the verified within-participant numbers; no superiority-margin claim. |
| Personalized LLM models achieved the best performance across domains. | Current strict results do not support this as a general statement. | Best method depends on endpoint and domain: RF leads aggregate prediction (and Content); LLM-PP leads coping/quitting message selection. |
| Supervised baselines are weak because they use patient characteristics only. | Current revision compares both methods on the same within-participant rows. | Supervised RF is a strong participant-calibrated baseline and the aggregate winner; this reflects participant identification, not demographic prediction. |
| LLM-PP achieves within-participant Spearman ~0.27-0.43. | Unsupported by current data; current LLM-PP within-participant Spearman is small (~0.06). | Report the small but defined LLM-PP rank signal; note RF's is undefined (tied predictions). |
| Directional accuracy demonstrates personalized ranking. | Reviewers may see it as vague; Spearman/QWK are clearer. | Use QWK, confusion matrices, and within-participant Spearman. |
| Digital twin models capture individual behavior. | PME is a rating endpoint, not behavior change. | Models predict perceived message effectiveness ratings. |
| LLM-only message selection shows model-level superiority. | Old LLM-only selection uses a different message pool. | Use the RF-vs-LLM-PP gain-over-random plot as the main selection result. |

## Response-Letter Updates Needed

Files that need cleanup before sending:

- `revision/reviewer_response_technical_draft.md`
  - Update the sensitivity section: learning curves across `k_train=1,3,7`
    now exist.
  - Replace stale QWK/class-imbalance claims with the strict single-Demographics-block
    QWK results (RF 0.527, LLM-PP 0.488).
  - Update the baseline section to say the new primary comparison is supervised
    RF vs LLM-PP on the same shared held-out rows (dt10, within-participant).

- `revision/reply_remaining_comments.md`
  - Delete old claims that LLM-PP methods achieve within-participant Spearman
    values around 0.27-0.43. The current LLM-PP within-participant Spearman is
    small (~0.06); RF's is undefined (tied per-participant predictions).
  - Remove any unverified minority-class superiority claims unless the exact
    table is regenerated.
  - Update the R4 baseline answer so it does not imply LLM dominance where
    supervised RF now leads aggregate prediction.

- Main manuscript/proof
  - Rewrite title, abstract, first Results paragraph, methods baseline
    description, metric description, and Discussion thesis.

- `docs/prompt_templates.md`
  - Prompt/template descriptions now use PP/persona-conditioned terminology
    while preserving code-level prompt-config identifiers.

## Figure Set To Use

The MAIN method-comparison figure must be the unified dt10 panel showing only
the two methods (RF vs LLM-PP), Demographics block, N = 898:

- `figures/figure2/figure2_main_dt10_rf_vs_llmpp.{png,pdf}` — Supervised RF vs
  LLM-PP on Accuracy / Macro-F1 / QWK by domain.
- `figures/figure2/figure2_context_generic_llm.{png,pdf}` — generic zero/few-shot
  context panel (N ≈ 87), explicitly labeled as different coverage.

Any progress-summary method panels (`ai_vs_ml_strict_snapshot`,
`learning_curves`, `ordinal_qwk_spearman_snapshot`, `confusion_matrices_k7`,
`message_selection_gain`) must be regenerated to show only RF vs LLM-PP before
use; do not reuse versions that still plot anchor methods.

Supporting data-context figures (use as-is, not as proof of model superiority):

- `revision/figures/class_distribution_by_domain.png`
- `revision/figures/all_rating_subgroup_histograms.png`
- `revision/figures/profile_test_similarity_distribution.png`
- `revision/figures/figure3_score_distributions_content.png`
- `revision/figures/figure3_score_distributions_coping.png`
- `revision/figures/figure3_score_distributions_quitting.png`

The histograms should be used for data context, class imbalance, and subgroup
checks. They should not be presented as proof of model superiority.

## Clean Narrative For Reviewers

The clean reviewer answer is:

> We agree that the original framing overstated the evidence for LLM
> superiority. We therefore rebuilt the comparison around strict shared-row
> evaluations on a single within-participant split (dt10) with a stronger
> supervised baseline. In these updated analyses, supervised RF is the
> strongest aggregate classifier (QWK 0.527); LLM-PP is competitive (QWK 0.488)
> and is the clear winner for message selection in the coping and quitting
> helpfulness domains and at low history / cold start. Because the split is
> within-participant, we frame supervised RF as a strong participant-calibrated
> baseline rather than evidence that demographics predict PME. We revised the
> manuscript to make this benchmark result central, added QWK and confusion
> matrices to address ordinal class imbalance, and included supervised RF
> directly in the message-selection comparison.

This is more defensible than the original narrative and directly answers the
reviewers' main concern: whether the LLM method is being compared against an
adequate supervised baseline on the same data.
