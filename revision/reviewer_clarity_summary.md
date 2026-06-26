# Reviewer Clarity Summary — one coherent story, heterogeneity made legible

Date: 2026-06-19
Purpose: a single page that prevents the predictable reviewer confusions. Every
number is recomputed from committed CSVs (dt10 split). Companion detail:
`SPLIT_DEFINITION_AND_UNIFIED_RESULTS.md`, `why_new_numbers_higher_reviewer_explanation.md`.

## The design, in three sentences

1. **Within-participant personalization on the `dt10` split.** Each of 301
   participants rated 10 messages; we give each model *k* of them as history and
   predict that participant's held-out (10 − *k*) ratings. Primary *k* = 7 (N = 898).
2. **Same held-out rows for every method** (supervised RF and LLM-PP are compared
   on the identical 898 rows; generic zero/few-shot cover a smaller subset, N ≈ 87,
   and are reported separately).
3. **Supervised RF uses 7 demographic attributes only.** Because, in a
   within-participant split, those attributes nearly uniquely identify each
   participant, RF functions as a strong *"predict this person's typical rating"*
   baseline — not as demographic prediction.

## The single take-home (say this first, everywhere)

> For aggregate rating **prediction**, the person-calibrated supervised RF is hard
> to beat. For the deployment-relevant task of **message selection** in the
> helpfulness domains (coping, quitting), the LLM is the clear winner. The methods
> are **complementary**; neither uniformly dominates.

## It differs by DOMAIN — state it plainly, don't bury it

| Domain | Aggregate prediction, k=7 (RF / LLM-PP) | Message selection, gain@K=5 (RF / LLM-PP) | Who to feature |
|---|---|---|---|
| **Content** (message quality) | Acc .523 / .499 · QWK .449 / .402 → **RF** | .389 / .184 → **RF** | RF on both |
| **Coping** (helpfulness) | Acc .450 / .431 · QWK .543 / .503 → **RF** | .245 / **.570** → **LLM-PP** | split |
| **Quitting** (helpfulness) | Acc .450 / **.457** · QWK .589 / .559 → ~tie | .123 / **.540** → **LLM-PP** | LLM-PP |

**Pattern:** message *quality* (Content) is best handled by the person-baseline RF;
message *helpfulness* (Coping, Quitting) is where the LLM earns its keep — it
selects far better messages than RF (≈0.55 vs ≈0.18 gain over random) even though
RF edges it on aggregate accuracy. This is the honest, domain-specific version of
"LLMs help."

## It differs by HISTORY LENGTH — the LLM's edge is at cold start

| Metric | k=1 (10/90) | k=3 (30/70) | k=7 (70/30) |
|---|---|---|---|
| Accuracy gap (RF − LLM-PP) | +0.003 | +0.010 | +0.012 |
| QWK gap (RF − LLM-PP) | +0.001 (tie) | **−0.010 (LLM wins)** | +0.039 (RF) |

RF's advantage **grows with history** because its advantage *is* history-memorization.
The LLM is most competitive when history is scarce — it ties/edges RF on ordinal
agreement at low *k*. Since real deployments rarely have 7 prior ratings per new
user, the **low-history regime is the realistic one**, and that is where the LLM is
strongest relative to RF.

## Anticipated reviewer questions → crisp answers

1. **"Why are accuracies higher than the original submission?"**
   We standardized every method on a within-participant evaluation; this lifts all
   methods. No leakage (headline RF uses demographics only; history features are
   built from training rows only). The original analysis gave *only the LLM* this
   within-participant advantage while the supervised baseline did not get it —
   placing both on the same split is why the baseline is now competitive.

2. **"How can a demographics-only model be the best?"**
   It is not using demographic signal. In a within-participant split, the 7
   demographic attributes (~46 one-hot columns; ~55k possible profiles for 301
   people) act as a participant fingerprint, so RF predicts each person's modal
   past rating. We show a trivial "predict the person's modal training rating"
   baseline reproduces RF (Content .53/.52, Coping .45/.45, Quitting .47/.45), and
   that it collapses to chance-level global mode (~.32) — matching the
   unseen-participant result (~.35) — once person identity is removed.

3. **"Then what does the LLM add?"**
   (a) Message selection for coping/quitting (gain ≈0.55 vs ≈0.18). (b) Competitiveness
   at low history / cold start. (c) Unlike RF, it can rank a participant's messages —
   RF's within-participant Spearman is undefined because it assigns one value per person.

4. **"Are the comparisons apples-to-apples?"**
   RF and LLM-PP use the identical 898 held-out rows. Generic zero/few-shot cover a
   different, smaller subset (N ≈ 87) and are shown only as labeled context, never
   merged into the main comparison.

5. **"Which split, and why dt10?"**
   All results are on dt10 (full 10-message record, exactly *k* history messages).
   The earlier 70/30 partition used only 3 of 10 messages with uneven/limited
   history; conclusions are unchanged, but dt10 is cleaner and supports the
   k = 1/3/7 history-length analysis.

## What NOT to say (each invites a confused reviewer)

- ✗ "LLM digital twins outperform supervised baselines." (False on aggregate.)
- ✗ "RF is the best model" — without the participant-baseline caveat (invites
  "so demographics predict PME?").
- ✗ Presenting within-participant numbers as generalization to new individuals.
- ✗ Mixing splits or feature blocks inside a single comparison.
- ✗ The legacy "+12pp / +13pp" superiority claim (unsupported by current data).
