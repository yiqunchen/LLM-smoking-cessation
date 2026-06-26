# Why the revised numbers are higher than the original — and why it is not an error

Date: 2026-06-19

## Short answer

The increase is **not** a computational mistake or a leakage bug. It reflects a
deliberate change in the evaluation question — from *unseen-participant
generalization* to *within-participant personalization* — applied **equally to
every method**. We verified there is no held-out-rating leakage. The only
asymmetry that existed was in the **original** comparison, and the revision
removes it; that is why the supervised baseline "caught up."

## The decomposition (all values recomputed from committed CSVs)

| Version | Split | Eval regime | N | Supervised RF acc | Best LLM acc |
|---|---|---|---:|---:|---:|
| Original / V1 | `participant_7030` | **unseen participant** (hold out whole people) | 274 | **0.345** | ~0.41–0.45 |
| Revised | `dt10_k7` | **within participant** (hold out 3 of each person's 10 messages) | 898 | **0.474** | 0.462 (LLM-PP) |

**The +0.13 jump is dominated by the regime change.** Predicting a rating for a
person from whom the model has *already seen 7 ratings* is far easier than
predicting for a brand-new person. Every method rose together — supervised and
LLM alike — so the rise is not a thumb on the scale for the LLM.

## We checked the three places a real mistake could hide

1. **Held-out ratings leaking into training?** No. The headline supervised model
   uses the `Demographics` feature block only — `lc_dt10_rf.py:52`:
   `{"demo": True, "history": False, "embedding": False}`. It has no access to any
   message or any rating.
2. **History feature leaking test ratings?** No. Where history *is* used, the
   lookup is built from the training rows only — `lc_dt10_rf.py:81`
   `_build_history_lookup(train)`, with `exclude_self=True` on train. Held-out
   ratings never enter the feature.
3. **Same message-rating in train and test?** No. dt10 holds out *different*
   messages per participant (10 = k train + (10−k) test), and duplicate test
   items were filtered.

## Why "demographics-only RF" looks strong (the honest mechanism — demonstrated)

Because evaluation is within-participant, a participant's (constant, near-unique)
demographic vector acts as a **participant identifier** (7 attributes → ~46 one-hot
columns → ~55,000 possible profiles for 301 people; 228/307 rows map to a single
participant). The RF therefore learns each participant's **most-common training
rating** and reuses it — a per-participant lookup, not demographic prediction.

**Direct demonstration (recomputed on `test_dt10_k7.json`, N=898):** a trivial
"predict each person's modal past rating" baseline reproduces RF-demographics; and
removing person knowledge (global mode, i.e. what an unseen participant gets) drops
it to the unseen-participant floor.

| Domain | Predict person's modal past rating | RF-demographics (actual) | Global mode (no person knowledge) |
|---|---:|---:|---:|
| Content | 0.532 | 0.523 | 0.347 |
| Coping | 0.453 | 0.450 | 0.316 |
| Quitting | 0.470 | 0.450 | 0.304 |

The ~0.13 gap between "global mode" (~0.32) and RF (~0.47) is **entirely** the value
of having the test person's own prior ratings in the training set. It matches the
unseen→within-participant jump (0.345 → 0.474). No held-out label is ever seen; the
held-out test set protects the test *rows'* labels, not the test *person's* identity.

Further tell: RF-demographics' **within-participant Spearman is undefined** in all
three domains (it assigns one value per person and cannot rank a person's messages).

## The real "mistake" was in the original framing — and the revision fixes it

The original paper's headline "LLM personalization beats supervised baselines by
~13 points" came from comparing the LLM under the easy within-participant setup
against a supervised baseline that was **not** given the same participant-history
advantage. Once both are placed on the identical within-participant split (dt10),
the supervised baseline rises to ~0.47 and the gap disappears. So the higher
supervised numbers are the **correction** of an inadvertent asymmetry, not a new
error introduced in the revision.

## What this means we must disclose (so the numbers are not misread)

- State explicitly that the benchmark is **within-participant**: each model gets
  *k* of a participant's ratings and predicts their held-out ratings.
- Frame supervised RF as a **strong participant-calibrated reference**, not as
  evidence that demographics predict PME.
- Keep the unseen-participant result (~0.345) reported somewhere as the
  generalization contrast, so readers see the gap between personalization and
  generalization.
- Do not carry any original-version language that implies these within-participant
  numbers represent generalization to new individuals.

## One-paragraph version for the response letter

> The revised accuracies are higher than in the original submission because we
> standardized every method on a within-participant evaluation (dt10): each model
> receives k of a participant's ratings as history and predicts that participant's
> remaining held-out ratings. This is easier than the unseen-participant setting
> and raises all methods, supervised and LLM alike. We confirmed that the headline
> supervised model uses demographic features only and that history features are
> computed exclusively from training ratings, so held-out ratings never leak into
> training. Importantly, the original comparison gave the LLM this within-participant
> advantage while the supervised baseline did not receive it; placing both on the
> same partition is why the supervised baseline is now competitive. We interpret
> the supervised model as a strong participant-calibrated reference and report the
> unseen-participant result as the generalization contrast.
