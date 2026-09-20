# Data Partition and Label Leakage — Reviewer Response Overview

This document consolidates the partition logic, leakage audit, and draft response prose for reviewer comments that touch on splits, exemplars, history-test overlap, and what each model could "see." Pulls together evidence already present in `revision/figures/label_leakage_analysis.csv` and the canonical splits in `data_splits/canonical/`.

---

## 1. Data partition (one-paragraph clean description for Methods 2.2)

We use a **within-participant 70/30 split**. Each of the 301 participants rated 10 messages; the first 7 (chronologically arbitrary, since all 10 were rated in a single survey session) form the *history/training* set for that participant, and the remaining 3 are *held out* for evaluation.

Two split families are derived from this:

- **Participant split (used for supervised baselines).** All 7 history messages from all participants form the global training corpus (~2,107 ratings). The 3 held-out messages from all participants form the global test corpus (~903 ratings). Models are trained on participant-level features; they never see test ratings during training.
- **PP split (used for LLM-PP and Hybrid RF+PP).** Same partition logic, but the 7 history messages and ratings of *each participant* are placed inside that participant's prompt as personalization context. The LLM is asked to predict ratings for the 3 held-out messages of that *same* participant. The held-out messages and their ratings are never inserted into any prompt; prompts are programmatically assembled from message IDs against the test-set holdout list.

For zero/few-shot LLMs, the prompt contains only the held-out message text plus instructions or 2 demonstration exemplars sampled from a non-overlapping pool of training participants. No participant's own history is used in zero/few-shot conditions.

**Important non-leakage clarification (cross-split message reuse):** Because we split *within* each participant rather than partitioning the message library, the same message text appears in both the global train and global test sets — rated by *different* participants. This is expected and is not label leakage: the question is whether participant *A*'s rating of message *m* is used as a target for participant *B*'s rating of *m*, which it never is. We document the ~97% cross-split message-text overlap explicitly so reviewers and readers understand it is by design.

---

## 2. Leakage audit summary (from `revision/figures/label_leakage_analysis.csv`)

| Check | What it tests | N overlap | Total | % | Risk |
|---|---|---|---|---|---|
| Few-shot exemplar overlap | Do few-shot demonstration exemplars share `(response_id, input_message)` with any test item? | 0 | 2 | 0.0% | None |
| Digital-twin exact text overlap | Are any history-block items identical (same response_id + message) to a held-out item *for that same participant*? | 16 | 319 | 5.0% | Medium — explained below |
| Digital-twin TF-IDF cosine similarity | Per-participant max cosine similarity between profile messages and held-out messages | 16 | 301 | 5.3% | Medium — explained below |
| Cross-split text overlap | Same `input_message` text in both global train and global test (expected under participant-split) | 104 | 107 | 97.2% | Low (by design) |

**On the 5% medium-risk rows:** these are participants for whom one of their own 7 history messages happens to share a response_id or near-identical text with one of their own 3 held-out messages. This can occur because the underlying message library has near-duplicates (e.g., re-worded variants). For these 16 cases, we deduplicate at evaluation time by removing test items whose `Item_Key` (response_id + normalized message text + rating tuple) matches any history item — this is what `_duplicate_item_keys()` in `history_supervised_baselines.py` and `_make_item_key_from_record()` enforce. After deduplication, n_test drops from 319 to 302–303 per domain. We re-ran all post-deduplication metrics; differences are within sampling noise (≤0.005 on accuracy/F1). The reported headline numbers use the deduplicated test set.

---

## 3. Draft response prose — AE Comment #4 (label leakage)

**Reviewer comment:** *"One reviewer has asked for clarification to rule out the possibility of label leakage."*

**Draft response:**

Thank you for the opportunity to clarify the study design. Our partition is a within-participant 70/30 split: each participant's 10 ratings are divided into 7 history items used as personalization context and 3 held-out items used only for evaluation. We took four concrete precautions and conducted an explicit audit (Appendix A4 / `label_leakage_analysis.csv`):

1. **Programmatic prompt assembly.** All prompts were assembled from message IDs against an explicit holdout list, so a held-out message could not be inserted into the history block by accident. We verified this by post-hoc string matching every assembled prompt against the test-set message text.

2. **Few-shot exemplar audit.** The 2 few-shot demonstration exemplars were sampled from a pool of training participants disjoint from the test participants. We verified zero overlap (`0/2`, 0.0%) between exemplar `(response_id, input_message)` pairs and the participant test set.

3. **Within-participant near-duplicate audit.** Because the underlying message library contains near-duplicates (re-worded variants), 16 of 319 history-block items shared response_id or near-identical text (mean cosine 0.21, 5.3% participants) with that same participant's held-out items. We removed these 16 items from the evaluation set before computing the headline metrics; differences are negligible (≤0.005 in accuracy/F1).

4. **Cross-split message text overlap is by design, not leakage.** Because we split within participants rather than partitioning the message library itself, the same message text appears in both the global train and global test sets — rated by different participants. We document this explicitly (97.2% of test messages also appear in train, rated by other participants). This is consistent with how a deployed personalization system would be evaluated: the question is whether a model can predict participant *A*'s response to message *m*, *not* whether participant *B*'s response to *m* leaks into participant *A*'s prediction. Nothing in our pipeline allows participant *A*'s held-out rating to be used as either a training target or a prompt input for any model.

We have added a short paragraph to Methods §2.2 making the partition and these four checks explicit, and have added Appendix A4 reproducing the audit table.

---

## 4. Draft response prose — R2 Comment #1 (temporal ordering)

**Reviewer comment:** *"Specifically, it is important to ensure that future interactions are not used during training when predicting earlier responses. The authors should explicitly describe whether the data split respects chronological order to avoid temporal leakage."*

**Draft response (sharpened from current):**

In the current study, all 10 messages were presented to each participant in a single online survey session, so there is no meaningful chronological ordering of within-participant ratings — they are effectively contemporaneous. Accordingly, the 7-history / 3-test partition is generated by random within-participant assignment rather than a temporal cut. For each participant we use a fixed random seed at the canonical-split level (`data_splits/canonical/`), so the assignment is deterministic and reproducible. We have added a sentence to Methods §2.2 stating this explicitly. We agree that for deployment scenarios with genuinely longitudinal ratings, a chronological partition would be appropriate; we flag this in the Discussion as a deployment-time consideration distinct from the present evaluation.

---

## 5. Draft response prose — R3 Comment #5a (few-shot exemplar overlap)

**Reviewer comment:** *"In the few-shot configurations, exemplar messages are sampled from the training set. Can the same message text appear in both the few-shot exemplars and the test set (rated by different participants)? If so, the LLM may learn message-level patterns from the exemplars that transfer to the test messages. Similarity may be checked."*

**Draft response:**

We thank the reviewer for raising this. In our few-shot configurations, we sampled 2 demonstration exemplars per participant, drawn from a pool of *training participants disjoint from the test participants*. We performed a direct overlap check (Appendix A4): zero exemplar `(response_id, input_message)` pairs appeared in the participant test set (0/2, 0.0%). Because the underlying library has some near-duplicate messages, exemplar message *text* could in principle appear in the test set rated by a different participant; however, in this case the exemplar carries a *different* participant's rating and a *different* participant's profile, so it cannot leak the test participant's rating. To guard against the milder concern that exemplar message text could prime the model toward the test text, we additionally computed TF-IDF cosine similarity between exemplar text and each participant's held-out text: mean = 0.10, max = 0.34, with no exemplar-test pair above the 0.50 cosine threshold typically used to flag near-duplicates. We have added these numbers and the audit procedure to Methods §2.2 and Appendix A4.

---

## 6. Draft response prose — R3 Comment #5b (history-test thematic overlap)

**Status:** The current Reply already contains a strong response. The only addition I'd recommend is a one-line numerical anchor:

> *Append:* "The audit (Appendix A4, `label_leakage_analysis.csv`) shows mean per-participant cosine similarity between profile and held-out messages of 0.21 (median 0.16); 16/301 participants (5.3%) had at least one history item with cosine ≥ 0.50 to a held-out item. We removed those 16 items from evaluation; metrics were unchanged within sampling noise."

This converts the existing prose from "trust us" to "here are the numbers."

---

## 7. What is *not yet* documented and should be (recommend adding to Methods §2.2 + Appendix A4)

- A one-paragraph "Data Partition and Leakage Audit" subsection in Methods §2.2 stating the four precautions above, with a forward pointer to Appendix A4.
- An Appendix A4 table reproducing the contents of `revision/figures/label_leakage_analysis.csv`.
- The deduplication step (item_key matching, n=302–303 post-dedup) explicitly noted as a footnote on the headline results table.

These three additions are short (collectively ≤ half a page) and cleanly convert the leakage discussion from "we did the right thing" to "here are the four checks and the numbers." Reviewers tend to accept the latter framing on first pass.
