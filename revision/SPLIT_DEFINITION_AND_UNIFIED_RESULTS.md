# Canonical Split Definition & Unified Single-Split Results

Date: 2026-06-19
Status: **THIS is the single source of truth for the split and the headline numbers.**
It supersedes scattered/mixed numbers in the older revision docs.

Decisions locked by the author (2026-06-18/19):
- **One split for everything: `dt10` (per-participant k-of-10).**
- **One supervised feature block for the headline: `Demographics`.**
- **Anchor methods (RF-anchor / LLM-anchor) are dropped** from the manuscript.
- Primary comparison is **Supervised RF vs LLM-PP**.

---

## 1. What the `dt10` split is (and why we use it)

Each of **301 participants** rated **10 messages** across three PME domains
(content, coping, quitting). For a chosen history length `k_train = k`:

- **k of each participant's 10 messages** go to **train / personalization history**.
- the remaining **(10 − k)** are **held out for evaluation**.

At the primary **k_train = 7**: 7 history messages, 3 held out per participant
→ **N = 898 held-out ratings** (301 × ~3, minus rows with missing ratings).

`dt10` was built (`analysis-script/build_dt10_splits.py`) specifically to **fix**
the older `digital_twin_7030` split. Per that script's own docstring, the old
canonical 1090/3070/.../9010 splits used only 3 of each participant's 10 messages
and "effectively span 0–2 prior messages per participant with floor-rounding
artifacts." `dt10` unifies the full 10-message dataset so each participant has
exactly k history messages and a clean (10 − k) held out.

**Verification (recomputed 2026-06-19):**
`test_dt10_k7.json` = 898 rows, 301 unique participants;
`train_dt10_k7.json` = 2107 rows, 301 unique participants;
**participant overlap = 301/301.**

## 2. Why "Demographics-only RF performs OK" (0.474) — the key caveat

`dt10` is **within-participant by design**: all 301 test participants also appear
in training (they must — the whole point is to use a participant's own history).

Therefore a **demographics-only** RF is *not* doing demographic generalization.
The demographic vector (age, gender, race/ethnicity, Hispanic/Latino status,
sexual orientation, education, household income) frequently **uniquely identifies
a participant**, so RF effectively learns "participant X rates around 3.5" from
that participant's own training rows and reuses it on their held-out rows.

> **This is a same-participant / participant-identification effect, not demographic
> generalization.** It is why a "demographics-only" model looks surprisingly strong.
> The manuscript MUST state this. (In the older `digital_twin_7030` split the same
> issue was quantified: 228/307 test rows mapped to a unique training participant.)

The honest framing: **Supervised RF is a strong *same-participant* reference**, not
an independent-participant demographic predictor.

## 3. Unified headline numbers — ALL from `dt10`, k_train = 7, Demographics block, N = 898

Recomputed directly from `revision/figures/lc_dt10_ensemble_k7.csv` (means across
content/coping/quitting):

| Method | Accuracy | Macro-F1 | QWK |
|---|---:|---:|---:|
| **Supervised RF** | **0.474** | **0.403** | **0.527** |
| **LLM-PP** | 0.462 | 0.370 | 0.488 |

Reading: RF is the strongest aggregate classifier; LLM-PP is close but does not
beat it on any of accuracy / macro-F1 / QWK at the headline block.

## 4. Learning curve — ALL from `dt10`, Demographics block (same split, all k)

Recomputed from `lc_dt10_ensemble_k{1,3,7}.csv`:

| | k=1 | k=3 | k=7 |
|---|---:|---:|---:|
| RF accuracy | 0.420 | 0.448 | 0.474 |
| LLM-PP accuracy | 0.417 | 0.438 | 0.462 |
| RF QWK | 0.418 | 0.447 | 0.527 |
| LLM-PP QWK | 0.417 | 0.457 | 0.488 |

> **These REPLACE the old draft line-307 numbers** (RF acc 0.420/0.459/0.477,
> QWK 0.418/0.508/0.554), which did NOT reproduce from any committed CSV and
> silently pulled QWK from the `Avg-History` block. Use the table above.

## 5. Generic zero-/few-shot LLMs — same split, but DIFFERENT coverage (report honestly)

Generic zero/few-shot were also run on `dt10` (`lc_dt10_generic_llm.csv`), but they
**do not cover all 898 rows** at k=7 — coverage is **N≈87 per cell** at k=7
(larger at k=0). So they are NOT on the same N as RF/LLM-PP and must be reported
as a **contextual** comparison with their own N stated, never merged into the
898-row headline table.

| Method (mean of 5 LLMs, k=7) | Accuracy | Macro-F1 | QWK | Coverage N |
|---|---:|---:|---:|---:|
| Zero-shot | 0.295 | 0.157 | 0.019 | ~87 |
| Few-shot | 0.326 | 0.220 | 0.063 | ~87 |

## 6. Figures — now unified on `dt10` (DONE 2026-06-19)

Figure 2 has been **regenerated on `dt10`** (author chose "regenerate", not relabel).
Generator: `analysis-script/make_figure2_dt10.py`. Outputs:

- `figures/figure2/figure2_main_dt10_rf_vs_llmpp.{png,pdf}` — MAIN panel:
  Supervised RF vs LLM-PP, Demographics block, **N = 898**, Content/Coping/Quitting
  on Accuracy / Macro-F1 / QWK. Same held-out rows for both methods.
- `figures/figure2/figure2_context_generic_llm.{png,pdf}` — CONTEXT panel:
  generic zero-/few-shot (mean of 5 LLMs), **N ≈ 87**, explicitly labeled as
  different coverage and NOT comparable to the 898-row panel.
- `figures/figure2/figure2_dt10_source_table.csv` — provenance for every plotted value.

The OLD `digital_twin_7030` Figure 2 panels (`bars_all_methods_*`, N≈306) are now
superseded for the main text. Keep them only if a clearly-labeled supplementary
"older 70/30 prompt-family" panel is desired; do not present them as the main split.

> Still TODO in the manuscript prose (`Draft_final_main_text_before_after_edits.md`):
> drop anchor methods, replace line-307 learning-curve numbers with §4 above,
> switch QWK to the single Demographics block (0.527, not 0.547/0.555),
> remove legacy "+12pp/+13pp" and high-Spearman (~0.27–0.43) claims, and add the
> §2 same-participant-proxy caveat.

## 7. Provenance (every number above is reproducible)

- Split structure: `data_splits/canonical/{train,test}_dt10_k{1,3,5,7}.json`
- Supervised + LLM-PP + (dropped) anchors: `revision/figures/lc_dt10_ensemble_k{1,3,7}.csv`
- Supervised-only all blocks: `revision/figures/lc_dt10_rf.csv`
- Generic zero/few-shot: `revision/figures/lc_dt10_generic_llm.csv`
- LLM digital-twin per model: `revision/figures/lc_dt10_llmdt.csv`
