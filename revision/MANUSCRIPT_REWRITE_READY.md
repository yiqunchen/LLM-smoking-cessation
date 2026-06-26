# Manuscript Rewrite Ready

Date: 2026-06-03 (numbers/claims updated 2026-06-19)

> **SUPERSEDED FRAMING UPDATED (2026-06-19).** The author has locked the final
> design: **anchor methods (RF-anchor / LLM-anchor) are DROPPED entirely**; the
> comparison is now exactly **two methods — supervised RF vs LLM-PP** (persona-
> conditioned LLM). Everything uses **one split, `dt10`** (within-participant; 301
> participants; **N = 898** held-out rows at `k_train = 7`) and **one supervised
> feature block for the headline, `Demographics`**. The numbers and tables below
> have been rewritten accordingly. The single source of truth is
> `revision/SPLIT_DEFINITION_AND_UNIFIED_RESULTS.md` (companions:
> `why_new_numbers_higher_reviewer_explanation.md`, `reviewer_clarity_summary.md`).
> Any earlier "+12pp / +13pp", anchor-method, or mixed-block QWK figures are void.

## Status

The revision package is ready for manual manuscript rewriting. Use this file as
the final source-of-truth checklist before editing the Word document.

## Core Claim To Use

On the within-participant `dt10` benchmark, the two compared methods are
**supervised RF** and **LLM-PP**. In strict shared-row comparisons (same 898
held-out rows), the supervised RF is the strongest aggregate classifier
(QWK 0.527), and LLM-PP is competitive — especially at low history / cold start
and for coping/quitting message selection. The persona-conditioned LLM thus
contributes complementary value (message selection in the helpfulness domains,
competitiveness with little history) rather than uniform aggregate superiority.

**Critical caveat (state wherever RF/demographics looks strong):** `dt10` is
within-participant by design — all 301 participants appear in BOTH train and test.
Demographics (7 attributes → ~46 one-hot columns) nearly uniquely identify a
participant, so RF effectively predicts each person's own modal prior rating. A
trivial "predict the participant's most-frequent prior rating" baseline reproduces
RF accuracy (Content .53, Coping .45, Quitting .47); on a strict unseen-participant
split RF falls to ~0.35. So supervised RF is a strong **participant-calibrated
reference**, NOT evidence that demographics predict PME.

Do not claim that LLM-PP methods broadly outperform supervised baselines.
Do not claim that historical ratings are causally or universally most important.
Do not claim that these within-participant numbers represent generalization to
new individuals.

## Final Strict `k_train=7` Numbers

All rows use the single **Demographics** feature block on the `dt10` split, same
898 held-out rows for both methods.

| Endpoint | Feature block | N | Supervised RF | LLM-PP |
|---|---|---:|---:|---:|
| Accuracy | Demographics | 898 | 0.474 | 0.462 |
| Macro-F1 | Demographics | 898 | 0.403 | 0.370 |
| QWK | Demographics | 898 | 0.527 | 0.488 |
| Within-participant Spearman rho | Demographics | 898 | NA (undefined) | 0.058 |

Reading: RF is the strongest aggregate classifier; LLM-PP is close behind. RF's
within-participant Spearman is undefined because it is message-blind and assigns
one tied prediction per participant; LLM-PP's is small but defined (~0.06). The
QWK here is the single-Demographics-block value (0.527 / 0.488) — it replaces the
old 0.547 / 0.555 figures, which mixed in the Avg-History block.

## History-Length Sensitivity

Use these values (Demographics block, `dt10`) when describing `k_train=1,3,7`
learning curves:

| Metric | Method | k=1 | k=3 | k=7 |
|---|---|---:|---:|---:|
| Accuracy | Supervised RF | 0.420 | 0.448 | 0.474 |
| Accuracy | LLM-PP | 0.417 | 0.438 | 0.462 |
| QWK | Supervised RF | 0.418 | 0.447 | 0.527 |
| QWK | LLM-PP | 0.417 | 0.457 | 0.488 |

LLM-PP ties RF on QWK at k=1 (0.417 vs 0.418) and slightly exceeds it at k=3
(0.457 vs 0.447); RF pulls ahead at k=7. RF's edge grows with history because that
edge *is* history-memorization; LLM-PP is most competitive at low history / cold
start, which is the realistic deployment regime.

These columns use the shared-row ensemble files at each `k_train`; row counts
change across `k_train`, so treat them as history-length sensitivity evidence,
not a single fixed-cohort longitudinal curve.

## Message Selection

The supporting method-level benchmark uses the fixed
`Demographics + History + Message Embedding` feature block across all domains.
At `K=5`, the largest gain over random is:

| Domain | Best method | Gain over random (best / other) |
|---|---|---:|
| Content | Supervised RF | 0.389 (RF) vs 0.184 (LLM-PP) |
| Coping | LLM-PP | 0.570 (LLM-PP) vs 0.245 (RF) |
| Quitting | LLM-PP | 0.540 (LLM-PP) vs 0.123 (RF) |

At `K=10` for Quitting, LLM-PP remains highest with gain `0.521`. Pattern: message
*quality* (Content) is best handled by the person-baseline RF; message
*helpfulness* (Coping, Quitting) is where LLM-PP earns its keep, selecting far
better messages than RF (~0.55 vs ~0.18 gain over random).

## Figure Files For Word

- Figure 1: `figures/figure1/llm-message-paper-figure1.pdf` or `.png`
- Figure 2 main (RF vs LLM-PP, dt10, N=898, Demographics block):
  `figures/figure2/figure2_main_dt10_rf_vs_llmpp.{png,pdf}`
- Figure 2 context (generic zero/few-shot, N≈87, labeled different coverage):
  `figures/figure2/figure2_context_generic_llm.{png,pdf}`
- Figure 2 provenance for every plotted value: `figures/figure2/figure2_dt10_source_table.csv`
- NOTE: the old `bars_all_methods_*` and `figure2_main_top3_vertical_300dpi`
  panels are the superseded four-method / 70-30 figures — do NOT use them in the
  main text; keep only as a clearly-labeled supplementary "older 70/30" panel if
  desired.
- Figure 3 assembled vertical: `figures/figure3/figure3_assembled_vertical_300dpi.png`
- Figure 3 individual panels: `figures/figure3/figure3_score_distributions_*.png`
- Figure 4 original slot: `figures/figure4/llm_selection_quality.png`
- Figure 4 supporting benchmark: `figures/figure4/supporting_message_selection_gain.png`
- Supplementary diagnostics: `figures/supplementary/`

## Rewrite Guides Now Aligned

- `revision/START_HERE_revision_edit_checklist.md`
- `revision/main_paper_docx_concrete_edits.md`
- `revision/Draft_final_main_text_before_after_edits.md`
- `revision/main_paper_claim_update_map.md`
- `revision/latest_revision_narrative.md`
- `revision/reviewer_response_technical_draft.md`
- `revision/reply_remaining_comments.md`

## Final QA

- Figure and supplementary manifests are current.
- No May-dated supplementary outputs remain in `figures/supplementary/`.
- Pairwise significance was refreshed after duplicate filtering:
  `23/120` Bonferroni-significant tests, with Content `1`, Coping `13`,
  Quitting `9`.
- No placeholder or fabricated revision rows were generated.
