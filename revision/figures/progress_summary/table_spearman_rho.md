# Within-participant Spearman rho — apples-to-apples by k_train

All values are sourced from the ensemble files. Within a fixed 
`(k_train, feature_set)` block, all method rows use the same test 
rows. The table below picks the best feature set separately for 
each method × domain × metric, so the exact feature set and N are 
reported in the details block. Use `clean_results.md` for the 
strict fixed-feature apples-to-apples read.

## k_train = 1  (test N = 2670)

| Method | Content | Coping | Quitting | Mean |
|---|---|---|---|---|
| Supervised RF | -0.006 | 0.015 | -0.003 | 0.002 |
| LLM-PP (Grok-4-Fast) | 0.056 | 0.054 | 0.076 | 0.062 |

<details><summary>Best feature set used per cell</summary>

| Method | best feature set (Content) | best feature set (Coping) | best feature set (Quitting) | N (Content) | N (Coping) | N (Quitting) |
|---|---|---|---|---|---|---|
| Supervised RF | Embedding+Demo | Embedding+Demo | Embedding | 2670 | 2670 | 2670 |
| LLM-PP (Grok-4-Fast) | Embedding | Embedding | Embedding | 2670 | 2670 | 2670 |

</details>

## k_train = 3  (test N range = 2082-2102)

| Method | Content | Coping | Quitting | Mean |
|---|---|---|---|---|
| Supervised RF | 0.028 | 0.074 | 0.045 | 0.049 |
| LLM-PP (Grok-4-Fast) | 0.033 | 0.016 | -0.004 | 0.015 |

<details><summary>Best feature set used per cell</summary>

| Method | best feature set (Content) | best feature set (Coping) | best feature set (Quitting) | N (Content) | N (Coping) | N (Quitting) |
|---|---|---|---|---|---|---|
| Supervised RF | Embedding | Demographics + History + Message Embedding | Demographics + History + Message Embedding | 2082 | 2082 | 2082 |
| LLM-PP (Grok-4-Fast) | Demographics | Demographics | Demographics | 2102 | 2102 | 2102 |

</details>

## k_train = 7  (test N = 883)

| Method | Content | Coping | Quitting | Mean |
|---|---|---|---|---|
| Supervised RF | -0.052 | 0.051 | -0.011 | -0.004 |
| LLM-PP (Grok-4-Fast) | 0.061 | 0.060 | 0.092 | 0.071 |

<details><summary>Best feature set used per cell</summary>

| Method | best feature set (Content) | best feature set (Coping) | best feature set (Quitting) | N (Content) | N (Coping) | N (Quitting) |
|---|---|---|---|---|---|---|
| Supervised RF | Embedding+Demo | Demographics + History + Message Embedding | Demographics + History + Message Embedding | 883 | 883 | 883 |
| LLM-PP (Grok-4-Fast) | Embedding | Embedding | Embedding | 883 | 883 | 883 |

</details>

---
## Generic LLM (separate test set — DO NOT directly compare)

Generic LLM zero-/few-shot was evaluated on the dt10 message subset (much smaller test set than the ensemble grid above). 
Reported here for context only.

Generic LLM Spearman rows are omitted here because the aggregate dt10 CSV does not carry the valid-participant counts needed to interpret tied/degenerate rank correlations. Use the dedicated `spearman_rank_summary.csv` artifact for that separate analysis.
