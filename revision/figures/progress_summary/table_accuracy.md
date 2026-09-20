# Accuracy — apples-to-apples by k_train

All values are sourced from the ensemble files. Within a fixed 
`(k_train, feature_set)` block, all method rows use the same test 
rows. The table below picks the best feature set separately for 
each method × domain × metric, so the exact feature set and N are 
reported in the details block. Use `clean_results.md` for the 
strict fixed-feature apples-to-apples read.

## k_train = 1  (test N range = 2670-2704)

| Method | Content | Coping | Quitting | Mean |
|---|---|---|---|---|
| Supervised RF | 0.438 | 0.407 | 0.416 | 0.420 |
| LLM-PP (Grok-4-Fast) | 0.462 | 0.392 | 0.401 | 0.419 |

<details><summary>Best feature set used per cell</summary>

| Method | best feature set (Content) | best feature set (Coping) | best feature set (Quitting) | N (Content) | N (Coping) | N (Quitting) |
|---|---|---|---|---|---|---|
| Supervised RF | Demo+History | Demographics | Demographics | 2704 | 2704 | 2704 |
| LLM-PP (Grok-4-Fast) | Demographics | Embedding | Embedding | 2704 | 2670 | 2670 |

</details>

## k_train = 3  (test N range = 2082-2102)

| Method | Content | Coping | Quitting | Mean |
|---|---|---|---|---|
| Supervised RF | 0.500 | 0.442 | 0.434 | 0.459 |
| LLM-PP (Grok-4-Fast) | 0.486 | 0.414 | 0.414 | 0.438 |

<details><summary>Best feature set used per cell</summary>

| Method | best feature set (Content) | best feature set (Coping) | best feature set (Quitting) | N (Content) | N (Coping) | N (Quitting) |
|---|---|---|---|---|---|---|
| Supervised RF | Demographics | Demo+History | Demographics | 2102 | 2102 | 2102 |
| LLM-PP (Grok-4-Fast) | Embedding | Demographics | Demographics | 2082 | 2102 | 2102 |

</details>

## k_train = 7  (test N range = 883-898)

| Method | Content | Coping | Quitting | Mean |
|---|---|---|---|---|
| Supervised RF | 0.523 | 0.450 | 0.459 | 0.477 |
| LLM-PP (Grok-4-Fast) | 0.499 | 0.433 | 0.458 | 0.463 |

<details><summary>Best feature set used per cell</summary>

| Method | best feature set (Content) | best feature set (Coping) | best feature set (Quitting) | N (Content) | N (Coping) | N (Quitting) |
|---|---|---|---|---|---|---|
| Supervised RF | Demographics | Demographics | Avg-History | 898 | 898 | 898 |
| LLM-PP (Grok-4-Fast) | Demographics | Embedding | Embedding | 898 | 883 | 883 |

</details>

---
## Generic LLM (separate test set — DO NOT directly compare)

Generic LLM zero-/few-shot was evaluated on the dt10 message subset (much smaller test set than the ensemble grid above). 
Reported here for context only.

| k_train | variant | Content | Coping | Quitting | Mean | N (any domain) |
|---|---|---|---|---|---|---|
| 0 | Zero-shot | 0.398 | 0.320 | 0.311 | 0.343 | 322 |
| 0 | Few-shot | 0.398 | 0.323 | 0.360 | 0.360 | 322 |
| 1 | Zero-shot | 0.406 | 0.324 | 0.311 | 0.347 | 293 |
| 1 | Few-shot | 0.410 | 0.328 | 0.372 | 0.370 | 293 |
| 3 | Zero-shot | 0.408 | 0.303 | 0.325 | 0.345 | 228 |
| 3 | Few-shot | 0.408 | 0.307 | 0.373 | 0.363 | 228 |
| 5 | Zero-shot | 0.399 | 0.333 | 0.327 | 0.353 | 153 |
| 5 | Few-shot | 0.399 | 0.327 | 0.392 | 0.373 | 153 |
| 7 | Zero-shot | 0.425 | 0.356 | 0.345 | 0.375 | 87 |
| 7 | Few-shot | 0.402 | 0.322 | 0.391 | 0.372 | 87 |
