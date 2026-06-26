# Macro-F1 — apples-to-apples by k_train

All values are sourced from the ensemble files. Within a fixed 
`(k_train, feature_set)` block, all method rows use the same test 
rows. The table below picks the best feature set separately for 
each method × domain × metric, so the exact feature set and N are 
reported in the details block. Use `clean_results.md` for the 
strict fixed-feature apples-to-apples read.

## k_train = 1  (test N range = 2670-2704)

| Method | Content | Coping | Quitting | Mean |
|---|---|---|---|---|
| Supervised RF | 0.336 | 0.369 | 0.364 | 0.356 |
| LLM-PP (Grok-4-Fast) | 0.327 | 0.349 | 0.357 | 0.345 |

<details><summary>Best feature set used per cell</summary>

| Method | best feature set (Content) | best feature set (Coping) | best feature set (Quitting) | N (Content) | N (Coping) | N (Quitting) |
|---|---|---|---|---|---|---|
| Supervised RF | Demographics | Demo+History | Demographics | 2704 | 2704 | 2704 |
| LLM-PP (Grok-4-Fast) | Demographics | Embedding | Embedding | 2704 | 2670 | 2670 |

</details>

## k_train = 3  (test N = 2102)

| Method | Content | Coping | Quitting | Mean |
|---|---|---|---|---|
| Supervised RF | 0.354 | 0.402 | 0.392 | 0.383 |
| LLM-PP (Grok-4-Fast) | 0.334 | 0.361 | 0.363 | 0.353 |

<details><summary>Best feature set used per cell</summary>

| Method | best feature set (Content) | best feature set (Coping) | best feature set (Quitting) | N (Content) | N (Coping) | N (Quitting) |
|---|---|---|---|---|---|---|
| Supervised RF | Demographics | Demo+History | Demo+History | 2102 | 2102 | 2102 |
| LLM-PP (Grok-4-Fast) | Demographics | Demographics | Demographics | 2102 | 2102 | 2102 |

</details>

## k_train = 7  (test N range = 883-898)

| Method | Content | Coping | Quitting | Mean |
|---|---|---|---|---|
| Supervised RF | 0.393 | 0.407 | 0.408 | 0.403 |
| LLM-PP (Grok-4-Fast) | 0.337 | 0.370 | 0.406 | 0.371 |

<details><summary>Best feature set used per cell</summary>

| Method | best feature set (Content) | best feature set (Coping) | best feature set (Quitting) | N (Content) | N (Coping) | N (Quitting) |
|---|---|---|---|---|---|---|
| Supervised RF | Demographics | Demographics | Demographics | 898 | 898 | 898 |
| LLM-PP (Grok-4-Fast) | Demographics | Embedding | Demographics | 898 | 883 | 898 |

</details>

---
## Generic LLM (separate test set — DO NOT directly compare)

Generic LLM zero-/few-shot was evaluated on the dt10 message subset (much smaller test set than the ensemble grid above). 
Reported here for context only.

| k_train | variant | Content | Coping | Quitting | Mean | N (any domain) |
|---|---|---|---|---|---|---|
| 0 | Zero-shot | 0.227 | 0.195 | 0.187 | 0.203 | 322 |
| 0 | Few-shot | 0.272 | 0.209 | 0.280 | 0.254 | 322 |
| 1 | Zero-shot | 0.234 | 0.201 | 0.185 | 0.207 | 293 |
| 1 | Few-shot | 0.284 | 0.220 | 0.293 | 0.266 | 293 |
| 3 | Zero-shot | 0.218 | 0.185 | 0.189 | 0.197 | 228 |
| 3 | Few-shot | 0.257 | 0.211 | 0.308 | 0.259 | 228 |
| 5 | Zero-shot | 0.195 | 0.192 | 0.187 | 0.191 | 153 |
| 5 | Few-shot | 0.235 | 0.189 | 0.285 | 0.236 | 153 |
| 7 | Zero-shot | 0.201 | 0.206 | 0.180 | 0.196 | 87 |
| 7 | Few-shot | 0.299 | 0.226 | 0.296 | 0.274 | 87 |
