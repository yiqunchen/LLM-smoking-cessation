# QWK — apples-to-apples by k_train

All values are sourced from the ensemble files. Within a fixed 
`(k_train, feature_set)` block, all method rows use the same test 
rows. The table below picks the best feature set separately for 
each method × domain × metric, so the exact feature set and N are 
reported in the details block. Use `clean_results.md` for the 
strict fixed-feature apples-to-apples read.

## k_train = 1  (test N range = 2670-2704)

| Method | Content | Coping | Quitting | Mean |
|---|---|---|---|---|
| Supervised RF | 0.350 | 0.435 | 0.470 | 0.418 |
| LLM-PP (Grok-4-Fast) | 0.350 | 0.428 | 0.476 | 0.418 |

<details><summary>Best feature set used per cell</summary>

| Method | best feature set (Content) | best feature set (Coping) | best feature set (Quitting) | N (Content) | N (Coping) | N (Quitting) |
|---|---|---|---|---|---|---|
| Supervised RF | Demo+History | Demographics | Demographics | 2704 | 2704 | 2704 |
| LLM-PP (Grok-4-Fast) | Demographics | Embedding | Embedding | 2704 | 2670 | 2670 |

</details>

## k_train = 3  (test N = 2102)

| Method | Content | Coping | Quitting | Mean |
|---|---|---|---|---|
| Supervised RF | 0.443 | 0.524 | 0.556 | 0.508 |
| LLM-PP (Grok-4-Fast) | 0.381 | 0.475 | 0.516 | 0.457 |

<details><summary>Best feature set used per cell</summary>

| Method | best feature set (Content) | best feature set (Coping) | best feature set (Quitting) | N (Content) | N (Coping) | N (Quitting) |
|---|---|---|---|---|---|---|
| Supervised RF | Demo+History | Avg-History | Demo+History | 2102 | 2102 | 2102 |
| LLM-PP (Grok-4-Fast) | Demographics | Demographics | Demographics | 2102 | 2102 | 2102 |

</details>

## k_train = 7  (test N range = 883-898)

| Method | Content | Coping | Quitting | Mean |
|---|---|---|---|---|
| Supervised RF | 0.479 | 0.566 | 0.617 | 0.554 |
| LLM-PP (Grok-4-Fast) | 0.403 | 0.508 | 0.560 | 0.490 |

<details><summary>Best feature set used per cell</summary>

| Method | best feature set (Content) | best feature set (Coping) | best feature set (Quitting) | N (Content) | N (Coping) | N (Quitting) |
|---|---|---|---|---|---|---|
| Supervised RF | Avg-History | Avg-History | Demo+History | 898 | 898 | 898 |
| LLM-PP (Grok-4-Fast) | Embedding | Embedding | Embedding | 883 | 883 | 883 |

</details>

---
## Generic LLM (separate test set — DO NOT directly compare)

Generic LLM zero-/few-shot was evaluated on the dt10 message subset (much smaller test set than the ensemble grid above). 
Reported here for context only.

| k_train | variant | Content | Coping | Quitting | Mean | N (any domain) |
|---|---|---|---|---|---|---|
| 0 | Zero-shot | 0.121 | 0.107 | 0.099 | 0.109 | 322 |
| 0 | Few-shot | 0.168 | 0.134 | 0.234 | 0.179 | 322 |
| 1 | Zero-shot | 0.129 | 0.117 | 0.105 | 0.117 | 293 |
| 1 | Few-shot | 0.185 | 0.150 | 0.246 | 0.194 | 293 |
| 3 | Zero-shot | 0.097 | 0.069 | 0.092 | 0.086 | 228 |
| 3 | Few-shot | 0.187 | 0.123 | 0.234 | 0.181 | 228 |
| 5 | Zero-shot | 0.178 | 0.161 | 0.137 | 0.159 | 153 |
| 5 | Few-shot | 0.172 | 0.132 | 0.253 | 0.186 | 153 |
| 7 | Zero-shot | 0.079 | 0.086 | 0.022 | 0.062 | 87 |
| 7 | Few-shot | 0.150 | 0.151 | 0.200 | 0.167 | 87 |
