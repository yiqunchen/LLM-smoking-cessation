# Mean-across-domains by k_train (apples-to-apples)

Source: ensemble files. Each cell = mean over {Content, Coping, 
Quitting} of the per-domain best feature set, with N tracked in 
the metric-specific tables. Each k_train column corresponds to a 
different test set (different intersection of users with both RF 
and LLM-PP predictions at that history depth), so absolute values 
across columns are not strictly comparable — only the ranking of 
rows within a column is. For the strict feature-matched read, see 
`clean_results.md`.

## Accuracy

| Method | k=1 | k=3 | k=7 |
|---|---|---|---|
| Supervised RF | 0.420 | 0.459 | 0.477 |
| LLM-PP (Grok-4-Fast) | 0.419 | 0.438 | 0.463 |

## Macro-F1

| Method | k=1 | k=3 | k=7 |
|---|---|---|---|
| Supervised RF | 0.356 | 0.383 | 0.403 |
| LLM-PP (Grok-4-Fast) | 0.345 | 0.353 | 0.371 |

## QWK

| Method | k=1 | k=3 | k=7 |
|---|---|---|---|
| Supervised RF | 0.418 | 0.508 | 0.554 |
| LLM-PP (Grok-4-Fast) | 0.418 | 0.457 | 0.490 |

## Within-participant Spearman rho

| Method | k=1 | k=3 | k=7 |
|---|---|---|---|
| Supervised RF | 0.002 | 0.049 | -0.004 |
| LLM-PP (Grok-4-Fast) | 0.062 | 0.015 | 0.071 |
