# Test-set size audit (max N per source × k_train)

These columns show the test-set size each method was evaluated on. 
When the numbers in a row of the comparison tables differ across 
*sources*, they are NOT on the same test set and should not be 
directly compared.

| source | k=0 | k=1 | k=3 | k=5 | k=7 |
|---|---|---|---|---|---|
| Ensemble k=1 (lc_dt10_ensemble_k1.csv) | — | 2704 | — | — | — |
| Ensemble k=3 (lc_dt10_ensemble_k3.csv) | — | — | 2102 | — | — |
| Ensemble k=7 (lc_dt10_ensemble_k7.csv) | — | — | — | — | 898 |
| Generic LLM (lc_dt10_generic_llm.csv) | 322 | 293 | 228 | 153 | 87 |
| LLM-PP     (lc_dt10_llmdt.csv) | 892 | 790 | 628 | 439 | 265 |
| RF         (lc_dt10_rf.csv) | — | 2704 | 2102 | 1500 | 898 |

**Implication.** The cross-method comparison tables in this folder 
are sourced ONLY from the ensemble files, where all method rows 
share a single test set within a fixed `(k_train, feature_set)` 
block. Generic LLM rows live in their own table because they are 
evaluated on a different (smaller) population.