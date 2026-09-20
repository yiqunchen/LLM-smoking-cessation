# Cross-method progress summary

**Important caveat.** Earlier versions of this summary mixed 
rows from different files at the same k_train. Those files use 
DIFFERENT test sets at the same k_train (Generic LLM ≈ 300, 
LLM-PP ≈ 800, RF/Ensemble ≈ 2700 at k=1), so the apples-to-
apples comparison was broken. See `test_set_audit.md`.

This version sources the comparison only from the ensemble 
files (`lc_dt10_ensemble_k1.csv`, `lc_dt10_ensemble_k3.csv`, 
and `lc_dt10_ensemble_k7.csv`), where all method rows are 
evaluated on the same test set inside each fixed `(k_train, 
feature_set)` block. Generic LLM is reported separately with 
its own (smaller) test set.

## Clean updated read
- [`clean_results.md`](clean_results.md) — concise messaging, 
strict fixed-feature AI-vs-ML summary, and message-selection 
headline table.

## Tables (apples-to-apples, one panel per k_train)
- Accuracy: [`table_accuracy.md`](table_accuracy.md) (CSV: `table_accuracy.csv`)
- Macro-F1: [`table_macro_f1.md`](table_macro_f1.md) (CSV: `table_macro_f1.csv`)
- QWK: [`table_qwk.md`](table_qwk.md) (CSV: `table_qwk.csv`)
- Within-participant Spearman rho: [`table_spearman_rho.md`](table_spearman_rho.md) (CSV: `table_spearman_rho.csv`)

## Mean across domains, by k_train
- [`sweep_by_k_train.md`](sweep_by_k_train.md)

## Learning curves
- `learning_curves.png` / `learning_curves.pdf` — apples-to-
apples scaling at k_train in {1, 3, 7}.

## Clean plots
- `ai_vs_ml_strict_snapshot.png` / `.pdf` — latest strict 
fixed-feature AI-vs-ML snapshot.
- `ai_vs_ml_strict_scaling.png` / `.pdf` — strict fixed-feature 
scaling across k.
- `message_selection_gain.png` / `.pdf` — supervised RF and 
LLM-PP message-selection gain over random.

## Test-set audit
- [`test_set_audit.md`](test_set_audit.md) — N per source × k_train.

## Notes
- Each k_train evaluates on a different shared test set (the 
intersection of users with both an RF and an LLM-PP prediction 
at that history depth: N≈2700 at k=1, N≈2100 at k=3, N≈900 at 
k=7). Within a fixed feature-set block, all method rows share 
that test set; absolute values across k columns are not strictly 
comparable, so read the curves as the relative ranking of 
methods at each k.
- If you want Generic LLM in the apples-to-apples grid, score 
the Generic LLM on the ensemble file's test set rather than 
the smaller dt10 set.