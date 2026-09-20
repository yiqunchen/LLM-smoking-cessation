# dt10 source-of-truth policy

For every personalized-prompt (PP) result, table, and figure in this revision:

- Use `data/splits/canonical/train_dt10_k7.json` and
  `data/splits/canonical/test_dt10_k7.json` for the primary analysis.
- Use only the matching `dt10_k1`, `dt10_k3`, or `dt10_k5` files for declared
  learning-curve sensitivity analyses.
- Do not use, plot, or combine any `digital_twin_7030`, `dt7030`, or
  `*_7030.json` result. Those artifacts came from a three-message evaluation
  pool and do not represent a 30% holdout of the full ten-message dataset.

The k=7 primary evaluation contains 2,107 history rows (seven for each of 301
participants) and 898 held-out ratings. This is the only test source for the
reviewer-requested PP configurations.
