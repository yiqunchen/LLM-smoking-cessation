# Canonical dt10 Evaluation Splits

**This directory is the sole source of truth for personalized-prompt test
evaluation.** Do not use the retired `digital_twin_7030` 323-row pool.

All participant-level reviewer analyses must use the full ten-message,
within-participant splits built with seed `202509`.

| Evaluation | Training histories | Held-out rows | Files |
| --- | ---: | ---: | --- |
| Primary reviewer evaluation | 7 per participant | 898 | `train_dt10_k7.json`, `test_dt10_k7.json`, `metadata_dt10_k7.json` |
| Learning-curve sensitivity | 1, 3, or 5 per participant | split-specific | corresponding `train_dt10_k*.json`, `test_dt10_k*.json`, `metadata_dt10_k*.json` |

The k=7 split has 2,107 training rows and 898 held-out rows across 301
participants. Five participants have nine valid ratings; therefore the full
analysis dataset contains 3,005 usable rows, not 3,010.

The archival raw files are retained outside Git under `archive/data/`. They are
needed only to rebuild a split, not to run a prompt evaluation when both dt10
JSON files are supplied with `--data-file` and `--train-file`.
