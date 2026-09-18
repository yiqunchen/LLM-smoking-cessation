# Revision figures: dt10-only

Every retained figure and source table in this directory is tied to the full
ten-message `dt10` evaluation. The primary test source is
`data_splits/canonical/test_dt10_k7.json` (898 held-out rows), with its matched
seven-message profile file `train_dt10_k7.json`.

Retained outputs:

- `progress_summary/`: strict shared-row dt10 summaries, learning curves, and
  k=7 diagnostics.
- `lc_dt10_*.csv`: learning-curve source data.
- `message_selection_*`: dt10 k=7 supporting message-selection analysis.

No `digital_twin_7030` results, figures, or cached tables are retained here.
