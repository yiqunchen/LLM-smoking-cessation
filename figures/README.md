# Figures

All figures use the dt10 splits (`docs/DATA_POLICY.md`); each directory holds
the PNG/PDF files and the CSV tables they were drawn from.

| Directory | Content | Script |
|---|---|---|
| `figure2/` | Main dt10 comparison: supervised RF vs LLM personalized prompting | `analysis/make_figure2_dt10.py` |
| `figure4/` | Message-selection quality and gain at k = 7 | `analysis/make_figure4_supporting_dt10.py` |
| `progress_summary/` | Strict shared-row summaries, learning curves over k, k = 7 diagnostics | `analysis/build_progress_summary.py` |
| `prompt_ablations/` | Prompt-component ablations: metric grids, rating distributions, pairwise comparisons, bootstrap tables | `analysis/plot_prompt_ablations.py`, `analysis/bootstrap_prompt_ablations.py` |
