# Working in this repository

Read this before changing anything. `README.md` explains the project; this
file is the contract for agents and contributors.

## Ground rules

1. **Data policy (`docs/DATA_POLICY.md`)** – all personalized-prompt (PP)
   results, tables, and figures use the ten-message `dt10` splits under
   `data/splits/canonical/`. Primary analysis is `dt10_k7`. Never use, plot,
   or cite anything from the retired 70/30 family (`7030`, `1090`, `3070`,
   `5050`, `9010`, `participant_3070`). Those files live only in gitignored
   `archive/`.
2. **Three outcomes** – content, coping, quitting. The `design` rating is
   excluded from every analysis and must never appear in results, figures, or
   reports. (`design` remains a raw field inside the split JSONs and in
   `analysis/legacy/`; that is source data / provenance, not a result.)
3. **No placeholders** – never fabricate or ship mock rows, preview values, or
   partial results presented as complete. If something is missing, say so and
   give the exact rerun command.
4. **Never delete result data.** Result JSONs are expensive API runs. Move
   things into `archive/` if they must leave the active tree; do not `rm`.
5. **Don't edit `analysis/legacy/` or `scripts/legacy/`** except to fix an
   import path. They are frozen provenance.

## Layout

| Path | Purpose |
|---|---|
| `analysis/` | Active Python. One script per step; each has a docstring and `--help`. |
| `analysis/legacy/` | Frozen manuscript-era code. |
| `scripts/` | Shell entry points. `run_prompt_ablations.sh`, `build_prompt_ablation_package.sh`. |
| `data/splits/canonical/` | dt10 `k1/k3/k5/k7` train/test/metadata JSON (tracked). |
| `data/raw/` | Restricted participant-level inputs (gitignored except embeddings). |
| `results/prompt_ablations/<model>/` | Prompt-ablation predictions: 4 conditions x 898 rows + manifest (tracked). |
| `results/manuscript/` | Manuscript-era runs; only dt10 Grok PP files tracked. |
| `figures/` | All figures and their source CSVs. `figures/README.md` maps figure -> data -> script. |
| `reports/` | Generated Word report and ZIP package for the prompt ablations. |
| `docs/` | `DATA_POLICY.md`, `prompt_templates.md`, `revision/` (reviewer-response drafts). |
| `archive/` | Gitignored. Retired splits/results/code and handoff bundles. |
| `logs/` | Gitignored run logs, one per tmux session. |

## Conventions

- Python via `uv` (`uv sync`; run with `.venv/bin/python`). Python 3.12.
- All LLM calls go through OpenRouter (`OPENROUTER_API_KEY`).
- Long runs go in tmux (`rerun-<model>-<condition>`) and are checkpointed;
  re-running the launcher resumes and skips completed conditions.
- Result JSONs are `{row_index: record}` dictionaries keyed by the canonical
  test-row index; `manifest_dt10_k7.json` records split SHA-256s and the
  prompt spec. Validate with `analysis/verify_prompt_ablations.py` and
  `analysis/audit_prompt_ablations.py` before analysing.
- Figures: bold Helvetica (Arial, DejaVu Sans fallbacks), white backgrounds,
  colorblind-safe model colors — GPT-4o-mini `#0173B2`, GPT-5 `#DE8F05`,
  DeepSeek-R1 `#029E73`, Grok `#CC78BC`, Gemini-2.5-Pro `#CA9161`,
  Logistic Regression `#E02020`, Random Forest `#7F7F7F`; neutral gray
  `#4D4D4D` dashed reference lines, light grid. Export PNG + PDF.
- Commit result data together with the code that produced it; keep commit
  messages descriptive of what changed in the analysis, not of who asked.
