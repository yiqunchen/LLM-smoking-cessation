# Working in this repository

1. **dt10 only.** Every result, table, and figure uses the splits in
   `data/splits/canonical/` (`docs/DATA_POLICY.md`). The retired 70/30-era
   splits and results exist only in gitignored `archive/`; never use them.
2. **Three outcomes**: content, coping, quitting. The raw `design` rating is
   never analysed or reported.
3. **No placeholders.** Never fabricate or ship mock, preview, or partial
   results as complete. If something is missing, say so and give the rerun
   command.
4. **Never delete result data.** Result JSONs are expensive API runs. Move
   things to `archive/` if they must leave the tree.
5. **Validate before analysing**: `analysis/verify_prompt_ablations.py` and
   `analysis/audit_prompt_ablations.py` for prompt ablations;
   `analysis/verify_dt10_artifacts.py` for the main dt10 outputs.

Conventions

- `uv sync`; run with `.venv/bin/python` (Python 3.12). All LLM calls go
  through OpenRouter (`OPENROUTER_API_KEY`).
- Long runs go in tmux and are checkpointed; `scripts/run_prompt_ablations.sh`
  resumes and skips completed conditions. Result JSONs are `{row_index:
  record}` keyed by canonical test-row index; `manifest_dt10_k7.json` records
  split SHA-256s and per-condition run settings.
- Figures: bold Helvetica (Arial, DejaVu Sans fallbacks), white background,
  light gray grid, PNG + PDF. Model colors: GPT-4o-mini `#0173B2`, GPT-5
  `#DE8F05`, DeepSeek-R1 `#029E73`, Grok `#CC78BC`, Gemini-2.5-Pro `#CA9161`,
  Logistic Regression `#E02020`, Random Forest `#7F7F7F`; reference gray
  `#4D4D4D`.
- Commit result data with the code that produced it.
