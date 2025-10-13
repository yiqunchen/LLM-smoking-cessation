## Title

Brief, action‑oriented title (e.g., feat: add calibration plot builder)

## Summary

- Why is this change needed?
- What problem does it solve for the LLM Smoking Cessation project?

## Changes

- Bullet list of key changes
- Note any breaking changes or follow‑ups

## How to Test

Provide exact commands and expected artifacts:

```bash
pip install -r requirements.txt
export CHEN_OPENAI_API_KEY=...  # required for model calls
python analysis-script/preprocess_data.py
python analysis-script/main_eval.py --mode text-only --model gpt-4o --prompt-config few-shot-feature-select-balanced --sample-size 10
```

## Data / Environment

- Inputs required (e.g., files in `data/`)
- Secrets/config (e.g., `CHEN_OPENAI_API_KEY`)

## Screenshots / Plots (optional)

Attach key plots or console output to verify results.

## Checklist

- [ ] Scoped and reviewable
- [ ] No secrets or PII committed
- [ ] Docs updated (`README.md`, `AGENTS.md` if needed)
- [ ] Repro steps provided

