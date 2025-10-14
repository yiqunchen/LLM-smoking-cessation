# Digital Twin Simulation: Smoking‑Message Reward Modeling

This folder provides a plug‑and‑play scaffold to apply a Digital‑Twin style simulation (inspired by the referenced paper) to your smoking‑cessation message setting. It focuses on:

- Persona building (per participant JSON + optional natural‑language persona)
- Prompt templates for per‑message “New Survey Question” simulation (4 dimensions)
- Evaluation: direct 5‑class accuracy vs. a test–retest ceiling and per‑participant ranking quality (pairwise/Kendall‑τ)
- Residual fusion: separate participant baseline (twin) from message residuals learned from embeddings

## Folder Structure

- `persona_schema.json` — Canonical JSON schema for a participant persona
- `build_personas.py` — Creates per‑participant persona JSONs from `data/processed_llm_data.json`
- `prompt_templates.md` — System/instruction templates for 4 dimensions (content/design/coping/quitting)
- `make_question_items.py` — Generates JSONL prompts per (participant, message, dimension)
- `evaluate_twins.py` — Computes direct 5‑class metrics and ranking metrics (plus ceiling normalization)
- `residual_model.py` — Learns message residuals on top of per‑participant baseline using embeddings

Outputs (created at run time):
- `personas/` — One JSON persona per participant + an aggregated `personas.json`
- `prompts/` — JSONL prompt files per domain (for LLM calls outside this repo)
- `eval/` — Evaluation artifacts (`twin_eval_summary.md`, CSVs)

## Quick Start

1) Build personas

```
python digital-twin/build_personas.py \
  --input data/processed_llm_data.json \
  --out-dir digital-twin/personas
```

2) Generate prompts (questions) per dimension

```
python digital-twin/make_question_items.py \
  --personas-dir digital-twin/personas \
  --data-json data/processed_llm_data.json \
  --out-dir digital-twin/prompts
```

This writes JSONL files you can feed to an LLM client. Each line contains:
- persona_json (string), message_text, image_tags (if available), question (1–5 options), dimension, participant_id, input_message.

3) Run twins (outside this repo)

Call your LLM with the JSONL prompts to produce predicted 1–5 labels per dimension. Save a CSV:
`digital-twin/predictions.csv` with columns: `participant_id,input_message,dimension,predicted` (1–5).

4) Evaluate twins

```
python digital-twin/evaluate_twins.py \
  --pred-csv digital-twin/predictions.csv \
  --data-json data/processed_llm_data.json \
  --out-dir digital-twin/eval
```

This computes:
- Direct 5‑class metrics: Accuracy, Cohen’s κ, Spearman’s ρ (per dimension and overall)
- Test–retest ceiling (when repeated ratings exist) and normalized score = twin / ceiling
- Ranking metrics: per‑participant pairwise accuracy and Kendall‑τ

5) Residual fusion (optional)

```
python digital-twin/residual_model.py \
  --data-json data/processed_llm_data.json \
  --emb-csv data/message_embeddings.csv \
  --out-dir digital-twin/eval
```

Learns a per‑participant baseline (twin) and a message residual model on embeddings (PCA‑reduced) to predict deviations.

## Notes
- Design is a negative control without vision features; expect limited gains there.
- If you lack exact test–retest repeats, the evaluator reports N/A for ceiling; you can optionally estimate split‑half reliability as a proxy.
- All scripts are self‑contained and do not call external APIs; you’ll run the LLM client separately using the generated JSONL prompts.

