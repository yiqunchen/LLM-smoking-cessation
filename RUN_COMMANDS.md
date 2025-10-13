# Commands to Run Full Pipeline for Each Model

## 1. GPT-4o-mini (OpenAI)
```bash
bash run_full_manuscript_pipeline.sh gpt-4o-mini
```

## 2. GPT-4o (OpenAI)
```bash
bash run_full_manuscript_pipeline.sh gpt-4o
```

## 3. DeepSeek-R1 (OpenRouter)
```bash
export OPENROUTER_API_KEY="your-key-here"
bash run_full_manuscript_pipeline.sh deepseek/deepseek-r1 openrouter
```

## 4. Grok-2 (OpenRouter - Free)
```bash
export OPENROUTER_API_KEY="your-key-here"
bash run_full_manuscript_pipeline.sh x-ai/grok-2-1212 openrouter
```

## 5. Gemini 2.0 Flash (Google)
```bash
export GEMINI_API_KEY="your-key-here"
bash run_full_manuscript_pipeline.sh gemini-2.0-flash-exp gemini
```

---

## What Each Command Does

Each command runs the **full manuscript pipeline** including:

1. ✅ **Generic LLM Methods** (6 configurations):
   - Zero-shot (all features)
   - Zero-shot (selected features)
   - Few-shot (all features)
   - Few-shot (selected features)
   - Continuous rating (natural language)
   - Continuous rating (with probabilities)

2. ✅ **Digital Twin Methods** (7 configurations):
   - Full features (70/30 split)
   - Selected features (70/30 split)
   - With feedback (70/30 split)
   - CBT/ACT-informed (10/90, 30/70, 70/30, 90/10 splits)

3. ✅ **Traditional ML Comparison**:
   - Random Forest on participant characteristics
   - Ensemble methods (voting, stacking)

4. ✅ **Comprehensive Analysis**:
   - 8 metrics: accuracy, accuracy±1, κ, τ, ρ, F1, precision, recall
   - Visualizations and summary tables

---

## Output Structure

Results saved to: `results_manuscript_<model_name>/`

Example for gpt-4o-mini:
```
results_manuscript_gpt-4o-mini/
├── generic_llm_1_zero_shot.json
├── generic_llm_2_zero_shot_select.json
├── generic_llm_3_few_shot.json
├── generic_llm_4_few_shot_select.json
├── generic_llm_5_continuous.json
├── generic_llm_6_continuous_select.json
├── digital_twin_1_full_7030.json
├── digital_twin_2_select_7030.json
├── digital_twin_3_feedback_7030.json
├── digital_twin_4_cbtact_1090.json
├── digital_twin_4_cbtact_3070.json
├── digital_twin_4_cbtact_7030.json
├── digital_twin_4_cbtact_9010.json
├── summary_table.csv
├── summary_table.md
├── summary_table_pivot.csv
├── method_comparison_all_metrics.png
├── checkpoints/  (for resuming interrupted runs)
├── comparisons/  (traditional ML vs LLM)
└── logs/  (full execution logs)
```

---

## Features

- ✅ **Checkpointing**: Resume from any interruption (Ctrl+C safe)
- ✅ **API Robustness**: 5 retries with exponential backoff
- ✅ **Progress Tracking**: tqdm shows completion status
- ✅ **Reproducible**: seed=202509, canonical 70/30 splits
- ✅ **Full Logging**: Everything logged for debugging

---

## Estimated Runtime

- **GPT-4o-mini**: ~15-20 minutes (fast, cheap)
- **GPT-4o**: ~30-40 minutes (slower, more expensive)
- **DeepSeek-R1**: ~20-30 minutes (depends on OpenRouter load)
- **Grok-2**: ~25-35 minutes (free tier may be slower)
- **Gemini 2.0 Flash**: ~15-25 minutes (fast, competitive pricing)

Total API calls per model: ~3,500-4,000 calls

