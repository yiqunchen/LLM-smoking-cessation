# bars_all_methods Source Manifest

- Split: canonical PP `7030` source.
- Duplicate policy: known 70/30 history/test duplicate items are removed from every plotted LLM method.
- Main supervised comparator retained: LR/RF with demographics only.
- Revision comparator: LR/RF with `Demographics + History + Message Embedding`.
- Demographics include age, gender, race/ethnicity, Hispanic/Latino status, sexual orientation, education, and household income.
- Demographics exclude smoking behavior, quit-readiness/support, and psychosocial items from the broader metadata extractor.
- History uses the single `avg_history_overall` prior-rating feature; message embedding uses the OpenAI text embedding matched to each message.
- No rows are borrowed from participant 70/30, 50/50, archived embedding-only runs, or mixed split files.

Missing consistent-split sources, omitted rather than backfilled:

```csv
Model,Method,Source_File,Duplicate_Filter
GPT-4o-mini,Zero-shot (w/ prob),results_manuscript_gpt-4o-mini/generic_llm_5_continuous_dt7030.json,known_dt7030_duplicates
GPT-5,Zero-shot (w/ prob),results_manuscript_gpt-5/generic_llm_5_continuous_dt7030.json,known_dt7030_duplicates
DeepSeek-R1,Zero-shot (w/ prob),results_manuscript_deepseek_deepseek-r1-0528/generic_llm_5_continuous_dt7030.json,known_dt7030_duplicates
Grok-4-Fast,Zero-shot (w/ prob),results_manuscript_x-ai_grok-4-fast/generic_llm_5_continuous_dt7030.json,known_dt7030_duplicates
Gemini-2.5-Pro,Zero-shot (w/ prob),results_manuscript_gemini-2.5-pro/generic_llm_5_continuous_dt7030.json,known_dt7030_duplicates
```

