# LLM-Based Smoking Cessation Message Evaluation

Evaluating smoking cessation messages using Large Language Models (LLMs) to predict participant ratings across multiple dimensions.

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
conda activate research
source ~/.bash_profile  # Loads API keys
```

### 2. Run Complete E2E Pipeline (All 5 Models)
```bash
# Run full pipeline for all models (GPT-4o-mini, GPT-5, DeepSeek-R1, Grok-4-Fast, Gemini-2.5-Pro)
bash e2e_pipeline.sh
```

This will:
- Run all 6 Generic LLM methods + 1 PP method per model
- Generate comprehensive results and figures
- Output to `figures/` directory

### 3. Run Single Model Pipeline
```bash
# Run full pipeline for a single model
bash run_full_manuscript_pipeline.sh gpt-4o-mini openai 10
```

### 4. Run Hybrid RF+PP
```bash
# Run hybrid Random Forest + PP for all models
bash run_hybrid_rf_all_models.sh
```
---

## 🤝 Handoff Note: What is and isn't in this repo

This repository is **self-contained for the revision deliverable** but **not self-regenerating from scratch**. Read this before assuming a script runs end-to-end.

**Committed and ready to use:**
- Revision narrative, reviewer responses, and edit maps — `revision/*.md`
- Final revision graphs (PNG + PDF) — `revision/figures/`
- Figure-level underlying data (CSV) — `revision/figures/` (see `revision/figures/README.md` for the figure → data → script map)
- Raw plot-source inputs needed by the current figure scripts:
  - canonical plot splits under `data_splits/canonical/` for `digital_twin_7030`, `participant_7030`, and `dt10_k1/k3/k5/k7`
  - five-model 70/30 result JSONs used by Figure 2, Figure 4, demographic subgroup, pairwise-significance, and uncertainty plots
  - five hybrid RF+PP 70/30 result JSONs used by Figure 2
  - Grok dt10 result JSONs used by the strict AI-vs-ML/message-selection support
  - `archive/data/message_embeddings.pkl` and metadata used by the supervised embedding/history baselines
  - `archive_results/embedding_results_test30/results.json` used by legacy embedding-baseline plotting helpers
- All analysis/plotting code — `analysis-script/*.py`
- Graph reproducibility map — see [Graph Reproducibility](#graph-reproducibility) below

**NOT committed (gitignored — must be obtained or regenerated locally):**
- old/non-plot split variants, raw survey spreadsheets, old archives, checkpoints, logs, and comparison/replicate result files
- `results_manuscript_*/generic_llm_5_continuous_dt7030.json` — referenced by the Figure 2 audit path but not present locally for any model, so no placeholder was committed
- `results_hybrid_*/` and stale `results_manuscript_*` files not directly consumed by the current committed figures
- `archive*/`, checkpoint/evaluation `*.json` — old experiments and intermediate caches.

**Bottom line for regeneration:** the plotting scripts in the [Graph Reproducibility](#graph-reproducibility) order now have the current plot-source inputs committed for the revision figures. A full new model rerun from scratch still requires API keys and will create gitignored checkpoints/logs; do not substitute old split variants or fabricate missing rows.

---

## 📊 Evaluation Methods

### Generic LLM Methods (6 variants)

| # | Method | Prompt Config | Features | Examples |
|---|--------|---------------|----------|----------|
| 1 | Zero-shot | `zero-shot` | All | None |
| 2 | Zero-shot | `zero-shot-feature-select` | Selected | None |
| 3 | Few-shot | `few-shot` | All | 3 examples |
| 4 | Few-shot | `few-shot-feature-select` | Selected | 3 examples |
| 5 | Continuous | `zero-shot-natural-lang` | All (natural language) | None |
| 6 | Continuous | `zero-shot-prob` | Selected + probabilities | None |

### Personalized Prompt (PP) Method
Uses the PP split (`test_digital_twin_7030.json`) - each participant has messages in both train (profile) and test.

| # | Method | Prompt Config | Split | Description |
|---|--------|---------------|-------|-------------|
| 4 | CBT/ACT-informed | `digital-twin-cbtact` | 70/30 | Personalized using participant's historical ratings |

### Hybrid RF+PP Method
Combines Random Forest predictions with PP prompts.

### Supervised ML Baselines
- **Logistic Regression**: Trained on participant metadata (age, gender, race, education, income, smoking status, quit motivation, cigarettes/day)
- **Random Forest**: Same features as Logistic Regression

---

## 📈 Evaluation Domains & Metrics

### Domains
Per manuscript, we evaluate **3 domains**:

1. **Content**: Quality of message words/meaning (5-point scale: Very poor → Very good)
2. **Coping**: Helpfulness for coping with cravings (5-point scale: Not at all → Extremely helpful)
3. **Quitting**: Helpfulness for quitting smoking (5-point scale: Not at all → Extremely helpful)

### Metrics
- **Accuracy**: Exact match accuracy (all five ratings)
- **F1 Macro**: Multi-class F1 score (all five ratings)
- **Directional Accuracy**: 3-bucket accuracy (low/neutral/high ratings)
- **Directional Macro-F1**: F1 score for 3-bucket classification
- **Cohen's Kappa**: Inter-rater agreement accounting for chance
- **Quadratic Weighted Kappa (QWK)**: Ordinal agreement with larger penalties for larger rating disagreements
- **Spearman's ρ**: Per-participant ranking correlation

---

## Output Files

### Figures (`figures/`)
- `bars_all_methods_accuracy.png` - Bar chart comparing accuracy across models
- `bars_all_methods_kappa.png` - Bar chart comparing Cohen's Kappa
- `bars_all_methods_qwk.png` - Bar chart comparing Quadratic Weighted Kappa
- `bars_all_methods_acc_within_1.png` - Bar chart comparing directional accuracy
- `bars_all_methods_f1.png` - Bar chart comparing F1 macro
- `heatmap_all_methods_*.png` - Heatmaps showing performance by model/domain
- `scatter_content.png`, `scatter_coping.png`, `scatter_quitting.png` - Scatter plots (accuracy vs other metrics)
- `learning_curve_*.png` - PP learning curves (10%, 30%, 70%, 90% training data)

---

## Graph Reproducibility

The revision figures use several supervised-result families. Do not mix rows
across these families unless the script explicitly aligns them.

| Artifact family | Main outputs | Script | Primary source data | Split / row policy | Uncertainty |
|---|---|---|---|---|---|
| Main model comparison / Figure 2 | `figures/figure2/*`, `figures/bars_all_methods_*`, `figures/bars_all_methods_source_table.csv` | `analysis-script/create_comprehensive_figures.py` | LLM result JSONs plus `revision/figures/history_supervised_baselines.csv` | cleaned canonical `digital_twin_7030`; known train/test duplicate rows removed; supervised rows are LR/RF for `Demographics` and `Demographics + History + Message Embedding` | point estimates only in bars |
| Supervised 70/30 baselines | `revision/figures/history_supervised_baselines.csv`, `revision/figures/history_supervised_predictions.csv`, `revision/figures/history_supervised_best.*` | `analysis-script/history_supervised_baselines.py` | `data_splits/canonical/train_digital_twin_7030.json`, `test_digital_twin_7030.json`, embeddings | cleaned canonical `digital_twin_7030`; duplicate test rows removed before fitting/evaluation | point estimates |
| dt10 RF learning curves | `revision/figures/lc_dt10_rf.csv`, `revision/figures/lc_dt10_rf_predictions.csv` | `analysis-script/lc_dt10_rf.py` | `data_splits/canonical/train_dt10_k*.json`, `test_dt10_k*.json` | within-participant dt10, `k_train` in 1/3/5/7; RF feature sets kept separate | point estimates |
| Strict AI-vs-ML summary | `revision/figures/progress_summary/*`, copied to `figures/supplementary/*` | `analysis-script/build_progress_summary.py` | `lc_dt10_rf_predictions.csv`, `lc_dt10_ensemble_k*.csv`, Grok `digital_twin_dt10_k*.json` | shared-row dt10 comparison; two headline methods are Supervised RF and LLM-PP | normal-approximation CIs where shown |
| Figure 4 LLM-only context | `figures/figure4/llm_selection_quality.*`, `figures/figure4/top_k_agreement_line.*` | `analysis-script/create_practical_analysis_figures.py --selection-only --output-dir figures/figure4` | five-model PP `digital_twin_4_cbtact_7030.json` files | LLM-only 108-message digital-twin 70/30 message pool; context only, not an apples-to-apples supervised comparison | 2000 message-bootstrap resamples |
| Figure 4 supervised support | `figures/figure4/supporting_message_selection_quality_dt10.*`, `figures/figure4/supporting_message_selection_gain_dt10.*`, `figures/figure4/supporting_message_selection_methods_k7.csv` | `analysis-script/make_figure4_supporting_dt10.py` | `revision/figures/progress_summary/message_selection_methods_k7.csv` | dt10 `k_train=7`, 122-message pool, fixed `Demographics + History + Message Embedding` RF feature block; anchor hybrids excluded | normal-approximation message-level SE CIs |
| CBT vs ACT figure | `revision/figures/cbt_vs_act_performance.*`, `revision/figures/cbt_act_nonllm_accuracy_summary.csv`, `revision/figures/cbt_act_nonllm_accuracy_significance.csv` | `analysis-script/cbt_act_comparison.py --plot-only` | cached LLM/PP `cbt_act_comparison.csv` plus supervised `history_supervised_predictions.csv` | supervised panel uses cleaned `digital_twin_7030` predictions; do not substitute participant-split text baselines | row bootstrap within ACT/CBT cells |
| Supplementary diagnostics | `figures/supplementary/*` | see `figures/supplementary/SUPPLEMENTARY_REFRESH_AUDIT.md` | mixed by diagnostic, documented per file | use only for the diagnostic named in the audit | documented per diagnostic |

Recommended regeneration order for revision graphs:

```bash
uv run python analysis-script/history_supervised_baselines.py
uv run python analysis-script/create_comprehensive_figures.py
uv run python analysis-script/lc_dt10_rf.py
uv run python analysis-script/build_progress_summary.py
uv run python analysis-script/make_figure4_supporting_dt10.py
uv run python analysis-script/cbt_act_comparison.py --plot-only
```

The old Figure 4 LLM-only curve and the supervised dt10 Figure 4 support use
different message pools (`108` versus `122` messages per domain). Keep them as
separate panels/artifacts; do not add supervised RF directly to the old LLM-only
curve unless all methods are recomputed on one shared message pool. In the
support CSV, `method` is normalized to `LLM-PP` / `Supervised RF`; the raw
progress-summary label is retained in `method_source`.

---

## Requirements

### API Keys
Set in `~/.bash_profile` or `~/.bashrc`:
```bash
export OPENAI_API_KEY='sk-...'
export DEEPSEEK_API_KEY='sk-...'
export XAI_API_KEY='xai-...'
export GOOGLE_API_KEY='...'
```

### Python Packages
Key dependencies:
- `openai`, `anthropic` - API clients
- `pandas`, `numpy` - Data processing
- `scikit-learn` - Metrics & ML models
- `scipy` - Statistical tests
- `matplotlib`, `seaborn` - Visualization

---
