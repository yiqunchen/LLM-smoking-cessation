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
