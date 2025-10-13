# LLM-Based Smoking Cessation Message Evaluation

Evaluating smoking cessation messages using Large Language Models (LLMs) to predict participant ratings across multiple dimensions.

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
conda activate research
source ~/.bash_profile  # Loads OPENAI_API_KEY
```

### 2. Run Generic LLM Experiments (Ready Now!)
```bash
# Run all 6 Generic LLM methods at once
bash run_manuscript_evaluations.sh --methods generic_llm

# OR run individually
python analysis-script/main_eval.py \
  --mode text-only \
  --model gpt-4o-mini \
  --prompt-config zero-shot \
  --data-file data_splits/canonical/test_participant_7030.json \
  --max-concurrent 10 \
  --output-file results_manuscript/generic_llm_1_zero_shot_all.json
```

### 3. Analyze Results
```bash
python analysis-script/analyze_manuscript_results.py
```

Output: `results_manuscript/summary_table.csv`, `summary_table.md`, `method_comparison.png`

---

## 📊 Manuscript Methods (Section 2.2)

Based on `wip-manuscript.md`. All methods use canonical splits (seed=202509) for fair comparison.

### 2.2.1 Benchmark Models
- **Random Guess**: Baseline using uniform random selection
- **Regression**: Logistic regression on participant characteristics
- **Status**: ⚠️ Needs implementation (`analysis-script/run_benchmarks.py`)

### 2.2.2 Generic LLM Models (Using 70/30 Participant Split)
All use `data_splits/canonical/test_participant_7030.json`

| # | Method | Config | Status |
|---|--------|--------|--------|
| 1 | Zero-shot + all features | `--prompt-config zero-shot` | ✅ Ready |
| 2 | Zero-shot + selected features | `--prompt-config zero-shot-feature-select` | ✅ Ready |
| 3 | Few-shot + all features | `--prompt-config few-shot` | ✅ Ready |
| 4 | Few-shot + selected features | `--prompt-config few-shot-feature-select` | ✅ Ready |
| 5 | Continuous + all features | `--prompt-config zero-shot-natural-lang` | ✅ Ready |
| 6 | Continuous + selected features | `--prompt-config zero-shot-prob` | ✅ Ready |

### 2.2.3 Hybrid ML-LLM Models (Using 70/30 Participant Split)
- **ML + LLM embeddings**: Combine participant features (ML) with message embeddings (LLM)
- **Status**: ⚠️ Needs implementation (extend `compare_llm_vs_individual.py`)

### 2.2.4 Digital Twin Models (Using Digital Twin Splits)
Each participant has messages split into "profile" (train) and "test" sets.

| # | Method | Split | Config | Status |
|---|--------|-------|--------|--------|
| 1 | Full-feature | 50/50 | `--prompt-config digital-twin` | ⚠️ Needs script fix |
| 2 | Selected-feature | 50/50 | `--prompt-config digital-twin-select` | ⚠️ Needs script fix |
| 3 | Full + feedback | 50/50 | `--prompt-config digital-twin-feedback` | ⚠️ Needs script fix |
| 4a | CBT/ACT-informed | 50/50 | `--prompt-config digital-twin-cbtact` | ⚠️ Needs script fix |
| 4b | CBT/ACT-informed | 70/30 | `--prompt-config digital-twin-cbtact` | ⚠️ Needs script fix |
| 4c | CBT/ACT-informed | 90/10 | `--prompt-config digital-twin-cbtact` | ⚠️ Needs script fix |

**Issue**: Functions in `prompt_config.py` load from `data/digitalTwin_msg.xlsx` instead of canonical split JSONs.

---

## 📁 Directory Structure

```
LLM-smoking-cessation/
├── README.md                          # This file
├── wip-manuscript.md                  # Paper draft
├── run_manuscript_evaluations.sh      # Master evaluation script
│
├── data/                              # Original data
│   ├── processed_llm_data.json        # Full dataset
│   ├── message_embeddings.pkl         # Message embeddings
│   └── digitalTwin_msg.xlsx           # Digital twin training data
│
├── data_splits/
│   └── canonical/                     # ⭐ Canonical splits (NEVER MODIFY)
│       ├── train_participant_7030.json
│       ├── test_participant_7030.json
│       ├── train_digital_twin_5050.json
│       └── ... (all splits with metadata)
│
├── analysis-script/                   # Analysis & evaluation scripts
│   ├── main_eval.py                   # Main evaluation script
│   ├── prompt_config.py               # Prompt templates
│   ├── create_canonical_splits.py     # Generate data splits
│   ├── analyze_manuscript_results.py  # Analyze all methods
│   ├── analyze_ranking_performance.py # Cohen's Kappa & Spearman
│   ├── analyze_llm_calibration.py     # Calibration analysis
│   ├── compare_llm_vs_individual.py   # Compare LLM vs regression
│   └── plot_results_3cat.py           # Plotting utilities
│
├── results_manuscript/                # ⭐ New results go here
│   ├── generic_llm_*.json             # LLM experiment results
│   ├── summary_table.csv              # Final comparison table
│   └── method_comparison.png          # Visualization
│
├── digital-twin/                      # Digital twin experiments
│
├── archive_code/                      # Old experimental code
├── archive_results/                   # Old experimental results
└── archive/                           # Legacy archive
```

---

## 🔧 Main Scripts

### Evaluation: `main_eval.py`
```bash
python analysis-script/main_eval.py \
  --mode text-only \                    # or 'vision'
  --model gpt-4o-mini \                 # Model name
  --prompt-config zero-shot \           # Prompt type
  --data-file data_splits/canonical/test_participant_7030.json \  # ⚠️ USE SPLITS
  --max-concurrent 10 \                 # Parallel API calls
  --output-file results_manuscript/output.json
```

**Prompt configs:**
- `zero-shot`: Basic zero-shot with all features
- `zero-shot-feature-select`: Zero-shot with selected features
- `zero-shot-natural-lang`: Natural language profile + probabilities
- `zero-shot-prob`: Feature-select + probabilities
- `few-shot`: Few-shot with all features
- `few-shot-feature-select`: Few-shot with selected features
- `digital-twin*`: Digital twin variants (needs fix)

### Analysis: `analyze_manuscript_results.py`
```bash
python analysis-script/analyze_manuscript_results.py
```

Generates:
- `results_manuscript/summary_table.csv` - Metrics by method & domain
- `results_manuscript/summary_table.md` - Markdown table
- `results_manuscript/method_comparison.png` - Bar chart comparison

Metrics: Accuracy, Cohen's Kappa, Spearman's Rho (per-participant ranking)

### Ranking Analysis: `analyze_ranking_performance.py`
```bash
python analysis-script/analyze_ranking_performance.py <result_file.json>
```

Outputs:
- Cohen's Kappa: Agreement accounting for chance
- Spearman's Rho: Per-participant rank correlation (directional correctness)

### Calibration Analysis: `analyze_llm_calibration.py`
```bash
python analysis-script/analyze_llm_calibration.py <result_file_with_probabilities.json>
```

Outputs:
- Accuracy & AUC
- ECE (Expected Calibration Error)
- Reliability diagrams
- Temperature scaling for calibration

---

## 📊 Evaluation Domains

Per manuscript, we focus on **3 domains** (excluding design):

1. **Content**: Quality of message words/meaning (Very poor → Very good)
2. **Coping**: Helpfulness for coping with cravings (Not at all helpful → Extremely helpful)
3. **Quitting**: Helpfulness for quitting smoking (Not at all helpful → Extremely helpful)

---

## 🔑 Requirements

### Environment
```bash
conda activate research
```

### API Keys
Set in `~/.bash_profile`:
```bash
export OPENAI_API_KEY='sk-...'
export OPENROUTER_API_KEY='sk-or-...'  # For alternative models
```

### Python Packages
See `requirements.txt`. Key dependencies:
- `openai` - API client
- `pandas`, `numpy` - Data processing
- `scikit-learn` - Metrics & ML models
- `scipy` - Statistical tests
- `matplotlib`, `seaborn` - Visualization

---

## 🎯 To-Do List

### ✅ Can Run Now (6 experiments)
- [ ] Generic LLM 1: Zero-shot + all features
- [ ] Generic LLM 2: Zero-shot + selected features
- [ ] Generic LLM 3: Few-shot + all features
- [ ] Generic LLM 4: Few-shot + selected features
- [ ] Generic LLM 5: Continuous + all features
- [ ] Generic LLM 6: Continuous + selected features

**Command:** `bash run_manuscript_evaluations.sh --methods generic_llm`

### ⚠️ Needs Implementation (9 experiments)
- [ ] Implement benchmarks (random guess, regression)
- [ ] Fix digital twin scripts to use canonical splits
- [ ] Run Digital Twin 1-6
- [ ] Implement hybrid ML-LLM model
- [ ] Run hybrid model

---

## ⚠️ Critical Notes

### ALWAYS Use Canonical Splits
✅ **Correct:**
```bash
--data-file data_splits/canonical/test_participant_7030.json
```

❌ **Wrong (uses full dataset):**
```bash
# Missing --data-file argument
```

### Data Split Types
- **Participant splits** (`*_participant_*.json`): For Generic LLM & Hybrid models
  - No participant appears in both train and test
  - Use: `test_participant_7030.json` for evaluation
  
- **Digital twin splits** (`*_digital_twin_*.json`): For Digital Twin models
  - Each participant has messages in both train (profile) and test
  - Use: `test_digital_twin_5050.json` (or 7030/9010) for evaluation

### Reproducibility
- All splits use seed: `202509`
- Fixed in `create_canonical_splits.py`
- Documented in split metadata files

---

## 💰 Cost Estimates

| Method Category | # Experiments | Runtime | Cost |
|----------------|---------------|---------|------|
| Generic LLM (gpt-4o-mini) | 6 | 2-3 hrs | $12-18 |
| Digital Twin (gpt-4o) | 6 | 2-3 hrs | $6-12 |
| Hybrid | 1 | 30 min | $2-3 |
| **Total** | **13** | **5-7 hrs** | **$20-35** |

---

## 🐛 Troubleshooting

### "No such file: data_splits/canonical/..."
```bash
python analysis-script/create_canonical_splits.py
```

### "OPENAI_API_KEY not found"
```bash
source ~/.bash_profile
echo $OPENAI_API_KEY  # Should not be empty
```

### "No results found" when analyzing
```bash
# Run evaluations first
bash run_manuscript_evaluations.sh --methods generic_llm
```

### Digital twin errors
Scripts need updating to load from JSON splits instead of Excel. See issue in `prompt_config.py` lines with `pd.read_excel("data/digitalTwin_msg.xlsx")`.

---

## 📚 Related Files

- **`wip-manuscript.md`**: Paper draft with methods descriptions
- **`run_manuscript_evaluations.sh`**: Master script to run all methods
- **`data_splits/canonical/README.md`**: Documentation of canonical splits

---

## 📝 Citation

If you use this code, please cite:
```
[Paper citation to be added]
```

---

**Last Updated**: 2025-10-13  
**Repository**: Clean and ready for manuscript experiments  
**Status**: Generic LLM methods ready to run
