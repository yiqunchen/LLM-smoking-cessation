# Implementation Summary: Train/Test Splits & Comprehensive Metrics

## ✅ Completed Changes

### 1. **Default Split Ratio: 70/30**
All methods now use a **70/30 train/test split** as the default:
- **Generic LLM models**: Use `test_participant_7030.json` (participant-based split)
- **Digital Twin models**: Use `test_digital_twin_7030.json` (message-within-participant split)
- **Traditional ML baselines**: Use `train_participant_7030.json` / `test_participant_7030.json`

### 2. **Digital Twin Additional Splits**
Digital twin models now have **5 split ratios** available:
- **10/90** (10% train, 90% test)
- **30/70** (30% train, 70% test)
- **50/50** (50% train, 50% test) - *original*
- **70/30** (70% train, 30% test) - **DEFAULT**
- **90/10** (90% train, 10% test)

All splits generated in: `data_splits/canonical/`

### 3. **Traditional ML Baselines**
Uses existing script: `analysis-script/compare_llm_vs_individual.py`

Implements:
- **Random Forest** on participant characteristics (cross-validated)
- Uses the same **70/30 split** as other methods when `--data-path` points to canonical test split
- Generates traditional ML results automatically alongside LLM comparison

### 4. **Comprehensive Metrics Implementation**
Updated `analysis-script/analyze_manuscript_results.py` to calculate:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Exact match accuracy |
| **Accuracy ±1** | Accuracy allowing ±1 error (e.g., predicting 3 when truth is 4) |
| **Cohen's κ (Kappa)** | Inter-rater agreement correcting for chance |
| **Kendall's τ (Tau)** | Rank correlation (ordinal agreement) |
| **Spearman's ρ (Rho)** | Per-participant rank correlation (averaged) |
| **Macro F1** | Macro-averaged F1 score |
| **Macro Precision** | Macro-averaged precision |
| **Macro Recall** | Macro-averaged recall |

### 5. **Updated Run Script**
`run_manuscript_evaluations.sh` now includes:
- Traditional ML benchmarks (2 methods)
- Generic LLM methods (6 configurations, all use 70/30)
- Hybrid ML-LLM (placeholder)
- Digital Twin methods (4 variants × 4 splits = 16 configurations, default 70/30)

---

## 🚀 How to Run

### Step 1: Run Traditional ML Baselines
```bash
bash run_manuscript_evaluations.sh --methods benchmarks
```

### Step 2: Run Generic LLM Models (70/30 split)
```bash
bash run_manuscript_evaluations.sh --methods generic_llm
```

### Step 3: Run Digital Twin Models (all splits, default 70/30)
```bash
bash run_manuscript_evaluations.sh --methods digital_twin
```

### Step 4: Analyze All Results
```bash
conda activate research
python analysis-script/analyze_manuscript_results.py
```

This will generate:
- `results_manuscript/summary_table.csv` (full results)
- `results_manuscript/summary_table.md` (markdown version)
- `results_manuscript/summary_table_pivot.csv` (pivot by method/domain)
- `results_manuscript/method_comparison_all_metrics.png` (visualization)

---

## 📊 Expected Output Structure

```
results_manuscript/
├── traditional_ml_logistic.json          # Logistic Regression results
├── traditional_ml_random_forest.json     # Random Forest results
├── generic_llm_1_zero_shot.json          # Zero-shot all features
├── generic_llm_2_zero_shot_select.json   # Zero-shot selected features
├── generic_llm_3_few_shot.json           # Few-shot all features
├── generic_llm_4_few_shot_select.json    # Few-shot selected features
├── generic_llm_5_continuous.json         # Continuous rating
├── generic_llm_6_continuous_select.json  # Continuous rating + selected
├── digital_twin_1_full_5050.json         # Digital twin 50/50 split
├── digital_twin_2_select_5050.json       # Digital twin selected features
├── digital_twin_3_feedback_5050.json     # Digital twin + feedback
├── digital_twin_4_cbtact_1090.json       # CBT/ACT 10/90
├── digital_twin_4_cbtact_3070.json       # CBT/ACT 30/70
├── digital_twin_4_cbtact_7030.json       # CBT/ACT 70/30 (DEFAULT)
├── digital_twin_4_cbtact_9010.json       # CBT/ACT 90/10
├── summary_table.csv                     # Full results table
├── summary_table.md                      # Markdown table
├── summary_table_pivot.csv               # Pivot summary
└── method_comparison_all_metrics.png     # Visualization
```

---

## 📝 Key Files Modified

1. **`analysis-script/create_canonical_splits.py`**
   - Added 10/90 and 30/70 splits for digital twin models
   - All splits now available: 1090, 3070, 5050, 7030, 9010

2. **`analysis-script/run_traditional_ml_baseline.py`** (NEW)
   - Traditional ML baseline using participant characteristics
   - Supports Logistic Regression and Random Forest

3. **`run_manuscript_evaluations.sh`**
   - Traditional ML now runs with 70/30 split
   - Digital twin loop expanded to include 1090, 3070, 7030, 9010
   - All methods consistently use canonical splits

4. **`analysis-script/analyze_manuscript_results.py`**
   - Added 8 comprehensive metrics
   - Enhanced visualization with 2×3 subplot grid
   - Pivot table for easier comparison

---

## ✨ Next Steps

1. **Run the experiments**:
   ```bash
   bash run_manuscript_evaluations.sh --methods all
   ```

2. **Analyze results**:
   ```bash
   python analysis-script/analyze_manuscript_results.py
   ```

3. **Review outputs** in `results_manuscript/`

4. **Iterate** based on findings:
   - Adjust hyperparameters
   - Try different prompt configurations
   - Implement hybrid ML-LLM methods

---

## 🔍 Verification

All train/test splits are now **consistent and canonical**:
- ✅ Generic LLM: Uses `test_participant_7030.json` (70/30 participant split)
- ✅ Digital Twin: Uses `test_digital_twin_XXXX.json` (message-within-participant splits)
- ✅ Traditional ML: Uses `train/test_participant_7030.json` (70/30 participant split)
- ✅ All splits generated with seed `202509` for reproducibility
- ✅ Metadata files document split statistics

No more inconsistent splits or hardcoded data paths! 🎉

