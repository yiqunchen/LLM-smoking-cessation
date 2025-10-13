# Experiments That MUST Be Re-run

## ⚠️ Critical Issue: Existing Results Use Wrong Data Split

**Problem:** All existing `evaluation_results_*.json` files were run on the FULL dataset without proper train/test splits. For fair comparison with regression models and to prevent data leakage, we MUST re-run everything using the canonical splits.

---

## 📋 Complete Re-run List

### 2.2.2 Generic LLM Models (6 experiments)
**Must use:** `data_splits/canonical/test_participant_7030.json` (70/30 split)

| # | Experiment | Command | Priority |
|---|------------|---------|----------|
| 1 | Zero-shot + all features | `bash run_manuscript_evaluations.sh --methods generic_llm` OR run individually ↓ | **HIGH** |
| | | `python analysis-script/main_eval.py --mode text-only --model gpt-4o-mini --prompt-config zero-shot --max-concurrent 10 --output-file results_manuscript/generic_llm_1_zero_shot_all.json` | |
| 2 | Zero-shot + selected features | Same as above, or: | **HIGH** |
| | | `python analysis-script/main_eval.py --mode text-only --model gpt-4o-mini --prompt-config zero-shot-feature-select --max-concurrent 10 --output-file results_manuscript/generic_llm_2_zero_shot_select.json` | |
| 3 | Few-shot + all features | Same as above, or: | **HIGH** |
| | | `python analysis-script/main_eval.py --mode text-only --model gpt-4o-mini --prompt-config few-shot --max-concurrent 10 --output-file results_manuscript/generic_llm_3_few_shot_all.json` | |
| 4 | Few-shot + selected features | Same as above, or: | **HIGH** |
| | | `python analysis-script/main_eval.py --mode text-only --model gpt-4o-mini --prompt-config few-shot-feature-select --max-concurrent 10 --output-file results_manuscript/generic_llm_4_few_shot_select.json` | |
| 5 | Continuous + all features | Same as above, or: | **MEDIUM** |
| | | `python analysis-script/main_eval.py --mode text-only --model gpt-4o-mini --prompt-config zero-shot-natural-lang --max-concurrent 10 --output-file results_manuscript/generic_llm_5_continuous_all.json` | |
| 6 | Continuous + selected features | Same as above, or: | **MEDIUM** |
| | | `python analysis-script/main_eval.py --mode text-only --model gpt-4o-mini --prompt-config zero-shot-prob --max-concurrent 10 --output-file results_manuscript/generic_llm_6_continuous_select.json` | |

### 2.2.3 Hybrid ML-LLM Models (1 experiment)
**Must use:** `data_splits/canonical/train_participant_7030.json` for training, `test_participant_7030.json` for testing

| # | Experiment | Command | Priority | Status |
|---|------------|---------|----------|--------|
| 7 | ML + LLM embeddings | **Needs implementation first** | **HIGH** | ❌ NOT IMPLEMENTED |

**Note:** This requires:
1. Training a regression/ML model on participant characteristics using train split
2. Using message embeddings from `message_embeddings.pkl`
3. Combining predictions
4. See `analysis-script/compare_llm_vs_individual.py` with `--run-joint-model` flag

### 2.2.4 Digital Twin Models (6 experiments)
**Critical:** These need script updates FIRST to load from canonical split files instead of `data/digitalTwin_msg.xlsx`

| # | Experiment | Split File | Command | Priority | Status |
|---|------------|------------|---------|----------|--------|
| 8 | Full-feature (50/50) | `train_digital_twin_5050.json` / `test_digital_twin_5050.json` | **Needs script update** | **HIGH** | ❌ SCRIPT NEEDS FIX |
| 9 | Selected-feature (50/50) | `train_digital_twin_5050.json` / `test_digital_twin_5050.json` | **Needs script update** | **MEDIUM** | ❌ SCRIPT NEEDS FIX |
| 10 | Full + feedback (50/50) | `train_digital_twin_5050.json` / `test_digital_twin_5050.json` | **Needs script update** | **MEDIUM** | ❌ SCRIPT NEEDS FIX |
| 11 | CBT/ACT (50/50) | `train_digital_twin_5050.json` / `test_digital_twin_5050.json` | **Needs script update** | **HIGH** | ❌ SCRIPT NEEDS FIX |
| 12 | CBT/ACT (70/30) | `train_digital_twin_7030.json` / `test_digital_twin_7030.json` | **Needs script update** | **HIGH** | ❌ SCRIPT NEEDS FIX |
| 13 | CBT/ACT (90/10) | `train_digital_twin_9010.json` / `test_digital_twin_9010.json` | **Needs script update** | **MEDIUM** | ❌ SCRIPT NEEDS FIX |

**Script Update Needed:** 
- Current: `analysis-script/prompt_config.py` loads from `data/digitalTwin_msg.xlsx`
- Required: Update functions to load training messages from split JSON files
- Functions to update: `generate_digital_twin_prompt()`, `generate_digital_twin_select_prompt()`, `generate_digital_twin_feedback_prompt()`, `generate_digital_twin_cbtact_prompt()`

### 2.2.1 Benchmark Models (2 experiments)

| # | Experiment | Command | Priority | Status |
|---|------------|---------|----------|--------|
| 14 | Random guess baseline | **Needs new script** | **HIGH** | ❌ NOT IMPLEMENTED |
| 15 | Regression baseline | **Needs new script** | **HIGH** | ❌ NOT IMPLEMENTED |

**Implementation needed:** Create `analysis-script/run_benchmarks.py` that:
1. Random guess: Randomly assigns ratings uniformly
2. Regression: Trains logistic/ordinal regression on participant features using train split

---

## 🚀 Quick Start Guide

### Step 1: Run Generic LLM Methods (Easiest - Can Start Now)
```bash
conda activate research

# Run ALL 6 generic LLM methods at once
bash run_manuscript_evaluations.sh --methods generic_llm

# OR run them individually (more control over rate limits)
python analysis-script/main_eval.py --mode text-only --model gpt-4o-mini \
  --prompt-config zero-shot --max-concurrent 10 \
  --output-file results_manuscript/generic_llm_1_zero_shot_all.json

python analysis-script/main_eval.py --mode text-only --model gpt-4o-mini \
  --prompt-config zero-shot-feature-select --max-concurrent 10 \
  --output-file results_manuscript/generic_llm_2_zero_shot_select.json

# ... repeat for configs 3-6
```

**Time estimate:** ~30-60 minutes per method (depending on API rate limits)  
**Cost estimate:** ~$2-5 per method with gpt-4o-mini

### Step 2: Implement & Run Benchmark Models
```bash
# TODO: Create this script first
python analysis-script/run_benchmarks.py \
  --train-split data_splits/canonical/train_participant_7030.json \
  --test-split data_splits/canonical/test_participant_7030.json \
  --output-dir results_manuscript/
```

### Step 3: Fix Digital Twin Scripts
**Update needed in `analysis-script/prompt_config.py`:**

Current code loads from Excel:
```python
df = pd.read_excel("data/digitalTwin_msg.xlsx")
```

Change to load from JSON split files:
```python
# Need to pass train_file path as parameter
with open(train_file, 'r') as f:
    train_data = json.load(f)
```

### Step 4: Run Digital Twin Methods
```bash
# After fixing scripts
bash run_manuscript_evaluations.sh --methods digital_twin
```

### Step 5: Implement & Run Hybrid Model
```bash
# TODO: This needs implementation
python analysis-script/compare_llm_vs_individual.py \
  --llm-results results_manuscript/generic_llm_1_zero_shot_all.json \
  --run-joint-model \
  --train-split data_splits/canonical/train_participant_7030.json \
  --test-split data_splits/canonical/test_participant_7030.json
```

---

## 📊 Priority Order

### Priority 1: Can Run Immediately (6 experiments)
✅ **Generic LLM methods 1-6** - Scripts ready, just need to run
```bash
bash run_manuscript_evaluations.sh --methods generic_llm
```

### Priority 2: Quick Implementation Needed (2 experiments)
⚠️ **Benchmark models** - Need simple script for random guess + regression
- Create `analysis-script/run_benchmarks.py`
- Should take ~1-2 hours to implement

### Priority 3: Script Updates Needed (6 experiments)
⚠️ **Digital Twin methods** - Need to update prompt_config.py
- Modify 4 functions to load from JSON instead of Excel
- Should take ~2-3 hours to implement and test

### Priority 4: More Complex Implementation (1 experiment)
⚠️ **Hybrid ML-LLM** - Need to extend existing script
- Modify `compare_llm_vs_individual.py`
- Should take ~3-4 hours to implement properly

---

## ⚠️ Important Notes

### Data Split Usage
**CRITICAL:** When running experiments, the script must:
1. Load data from the canonical split file
2. NOT load from `data/processed_llm_data.json` (that's the full dataset)

**Current issue:** `main_eval.py` might still be loading from the full dataset. Need to verify and update if needed.

### Checking Current Implementation
```bash
# Check what data file main_eval.py loads
grep -n "processed_llm_data.json" analysis-script/main_eval.py
```

If it hardcodes the data path, we need to:
1. Add `--data-file` argument to `main_eval.py`
2. Update `run_manuscript_evaluations.sh` to pass the correct split file

---

## 📝 Checklist

### Can Start Immediately
- [ ] Run Generic LLM method 1: Zero-shot + all features
- [ ] Run Generic LLM method 2: Zero-shot + selected features  
- [ ] Run Generic LLM method 3: Few-shot + all features
- [ ] Run Generic LLM method 4: Few-shot + selected features
- [ ] Run Generic LLM method 5: Continuous + all features
- [ ] Run Generic LLM method 6: Continuous + selected features

### Needs Implementation First
- [ ] Create `run_benchmarks.py` for random guess + regression
- [ ] Run benchmark: Random guess
- [ ] Run benchmark: Regression model
- [ ] Update `prompt_config.py` to load digital twin data from JSON splits
- [ ] Run Digital Twin method 1: Full-feature (50/50)
- [ ] Run Digital Twin method 2: Selected-feature (50/50)
- [ ] Run Digital Twin method 3: Full + feedback (50/50)
- [ ] Run Digital Twin method 4a: CBT/ACT (50/50)
- [ ] Run Digital Twin method 4b: CBT/ACT (70/30)
- [ ] Run Digital Twin method 4c: CBT/ACT (90/10)
- [ ] Implement hybrid ML-LLM model
- [ ] Run Hybrid ML-LLM method

### After All Experiments
- [ ] Run `python analysis-script/analyze_manuscript_results.py`
- [ ] Generate final comparison table
- [ ] Create publication figures
- [ ] Update manuscript with results

---

## 💰 Cost Estimate

**Generic LLM (gpt-4o-mini):** 6 methods × ~916 samples × ~$0.0003/sample = **~$2-3 per method = $12-18 total**

**Digital Twin (gpt-4o):** 6 methods × ~300-600 samples × ~$0.003/sample = **~$1-2 per method = $6-12 total**

**Total estimated cost:** $20-30 for all LLM-based experiments

**Time estimate:** 
- API calls: 6-8 hours total (can run overnight)
- Implementation: 6-9 hours (benchmarks + digital twin fixes + hybrid)
- Analysis: 2-3 hours

**Total time:** ~15-20 hours of work

