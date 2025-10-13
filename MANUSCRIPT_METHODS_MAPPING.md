# Manuscript Methods Mapping

This document maps existing evaluation results to the methods described in `wip-manuscript.md`.

## Manuscript Methods Structure

### 2.2.1 Benchmark Models
- **Random Guess**: Not yet implemented (needs separate script)
- **Regression Model**: Not yet implemented (needs separate script using participant characteristics)

### 2.2.2 Generic LLM Models  
All must use **canonical participant split (70/30)** from `data_splits/canonical/`

| Method | Description | Existing Results | Status | Notes |
|--------|-------------|------------------|--------|-------|
| 1. Zero-shot + all features | Text-only, all participant metadata | `evaluation_results_gpt-4o-mini_text-only_zero-shot.json` | ⚠️ **NEEDS RE-RUN** | Uses full dataset, not canonical split |
| 2. Zero-shot + selected features | Text-only, selected metadata | `evaluation_results_gpt-4o-mini_text-only_zero-shot-feature-select.json` | ⚠️ **NEEDS RE-RUN** | Uses full dataset, not canonical split |
| 3. Few-shot + all features | Text-only, few-shot examples, all metadata | `evaluation_results_gpt-4o-mini_text-only_few-shot.json` | ⚠️ **NEEDS RE-RUN** | Uses full dataset, not canonical split |
| 4. Few-shot + selected features | Text-only, few-shot examples, selected metadata | `evaluation_results_gpt-4o-mini_text-only_few-shot-feature-select.json` | ⚠️ **NEEDS RE-RUN** | Uses full dataset, not canonical split |
| 5. Continuous rating + all features | For calibration, all metadata | `evaluation_results_gpt-4o-mini_text-only_zero-shot-natural-lang.json` | ⚠️ **NEEDS RE-RUN** | Has probabilities, but needs canonical split |
| 6. Continuous rating + selected features | For calibration, selected metadata | ❌ **MISSING** | Need to run `zero-shot-prob` config |

### 2.2.3 Hybrid ML-LLM Models
Must use **canonical participant split (70/30)** from `data_splits/canonical/`

| Method | Description | Existing Results | Status | Notes |
|--------|-------------|------------------|--------|-------|
| ML + LLM embeddings | Individual characteristics + message embeddings | ❌ **MISSING** | Need to implement | Requires `message_embeddings.pkl` + train on canonical split |

### 2.2.4 Digital Twin Models
Must use **canonical digital twin splits** from `data_splits/canonical/`

| Method | Description | Existing Results | Status | Notes |
|--------|-------------|------------------|--------|-------|
| 1. Full-feature (50/50) | All metadata, 50% messages in profile | ❌ **MISSING** | Need `digital-twin` + 50/50 split |
| 2. Selected-feature (50/50) | Selected metadata, 50% messages in profile | ❌ **MISSING** | Need `digital-twin-select` + 50/50 split |
| 3. Full-feature + feedback (50/50) | All metadata + feedback, 50% messages | ❌ **MISSING** | Need `digital-twin-feedback` + 50/50 split |
| 4a. CBT/ACT-informed (50/50) | Domain-specific, 50% messages | `evaluation_results_gpt-4o_text-only_digital-twin-cbtact.json` | ⚠️ **CHECK SPLIT** | Verify using correct split |
| 4b. CBT/ACT-informed (70/30) | Domain-specific, 70% messages | ❌ **MISSING** | Need to run with 70/30 split |
| 4c. CBT/ACT-informed (90/10) | Domain-specific, 90% messages | ❌ **MISSING** | Need to run with 90/10 split |

## Files to Archive

### Experimental / Not in Manuscript
These results were exploratory and should be moved to `archive/old_results/`:

- `evaluation_results_gpt-4o-mini_vision*.json` (design dimension excluded from main analysis per manuscript)
- `evaluation_results_gpt-4o_vision*.json` (vision not main focus)
- `evaluation_results_o3-*.json` (exploratory model testing)
- `evaluation_results_*_enhanced-zero-shot.json` (experimental prompt)
- `evaluation_results_*_balanced.json` (experimental balancing)

### Checkpoint Files
These should stay in `temp_checkpoint/` but not be tracked in git:
- All `checkpoint_results_*.json` files

## Action Plan

### Phase 1: Setup (DONE ✓)
- [x] Create canonical data splits
- [x] Resolve git merge conflicts
- [x] Update .gitignore

### Phase 2: Archive Irrelevant Results
- [ ] Create `archive/old_results/` directory
- [ ] Move experimental results files
- [ ] Update .gitignore to exclude result files from tracking

### Phase 3: Implement Missing Methods
- [ ] Implement random guess baseline
- [ ] Implement regression baseline
- [ ] Re-run Generic LLM methods (1-4) with canonical split
- [ ] Implement continuous rating + selected features (method 6)
- [ ] Implement Hybrid ML-LLM method
- [ ] Run Digital Twin methods (1-4) with correct splits

### Phase 4: Create Master Evaluation Script
- [ ] Script that runs all methods with correct splits
- [ ] Automated result collection and comparison

### Phase 5: Generate Results Table
- [ ] Extract metrics from all methods
- [ ] Create publication-ready comparison table
- [ ] Generate visualizations

## Running Commands (Template)

### Generic LLM Methods (use canonical participant split)
```bash
conda activate research

# Method 1: Zero-shot + all features
python analysis-script/main_eval.py \\
  --mode text-only \\
  --model gpt-4o-mini \\
  --prompt-config zero-shot \\
  --data-file data_splits/canonical/test_participant_7030.json \\
  --output-file results_manuscript/generic_llm_1_zero_shot_all.json

# Method 2: Zero-shot + selected features
python analysis-script/main_eval.py \\
  --mode text-only \\
  --model gpt-4o-mini \\
  --prompt-config zero-shot-feature-select \\
  --data-file data_splits/canonical/test_participant_7030.json \\
  --output-file results_manuscript/generic_llm_2_zero_shot_select.json

# ... etc for methods 3-6
```

### Digital Twin Methods (use canonical digital twin splits)
```bash
# Method 4a: CBT/ACT-informed (50/50)
python analysis-script/main_eval.py \\
  --mode text-only \\
  --model gpt-4o \\
  --prompt-config digital-twin-cbtact \\
  --data-file data_splits/canonical/test_digital_twin_5050.json \\
  --profile-file data_splits/canonical/train_digital_twin_5050.json \\
  --output-file results_manuscript/digital_twin_4a_cbtact_5050.json
```

## Notes
- All text-only methods should exclude the `design` dimension per manuscript (vision not improving accuracy)
- Focus on 3 dimensions: content, coping, quitting
- Seed 202509 used for all splits for reproducibility
- GPT-4o-mini is the default model unless otherwise specified

