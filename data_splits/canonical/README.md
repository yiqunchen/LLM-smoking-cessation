# Canonical Data Splits

**DO NOT MODIFY THESE SPLITS!** All experiments must use these exact splits for fair comparison.

## Split Information
- **Seed**: 202509
- **Source Data**: `data/processed_llm_data.json`

## Split Strategy

### Generic LLM & Hybrid ML-LLM Models
**Split by PARTICIPANT** - ensures no participant appears in both train and test.

Use these splits for:
- 2.2.2 Generic LLM Models (zero-shot, few-shot, continuous)
- 2.2.3 Hybrid ML-LLM Models

Files:
- `train_participant_7030.json` / `test_participant_7030.json` (DEFAULT: 70% train, 30% test)
- `train_participant_3070.json` / `test_participant_3070.json` (ALTERNATIVE: 30% train, 70% test)

### Digital Twin Models
**Split by MESSAGE within each participant** - each person has some messages in their profile (train) 
and some held out for testing (test).

Use these splits for:
- 2.2.4 Digital Twin Models

Files:
- `train_digital_twin_5050.json` / `test_digital_twin_5050.json` (50% profile, 50% test)
- `train_digital_twin_7030.json` / `test_digital_twin_7030.json` (70% profile, 30% test)
- `train_digital_twin_9010.json` / `test_digital_twin_9010.json` (90% profile, 10% test)

## Metadata Files
Each split has a corresponding `metadata_*.json` file with:
- Exact participant IDs in train/test
- Sample counts
- Reproducibility information

## Usage in Evaluation Scripts
Always load the appropriate canonical split:
```python
with open('data_splits/canonical/train_participant_7030.json') as f:
    train_data = json.load(f)
```
