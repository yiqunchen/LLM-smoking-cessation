"""
Create canonical train/test splits for all experiments.

This script creates standardized splits that MUST be used by all methods
to ensure fair apples-to-apples comparison.

Split Strategy:
- For Generic LLM & Hybrid ML-LLM: Split by PARTICIPANT (people, not messages)
- For Digital Twin: Split by MESSAGE per participant (within-person)
- All splits use seed=202509 for reproducibility
"""

import json
import pandas as pd
import numpy as np
import os
from pathlib import Path

SEED = 202509

def load_data(data_path: str) -> list:
    """Loads the main processed data file."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


def create_participant_split(data: list, train_ratio: float, output_dir: str, split_name: str):
    """
    Creates a split by PARTICIPANT for generic LLM and hybrid models.
    This ensures no participant leakage between train and test.
    
    Args:
        data: Full dataset
        train_ratio: Proportion for training (e.g., 0.7 for 70/30)
        output_dir: Where to save splits
        split_name: Descriptive name (e.g., "7030", "3070")
    """
    print(f"\n{'='*60}")
    print(f"Creating PARTICIPANT split: {split_name} (train_ratio={train_ratio})")
    print(f"{'='*60}")
    
    df = pd.DataFrame(data)
    unique_participants = df['response_id'].unique()
    
    # Reproducible shuffle
    rng = np.random.RandomState(SEED)
    rng.shuffle(unique_participants)
    
    # Split participants
    split_point = int(len(unique_participants) * train_ratio)
    train_participants = set(unique_participants[:split_point])
    test_participants = set(unique_participants[split_point:])
    
    train_data = [item for item in data if item['response_id'] in train_participants]
    test_data = [item for item in data if item['response_id'] in test_participants]
    
    # Save splits
    train_path = os.path.join(output_dir, f'train_participant_{split_name}.json')
    test_path = os.path.join(output_dir, f'test_participant_{split_name}.json')
    
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(test_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    # Save metadata
    metadata = {
        'split_type': 'by_participant',
        'split_name': split_name,
        'train_ratio': train_ratio,
        'seed': SEED,
        'total_participants': len(unique_participants),
        'train_participants': len(train_participants),
        'test_participants': len(test_participants),
        'train_samples': len(train_data),
        'test_samples': len(test_data),
        'train_participant_ids': sorted(list(train_participants)),
        'test_participant_ids': sorted(list(test_participants))
    }
    
    metadata_path = os.path.join(output_dir, f'metadata_participant_{split_name}.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Total participants: {len(unique_participants)}")
    print(f"  Train: {len(train_participants)} participants, {len(train_data)} samples")
    print(f"  Test:  {len(test_participants)} participants, {len(test_data)} samples")
    print(f"  ✓ Saved to {train_path} and {test_path}")
    
    return train_participants, test_participants


def create_digital_twin_split(data: list, train_ratio: float, output_dir: str, split_name: str, 
                               participant_ids: set = None):
    """
    Creates a split by MESSAGE within each participant for digital twin models.
    Each participant has some messages in "profile" (train) and some in "test".
    
    Args:
        data: Full dataset
        train_ratio: Proportion for training (e.g., 0.5 for 50/50, 0.7 for 70/30)
        output_dir: Where to save splits
        split_name: Descriptive name (e.g., "5050", "7030", "9010")
        participant_ids: Optional set of participants to include (for consistency with generic LLM splits)
    """
    print(f"\n{'='*60}")
    print(f"Creating DIGITAL TWIN split: {split_name} (train_ratio={train_ratio})")
    print(f"{'='*60}")
    
    df = pd.DataFrame(data)
    
    # Optionally filter to specific participants
    if participant_ids is not None:
        df = df[df['response_id'].isin(participant_ids)]
        data = df.to_dict('records')
    
    train_data = []
    test_data = []
    
    rng = np.random.RandomState(SEED)
    
    participant_stats = []
    
    # For each participant, split their messages
    for participant_id in df['response_id'].unique():
        participant_messages = df[df['response_id'] == participant_id]
        messages = participant_messages.to_dict('records')
        
        # Shuffle this participant's messages
        indices = np.arange(len(messages))
        rng.shuffle(indices)
        
        # Split
        split_point = int(len(messages) * train_ratio)
        train_indices = indices[:split_point]
        test_indices = indices[split_point:]
        
        participant_train = [messages[i] for i in train_indices]
        participant_test = [messages[i] for i in test_indices]
        
        train_data.extend(participant_train)
        test_data.extend(participant_test)
        
        participant_stats.append({
            'participant_id': participant_id,
            'total_messages': len(messages),
            'train_messages': len(participant_train),
            'test_messages': len(participant_test)
        })
    
    # Save splits
    train_path = os.path.join(output_dir, f'train_digital_twin_{split_name}.json')
    test_path = os.path.join(output_dir, f'test_digital_twin_{split_name}.json')
    
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(test_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    # Save metadata
    metadata = {
        'split_type': 'by_message_within_participant',
        'split_name': split_name,
        'train_ratio': train_ratio,
        'seed': SEED,
        'total_participants': len(participant_stats),
        'train_samples': len(train_data),
        'test_samples': len(test_data),
        'participant_stats': participant_stats
    }
    
    metadata_path = os.path.join(output_dir, f'metadata_digital_twin_{split_name}.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Total participants: {len(participant_stats)}")
    print(f"  Train: {len(train_data)} samples (profile messages)")
    print(f"  Test:  {len(test_data)} samples (held-out messages)")
    print(f"  ✓ Saved to {train_path} and {test_path}")


def main():
    # Setup
    data_path = 'data/processed_llm_data.json'
    output_dir = 'data_splits/canonical'
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("CREATING CANONICAL TRAIN/TEST SPLITS FOR ALL EXPERIMENTS")
    print("="*80)
    print(f"Seed: {SEED}")
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    
    # Load data
    data = load_data(data_path)
    print(f"\nTotal samples: {len(data)}")
    
    # =========================================================================
    # PART 1: Generic LLM & Hybrid ML-LLM Models (Split by PARTICIPANT)
    # =========================================================================
    print("\n" + "="*80)
    print("PART 1: Generic LLM & Hybrid ML-LLM Models")
    print("Split strategy: BY PARTICIPANT (no participant leakage)")
    print("="*80)
    
    # Create 70/30 split (default for all generic LLM and hybrid methods)
    train_participants_70, test_participants_70 = create_participant_split(
        data, train_ratio=0.7, output_dir=output_dir, split_name="7030"
    )
    
    # Create 30/70 split (alternative)
    train_participants_30, test_participants_30 = create_participant_split(
        data, train_ratio=0.3, output_dir=output_dir, split_name="3070"
    )
    
    # =========================================================================
    # PART 2: Digital Twin Models (Split by MESSAGE within participant)
    # =========================================================================
    print("\n" + "="*80)
    print("PART 2: Digital Twin Models")
    print("Split strategy: BY MESSAGE WITHIN PARTICIPANT (personalized profiles)")
    print("="*80)
    
    # Create 50/50 split
    create_digital_twin_split(
        data, train_ratio=0.5, output_dir=output_dir, split_name="5050"
    )
    
    # Create 70/30 split
    create_digital_twin_split(
        data, train_ratio=0.7, output_dir=output_dir, split_name="7030"
    )
    
    # Create 90/10 split
    create_digital_twin_split(
        data, train_ratio=0.9, output_dir=output_dir, split_name="9010"
    )
    
    # =========================================================================
    # PART 3: Create README documenting the splits
    # =========================================================================
    readme_content = f"""# Canonical Data Splits

**DO NOT MODIFY THESE SPLITS!** All experiments must use these exact splits for fair comparison.

## Split Information
- **Seed**: {SEED}
- **Source Data**: `{data_path}`

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
"""
    
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print("\n" + "="*80)
    print("✓ ALL CANONICAL SPLITS CREATED SUCCESSFULLY")
    print("="*80)
    print(f"\nSplits saved to: {output_dir}/")
    print(f"Documentation: {readme_path}")
    print("\nNext steps:")
    print("1. Update all evaluation scripts to use these canonical splits")
    print("2. Re-run all experiments with the same train/test sets")
    print("3. Compare results with confidence that splits are consistent")


if __name__ == '__main__':
    main()

