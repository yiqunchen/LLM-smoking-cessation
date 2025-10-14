import json
import pandas as pd
import numpy as np
import argparse
import os

def load_data(data_path: str) -> list:
    """Loads the main processed data file."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def split_by_message(data: list, seed: int, output_dir: str, train_ratio: float = 0.5):
    """Splits data based on unique messages."""
    print(f"Splitting by message with seed {seed} (train ratio: {train_ratio})...")
    
    df = pd.DataFrame(data)
    unique_messages = df['input_message'].unique()
    
    # Reproducible shuffle
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_messages)
    
    # Split messages based on ratio
    split_point = int(len(unique_messages) * train_ratio)
    train_messages = set(unique_messages[:split_point])
    
    train_data = [item for item in data if item['input_message'] in train_messages]
    test_data = [item for item in data if item['input_message'] not in train_messages]
    
    print(f"  Total unique messages: {len(unique_messages)}")
    print(f"  Train messages: {len(train_messages)} | Test messages: {len(unique_messages) - len(train_messages)}")
    print(f"  Train samples: {len(train_data)} | Test samples: {len(test_data)}")
    
    # Create filenames based on split ratio
    ratio_str = f"{int(train_ratio*100)}{int((1-train_ratio)*100)}"
    train_path = os.path.join(output_dir, f'train_by_message_{ratio_str}.json')
    test_path = os.path.join(output_dir, f'test_by_message_{ratio_str}.json')
    
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(test_path, 'w') as f:
        json.dump(test_data, f, indent=2)
        
    print(f"  Saved splits to {train_path} and {test_path}")

def split_by_participant(data: list, seed: int, output_dir: str, train_ratio: float = 0.5):
    """Splits data based on unique participants (response_id)."""
    print(f"Splitting by participant with seed {seed} (train ratio: {train_ratio})...")
    
    df = pd.DataFrame(data)
    unique_participants = df['response_id'].unique()
    
    # Reproducible shuffle
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_participants)
    
    # Split participants based on ratio
    split_point = int(len(unique_participants) * train_ratio)
    train_participants = set(unique_participants[:split_point])
    
    train_data = [item for item in data if item['response_id'] in train_participants]
    test_data = [item for item in data if item['response_id'] not in train_participants]
    
    print(f"  Total unique participants: {len(unique_participants)}")
    print(f"  Train participants: {len(train_participants)} | Test participants: {len(unique_participants) - len(train_participants)}")
    print(f"  Train samples: {len(train_data)} | Test samples: {len(test_data)}")
    
    # Create filenames based on split ratio
    ratio_str = f"{int(train_ratio*100)}{int((1-train_ratio)*100)}"
    train_path = os.path.join(output_dir, f'train_by_participant_{ratio_str}.json')
    test_path = os.path.join(output_dir, f'test_by_participant_{ratio_str}.json')
    
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(test_path, 'w') as f:
        json.dump(test_data, f, indent=2)
        
    print(f"  Saved splits to {train_path} and {test_path}")

def main():
    parser = argparse.ArgumentParser(description="Create train/test splits from the processed data.")
    parser.add_argument('--data-path', default='data/processed_llm_data.json', help="Path to the main data file.")
    parser.add_argument('--output-dir', default='data_splits', help="Directory to save the output split files.")
    parser.add_argument('--split-by', required=True, choices=['message', 'participant'], help="How to create the splits.")
    parser.add_argument('--seed', type=int, default=202509, help="Random seed for reproducibility.")
    parser.add_argument('--train-ratio', type=float, default=0.7, 
                        help="Ratio of data for training (default: 0.7 for 70/30 split). Use 0.3 for 30/70 split.")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    data = load_data(args.data_path)
    
    if args.split_by == 'message':
        split_by_message(data, args.seed, args.output_dir, args.train_ratio)
    elif args.split_by == 'participant':
        split_by_participant(data, args.seed, args.output_dir, args.train_ratio)
        
if __name__ == '__main__':
    main()


