"""
Traditional ML Baseline for Manuscript

Trains regression models on participant characteristics using the canonical 70/30 split.
Provides baseline comparison for LLM methods.
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import argparse
import os


def load_split(split_path):
    """Load a data split."""
    with open(split_path, 'r') as f:
        return json.load(f)


def prepare_features(data_items):
    """Extract participant features and labels."""
    records = []
    for item in data_items:
        metadata = item.get('metadata', {})
        ratings = item.get('ratings', {})
        
        record = {
            'response_id': item.get('response_id'),
            'age_years': metadata.get('age_years'),
            'gender_identity': metadata.get('gender_identity'),
            'smoking_status': metadata.get('smoking_status'),
            'cigs_per_day': metadata.get('cigs_per_day'),
            'quit_attempts_count': metadata.get('quit_attempts_count'),
            'quit_motivation_level': metadata.get('quit_motivation_level'),
            'social_support_to_quit': metadata.get('social_support_to_quit'),
            'content': ratings.get('content'),
            'design': ratings.get('design'),
            'coping': ratings.get('coping'),
            'quitting': ratings.get('quitting')
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Encode categorical variables
    categorical_cols = ['gender_identity', 'smoking_status', 'quit_motivation_level', 'social_support_to_quit']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Rating mappings
    rating_maps = {
        'content': {'Very poor': 1, 'Poor': 2, 'Acceptable': 3, 'Good': 4, 'Very good': 5},
        'design': {'Very poor': 1, 'Poor': 2, 'Acceptable': 3, 'Good': 4, 'Very good': 5},
        'coping': {'Not at all helpful': 1, 'Somewhat helpful': 2, 'Moderately helpful': 3, 'Very helpful': 4, 'Extremely helpful': 5},
        'quitting': {'Not at all helpful': 1, 'Somewhat helpful': 2, 'Moderately helpful': 3, 'Very helpful': 4, 'Extremely helpful': 5}
    }
    
    for domain in ['content', 'design', 'coping', 'quitting']:
        df_encoded[f'{domain}_num'] = df[domain].map(rating_maps[domain])
    
    return df_encoded


def train_and_evaluate(train_df, test_df, domains, model_type='logistic'):
    """Train and evaluate models for each domain."""
    feature_cols = [col for col in train_df.columns if col not in 
                    ['response_id', 'content', 'design', 'coping', 'quitting', 
                     'content_num', 'design_num', 'coping_num', 'quitting_num']]
    
    results = {}
    
    for domain in domains:
        label_col = f'{domain}_num'
        
        # Prepare data
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[label_col].dropna()
        X_train = X_train.loc[y_train.index]
        
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df[label_col].dropna()
        X_test = X_test.loc[y_test.index]
        
        if len(y_train) == 0 or len(y_test) == 0:
            continue
        
        # Train model
        if model_type == 'logistic':
            model = LogisticRegression(max_iter=1000, random_state=202509)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=202509)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[domain] = {
            'accuracy': accuracy,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'predictions': y_pred.tolist(),
            'ground_truth': y_test.tolist()
        }
        
        print(f"  {domain.capitalize()}: Accuracy = {accuracy:.3f} (n_test={len(y_test)})")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run traditional ML baseline")
    parser.add_argument('--train-split', default='data_splits/canonical/train_participant_7030.json',
                        help="Path to training split")
    parser.add_argument('--test-split', default='data_splits/canonical/test_participant_7030.json',
                        help="Path to test split")
    parser.add_argument('--model-type', choices=['logistic', 'random_forest'], default='logistic',
                        help="Type of model to train")
    parser.add_argument('--output-dir', default='results_manuscript',
                        help="Directory to save results")
    args = parser.parse_args()
    
    print("="*80)
    print("TRADITIONAL ML BASELINE")
    print("="*80)
    print(f"Model type: {args.model_type}")
    print(f"Train split: {args.train_split}")
    print(f"Test split: {args.test_split}")
    print()
    
    # Load data
    train_data = load_split(args.train_split)
    test_data = load_split(args.test_split)
    
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print()
    
    # Prepare features
    train_df = prepare_features(train_data)
    test_df = prepare_features(test_data)
    
    # Train and evaluate
    domains = ['content', 'coping', 'quitting']  # Excluding design per manuscript
    print("Training and evaluating models...")
    results = train_and_evaluate(train_df, test_df, domains, args.model_type)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f'traditional_ml_{args.model_type}.json')
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print(f"✓ Results saved to: {output_path}")
    print()
    print("Summary:")
    for domain, metrics in results.items():
        print(f"  {domain.capitalize()}: {metrics['accuracy']:.3f}")


if __name__ == '__main__':
    main()

