"""
Create a hybrid model: Random Forest (all features) + Grok-4 Digital Twin (70/30).

This combines:
1. Random Forest predictions on ALL participant metadata features
2. Grok-4 Digital Twin 70/30 predictions
3. Ensemble: Take RF prediction as an additional "confidence" feature for the LLM

Strategy: Run RF first, then pass its prediction + confidence as additional context
to the digital twin prompt.
"""

import json
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, cohen_kappa_score
from scipy.stats import spearmanr

# Rating mappings
RATING_MAPS = {
    'content': {'Very poor': 1, 'Poor': 2, 'Acceptable': 3, 'Good': 4, 'Very good': 5},
    'coping': {
        'Not at all helpful': 1, 'Somewhat helpful': 2, 'Moderately helpful': 3,
        'Very helpful': 4, 'Extremely helpful': 5, 'Not Helpful': 1
    },
    'quitting': {
        'Not at all helpful': 1, 'Somewhat helpful': 2, 'Moderately helpful': 3,
        'Very helpful': 4, 'Extremely helpful': 5, 'Not Helpful': 1
    }
}

DOMAINS = ['content', 'coping', 'quitting']


def train_rf_on_all_features():
    """Train Random Forest on ALL metadata features using Participant 70/30 split."""
    
    print("=" * 80)
    print("TRAINING RANDOM FOREST ON ALL FEATURES")
    print("Using Participant 70/30 split (SAME AS ALL OTHER METHODS)")
    print("=" * 80)
    
    # Load train and test data (MUST match all other methods!)
    with open('data_splits/canonical/train_participant_7030.json') as f:
        train_data = json.load(f)
    with open('data_splits/canonical/test_participant_7030.json') as f:
        test_data = json.load(f)
    
    train_df = pd.DataFrame([item for item in train_data if isinstance(item, dict)])
    test_df = pd.DataFrame([item for item in test_data if isinstance(item, dict)])
    
    print(f"\nTrain size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    
    # Extract ALL metadata features
    def extract_all_features(df):
        features = []
        for _, row in df.iterrows():
            metadata = row.get('metadata', {})
            if not isinstance(metadata, dict):
                metadata = {}
            feat = {}
            
            # Numeric features
            numeric_features = [
                'age_years', 'days_smoked_past_30d', 'cigs_per_day',
                'quit_attempts_count', 'pain_blocks_valued_life', 'fear_of_feelings',
                'worry_about_control', 'memories_block_fulfillment', 'emotions_cause_problems'
            ]
            for col in numeric_features:
                val = metadata.get(col)
                try:
                    feat[col] = float(val) if val is not None else 0.0
                except:
                    feat[col] = 0.0
            
            # Categorical features (one-hot encode)
            categorical_features = [
                'gender_identity', 'race_ethnicity', 'is_hispanic_latino',
                'quit_intention', 'sexual_orientation_identity', 'education_level',
                'household_income', 'time_to_first_cig', 'household_smokers',
                'friends_smoke_level', 'quit_attempt_past_year'
            ]
            for col in categorical_features:
                val = metadata.get(col)
                if val is not None:
                    feat[f'{col}_{val}'] = 1.0
            
            features.append(feat)
        return pd.DataFrame(features).fillna(0)
    
    X_train = extract_all_features(train_df)
    X_test = extract_all_features(test_df)
    
    print(f"\nFeature count: {X_train.shape[1]}")
    print(f"Sample features: {list(X_train.columns)[:10]}")
    
    # Align columns
    all_cols = list(set(X_train.columns) | set(X_test.columns))
    for col in all_cols:
        if col not in X_train.columns:
            X_train[col] = 0
        if col not in X_test.columns:
            X_test[col] = 0
    X_train = X_train[sorted(all_cols)]
    X_test = X_test[sorted(all_cols)]
    
    # Train RF for each domain and save predictions
    rf_predictions = {}
    
    for domain in DOMAINS:
        print(f"\n{'='*60}")
        print(f"Training Random Forest for {domain.upper()}")
        print(f"{'='*60}")
        
        # Extract labels
        y_train = train_df['ratings'].apply(
            lambda x: RATING_MAPS[domain].get(x.get(domain)) if isinstance(x, dict) else None
        )
        y_test = test_df['ratings'].apply(
            lambda x: RATING_MAPS[domain].get(x.get(domain)) if isinstance(x, dict) else None
        )
        
        # Remove NaN
        train_mask = y_train.notna()
        test_mask = y_test.notna()
        
        X_train_clean = X_train[train_mask]
        y_train_clean = y_train[train_mask]
        X_test_clean = X_test[test_mask]
        y_test_clean = y_test[test_mask]
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train_clean, y_train_clean)
        
        # Predict on test set
        pred_rf = rf.predict(X_test_clean)
        pred_rf_proba = rf.predict_proba(X_test_clean)
        
        # Calculate metrics
        acc_rf = accuracy_score(y_test_clean, pred_rf)
        acc_within_1 = np.mean(np.abs(y_test_clean - pred_rf) <= 1)
        kappa_rf = cohen_kappa_score(y_test_clean, pred_rf)
        
        print(f"  Accuracy: {acc_rf:.3f}")
        print(f"  Accuracy ±1: {acc_within_1:.3f}")
        print(f"  Cohen's Kappa: {kappa_rf:.3f}")
        
        # Store predictions with response IDs
        test_df_clean = test_df[test_mask].copy()
        test_df_clean['rf_prediction'] = pred_rf
        test_df_clean['rf_confidence'] = pred_rf_proba.max(axis=1)
        test_df_clean['ground_truth'] = y_test_clean.values
        
        rf_predictions[domain] = test_df_clean[['response_id', 'input_message', 
                                                  'rf_prediction', 'rf_confidence', 'ground_truth']]
    
    # Save RF predictions for use in hybrid model
    # Structure: {response_id: {rf_pred_content: X, rf_pred_design: X, rf_pred_coping: X, rf_pred_quitting: X}}
    output_dir = 'results_manuscript_hybrid_rf_grok4'
    os.makedirs(output_dir, exist_ok=True)
    
    # Reorganize by response_id
    predictions_by_id = {}
    for domain, predictions_df in rf_predictions.items():
        for _, row in predictions_df.iterrows():
            response_id = row['response_id']
            if response_id not in predictions_by_id:
                predictions_by_id[response_id] = {}
            predictions_by_id[response_id][f'rf_pred_{domain}'] = int(row['rf_prediction'])
    
    rf_pred_file = os.path.join(output_dir, 'rf_predictions_all_features.json')
    with open(rf_pred_file, 'w') as f:
        json.dump(predictions_by_id, f, indent=2)
    
    print(f"\n✓ Saved RF predictions to: {rf_pred_file}")
    print(f"  Format: {{response_id: {{rf_pred_content: X, rf_pred_design: X, ...}}}}")
    print(f"  Total response IDs: {len(predictions_by_id)}")
    print("\nNext step: Run Grok-4 Digital Twin with RF predictions as additional context")
    
    return rf_predictions


def main():
    print("\n" + "="*80)
    print("HYBRID MODEL: Random Forest (All Features) + Grok-4 Digital Twin")
    print("="*80 + "\n")
    
    # Step 1: Train RF on all features
    rf_predictions = train_rf_on_all_features()
    
    print("\n" + "="*80)
    print("✓ RF Training Complete!")
    print("="*80)
    print("\nTo complete the hybrid model:")
    print("1. RF predictions saved to results_manuscript_hybrid_rf_grok4/")
    print("2. Next: Modify digital twin prompt to include RF prediction as context")
    print("3. Run: bash run_hybrid_eval.sh")


if __name__ == '__main__':
    main()

