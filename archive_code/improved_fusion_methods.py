#!/usr/bin/env python3
"""
Improved Fusion Methods for Individual Covariates + Message Quality

This script implements better ways to combine individual characteristics 
with message quality features, addressing the noise accumulation problem
in the naive weighted average approach.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import json
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load the evaluation data and prepare features."""
    print("Loading evaluation data...")
    
    # Load the main evaluation results
    with open('evaluation_results_gpt-4o-mini_vision_zero-shot.json', 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    rows = []
    for key, item in data.items():
        row = {
            'response_id': item['response_id'],
            'input_message': item['input_message']
        }
        
        # Add metadata (individual characteristics)
        if 'metadata' in item:
            row.update(item['metadata'])
        
        # Add ground truth ratings
        for rating_type in ['content', 'design', 'coping', 'quitting']:
            gt_key = f'ground_truth_{rating_type}'
            if gt_key in item:
                # Convert categorical to numeric
                rating_map = {
                    'Very poor': 1, 'Poor': 2, 'Acceptable': 3, 'Good': 4, 'Very good': 5,
                    'Not at all helpful': 1, 'Slightly helpful': 2, 'Moderately helpful': 3,
                    'Very helpful': 4, 'Extremely helpful': 5
                }
                row[f'{rating_type}_rating'] = rating_map.get(item[gt_key], 3)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} data points")
    
    # Prepare individual characteristic features
    individual_features = []
    
    # Numeric features
    if 'age_years' in df.columns:
        df['age_years'] = pd.to_numeric(df['age_years'], errors='coerce')
        individual_features.append('age_years')
    
    if 'days_smoked_past_30d' in df.columns:
        df['days_smoked_past_30d'] = pd.to_numeric(df['days_smoked_past_30d'], errors='coerce')
        individual_features.append('days_smoked_past_30d')
    
    # Categorical features (one-hot encoded)
    categorical_features = ['gender_identity', 'race_ethnicity', 'education_level', 
                           'household_income', 'quit_intention', 'quit_motivation_level',
                           'social_support_to_quit']
    
    for feature in categorical_features:
        if feature in df.columns:
            # One-hot encode
            dummies = pd.get_dummies(df[feature], prefix=feature, dummy_na=True)
            df = pd.concat([df, dummies], axis=1)
            individual_features.extend(dummies.columns.tolist())
    
    return df, individual_features

def calculate_message_quality_proxies(df, individual_features):
    """Calculate better message quality proxies using multiple methods."""
    print("Calculating message quality proxies...")
    
    # Method 1: Participant-adjusted average ratings per message
    message_quality_features = []
    
    for rating_type in ['content', 'design', 'coping', 'quitting']:
        rating_col = f'{rating_type}_rating'
        if rating_col in df.columns:
            
            # Calculate raw message averages
            raw_avg = df.groupby('input_message')[rating_col].mean()
            df[f'msg_raw_avg_{rating_type}'] = df['input_message'].map(raw_avg)
            message_quality_features.append(f'msg_raw_avg_{rating_type}')
            
            # Calculate participant-type-adjusted averages
            # Group by key demographics and calculate residuals
            if 'gender_identity' in df.columns and 'age_years' in df.columns:
                df['age_group'] = pd.cut(df['age_years'], bins=[0, 25, 35, 50, 100], labels=['18-25', '26-35', '36-50', '50+'])
                
                # Calculate expected rating based on participant characteristics
                participant_avg = df.groupby(['gender_identity', 'age_group'])[rating_col].mean()
                df['participant_expected'] = df.set_index(['gender_identity', 'age_group']).index.map(participant_avg)
                df['participant_expected'] = df['participant_expected'].fillna(df[rating_col].mean())
                
                # Calculate residuals (how much better/worse than expected)
                df[f'rating_residual_{rating_type}'] = df[rating_col] - df['participant_expected']
                
                # Message quality = average residual for that message
                msg_residual_avg = df.groupby('input_message')[f'rating_residual_{rating_type}'].mean()
                df[f'msg_quality_{rating_type}'] = df['input_message'].map(msg_residual_avg)
                message_quality_features.append(f'msg_quality_{rating_type}')
            
            # Method 2: Variance-weighted message quality
            # Messages with high agreement (low variance) get higher weight
            msg_variance = df.groupby('input_message')[rating_col].var()
            msg_count = df.groupby('input_message')[rating_col].count()
            
            # Reliability weight = count / (1 + variance)
            reliability_weight = msg_count / (1 + msg_variance.fillna(1))
            df[f'msg_reliability_{rating_type}'] = df['input_message'].map(reliability_weight)
            message_quality_features.append(f'msg_reliability_{rating_type}')
    
    print(f"Created {len(message_quality_features)} message quality features")
    return df, message_quality_features

class HierarchicalPredictor:
    """
    Hierarchical approach: First predict message quality, then individual variation.
    """
    def __init__(self):
        self.message_model = LogisticRegression(random_state=42, max_iter=1000)
        self.individual_model = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler_msg = StandardScaler()
        self.scaler_ind = StandardScaler()
        
    def fit(self, X_individual, X_message, y):
        """
        Fit hierarchical model.
        Step 1: Predict using message features
        Step 2: Use individual features to predict residuals
        """
        # Step 1: Train message quality model
        X_msg_scaled = self.scaler_msg.fit_transform(X_message)
        self.message_model.fit(X_msg_scaled, y)
        
        # Get message predictions
        msg_pred_proba = self.message_model.predict_proba(X_msg_scaled)
        msg_pred = self.message_model.predict(X_msg_scaled)
        
        # Step 2: Train individual model to predict residuals/corrections
        # Use both individual features AND message predictions as input
        residuals = (y != msg_pred).astype(int)  # Binary: was message model wrong?
        
        X_combined = np.concatenate([X_individual, msg_pred_proba], axis=1)
        X_combined_scaled = self.scaler_ind.fit_transform(X_combined)
        self.individual_model.fit(X_combined_scaled, residuals)
        
        return self
    
    def predict(self, X_individual, X_message):
        """Make hierarchical predictions."""
        # Step 1: Message predictions
        X_msg_scaled = self.scaler_msg.transform(X_message)
        msg_pred_proba = self.message_model.predict_proba(X_msg_scaled)
        msg_pred = self.message_model.predict(X_msg_scaled)
        
        # Step 2: Individual corrections
        X_combined = np.concatenate([X_individual, msg_pred_proba], axis=1)
        X_combined_scaled = self.scaler_ind.transform(X_combined)
        correction_prob = self.individual_model.predict_proba(X_combined_scaled)
        
        # Final prediction: blend based on correction confidence
        final_pred = msg_pred.copy()
        # If correction model is confident the message model is wrong, adjust
        high_correction_confidence = correction_prob[:, 1] > 0.7
        
        # For high-confidence corrections, shift prediction based on individual characteristics
        # This is a simplified approach - in practice you'd want more sophisticated blending
        return final_pred

class ResidualPredictor:
    """
    Residual approach: Predict individual baseline, then message adds/subtracts.
    """
    def __init__(self):
        self.individual_model = LogisticRegression(random_state=42, max_iter=1000)
        self.message_model = Ridge(random_state=42)  # Predicts continuous adjustment
        self.scaler_ind = StandardScaler()
        self.scaler_msg = StandardScaler()
        
    def fit(self, X_individual, X_message, y):
        """
        Step 1: Predict individual baseline rating tendency
        Step 2: Use message features to predict adjustment from baseline
        """
        # Step 1: Individual baseline model
        X_ind_scaled = self.scaler_ind.fit_transform(X_individual)
        self.individual_model.fit(X_ind_scaled, y)
        individual_pred = self.individual_model.predict(X_ind_scaled)
        
        # Step 2: Message adjustment model
        # Predict how much message quality adjusts from individual baseline
        adjustment = y - individual_pred  # Residual
        X_msg_scaled = self.scaler_msg.fit_transform(X_message)
        self.message_model.fit(X_msg_scaled, adjustment)
        
        return self
    
    def predict(self, X_individual, X_message):
        """Make residual-based predictions."""
        # Individual baseline
        X_ind_scaled = self.scaler_ind.transform(X_individual)
        individual_baseline = self.individual_model.predict(X_ind_scaled)
        
        # Message adjustment
        X_msg_scaled = self.scaler_msg.transform(X_message)
        message_adjustment = self.message_model.predict(X_msg_scaled)
        
        # Combined prediction
        combined = individual_baseline + message_adjustment
        
        # Clip to valid range and round
        final_pred = np.clip(np.round(combined), 1, 5).astype(int)
        return final_pred

class EnsembleWithGating:
    """
    Ensemble with gating: Use individual characteristics to decide whether to trust message model.
    """
    def __init__(self):
        self.individual_model = LogisticRegression(random_state=42, max_iter=1000)
        self.message_model = LogisticRegression(random_state=42, max_iter=1000)
        self.gating_model = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler_ind = StandardScaler()
        self.scaler_msg = StandardScaler()
        self.scaler_gate = StandardScaler()
        
    def fit(self, X_individual, X_message, y):
        """Train ensemble with gating mechanism."""
        # Train individual and message models separately
        X_ind_scaled = self.scaler_ind.fit_transform(X_individual)
        X_msg_scaled = self.scaler_msg.fit_transform(X_message)
        
        self.individual_model.fit(X_ind_scaled, y)
        self.message_model.fit(X_msg_scaled, y)
        
        # Get predictions from both models
        ind_pred = self.individual_model.predict(X_ind_scaled)
        msg_pred = self.message_model.predict(X_msg_scaled)
        
        # Train gating model to predict which model to trust
        # Features: individual characteristics + difference between model predictions
        pred_diff = np.abs(ind_pred - msg_pred).reshape(-1, 1)
        X_gating = np.concatenate([X_individual, pred_diff], axis=1)
        X_gating_scaled = self.scaler_gate.fit_transform(X_gating)
        
        # Target: which model was more accurate (1 = message model, 0 = individual model)
        ind_correct = (ind_pred == y).astype(int)
        msg_correct = (msg_pred == y).astype(int)
        use_message_model = (msg_correct >= ind_correct).astype(int)
        
        self.gating_model.fit(X_gating_scaled, use_message_model)
        return self
    
    def predict(self, X_individual, X_message):
        """Make gated ensemble predictions."""
        X_ind_scaled = self.scaler_ind.transform(X_individual)
        X_msg_scaled = self.scaler_msg.transform(X_message)
        
        ind_pred = self.individual_model.predict(X_ind_scaled)
        msg_pred = self.message_model.predict(X_msg_scaled)
        
        # Gating decision
        pred_diff = np.abs(ind_pred - msg_pred).reshape(-1, 1)
        X_gating = np.concatenate([X_individual, pred_diff], axis=1)
        X_gating_scaled = self.scaler_gate.transform(X_gating)
        
        use_message = self.gating_model.predict(X_gating_scaled)
        
        # Final prediction: choose based on gating
        final_pred = np.where(use_message, msg_pred, ind_pred)
        return final_pred

def evaluate_fusion_methods(df, individual_features, message_features, rating_type):
    """Evaluate different fusion methods for a specific rating type."""
    print(f"\n=== Evaluating fusion methods for {rating_type.upper()} ===")
    
    rating_col = f'{rating_type}_rating'
    if rating_col not in df.columns:
        print(f"No data for {rating_type}")
        return {}
    
    # Prepare data
    model_df = df[df[rating_col].notna()].copy()
    if len(model_df) < 100:
        print(f"Insufficient data: {len(model_df)}")
        return {}
    
    # Features
    X_individual = model_df[individual_features].fillna(0).values
    
    # Message features (only use available ones)
    available_msg_features = [f for f in message_features if f in model_df.columns]
    if not available_msg_features:
        print("No message features available")
        return {}
    
    X_message = model_df[available_msg_features].fillna(model_df[available_msg_features].mean()).values
    y = model_df[rating_col].values
    
    # Train/test split
    X_ind_train, X_ind_test, X_msg_train, X_msg_test, y_train, y_test = train_test_split(
        X_individual, X_message, y, test_size=0.3, random_state=42, stratify=y
    )
    
    results = {}
    
    # Baseline: Individual only
    individual_model = LogisticRegression(random_state=42, max_iter=1000)
    scaler = StandardScaler()
    X_ind_train_scaled = scaler.fit_transform(X_ind_train)
    X_ind_test_scaled = scaler.transform(X_ind_test)
    individual_model.fit(X_ind_train_scaled, y_train)
    ind_pred = individual_model.predict(X_ind_test_scaled)
    results['individual_only'] = accuracy_score(y_test, ind_pred)
    
    # Baseline: Message only  
    message_model = LogisticRegression(random_state=42, max_iter=1000)
    scaler_msg = StandardScaler()
    X_msg_train_scaled = scaler_msg.fit_transform(X_msg_train)
    X_msg_test_scaled = scaler_msg.transform(X_msg_test)
    message_model.fit(X_msg_train_scaled, y_train)
    msg_pred = message_model.predict(X_msg_test_scaled)
    results['message_only'] = accuracy_score(y_test, msg_pred)
    
    # Method 1: Hierarchical Predictor
    try:
        hierarchical = HierarchicalPredictor()
        hierarchical.fit(X_ind_train, X_msg_train, y_train)
        hier_pred = hierarchical.predict(X_ind_test, X_msg_test)
        results['hierarchical'] = accuracy_score(y_test, hier_pred)
    except Exception as e:
        print(f"Hierarchical method failed: {e}")
        results['hierarchical'] = 0.0
    
    # Method 2: Residual Predictor
    try:
        residual = ResidualPredictor()
        residual.fit(X_ind_train, X_msg_train, y_train)
        res_pred = residual.predict(X_ind_test, X_msg_test)
        results['residual'] = accuracy_score(y_test, res_pred)
    except Exception as e:
        print(f"Residual method failed: {e}")
        results['residual'] = 0.0
    
    # Method 3: Ensemble with Gating
    try:
        ensemble = EnsembleWithGating()
        ensemble.fit(X_ind_train, X_msg_train, y_train)
        ens_pred = ensemble.predict(X_ind_test, X_msg_test)
        results['ensemble_gating'] = accuracy_score(y_test, ens_pred)
    except Exception as e:
        print(f"Ensemble method failed: {e}")
        results['ensemble_gating'] = 0.0
    
    # Method 4: Simple concatenation (for comparison)
    X_combined = np.concatenate([X_individual, X_message], axis=1)
    X_comb_train, X_comb_test, y_train_comb, y_test_comb = train_test_split(
        X_combined, y, test_size=0.3, random_state=42, stratify=y
    )
    combined_model = LogisticRegression(random_state=42, max_iter=1000)
    scaler_comb = StandardScaler()
    X_comb_train_scaled = scaler_comb.fit_transform(X_comb_train)
    X_comb_test_scaled = scaler_comb.transform(X_comb_test)
    combined_model.fit(X_comb_train_scaled, y_train_comb)
    comb_pred = combined_model.predict(X_comb_test_scaled)
    results['concatenation'] = accuracy_score(y_test_comb, comb_pred)
    
    # Print results
    print(f"Results for {rating_type}:")
    for method, accuracy in results.items():
        print(f"  {method:20s}: {accuracy:.3f}")
    
    return results

def main():
    """Main evaluation of improved fusion methods."""
    print("=== IMPROVED FUSION METHODS EVALUATION ===\n")
    
    # Load data
    df, individual_features = load_and_prepare_data()
    
    # Calculate message quality proxies
    df, message_features = calculate_message_quality_proxies(df, individual_features)
    
    print(f"\nFeature summary:")
    print(f"  Individual features: {len(individual_features)}")
    print(f"  Message features: {len(message_features)}")
    
    # Evaluate for each rating type
    all_results = {}
    for rating_type in ['content', 'design', 'coping', 'quitting']:
        results = evaluate_fusion_methods(df, individual_features, message_features, rating_type)
        all_results[rating_type] = results
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY OF ALL METHODS")
    print("="*60)
    
    # Calculate averages across rating types
    all_methods = set()
    for results in all_results.values():
        all_methods.update(results.keys())
    
    for method in sorted(all_methods):
        accuracies = [all_results[rt].get(method, 0) for rt in ['content', 'design', 'coping', 'quitting']]
        avg_accuracy = np.mean([acc for acc in accuracies if acc > 0])
        print(f"{method:20s}: {avg_accuracy:.3f} (avg)")
        for rt, acc in zip(['content', 'design', 'coping', 'quitting'], accuracies):
            if acc > 0:
                print(f"  {rt}: {acc:.3f}")
    
    print("\n=== CONCLUSIONS ===")
    print("Best performing methods:")
    method_avgs = {}
    for method in all_methods:
        accuracies = [all_results[rt].get(method, 0) for rt in all_results.keys()]
        method_avgs[method] = np.mean([acc for acc in accuracies if acc > 0])
    
    sorted_methods = sorted(method_avgs.items(), key=lambda x: x[1], reverse=True)
    for i, (method, avg_acc) in enumerate(sorted_methods[:5]):
        print(f"{i+1}. {method}: {avg_acc:.3f}")

if __name__ == "__main__":
    main()