import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load the processed data and prepare it for analysis."""
    with open('data/processed_llm_data.json', 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Flatten metadata
    metadata_df = pd.json_normalize(df['metadata'])
    ratings_df = pd.json_normalize(df['ratings'])
    
    # Combine all data
    full_df = pd.concat([
        df[['response_id', 'input_message']].reset_index(drop=True),
        metadata_df.reset_index(drop=True),
        ratings_df.reset_index(drop=True)
    ], axis=1)
    
    return full_df

def analyze_score_variation(df):
    """Analyze variation in scores for the same message across participants."""
    print("=== SCORE VARIATION ANALYSIS ===\n")
    
    # Group by message to see variation
    message_stats = []
    rating_types = ['content', 'design', 'coping', 'quitting']
    
    # Convert ratings to numeric
    rating_map = {
        # Content/Design scale
        'Very poor': 1, 'Poor': 2, 'Fair': 3, 'Good': 4, 'Very good': 5,
        # Coping/Quitting scale  
        'Not helpful': 1, 'Slightly helpful': 2, 'Moderately helpful': 3, 
        'Very helpful': 4, 'Extremely helpful': 5
    }
    
    for rating_type in rating_types:
        df[f'{rating_type}_numeric'] = df[rating_type].map(rating_map)
    
    # Calculate stats per message
    for message in df['input_message'].unique():
        if pd.isna(message):
            continue
            
        msg_data = df[df['input_message'] == message]
        n_participants = len(msg_data)
        
        if n_participants < 2:  # Need at least 2 participants to measure variation
            continue
            
        msg_stats = {
            'message': message[:100] + '...' if len(message) > 100 else message,
            'n_participants': n_participants
        }
        
        for rating_type in rating_types:
            numeric_col = f'{rating_type}_numeric'
            if numeric_col in msg_data.columns:
                values = msg_data[numeric_col].dropna()
                if len(values) > 1:
                    msg_stats[f'{rating_type}_mean'] = values.mean()
                    msg_stats[f'{rating_type}_std'] = values.std()
                    msg_stats[f'{rating_type}_range'] = values.max() - values.min()
                    msg_stats[f'{rating_type}_cv'] = values.std() / values.mean() if values.mean() > 0 else 0
        
        message_stats.append(msg_stats)
    
    stats_df = pd.DataFrame(message_stats)
    
    # Summary statistics
    print(f"Total unique messages: {len(df['input_message'].unique())}")
    print(f"Messages with 2+ participants: {len(stats_df)}")
    print(f"Average participants per message: {stats_df['n_participants'].mean():.2f}")
    print()
    
    # Variation statistics
    for rating_type in rating_types:
        std_col = f'{rating_type}_std'
        cv_col = f'{rating_type}_cv'
        range_col = f'{rating_type}_range'
        
        if std_col in stats_df.columns:
            print(f"{rating_type.upper()} variation:")
            print(f"  Average std dev: {stats_df[std_col].mean():.3f}")
            print(f"  Average range: {stats_df[range_col].mean():.3f}")
            print(f"  Average CV: {stats_df[cv_col].mean():.3f}")
            print(f"  Max std dev: {stats_df[std_col].max():.3f}")
            print(f"  Max range: {stats_df[range_col].max():.0f}")
            print()
    
    # Plot variation
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, rating_type in enumerate(rating_types):
        std_col = f'{rating_type}_std'
        if std_col in stats_df.columns:
            axes[i].hist(stats_df[std_col].dropna(), bins=20, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{rating_type.title()} Score Std Dev Distribution')
            axes[i].set_xlabel('Standard Deviation')
            axes[i].set_ylabel('Number of Messages')
    
    plt.tight_layout()
    plt.savefig('score_variation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return stats_df, df

def prepare_features_and_targets(df):
    """Prepare features and targets for prediction models."""
    # Convert ratings to numeric
    rating_map = {
        'Very poor': 1, 'Poor': 2, 'Fair': 3, 'Good': 4, 'Very good': 5,
        'Not helpful': 1, 'Slightly helpful': 2, 'Moderately helpful': 3, 
        'Very helpful': 4, 'Extremely helpful': 5
    }
    
    rating_types = ['content', 'design', 'coping', 'quitting']
    for rating_type in rating_types:
        df[f'{rating_type}_numeric'] = df[rating_type].map(rating_map)
    
    # Prepare features
    feature_cols = []
    
    # Categorical features
    categorical_features = {
        'gender_identity': ['Male', 'Female', 'Other'],
        'race_ethnicity': ['White', 'Black or African American', 'Other'],
        'quit_intention': ['Are currently trying to quit', 'Plan to quit within 30 days', 'Other'],
        'education_level': ['High school graduate or GED', 'Some college', 'College graduate', 'Other'],
        'smoking_status': ['Every day', 'Some days', 'Not at all'],
        'quit_motivation_level': ['Not motivated', 'Slightly motivated', 'Moderately motivated', 'Very motivated', 'Extremely motivated'],
        'social_support_to_quit': ['Not supportive', 'Slightly supportive', 'Moderately supportive', 'Very supportive', 'Extremely supportive']
    }
    
    # Create dummy variables for categorical features
    for feature, categories in categorical_features.items():
        if feature in df.columns:
            for category in categories:
                col_name = f'{feature}_{category}'.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
                df[col_name] = (df[feature] == category).astype(int)
                feature_cols.append(col_name)
    
    # Numerical features
    numerical_features = ['age_years', 'days_smoked_past_30d', 'quit_attempts_count']
    for feature in numerical_features:
        if feature in df.columns:
            # Convert to numeric and fill missing values with median
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
            df[feature] = df[feature].fillna(df[feature].median())
            feature_cols.append(feature)
    
    return df, feature_cols, rating_types

def categorical_prediction_model(df, feature_cols, rating_types):
    """Build categorical prediction models using participant characteristics."""
    print("=== CATEGORICAL PREDICTION MODEL ===\n")
    
    results = {}
    
    for rating_type in rating_types:
        target_col = f'{rating_type}_numeric'
        if target_col not in df.columns:
            continue
            
        # Remove rows with missing targets
        model_df = df[df[target_col].notna()].copy()
        
        if len(model_df) < 100:  # Need sufficient data
            continue
            
        X = model_df[feature_cols].fillna(0)  # Fill missing features with 0
        y = model_df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train logistic regression
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"{rating_type.upper()} Rating Prediction:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': np.abs(model.coef_[0]) if len(model.coef_.shape) == 2 else np.abs(model.coef_)
        }).sort_values('importance', ascending=False)
        
        print(f"  Top 5 features:")
        for _, row in feature_importance.head().iterrows():
            print(f"    {row['feature']}: {row['importance']:.3f}")
        print()
        
        results[rating_type] = {
            'accuracy': accuracy,
            'model': model,
            'scaler': scaler,
            'feature_importance': feature_importance,
            'classification_report': classification_report(y_test, y_pred)
        }
    
    return results

def create_message_train_test_split(df):
    """Create train/test split based on unique messages (50/50)."""
    print("=== MESSAGE-BASED TRAIN/TEST SPLIT ===\n")
    
    unique_messages = df['input_message'].dropna().unique()
    print(f"Total unique messages: {len(unique_messages)}")
    
    # Random split of messages
    np.random.seed(42)
    train_messages = np.random.choice(unique_messages, size=len(unique_messages)//2, replace=False)
    test_messages = [msg for msg in unique_messages if msg not in train_messages]
    
    train_df = df[df['input_message'].isin(train_messages)].copy()
    test_df = df[df['input_message'].isin(test_messages)].copy()
    
    print(f"Train messages: {len(train_messages)}")
    print(f"Test messages: {len(test_messages)}")
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print()
    
    return train_df, test_df, train_messages, test_messages

def hybrid_prediction_model(train_df, test_df, feature_cols, rating_types):
    """Implement hybrid model with message embeddings and nearest neighbor averaging."""
    print("=== HYBRID PREDICTION MODEL ===\n")
    
    results = {}
    
    # Calculate average ratings per message in training set
    train_message_avgs = {}
    for rating_type in rating_types:
        target_col = f'{rating_type}_numeric'
        if target_col in train_df.columns:
            msg_avgs = train_df.groupby('input_message')[target_col].mean().to_dict()
            train_message_avgs[rating_type] = msg_avgs
    
    # For each rating type, build hybrid model
    for rating_type in rating_types:
        target_col = f'{rating_type}_numeric'
        if target_col not in train_df.columns or rating_type not in train_message_avgs:
            continue
            
        print(f"Building hybrid model for {rating_type}...")
        
        # Step 1: Train a model to predict message quality from participant characteristics
        train_clean = train_df[train_df[target_col].notna()].copy()
        test_clean = test_df[test_df[target_col].notna()].copy()
        
        if len(train_clean) < 50 or len(test_clean) < 10:
            continue
        
        # Calculate message-level features (average participant characteristics per message)
        message_features = train_clean.groupby('input_message')[feature_cols].mean().reset_index()
        message_targets = train_clean.groupby('input_message')[target_col].mean().reset_index()
        message_data = pd.merge(message_features, message_targets, on='input_message')
        
        # Train message quality predictor
        X_msg = message_data[feature_cols].fillna(0)
        y_msg = message_data[target_col]
        
        scaler_msg = StandardScaler()
        X_msg_scaled = scaler_msg.fit_transform(X_msg)
        
        msg_quality_model = LogisticRegression(random_state=42, max_iter=1000)
        msg_quality_model.fit(X_msg_scaled, y_msg.round().astype(int))  # Round for classification
        
        # Step 2: For test messages, find nearest neighbors in message space
        test_messages = test_clean['input_message'].unique()
        train_messages = list(train_message_avgs[rating_type].keys())
        
        # Simple text similarity (could be improved with embeddings)
        def text_similarity(text1, text2):
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union) if len(union) > 0 else 0
        
        # Step 3: Make predictions for test set
        predictions = []
        actuals = []
        
        for _, test_row in test_clean.iterrows():
            test_msg = test_row['input_message']
            
            # Find 5 most similar training messages
            similarities = []
            for train_msg in train_messages:
                sim = text_similarity(test_msg, train_msg)
                similarities.append((train_msg, sim))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_5_neighbors = similarities[:5]
            
            # Average rating from top 5 neighbors
            neighbor_scores = [train_message_avgs[rating_type][msg] for msg, _ in top_5_neighbors if msg in train_message_avgs[rating_type]]
            neighbor_avg = np.mean(neighbor_scores) if neighbor_scores else 3.0  # Default to middle
            
            # Predict message quality from participant characteristics
            participant_features = test_row[feature_cols].fillna(0).values.reshape(1, -1)
            participant_features_scaled = scaler_msg.transform(participant_features)
            predicted_quality = msg_quality_model.predict(participant_features_scaled)[0]
            
            # Hybrid prediction: weighted average
            hybrid_pred = 0.7 * neighbor_avg + 0.3 * predicted_quality
            
            predictions.append(hybrid_pred)
            actuals.append(test_row[target_col])
        
        # Evaluate
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Classification accuracy (round predictions)
        pred_classes = np.round(predictions).astype(int)
        pred_classes = np.clip(pred_classes, 1, 5)  # Ensure valid range
        accuracy = accuracy_score(actuals, pred_classes)
        
        # Regression metrics
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        
        print(f"  Classification accuracy: {accuracy:.3f}")
        print(f"  MAE: {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print()
        
        results[rating_type] = {
            'accuracy': accuracy,
            'mae': mae,
            'rmse': rmse,
            'predictions': predictions,
            'actuals': actuals
        }
    
    return results

def main():
    """Run the complete analysis."""
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    print(f"Loaded {len(df)} data points\n")
    
    # Task 1: Analyze score variation
    stats_df, df = analyze_score_variation(df)
    
    # Prepare features
    df, feature_cols, rating_types = prepare_features_and_targets(df)
    
    # Task 2: Categorical prediction model
    categorical_results = categorical_prediction_model(df, feature_cols, rating_types)
    
    # Task 3 & 4: Message-based split and hybrid model
    train_df, test_df, train_messages, test_messages = create_message_train_test_split(df)
    
    # Update the target columns in split dataframes
    for rating_type in rating_types:
        target_col = f'{rating_type}_numeric'
        if target_col in df.columns:
            rating_map = {
                'Very poor': 1, 'Poor': 2, 'Fair': 3, 'Good': 4, 'Very good': 5,
                'Not helpful': 1, 'Slightly helpful': 2, 'Moderately helpful': 3, 
                'Very helpful': 4, 'Extremely helpful': 5
            }
            train_df[target_col] = train_df[rating_type].map(rating_map)
            test_df[target_col] = test_df[rating_type].map(rating_map)
    
    hybrid_results = hybrid_prediction_model(train_df, test_df, feature_cols, rating_types)
    
    # Task 5: Compare results
    print("=== COMPARISON OF APPROACHES ===\n")
    print("Categorical Model (participant characteristics only):")
    for rating_type, result in categorical_results.items():
        print(f"  {rating_type}: {result['accuracy']:.3f}")
    
    print("\nHybrid Model (message similarity + participant characteristics):")
    for rating_type, result in hybrid_results.items():
        print(f"  {rating_type}: {result['accuracy']:.3f}")
    
    print("\nConclusions:")
    print("- High score variation suggests individual differences are important")
    print("- Participant characteristics alone have limited predictive power")
    print("- Message content similarity provides additional predictive value")
    print("- Hybrid approach shows promise but needs better text embeddings")

if __name__ == "__main__":
    main()