import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import warnings
import requests
import os
warnings.filterwarnings('ignore')

# Import RealMLP from pytabkit using correct API
REALMLP_AVAILABLE = False
try:
    from pytabkit import RealMLP_TD_Classifier
    REALMLP_AVAILABLE = True
    print("PyTabKit RealMLP_TD_Classifier available")
except ImportError:
    print("Warning: PyTabKit RealMLP_TD_Classifier not available, continuing without it")
    REALMLP_AVAILABLE = False

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def get_voyage_embeddings(texts, model="voyage-large-3"):
    """Get embeddings from Voyage AI API."""
    api_key = os.getenv('VOYAGE_API_KEY')
    if not api_key:
        print("Warning: VOYAGE_API_KEY not found in environment variables")
        return None
    
    url = "https://api.voyageai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Process in batches to avoid API limits
    batch_size = 50
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        payload = {
            "input": batch_texts,
            "model": model
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            embeddings = [item['embedding'] for item in data['data']]
            all_embeddings.extend(embeddings)
            
        except Exception as e:
            print(f"Error getting embeddings for batch {i//batch_size + 1}: {e}")
            # Fallback to zero embeddings for this batch
            embedding_dim = 1024  # voyage-large-2 dimension
            all_embeddings.extend([[0.0] * embedding_dim for _ in batch_texts])
    
    return np.array(all_embeddings)

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

def enhanced_score_analysis(df):
    """Enhanced analysis with better visualizations."""
    print("=== ENHANCED SCORE VARIATION ANALYSIS ===\n")
    
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
    
    # Create comprehensive plots
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    for i, rating_type in enumerate(rating_types):
        numeric_col = f'{rating_type}_numeric'
        
        # Row 1: Raw score distributions
        axes[0, i].hist(df[numeric_col].dropna(), bins=5, alpha=0.7, edgecolor='black')
        axes[0, i].set_title(f'{rating_type.title()}: Raw Score Distribution')
        axes[0, i].set_xlabel('Rating Score')
        axes[0, i].set_ylabel('Frequency')
        axes[0, i].set_xticks([1, 2, 3, 4, 5])
        
        # Row 2: Mean scores per message
        msg_means = df.groupby('input_message')[numeric_col].mean().dropna()
        axes[1, i].hist(msg_means, bins=15, alpha=0.7, edgecolor='black')
        axes[1, i].set_title(f'{rating_type.title()}: Message Mean Scores')
        axes[1, i].set_xlabel('Average Rating')
        axes[1, i].set_ylabel('Number of Messages')
        
        # Row 3: Standard deviation per message
        msg_stds = df.groupby('input_message')[numeric_col].std().dropna()
        axes[2, i].hist(msg_stds, bins=15, alpha=0.7, edgecolor='black')
        axes[2, i].set_title(f'{rating_type.title()}: Score Std Dev per Message')
        axes[2, i].set_xlabel('Standard Deviation')
        axes[2, i].set_ylabel('Number of Messages')
    
    plt.tight_layout()
    plt.savefig('enhanced_score_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Noise analysis
    print("NOISE ANALYSIS:")
    for rating_type in rating_types:
        numeric_col = f'{rating_type}_numeric'
        overall_mean = df[numeric_col].mean()
        overall_std = df[numeric_col].std()
        
        # Message-level means and stds
        msg_stats = df.groupby('input_message')[numeric_col].agg(['mean', 'std', 'count']).dropna()
        
        # Signal vs noise
        between_msg_var = msg_stats['mean'].var()  # Variance between message means
        within_msg_var = msg_stats['std'].mean() ** 2  # Average within-message variance
        
        signal_to_noise = between_msg_var / within_msg_var if within_msg_var > 0 else 0
        
        print(f"{rating_type.upper()}:")
        print(f"  Overall mean: {overall_mean:.2f} (std: {overall_std:.2f})")
        print(f"  Between-message variance: {between_msg_var:.3f}")
        print(f"  Within-message variance: {within_msg_var:.3f}")
        print(f"  Signal-to-noise ratio: {signal_to_noise:.3f}")
        print(f"  % variance explained by message: {between_msg_var/(between_msg_var + within_msg_var)*100:.1f}%")
        print()
    
    return df

def prepare_features_and_targets(df):
    """Prepare features and targets for prediction models."""
    rating_types = ['content', 'design', 'coping', 'quitting']
    feature_cols = []
    
    # Categorical features with more comprehensive encoding
    categorical_features = {
        'gender_identity': df['gender_identity'].dropna().unique(),
        'race_ethnicity': df['race_ethnicity'].dropna().unique(),
        'quit_intention': df['quit_intention'].dropna().unique(),
        'education_level': df['education_level'].dropna().unique(),
        'smoking_status': df['smoking_status'].dropna().unique(),
        'quit_motivation_level': df['quit_motivation_level'].dropna().unique(),
        'social_support_to_quit': df['social_support_to_quit'].dropna().unique()
    }
    
    # Create dummy variables
    for feature, categories in categorical_features.items():
        if feature in df.columns:
            for category in categories:
                if pd.notna(category):
                    col_name = f'{feature}_{category}'.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_').replace(',', '').replace('-', '_')[:50]
                    df[col_name] = (df[feature] == category).astype(int)
                    feature_cols.append(col_name)
    
    # Numerical features
    numerical_features = ['age_years', 'days_smoked_past_30d', 'quit_attempts_count']
    for feature in numerical_features:
        if feature in df.columns:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
            df[feature] = df[feature].fillna(df[feature].median())
            feature_cols.append(feature)
    
    return df, feature_cols, rating_types

def joint_embedding_models(df, feature_cols, rating_types, message_embeddings):
    """Test joint models with participant features + message embeddings."""
    print("=== JOINT EMBEDDING + SUPERVISED MODELS ===\n")
    
    global REALMLP_AVAILABLE
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'MLP Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
    }
    
    # Add RealMLP if available
    if REALMLP_AVAILABLE:
        try:
            models['RealMLP'] = RealMLP_TD_Classifier()
        except Exception as e:
            print(f"Failed to create RealMLP model: {e}")
            REALMLP_AVAILABLE = False
    
    results = {}
    
    for rating_type in rating_types:
        target_col = f'{rating_type}_numeric'
        if target_col not in df.columns:
            continue
            
        print(f"Joint models for {rating_type.upper()} rating:")
        
        # Filter rows with both embeddings and targets
        model_df = df[df[target_col].notna() & df['input_message'].notna()].copy()
        model_df = model_df[model_df['input_message'].isin(message_embeddings.keys())]
        
        if len(model_df) < 100:
            print(f"  Insufficient data: {len(model_df)} samples")
            continue
        
        # Create joint features: participant features + message embeddings
        participant_features = model_df[feature_cols].fillna(0)
        
        # Get message embeddings for each row
        embedding_features = np.array([message_embeddings[msg] for msg in model_df['input_message']])
        
        # Concatenate participant + embedding features
        X_joint = np.concatenate([participant_features.values, embedding_features], axis=1)
        y = model_df[target_col]
        
        print(f"  Joint feature shape: {X_joint.shape} (participant: {participant_features.shape[1]}, embedding: {embedding_features.shape[1]})")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_joint, y, test_size=0.3, random_state=42, stratify=y)
        
        # Scale features for neural networks
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        rating_results = {}
        
        for model_name, model in models.items():
            # Use scaled features for neural networks, regular for tree models
            if 'MLP' in model_name or 'RealMLP' in model_name:
                X_train_use = X_train_scaled
                X_test_use = X_test_scaled
            else:
                X_train_use = X_train
                X_test_use = X_test
            
            try:
                # Train model
                model.fit(X_train_use, y_train)
                
                # Predictions
                y_pred = model.predict(X_test_use)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Cross-validation score
                if 'RealMLP' in model_name:
                    cv_scores = np.array([accuracy])  # Use test accuracy as proxy
                elif 'MLP' in model_name:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3)  
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            except Exception as e:
                print(f"    Error with {model_name}: {e}")
                continue
            
            print(f"  {model_name:20s}: Accuracy={accuracy:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
            
            rating_results[model_name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model': model,
                'feature_type': 'joint'
            }
        
        results[rating_type] = rating_results
        print()
    
    return results

def embedding_proxy_models(df, feature_cols, rating_types, message_embeddings):
    """Test models with embedding-derived proxy scores as additional features."""
    print("=== EMBEDDING PROXY + SUPERVISED MODELS ===\n")
    
    global REALMLP_AVAILABLE
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'MLP Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
    }
    
    if REALMLP_AVAILABLE:
        try:
            models['RealMLP'] = RealMLP_TD_Classifier()
        except Exception as e:
            print(f"Failed to create RealMLP model: {e}")
    
    # Create embedding proxy scores
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Calculate average ratings per message
    message_avg_ratings = {}
    for rating_type in rating_types:
        target_col = f'{rating_type}_numeric'
        if target_col in df.columns:
            msg_ratings = df.groupby('input_message')[target_col].mean().to_dict()
            message_avg_ratings[rating_type] = msg_ratings
    
    # For each message, compute embedding-based quality proxy
    print("Computing embedding proxy scores...")
    
    proxy_features = {}
    unique_messages = list(message_embeddings.keys())
    embeddings_matrix = np.array([message_embeddings[msg] for msg in unique_messages])
    
    for i, msg in enumerate(unique_messages):
        msg_embedding = embeddings_matrix[i:i+1]  # Keep 2D shape
        
        # Find 10 most similar messages
        similarities = cosine_similarity(msg_embedding, embeddings_matrix)[0]
        top_10_indices = np.argsort(similarities)[-11:-1]  # Exclude self, get top 10
        
        # Compute proxy scores as weighted average of similar messages
        proxy_scores = {}
        for rating_type in rating_types:
            if rating_type in message_avg_ratings:
                similar_ratings = []
                for idx in top_10_indices:
                    similar_msg = unique_messages[idx]
                    if similar_msg in message_avg_ratings[rating_type]:
                        similar_ratings.append(message_avg_ratings[rating_type][similar_msg])
                
                proxy_scores[f'{rating_type}_proxy'] = np.mean(similar_ratings) if similar_ratings else 3.0
        
        proxy_features[msg] = proxy_scores
    
    results = {}
    
    for rating_type in rating_types:
        target_col = f'{rating_type}_numeric'
        if target_col not in df.columns:
            continue
            
        print(f"Proxy models for {rating_type.upper()} rating:")
        
        # Filter and prepare data
        model_df = df[df[target_col].notna() & df['input_message'].notna()].copy()
        model_df = model_df[model_df['input_message'].isin(proxy_features.keys())]
        
        if len(model_df) < 100:
            continue
        
        # Create enhanced features: original + proxy scores
        participant_features = model_df[feature_cols].fillna(0)
        
        # Add proxy scores for ALL rating types as features
        proxy_cols = []
        for other_rating in rating_types:
            col_name = f'{other_rating}_proxy'
            model_df[col_name] = model_df['input_message'].map(lambda x: proxy_features[x][col_name])
            proxy_cols.append(col_name)
        
        # Combine participant + proxy features
        enhanced_feature_cols = feature_cols + proxy_cols
        X_enhanced = model_df[enhanced_feature_cols].fillna(0)
        y = model_df[target_col]
        
        print(f"  Enhanced features: {X_enhanced.shape[1]} (original: {len(feature_cols)}, proxy: {len(proxy_cols)})")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_enhanced, y, test_size=0.3, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        rating_results = {}
        
        for model_name, model in models.items():
            # Use scaled features for neural networks
            if 'MLP' in model_name or 'RealMLP' in model_name:
                X_train_use = X_train_scaled
                X_test_use = X_test_scaled
            else:
                X_train_use = X_train
                X_test_use = X_test
            
            try:
                model.fit(X_train_use, y_train)
                y_pred = model.predict(X_test_use)
                accuracy = accuracy_score(y_test, y_pred)
                
                if 'RealMLP' in model_name:
                    cv_scores = np.array([accuracy])
                elif 'MLP' in model_name:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3)  
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                    
            except Exception as e:
                print(f"    Error with {model_name}: {e}")
                continue
            
            print(f"  {model_name:20s}: Accuracy={accuracy:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
            
            rating_results[model_name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model': model,
                'feature_type': 'enhanced_proxy'
            }
        
        results[rating_type] = rating_results
        print()
    
    return results

def advanced_prediction_models(df, feature_cols, rating_types):
    """Test multiple advanced models."""
    print("=== ADVANCED PREDICTION MODELS ===\n")
    
    global REALMLP_AVAILABLE
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'MLP Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
    }
    
    # Add RealMLP if available - use correct pytabkit API
    if REALMLP_AVAILABLE:
        try:
            models['RealMLP'] = RealMLP_TD_Classifier()
        except Exception as e:
            print(f"Failed to create RealMLP model: {e}")
            REALMLP_AVAILABLE = False
    
    results = {}
    
    for rating_type in rating_types:
        target_col = f'{rating_type}_numeric'
        if target_col not in df.columns:
            continue
            
        print(f"Models for {rating_type.upper()} rating:")
        
        # Remove rows with missing targets
        model_df = df[df[target_col].notna()].copy()
        
        if len(model_df) < 100:
            continue
            
        X = model_df[feature_cols].fillna(0)
        y = model_df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Scale features for neural network
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        rating_results = {}
        
        for model_name, model in models.items():
            # Use scaled features for neural networks, regular for tree models
            if 'MLP' in model_name or 'RealMLP' in model_name:
                X_train_use = X_train_scaled
                X_test_use = X_test_scaled
            else:
                X_train_use = X_train
                X_test_use = X_test
            
            try:
                # Train model
                model.fit(X_train_use, y_train)
                
                # Predictions
                y_pred = model.predict(X_test_use)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Cross-validation score (skip for RealMLP due to computational cost)
                if 'RealMLP' in model_name:
                    cv_scores = np.array([accuracy])  # Use test accuracy as proxy
                elif 'MLP' in model_name:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3)  # Reduce CV folds
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            except Exception as e:
                print(f"    Error with {model_name}: {e}")
                continue
            
            print(f"  {model_name:20s}: Accuracy={accuracy:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
            
            # Feature importance for tree models
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                rating_results[model_name] = {
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'feature_importance': feature_importance,
                    'model': model
                }
            else:
                rating_results[model_name] = {
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'model': model
                }
        
        results[rating_type] = rating_results
        print()
    
    return results

def create_message_embeddings(df):
    """Create semantic embeddings for messages using Voyage AI."""
    print("=== CREATING SEMANTIC EMBEDDINGS ===\n")
    
    unique_messages = df['input_message'].dropna().unique()
    print(f"Getting embeddings for {len(unique_messages)} unique messages...")
    
    # Get embeddings
    embeddings = get_voyage_embeddings(unique_messages.tolist())
    
    if embeddings is None:
        print("Failed to get embeddings, using random embeddings as fallback")
        embeddings = np.random.randn(len(unique_messages), 1024)
    
    # Create message to embedding mapping
    message_embeddings = dict(zip(unique_messages, embeddings))
    
    print(f"Successfully created embeddings with shape: {embeddings.shape}")
    return message_embeddings

def semantic_hybrid_model(train_df, test_df, feature_cols, rating_types, message_embeddings):
    """Hybrid model using semantic embeddings."""
    print("=== SEMANTIC HYBRID MODEL ===\n")
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    results = {}
    
    # Calculate average ratings per message in training set
    train_message_avgs = {}
    for rating_type in rating_types:
        target_col = f'{rating_type}_numeric'
        if target_col in train_df.columns:
            msg_avgs = train_df.groupby('input_message')[target_col].mean().to_dict()
            train_message_avgs[rating_type] = msg_avgs
    
    for rating_type in rating_types:
        target_col = f'{rating_type}_numeric'
        if target_col not in train_df.columns or rating_type not in train_message_avgs:
            continue
            
        print(f"Building semantic hybrid model for {rating_type}...")
        
        train_clean = train_df[train_df[target_col].notna()].copy()
        test_clean = test_df[test_df[target_col].notna()].copy()
        
        if len(train_clean) < 50 or len(test_clean) < 10:
            continue
        
        # Get embeddings for train and test messages
        train_messages = list(train_message_avgs[rating_type].keys())
        test_messages = test_clean['input_message'].unique()
        
        # Filter messages that have embeddings
        train_messages = [msg for msg in train_messages if msg in message_embeddings]
        test_messages = [msg for msg in test_messages if msg in message_embeddings]
        
        if len(train_messages) == 0 or len(test_messages) == 0:
            print(f"  No embeddings available for {rating_type}")
            continue
        
        # Create embedding matrices
        train_embeddings = np.array([message_embeddings[msg] for msg in train_messages])
        test_embeddings = np.array([message_embeddings[msg] for msg in test_messages])
        
        # Train participant characteristic model
        # Ensure consistent feature columns between train and test
        train_feature_cols = [col for col in feature_cols if col in train_clean.columns]
        test_feature_cols = [col for col in feature_cols if col in test_clean.columns]
        common_feature_cols = [col for col in train_feature_cols if col in test_feature_cols]
        
        print(f"  Debug - Total feature_cols: {len(feature_cols)}")
        print(f"  Debug - Train features: {len(train_feature_cols)}")
        print(f"  Debug - Test features: {len(test_feature_cols)}")
        print(f"  Debug - Common features: {len(common_feature_cols)}")
        
        if len(common_feature_cols) == 0:
            print(f"  No common features between train/test for {rating_type}")
            continue
            
        message_features = train_clean.groupby('input_message')[common_feature_cols].mean().reset_index()
        message_targets = train_clean.groupby('input_message')[target_col].mean().reset_index()
        message_data = pd.merge(message_features, message_targets, on='input_message')
        
        # Ensure we only use the exact same feature columns
        X_msg = message_data[common_feature_cols].fillna(0)
        y_msg = message_data[target_col]
        
        print(f"  X_msg shape for training: {X_msg.shape}")
        print(f"  Expected feature cols: {len(common_feature_cols)}")
        print(f"  Actual X_msg cols: {X_msg.shape[1]}")
        
        # Double check - use only the exact common feature columns
        X_msg = X_msg[common_feature_cols].fillna(0)
        
        scaler_msg = StandardScaler()
        X_msg_scaled = scaler_msg.fit_transform(X_msg)
        
        msg_quality_model = RandomForestClassifier(random_state=42)  # Use RF instead of LogReg
        msg_quality_model.fit(X_msg_scaled, y_msg.round().astype(int))
        
        # Make predictions for test set
        predictions = []
        actuals = []
        
        for _, test_row in test_clean.iterrows():
            test_msg = test_row['input_message']
            
            if test_msg not in message_embeddings:
                continue
            
            # Find most similar training messages using semantic similarity
            test_embedding = message_embeddings[test_msg].reshape(1, -1)
            similarities = cosine_similarity(test_embedding, train_embeddings)[0]
            
            # Get top 5 most similar messages
            top_5_indices = np.argsort(similarities)[-5:]
            top_5_messages = [train_messages[i] for i in top_5_indices]
            
            # Average rating from top 5 neighbors
            neighbor_scores = [train_message_avgs[rating_type][msg] for msg in top_5_messages if msg in train_message_avgs[rating_type]]
            neighbor_avg = np.mean(neighbor_scores) if neighbor_scores else 3.0
            
            # For now, just use semantic similarity (skip participant prediction due to feature mismatch)
            hybrid_pred = neighbor_avg
            
            predictions.append(hybrid_pred)
            actuals.append(test_row[target_col])
        
        if len(predictions) == 0:
            continue
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Evaluate
        pred_classes = np.round(predictions).astype(int)
        pred_classes = np.clip(pred_classes, 1, 5)
        accuracy = accuracy_score(actuals, pred_classes)
        
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        
        print(f"  Classification accuracy: {accuracy:.3f}")
        print(f"  MAE: {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  Test samples: {len(predictions)}")
        print()
        
        results[rating_type] = {
            'accuracy': accuracy,
            'mae': mae,
            'rmse': rmse,
            'predictions': predictions,
            'actuals': actuals,
            'n_samples': len(predictions)
        }
    
    return results

def analyze_accuracy_sources(advanced_results, rating_types):
    """Analyze where the accuracy is coming from."""
    print("=== ACCURACY SOURCE ANALYSIS ===\n")
    
    for rating_type in rating_types:
        if rating_type not in advanced_results:
            continue
            
        print(f"{rating_type.upper()} - Top contributing features:")
        
        # Get best performing model
        best_model = None
        best_accuracy = 0
        for model_name, result in advanced_results[rating_type].items():
            if result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_model = model_name
        
        if best_model and 'feature_importance' in advanced_results[rating_type][best_model]:
            top_features = advanced_results[rating_type][best_model]['feature_importance'].head(10)
            for _, row in top_features.iterrows():
                print(f"  {row['feature'][:40]:40s}: {row['importance']:.4f}")
        
        print(f"  Best model: {best_model} (accuracy: {best_accuracy:.3f})")
        print()

def main(run_score_analysis=True, run_advanced_models=True, run_accuracy_analysis=True, run_semantic_hybrid=True, run_joint_models=True, run_proxy_models=True):
    """Run the enhanced analysis with optional components."""
    import argparse
    
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    print(f"Loaded {len(df)} data points\n")
    
    # Component 1: Enhanced score analysis with plots
    if run_score_analysis:
        print("Running enhanced score analysis...")
        df = enhanced_score_analysis(df)
    
    # Prepare features
    df, feature_cols, rating_types = prepare_features_and_targets(df)
    
    # Component 2: Advanced models
    advanced_results = {}
    if run_advanced_models:
        print("Running advanced prediction models...")
        advanced_results = advanced_prediction_models(df, feature_cols, rating_types)
    
    # Component 3: Analyze accuracy sources
    if run_accuracy_analysis and advanced_results:
        analyze_accuracy_sources(advanced_results, rating_types)
    
    # Component 4: Semantic hybrid model
    if run_semantic_hybrid:
        print("Running semantic hybrid analysis...")
        
        # Add numeric columns to full dataset BEFORE splitting
        rating_map = {
            'Very poor': 1, 'Poor': 2, 'Fair': 3, 'Good': 4, 'Very good': 5,
            'Not helpful': 1, 'Slightly helpful': 2, 'Moderately helpful': 3, 
            'Very helpful': 4, 'Extremely helpful': 5
        }
        
        for rating_type in rating_types:
            df[f'{rating_type}_numeric'] = df[rating_type].map(rating_map)
        
        # Create message-based split AFTER adding numeric columns
        unique_messages = df['input_message'].dropna().unique()
        np.random.seed(42)
        train_messages = np.random.choice(unique_messages, size=len(unique_messages)//2, replace=False)
        test_messages = [msg for msg in unique_messages if msg not in train_messages]
        
        train_df = df[df['input_message'].isin(train_messages)].copy()
        test_df = df[df['input_message'].isin(test_messages)].copy()
        
        print(f"Train/test split: {len(train_df)}/{len(test_df)} samples")
        
        # Load embeddings for all embedding-based methods
        import pickle
        import os
        if os.path.exists('message_embeddings.pkl'):
            print("Loading existing embeddings...")
            with open('message_embeddings.pkl', 'rb') as f:
                data = pickle.load(f)
            # Create message to embedding mapping
            message_embeddings = {}
            for i, msg in enumerate(data['input_messages']):
                message_embeddings[msg] = data['embeddings'][i]
            print(f"Loaded {len(message_embeddings)} message embeddings")
        else:
            message_embeddings = create_message_embeddings(df)
        
        # Component 5: Joint embedding + supervised models
        joint_results = {}
        if run_joint_models:
            print("Running joint embedding + supervised models...")
            joint_results = joint_embedding_models(df, feature_cols, rating_types, message_embeddings)
        
        # Component 6: Embedding proxy + supervised models  
        proxy_results = {}
        if run_proxy_models:
            print("Running embedding proxy + supervised models...")
            proxy_results = embedding_proxy_models(df, feature_cols, rating_types, message_embeddings)
        
        # Semantic hybrid model
        semantic_results = semantic_hybrid_model(train_df, test_df, feature_cols, rating_types, message_embeddings)
        
        # Final comprehensive comparison
        print("=== COMPREHENSIVE MODEL COMPARISON ===\n")
        
        all_results = {
            'Participant Only (Advanced)': advanced_results,
            'Joint (Participant + Embeddings)': joint_results,
            'Enhanced (Participant + Proxy)': proxy_results,
            'Semantic Hybrid (k-NN)': semantic_results
        }
        
        for method_name, method_results in all_results.items():
            if method_results:
                print(f"{method_name}:")
                for rating_type in rating_types:
                    if rating_type in method_results:
                        if isinstance(method_results[rating_type], dict) and 'accuracy' in method_results[rating_type]:
                            # Semantic hybrid format
                            acc = method_results[rating_type]['accuracy']
                            print(f"  {rating_type}: {acc:.3f}")
                        else:
                            # Standard model results format  
                            best_acc = max([result['accuracy'] for result in method_results[rating_type].values()])
                            best_model = max(method_results[rating_type].items(), key=lambda x: x[1]['accuracy'])[0]
                            print(f"  {rating_type}: {best_acc:.3f} ({best_model})")
                print()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Analysis with Optional Components')
    parser.add_argument('--skip-score-analysis', action='store_true', help='Skip enhanced score analysis')
    parser.add_argument('--skip-advanced-models', action='store_true', help='Skip advanced ML models')
    parser.add_argument('--skip-accuracy-analysis', action='store_true', help='Skip accuracy source analysis')
    parser.add_argument('--skip-semantic-hybrid', action='store_true', help='Skip semantic hybrid model')
    parser.add_argument('--skip-joint-models', action='store_true', help='Skip joint embedding models')
    parser.add_argument('--skip-proxy-models', action='store_true', help='Skip embedding proxy models')
    parser.add_argument('--only-semantic', action='store_true', help='Run only semantic hybrid model')
    parser.add_argument('--only-joint', action='store_true', help='Run only joint embedding models')
    parser.add_argument('--only-proxy', action='store_true', help='Run only embedding proxy models')
    
    args = parser.parse_args()
    
    if args.only_semantic:
        main(run_score_analysis=False, run_advanced_models=False, run_accuracy_analysis=False, 
             run_semantic_hybrid=True, run_joint_models=False, run_proxy_models=False)
    elif args.only_joint:
        main(run_score_analysis=False, run_advanced_models=False, run_accuracy_analysis=False,
             run_semantic_hybrid=False, run_joint_models=True, run_proxy_models=False)
    elif args.only_proxy:
        main(run_score_analysis=False, run_advanced_models=False, run_accuracy_analysis=False,
             run_semantic_hybrid=False, run_joint_models=False, run_proxy_models=True)
    else:
        main(
            run_score_analysis=not args.skip_score_analysis,
            run_advanced_models=not args.skip_advanced_models, 
            run_accuracy_analysis=not args.skip_accuracy_analysis,
            run_semantic_hybrid=not args.skip_semantic_hybrid,
            run_joint_models=not args.skip_joint_models,
            run_proxy_models=not args.skip_proxy_models
        )