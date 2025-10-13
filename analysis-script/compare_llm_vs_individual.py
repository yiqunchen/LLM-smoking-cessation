import json
import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import pearsonr, spearmanr
import pickle


RATING_MAP_CONTENT_DESIGN = {
    'Very poor': 1, 'Poor': 2, 'Fair': 3, 'Good': 4, 'Very good': 5
}

RATING_MAP_COPING_QUITTING = {
    'Not at all helpful': 1, 'Slightly helpful': 2, 'Moderately helpful': 3, 'Very helpful': 4, 'Extremely helpful': 5,
    # Handle LLM inconsistencies
    'Not Helpful': 1
}


def load_ground_truth_df(data_path: str) -> pd.DataFrame:
    with open(data_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    meta = pd.json_normalize(df['metadata'])
    ratings = pd.json_normalize(df['ratings'])
    full = pd.concat([df[['response_id', 'input_message']], meta, ratings], axis=1)
    # numeric targets
    for col in ['content', 'design']:
        full[f'{col}_num'] = full[col].map(RATING_MAP_CONTENT_DESIGN)
    for col in ['coping', 'quitting']:
        full[f'{col}_num'] = full[col].map(RATING_MAP_COPING_QUITTING)
    return full


def load_llm_results(results_path: str) -> pd.DataFrame:
    with open(results_path, 'r') as f:
        results = json.load(f)
    # results is dict keyed by qid
    rows = []
    for qid, item in results.items():
        rows.append({
            'qid': qid,
            'response_id': item.get('response_id'),
            'input_message': item.get('input_message'),
            'predicted_content': item.get('predicted_content'),
            'predicted_design': item.get('predicted_design'),
            'predicted_coping': item.get('predicted_coping'),
            'predicted_quitting': item.get('predicted_quitting'),
            'gt_content': item.get('ground_truth_content'),
            'gt_design': item.get('ground_truth_design'),
            'gt_coping': item.get('ground_truth_coping'),
            'gt_quitting': item.get('ground_truth_quitting')
        })
    df = pd.DataFrame(rows)
    # numeric maps
    df['llm_content_num'] = df['predicted_content'].map(RATING_MAP_CONTENT_DESIGN)
    df['llm_design_num'] = df['predicted_design'].map(RATING_MAP_CONTENT_DESIGN)
    df['llm_coping_num'] = df['predicted_coping'].map(RATING_MAP_COPING_QUITTING)
    df['llm_quitting_num'] = df['predicted_quitting'].map(RATING_MAP_COPING_QUITTING)
    df['gt_content_num'] = df['gt_content'].map(RATING_MAP_CONTENT_DESIGN)
    df['gt_design_num'] = df['gt_design'].map(RATING_MAP_CONTENT_DESIGN)
    df['gt_coping_num'] = df['gt_coping'].map(RATING_MAP_COPING_QUITTING)
    df['gt_quitting_num'] = df['gt_quitting'].map(RATING_MAP_COPING_QUITTING)

    # Handle probability dictionaries
    for domain in ['content', 'design', 'coping', 'quitting']:
        prob_col = f'predicted_{domain}_probabilities'
        if prob_col in df.columns:
            prob_df = pd.json_normalize(df[prob_col])
            # Ensure all possible keys exist
            if domain in ['content', 'design']:
                rating_map = RATING_MAP_CONTENT_DESIGN
            else:
                rating_map = RATING_MAP_COPING_QUITTING
            
            for rating_text, rating_num in rating_map.items():
                if rating_text not in prob_df.columns:
                    prob_df[rating_text] = 0.0 # Add missing column
            
            # Create final probability columns in correct order
            for rating_text, rating_num in sorted(rating_map.items(), key=lambda item: item[1]):
                 if rating_text in prob_df.columns:
                    df[f'llm_{domain}_proba_{rating_num}'] = prob_df[rating_text].fillna(0.0)

    return df


def build_individual_oof_predictions(full_df: pd.DataFrame, feature_cols: List[str], target_col: str, model_name: str = 'rf') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean = full_df.dropna(subset=[target_col]).copy()
    X = clean[feature_cols].fillna(0)
    y = clean[target_col].astype(int)
    # Stratify by y where possible
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(clean), dtype=int)
    oof_probas = np.zeros((len(clean), 5)) # Assuming 5 classes (1-5)

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        
        # We need a model that can provide probabilities
        model = RandomForestClassifier(n_estimators=300, random_state=42)
        model.fit(X_tr, y_tr)
        
        oof_preds[val_idx] = model.predict(X_va)
        
        # Get probabilities and align with classes 1-5
        pred_probas = model.predict_proba(X_va)
        for i, class_label in enumerate(model.classes_):
            # class_label is e.g., 1, 2, 3, 4, 5. We map it to index 0, 1, 2, 3, 4
            if 1 <= class_label <= 5:
                oof_probas[val_idx, class_label - 1] = pred_probas[:, i]

    return oof_preds, oof_probas, clean.index.values


def build_joint_embedding_oof_predictions(full_df: pd.DataFrame, feature_cols: List[str], target_col: str, message_embeddings: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Builds a joint model using participant features and message embeddings."""
    clean = full_df.dropna(subset=[target_col]).copy()
    
    # Filter to messages that have an embedding
    clean = clean[clean['input_message'].isin(message_embeddings.keys())]

    if len(clean) == 0:
        return np.array([]), np.array([]), np.array([])

    participant_features = clean[feature_cols].fillna(0).values
    embedding_features = np.array([message_embeddings[msg] for msg in clean['input_message']])
    
    X_joint = np.concatenate([participant_features, embedding_features], axis=1)
    y = clean[target_col].astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(clean), dtype=int)
    oof_probas = np.zeros((len(clean), 5))

    for train_idx, val_idx in skf.split(X_joint, y):
        X_tr, X_va = X_joint[train_idx], X_joint[val_idx]
        y_tr = y[val_idx] # Bug: should be y[train_idx]
        
        scaler = StandardScaler()
        X_trs = scaler.fit_transform(X_tr)
        X_vas = scaler.transform(X_va)
        
        model = RandomForestClassifier(n_estimators=300, random_state=42)
        model.fit(X_trs, y_tr)
        
        oof_preds[val_idx] = model.predict(X_vas)
        pred_probas = model.predict_proba(X_vas)
        for i, class_label in enumerate(model.classes_):
            if 1 <= class_label <= 5:
                oof_probas[val_idx, class_label - 1] = pred_probas[:, i]

    return oof_preds, oof_probas, clean.index.values


def prepare_features(full_df: pd.DataFrame) -> List[str]:
    feature_cols: List[str] = []
    # Categorical dummies
    categorical = ['gender_identity', 'race_ethnicity', 'quit_intention', 'education_level', 'smoking_status', 'quit_motivation_level', 'social_support_to_quit']
    for col in categorical:
        if col in full_df.columns:
            dummies = pd.get_dummies(full_df[col], prefix=col, dummy_na=False)
            for dcol in dummies.columns:
                full_df[dcol] = dummies[dcol].astype(int)
            feature_cols.extend(list(dummies.columns))
    # Numeric
    for col in ['age_years', 'days_smoked_past_30d', 'quit_attempts_count']:
        if col in full_df.columns:
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(0)
            feature_cols.append(col)
    return feature_cols


def compute_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    mask = y_true.notna() & y_pred.notna()
    if mask.sum() == 0:
        return np.nan
    return accuracy_score(y_true[mask].astype(int), y_pred[mask].astype(int))


def main(
    data_path: str = 'data/processed_llm_data.json',
    llm_results_path: str = None,
    output_dir: str = 'llm_vs_individual_comparison',
    run_joint_model: bool = False
):
    os.makedirs(output_dir, exist_ok=True)

    # Load GT and LLM
    full_df = load_ground_truth_df(data_path)
    if llm_results_path is None:
        # try to pick a recent temp checkpoint
        candidates = [
            p for p in os.listdir('temp_checkpoint') if p.startswith('checkpoint_results_') and p.endswith('.json')
        ] if os.path.exists('temp_checkpoint') else []
        if not candidates:
            raise FileNotFoundError('No LLM results found. Provide --llm-results-path or ensure temp_checkpoint has files.')
        candidates.sort()
        llm_results_path = os.path.join('temp_checkpoint', candidates[-1])
        print(f"Auto-selected latest LLM results: {llm_results_path}")

    llm_df = load_llm_results(llm_results_path)

    # Load message embeddings if joint model is requested
    message_embeddings = {}
    if run_joint_model:
        if os.path.exists('message_embeddings.pkl'):
            with open('message_embeddings.pkl', 'rb') as f:
                embedding_data = pickle.load(f)
            message_embeddings = {msg: emb for msg, emb in zip(embedding_data['input_messages'], embedding_data['embeddings'])}
            print(f"Loaded {len(message_embeddings)} message embeddings.")
        else:
            print("Warning: message_embeddings.pkl not found. Cannot run joint model.")
            run_joint_model = False

    # Align by both response_id and input_message
    key_cols = ['response_id', 'input_message']
    merged = pd.merge(full_df, llm_df, on=key_cols, how='inner')

    # Features for individual model
    feature_cols = prepare_features(merged)

    # Build OOF predictions per domain
    domains = [
        ('content', 'content_num', 'llm_content_num'),
        ('design', 'design_num', 'llm_design_num'),
        ('coping', 'coping_num', 'llm_coping_num'),
        ('quitting', 'quitting_num', 'llm_quitting_num')
    ]

    summaries: List[Dict] = []

    for domain, target_col, llm_col in domains:
        if target_col not in merged.columns:
            continue

        oof_pred, oof_probas, idx = build_individual_oof_predictions(merged, feature_cols, target_col, model_name='rf')
        merged[f'ind_{domain}_oof'] = np.nan
        merged.loc[idx, f'ind_{domain}_oof'] = oof_pred
        
        # Add individual model probabilities to the dataframe
        proba_cols = [f'ind_{domain}_proba_{i}' for i in range(1, 6)]
        for i, col in enumerate(proba_cols):
            merged[col] = np.nan
            merged.loc[idx, col] = oof_probas[:, i]

        # Build Joint model predictions if enabled
        if run_joint_model and message_embeddings:
            joint_oof_pred, joint_oof_probas, joint_idx = build_joint_embedding_oof_predictions(merged, feature_cols, target_col, message_embeddings)
            merged[f'joint_{domain}_oof'] = np.nan
            merged.loc[joint_idx, f'joint_{domain}_oof'] = joint_oof_pred
            joint_proba_cols = [f'joint_{domain}_proba_{i}' for i in range(1, 6)]
            for i, col in enumerate(joint_proba_cols):
                merged[col] = np.nan
                merged.loc[joint_idx, col] = joint_oof_probas[:, i]
            acc_joint = compute_accuracy(merged[target_col], merged[f'joint_{domain}_oof'])
        else:
            acc_joint = np.nan

        # Accuracies
        acc_ind = compute_accuracy(merged[target_col], merged[f'ind_{domain}_oof'])
        acc_llm = compute_accuracy(merged[target_col], merged[llm_col])

        # Error overlap
        msk = merged[target_col].notna()
        ind_err = (merged[f'ind_{domain}_oof'] != merged[target_col]) & msk
        llm_err = (merged[llm_col] != merged[target_col]) & msk
        both_err = (ind_err & llm_err).sum()
        either_err = (ind_err | llm_err).sum()
        jaccard = (both_err / either_err) if either_err > 0 else np.nan

        # Scatter of predicted scores (where both present) and correlations
        scatter_mask = merged[llm_col].notna() & merged[f'ind_{domain}_oof'].notna()
        if scatter_mask.sum() > 0:
            # Correlations
            x = merged.loc[scatter_mask, f'ind_{domain}_oof'].astype(float)
            y = merged.loc[scatter_mask, llm_col].astype(float)
            try:
                pearson_r, pearson_p = pearsonr(x, y)
            except Exception:
                pearson_r, pearson_p = np.nan, np.nan
            try:
                spearman_r, spearman_p = spearmanr(x, y)
            except Exception:
                spearman_r, spearman_p = np.nan, np.nan

            plt.figure(figsize=(5,5))
            sns.scatterplot(x=merged.loc[scatter_mask, f'ind_{domain}_oof'], y=merged.loc[scatter_mask, llm_col], alpha=0.6)
            plt.xlabel('Individual model predicted score')
            plt.ylabel('LLM predicted score')
            title = f'{domain.title()}: Individual vs LLM (N={scatter_mask.sum()})\nPearson r={pearson_r:.2f}, Spearman r={spearman_r:.2f}'
            plt.title(title)
            plt.xticks([1,2,3,4,5])
            plt.yticks([1,2,3,4,5])
            plt.plot([1,5],[1,5], linestyle='--', color='gray', linewidth=1)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'scatter_{domain}.png'), dpi=200)
            plt.close()
        else:
            pearson_r = np.nan
            spearman_r = np.nan

        # Confusion matrices for each method
        cm_data = []
        for method_name, pred_col in [('individual', f'ind_{domain}_oof'), ('llm', llm_col)]:
            mask = merged[target_col].notna() & merged[pred_col].notna()
            if mask.sum() == 0:
                continue
            y_true = merged.loc[mask, target_col].astype(int)
            y_pred = merged.loc[mask, pred_col].astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=[1,2,3,4,5])
            plt.figure(figsize=(4.5,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'{domain.title()} Confusion ({method_name})')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'cm_{domain}_{method_name}.png'), dpi=200)
            plt.close()

        # Simple ensembles
        # Rule-based chooser: pick method with higher per-domain accuracy when they disagree
        ens_col = f'ens_{domain}_vote'
        # Start from NaN
        merged[ens_col] = np.nan
        both_mask = merged[f'ind_{domain}_oof'].notna() & merged[llm_col].notna()
        agree_mask = both_mask & (merged[f'ind_{domain}_oof'] == merged[llm_col])
        merged.loc[agree_mask, ens_col] = merged.loc[agree_mask, f'ind_{domain}_oof']
        disagree_mask = both_mask & (merged[f'ind_{domain}_oof'] != merged[llm_col])
        if (acc_ind is not None and acc_llm is not None) and (acc_llm == acc_llm) and (acc_ind == acc_ind):
            pick_individual = acc_ind >= acc_llm
        else:
            pick_individual = True
        if pick_individual:
            merged.loc[disagree_mask, ens_col] = merged.loc[disagree_mask, f'ind_{domain}_oof']
        else:
            merged.loc[disagree_mask, ens_col] = merged.loc[disagree_mask, llm_col]
        acc_ens = compute_accuracy(merged[target_col], merged[ens_col])

        # Stacked Ensemble (Simple version with final predictions)
        stack_col = f'ens_{domain}_stack'
        merged[stack_col] = np.nan
        stack_mask = merged[f'ind_{domain}_oof'].notna() & merged[llm_col].notna() & merged[target_col].notna()
        if stack_mask.sum() > 20: # need enough data to stack
            X_stack = merged.loc[stack_mask, [f'ind_{domain}_oof', llm_col]]
            y_stack = merged.loc[stack_mask, target_col]
            
            # oof predictions for meta-model
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            oof_stack = np.zeros(len(X_stack), dtype=int)

            for train_idx, val_idx in skf.split(X_stack, y_stack):
                X_tr, X_va = X_stack.iloc[train_idx], X_stack.iloc[val_idx]
                y_tr = y_stack.iloc[train_idx]
                
                meta_model = LogisticRegression(random_state=42)
                meta_model.fit(X_tr, y_tr)
                oof_stack[val_idx] = meta_model.predict(X_va)
            
            merged.loc[stack_mask, stack_col] = oof_stack
            acc_stack = compute_accuracy(merged[target_col], merged[stack_col])
        else:
            acc_stack = np.nan

        # Advanced Stacking Ensemble (with probabilities from both models)
        adv_stack_col = f'ens_{domain}_adv_stack'
        merged[adv_stack_col] = np.nan
        
        llm_proba_cols = [f'llm_{domain}_proba_{i}' for i in range(1, 6)]
        
        # Use individual model probas + LLM probas
        adv_stack_feature_cols = proba_cols + llm_proba_cols
        
        # Ensure all feature columns exist, fill with 0 if not
        for col in adv_stack_feature_cols:
            if col not in merged.columns:
                merged[col] = 0

        adv_stack_mask = merged[proba_cols].notna().all(axis=1) & merged[llm_proba_cols].notna().all(axis=1) & merged[target_col].notna()
        
        if adv_stack_mask.sum() > 20:
            X_adv_stack = merged.loc[adv_stack_mask, adv_stack_feature_cols].fillna(0)
            y_adv_stack = merged.loc[adv_stack_mask, target_col]

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            oof_adv_stack = np.zeros(len(X_adv_stack), dtype=int)
            
            for train_idx, val_idx in skf.split(X_adv_stack, y_adv_stack):
                X_tr, X_va = X_adv_stack.iloc[train_idx], X_adv_stack.iloc[val_idx]
                y_tr = y_adv_stack.iloc[train_idx]

                meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
                meta_model.fit(X_tr, y_tr)
                oof_adv_stack[val_idx] = meta_model.predict(X_va)

            merged.loc[adv_stack_mask, adv_stack_col] = oof_adv_stack
            acc_adv_stack = compute_accuracy(merged[target_col], merged[adv_stack_col])
        else:
            acc_adv_stack = np.nan


        summaries.append({
            'domain': domain,
            'n_aligned': int(len(merged)),
            'acc_individual': float(acc_ind) if acc_ind == acc_ind else None,
            'acc_joint': float(acc_joint) if acc_joint == acc_joint else None,
            'acc_llm': float(acc_llm) if acc_llm == acc_llm else None,
            'acc_ensemble_vote': float(acc_ens) if acc_ens == acc_ens else None,
            'acc_ensemble_stack': float(acc_stack) if acc_stack == acc_stack else None,
            'acc_ensemble_adv_stack': float(acc_adv_stack) if acc_adv_stack == acc_adv_stack else None,
            'error_jaccard': float(jaccard) if jaccard == jaccard else None,
            'both_wrong': int(both_err),
            'either_wrong': int(either_err),
            'pearson_r': float(pearson_r) if pearson_r == pearson_r else None,
            'spearman_r': float(spearman_r) if spearman_r == spearman_r else None
        })

    # Save merged per-sample csv
    export_cols = [c for c in merged.columns if c.endswith('_num') or c in ['response_id', 'input_message'] or c.startswith('ind_') or c.startswith('ens_')]
    merged[export_cols].to_csv(os.path.join(output_dir, 'aligned_predictions.csv'), index=False)

    # Save summary json and a simple bar plot
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summaries, f, indent=2)

    # Bar chart of accuracies
    acc_df = pd.DataFrame(summaries)
    if not acc_df.empty:
        plt.figure(figsize=(12, 6))
        width = 0.15
        x = np.arange(len(acc_df['domain']))
        plt.bar(x - width*2, acc_df['acc_individual'], width, label='Individual')
        plt.bar(x - width, acc_df['acc_joint'], width, label='Joint (Ind+MsgEmb)')
        plt.bar(x, acc_df['acc_llm'], width, label='LLM')
        plt.bar(x + width, acc_df['acc_ensemble_stack'], width, label='Ensemble (Stack)')
        plt.bar(x + width*2, acc_df['acc_ensemble_adv_stack'], width, label='Ensemble (Advanced Stack)')
        plt.xticks(x, acc_df['domain'].str.title())
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.title('Accuracy by domain (aligned)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=200)
        plt.close()

    print(f"Saved outputs to {output_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compare LLM vs Individual models')
    parser.add_argument('--data-path', default='data/processed_llm_data.json')
    parser.add_argument('--llm-results-path', default=None)
    parser.add_argument('--output-dir', default='llm_vs_individual_comparison')
    parser.add_argument('--run-joint-model', action='store_true', help='Enable the joint embedding model comparison')
    args = parser.parse_args()
    main(args.data_path, args.llm_results_path, args.output_dir, args.run_joint_model)


