"""
Create additional figures for manuscript:
1. Confusion matrix: How predictions changed from RF to Hybrid
2. Predicted vs True score distributions for 7 methods
3. Parallel line plots showing trends across methods

Uses the same Okabe-Ito colorblind-friendly palette as publication figures.
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import spearmanr
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# Enhanced colorblind-friendly palette with high contrast
COLORS = {
    'GPT-4o-mini': '#0173B2',      # Blue
    'GPT-5': '#DE8F05',            # Orange
    'DeepSeek-R1': '#029E73',      # Green
    'Grok-4-Fast': '#CC78BC',      # Purple
    'Gemini-2.5-Pro': '#CA9161',   # Brown
    'Logistic Regression': '#D55E00',  # Vermillion (distinct from RF)
    'LR': '#D55E00',               # Vermillion
    'Random Forest': '#0072B2',     # Dark blue (distinct from LR)
    'RF': '#0072B2',               # Dark blue
    'Zero-shot': '#56B4E9',         # Sky blue (lighter than RF)
    'Few-shot': '#E69F00',          # Orange (distinct from Zero-shot)
    'Hybrid': '#CC79A7',            # Reddish purple
    'Digital Twin': '#009E73',      # Bluish green
    'True Score': '#000000'         # Black
}

plt.style.use('seaborn-v0_8-whitegrid')
DPI = 400  # High DPI for publication

# Rating mappings
RATING_MAPS = {
    'content': {'Very poor': 1, 'Poor': 2, 'Acceptable': 3, 'Good': 4, 'Very good': 5},
    'design': {'Very poor': 1, 'Poor': 2, 'Acceptable': 3, 'Good': 4, 'Very good': 5},
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


def save_figure(fig, filepath_without_ext: str):
    """Save figure as both PNG and PDF with 400 DPI."""
    png_path = f"{filepath_without_ext}.png"
    fig.savefig(png_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved PNG: {png_path}")

    pdf_path = f"{filepath_without_ext}.pdf"
    fig.savefig(pdf_path, dpi=DPI, bbox_inches='tight', format='pdf')
    print(f"✓ Saved PDF: {pdf_path}")


def to_int_label(x):
    """Convert various label formats to integer."""
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    try:
        return int(str(x).strip())
    except:
        pass
    # Map known strings
    mapping = {
        "not helpful": 1, "not_helpful": 1, "not at all helpful": 1,
        "somewhat helpful": 2, "slightly": 2,
        "moderately helpful": 3, "moderately": 3,
        "very helpful": 4, "very": 4,
        "extremely helpful": 5, "extremely": 5,
        "very poor": 1,
        "poor": 2,
        "acceptable": 3,
        "good": 4,
        "very good": 5,
    }
    key = str(x).strip().lower()
    return mapping.get(key)


def load_results(filepath: str) -> pd.DataFrame:
    """Load evaluation results and convert to DataFrame."""
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r') as f:
        results = json.load(f)

    rows = []
    for item in results.values():
        if isinstance(item, dict) and item.get("predicted_content") != "ERROR":
            rows.append(item)

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # Convert to numeric
    for domain in DOMAINS:
        if f'ground_truth_{domain}' in df.columns and f'predicted_{domain}' in df.columns:
            df[f'gt_{domain}_num'] = df[f'ground_truth_{domain}'].map(RATING_MAPS[domain])
            df[f'pred_{domain}_num'] = df[f'predicted_{domain}'].map(RATING_MAPS[domain])

    return df


def load_rf_predictions():
    """Load Random Forest predictions."""
    rf_path = "results_manuscript_hybrid_rf_grok4/rf_predictions_all_features.json"
    if not os.path.exists(rf_path):
        return {}
    with open(rf_path, "r") as f:
        rf = json.load(f)
    return {str(k): v for k, v in rf.items() if isinstance(v, dict)}


def run_supervised_baselines(data_path: str = 'data_splits/canonical'):
    """Run supervised learning baselines and return predictions."""
    print("📊 Running supervised learning baselines for distributions...")

    # Load train and test data
    with open(os.path.join(data_path, 'train_participant_7030.json')) as f:
        train_data = json.load(f)
    with open(os.path.join(data_path, 'test_participant_7030.json')) as f:
        test_data = json.load(f)

    train_df = pd.DataFrame([item for item in train_data if isinstance(item, dict)])
    test_df = pd.DataFrame([item for item in test_data if isinstance(item, dict)])

    # Extract features
    feature_mapping = {
        'age': 'age_years',
        'gender': 'gender_identity',
        'race': 'race_ethnicity',
        'education': 'education_level',
        'income': 'household_income',
        'smoking_status': 'smoking_status',
        'quit_motivation': 'quit_motivation_level',
        'cigs_per_day': 'cigs_per_day'
    }

    def extract_features(df):
        features = []
        for _, row in df.iterrows():
            metadata = row.get('metadata', {})
            if not isinstance(metadata, dict):
                metadata = {}
            feat = {}
            for feat_name, meta_key in feature_mapping.items():
                val = metadata.get(meta_key)
                if feat_name == 'age':
                    feat[feat_name] = float(val) if val and str(val).replace('.', '').isdigit() else 30.0
                elif feat_name == 'cigs_per_day':
                    try:
                        feat[feat_name] = float(val) if val else 10.0
                    except:
                        feat[feat_name] = 10.0
                else:
                    if val:
                        feat[f'{feat_name}_{val}'] = 1.0
            features.append(feat)
        return pd.DataFrame(features).fillna(0)

    X_train = extract_features(train_df)
    X_test = extract_features(test_df)

    # Align columns
    all_cols = list(set(X_train.columns) | set(X_test.columns))
    for col in all_cols:
        if col not in X_train.columns:
            X_train[col] = 0
        if col not in X_test.columns:
            X_test[col] = 0
    X_train = X_train[sorted(all_cols)]
    X_test = X_test[sorted(all_cols)]

    results = {'LR': {}, 'RF': {}}

    for domain in DOMAINS:
        # Extract labels
        y_train = train_df['ratings'].apply(lambda x: RATING_MAPS[domain].get(x.get(domain)) if isinstance(x, dict) else None)
        y_test = test_df['ratings'].apply(lambda x: RATING_MAPS[domain].get(x.get(domain)) if isinstance(x, dict) else None)

        # Remove NaN
        train_mask = y_train.notna()
        test_mask = y_test.notna()

        X_train_clean = X_train[train_mask]
        y_train_clean = y_train[train_mask]
        X_test_clean = X_test[test_mask]
        y_test_clean = y_test[test_mask]

        if len(y_train_clean) < 10 or len(y_test_clean) < 2:
            continue

        # Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train_clean, y_train_clean)
        pred_lr = lr.predict(X_test_clean)

        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_clean, y_train_clean)
        pred_rf = rf.predict(X_test_clean)

        # Store predictions with ground truth
        results['LR'][domain] = {
            'predictions': pred_lr,
            'ground_truth': y_test_clean.values
        }
        results['RF'][domain] = {
            'predictions': pred_rf,
            'ground_truth': y_test_clean.values
        }

    return results


def create_hybrid_vs_rf_confusion_matrix(output_dir: str):
    """
    Task 1: Create confusion matrix showing how predictions changed from RF to Hybrid.
    For each RF prediction label, show distribution of Hybrid predictions.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load RF predictions
    rf_preds = load_rf_predictions()
    if not rf_preds:
        print("⚠️  No RF predictions found")
        return

    # We'll use Grok-4-Fast as representative model for hybrid
    hybrid_file = "results_manuscript_hybrid_x-ai_grok-4-fast/evaluation_results_x-ai_grok-4-fast_text-only_hybrid-rf-digital-twin.json"

    if not os.path.exists(hybrid_file):
        print(f"⚠️  Hybrid file not found: {hybrid_file}")
        return

    with open(hybrid_file, 'r') as f:
        hybrid_results = json.load(f)

    hybrid_entries = [v for v in hybrid_results.values() if isinstance(v, dict)]

    # Create confusion matrices for each domain
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, domain in enumerate(DOMAINS):
        ax = axes[idx]

        y_rf = []
        y_hybrid = []
        rf_key = f"rf_pred_{domain}"

        for entry in hybrid_entries:
            rid = str(entry.get("response_id"))
            if rid in rf_preds and rf_key in rf_preds[rid]:
                pred_hybrid = to_int_label(entry.get(f"predicted_{domain}"))
                pred_rf = to_int_label(rf_preds[rid].get(rf_key))

                if pred_hybrid is not None and pred_rf is not None:
                    y_rf.append(pred_rf)
                    y_hybrid.append(pred_hybrid)

        if len(y_rf) == 0:
            ax.axis('off')
            ax.set_title(f"{domain.capitalize()}: No data")
            continue

        # Create confusion matrix: rows=RF predictions, columns=Hybrid predictions
        cm = confusion_matrix(y_rf, y_hybrid, labels=[1, 2, 3, 4, 5])

        # Convert to percentages within each RF prediction label (row-wise)
        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # Create custom annotations with both percentage and count
        annot_array = np.empty_like(cm_pct, dtype=object)
        for i in range(5):
            for j in range(5):
                annot_array[i, j] = f'{cm_pct[i, j]:.1f}%\n({cm[i, j]})'

        # Plot heatmap - using Blues colormap for cleaner look
        sns.heatmap(cm_pct, annot=annot_array, fmt='', cmap='Blues', cbar=True, ax=ax,
                    xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5],
                    vmin=0, vmax=100, linewidths=1, linecolor='white',
                    cbar_kws={'label': '% of RF predictions'}, annot_kws={'fontsize': 9})

        ax.set_xlabel('Hybrid Prediction', fontsize=12, fontweight='bold')
        ax.set_ylabel('RF Prediction', fontsize=12, fontweight='bold')
        ax.set_title(f'{domain.capitalize()} (n={len(y_rf)})', fontsize=14, fontweight='bold', pad=10)

    plt.suptitle('How Predictions Changed: RF → Hybrid (Grok-4-Fast)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'hybrid_vs_rf_confusion_matrix')
    save_figure(fig, plot_path)
    plt.close()


def create_predicted_vs_true_distributions(output_dir: str):
    """
    Task 2: Create predicted vs true score distributions for 6 methods across ALL models.
    Methods: Zero-shot, Few-shot, Hybrid, Digital Twin, RF, LR (removed True Score)
    Show as 3x2 grid for each domain.
    """
    os.makedirs(output_dir, exist_ok=True)

    # All model configurations
    MODEL_CONFIGS_PRED = {
        'gpt-4o-mini': {'dir': 'results_manuscript_gpt-4o-mini', 'hybrid_dir': 'results_manuscript_hybrid_gpt-4o-mini', 'hybrid_file': 'evaluation_results_gpt-4o-mini_text-only_hybrid-rf-digital-twin.json'},
        'gpt-5': {'dir': 'results_manuscript_gpt-5', 'hybrid_dir': 'results_manuscript_hybrid_gpt-5', 'hybrid_file': 'evaluation_results_gpt-5_text-only_hybrid-rf-digital-twin.json'},
        'deepseek_deepseek-r1-0528': {'dir': 'results_manuscript_deepseek_deepseek-r1-0528', 'hybrid_dir': 'results_manuscript_hybrid_deepseek_deepseek-r1-0528', 'hybrid_file': 'evaluation_results_deepseek_deepseek-r1-0528_text-only_hybrid-rf-digital-twin.json'},
        'x-ai_grok-4-fast': {'dir': 'results_manuscript_x-ai_grok-4-fast', 'hybrid_dir': 'results_manuscript_hybrid_x-ai_grok-4-fast', 'hybrid_file': 'evaluation_results_x-ai_grok-4-fast_text-only_hybrid-rf-digital-twin.json'},
        'gemini-2.5-pro': {'dir': 'results_manuscript_gemini-2.5-pro', 'hybrid_dir': 'results_manuscript_hybrid_gemini-2.5-pro', 'hybrid_file': 'evaluation_results_gemini-2.5-pro_text-only_hybrid-rf-digital-twin.json'}
    }

    MODEL_DISPLAY_NAMES = {
        'gpt-4o-mini': 'GPT-4o-mini',
        'gpt-5': 'GPT-5',
        'deepseek_deepseek-r1-0528': 'DeepSeek-R1',
        'x-ai_grok-4-fast': 'Grok-4-Fast',
        'gemini-2.5-pro': 'Gemini-2.5-Pro'
    }

    # Get supervised baselines once (same for all models)
    supervised = run_supervised_baselines()

    # For each model
    for model_id, model_cfg in MODEL_CONFIGS_PRED.items():
        model_display = MODEL_DISPLAY_NAMES[model_id]
        model_dir = model_cfg['dir']

        # Load all methods for this model
        methods_data = {}

        # 1. Zero-shot
        df_zs = load_results(os.path.join(model_dir, "generic_llm_1_zero_shot.json"))
        if df_zs is not None:
            methods_data['Zero-shot'] = df_zs

        # 2. Few-shot
        df_fs = load_results(os.path.join(model_dir, "generic_llm_3_few_shot.json"))
        if df_fs is not None:
            methods_data['Few-shot'] = df_fs

        # 3. Digital Twin
        df_dt = load_results(os.path.join(model_dir, "digital_twin_4_cbtact_7030.json"))
        if df_dt is not None:
            methods_data['Digital Twin'] = df_dt

        # 4. Hybrid
        hybrid_file = os.path.join(model_cfg['hybrid_dir'], model_cfg['hybrid_file'])
        df_hybrid = load_results(hybrid_file)
        if df_hybrid is not None:
            methods_data['Hybrid'] = df_hybrid

        # For each domain, create a 2x3 figure with white background
        for domain in DOMAINS:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()

            # Set white background (no grey grid)
            fig.patch.set_facecolor('white')

            method_names = ['Zero-shot', 'Few-shot', 'Digital Twin', 'Hybrid', 'RF', 'LR']

            for idx, method in enumerate(method_names):
                ax = axes[idx]

                if method in ['RF', 'LR']:
                    # Supervised learning methods
                    ml_key = method
                    if ml_key in supervised and domain in supervised[ml_key]:
                        pred_values = supervised[ml_key][domain]['predictions']
                        gt_values = supervised[ml_key][domain]['ground_truth']

                        # Create confusion matrix style visualization
                        from sklearn.metrics import confusion_matrix
                        cm = confusion_matrix(gt_values, pred_values, labels=[1, 2, 3, 4, 5])

                        # Convert to percentages
                        cm_pct = cm.astype('float') / cm.sum() * 100 if cm.sum() > 0 else cm.astype('float')

                        # Create custom annotations with both percentage and count
                        annot_array = np.empty_like(cm_pct, dtype=object)
                        for i in range(5):
                            for j in range(5):
                                annot_array[i, j] = f'{cm_pct[i, j]:.1f}%\n({cm[i, j]})'

                        # Plot heatmap
                        sns.heatmap(cm_pct, annot=annot_array, fmt='', cmap='Blues', cbar=False, ax=ax,
                                   xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5],
                                   vmin=0, vmax=cm_pct.max() if cm_pct.max() > 0 else 1,
                                   linewidths=1, linecolor='white', annot_kws={'fontsize': 10, 'fontweight': 'bold'})

                        # Calculate and show systematic bias
                        bias = np.mean(pred_values - gt_values)
                        bias_text = f'Bias: {bias:+.2f}'
                        if bias > 0.05:
                            bias_text += ' (over-predict)'
                        elif bias < -0.05:
                            bias_text += ' (under-predict)'

                        ax.set_xlabel('True Score', fontsize=14, fontweight='bold')
                        ax.set_ylabel('Predicted Score', fontsize=14, fontweight='bold')
                        ax.set_title(f'{method}\n{bias_text}', fontsize=16, fontweight='bold', pad=10)
                        ax.set_facecolor('white')

                else:
                    # LLM methods
                    if method in methods_data:
                        gt_col = f'gt_{domain}_num'
                        pred_col = f'pred_{domain}_num'

                        if gt_col in methods_data[method].columns and pred_col in methods_data[method].columns:
                            valid_mask = methods_data[method][gt_col].notna() & methods_data[method][pred_col].notna()
                            gt_values = methods_data[method].loc[valid_mask, gt_col].values
                            pred_values = methods_data[method].loc[valid_mask, pred_col].values

                            # Create confusion matrix style visualization
                            from sklearn.metrics import confusion_matrix
                            cm = confusion_matrix(gt_values, pred_values, labels=[1, 2, 3, 4, 5])

                            # Convert to percentages
                            cm_pct = cm.astype('float') / cm.sum() * 100 if cm.sum() > 0 else cm.astype('float')

                            # Create custom annotations with both percentage and count
                            annot_array = np.empty_like(cm_pct, dtype=object)
                            for i in range(5):
                                for j in range(5):
                                    annot_array[i, j] = f'{cm_pct[i, j]:.1f}%\n({cm[i, j]})'

                            # Plot heatmap
                            sns.heatmap(cm_pct, annot=annot_array, fmt='', cmap='Blues', cbar=False, ax=ax,
                                       xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5],
                                       vmin=0, vmax=cm_pct.max() if cm_pct.max() > 0 else 1,
                                       linewidths=1, linecolor='white', annot_kws={'fontsize': 10, 'fontweight': 'bold'})

                            # Calculate and show systematic bias
                            bias = np.mean(pred_values - gt_values)
                            bias_text = f'Bias: {bias:+.2f}'
                            if bias > 0.05:
                                bias_text += ' (over-predict)'
                            elif bias < -0.05:
                                bias_text += ' (under-predict)'

                            ax.set_xlabel('True Score', fontsize=14, fontweight='bold')
                            ax.set_ylabel('Predicted Score', fontsize=14, fontweight='bold')
                            ax.set_title(f'{method}\n{bias_text}', fontsize=16, fontweight='bold', pad=10)
                            ax.set_facecolor('white')

            plt.suptitle(f'Predicted vs True Score: {domain.capitalize()} Domain ({model_display})',
                        fontsize=20, fontweight='bold', y=0.995)
            plt.tight_layout()

            plot_path = os.path.join(output_dir, f'predicted_vs_true_{domain}_{model_id}')
            save_figure(fig, plot_path)
            plt.close()

        print(f"  ✓ Completed {model_display}")


def create_score_density_distributions(output_dir: str):
    """
    Task 3: Create frequency histograms showing distribution of predicted scores.
    Rows: 4 method groups (True Score, Zero/Few-shot, DT/Hybrid, RF/LR)
    Columns: Models (GPT-4o-mini, GPT-5, DeepSeek-R1, Grok-4-Fast, Gemini-2.5-Pro)
    No fixed y-axis, using frequency histograms instead of density.
    """
    os.makedirs(output_dir, exist_ok=True)

    # All model configurations
    MODEL_CONFIGS_DENSITY = {
        'gpt-4o-mini': {'dir': 'results_manuscript_gpt-4o-mini', 'hybrid_dir': 'results_manuscript_hybrid_gpt-4o-mini', 'hybrid_file': 'evaluation_results_gpt-4o-mini_text-only_hybrid-rf-digital-twin.json'},
        'gpt-5': {'dir': 'results_manuscript_gpt-5', 'hybrid_dir': 'results_manuscript_hybrid_gpt-5', 'hybrid_file': 'evaluation_results_gpt-5_text-only_hybrid-rf-digital-twin.json'},
        'deepseek_deepseek-r1-0528': {'dir': 'results_manuscript_deepseek_deepseek-r1-0528', 'hybrid_dir': 'results_manuscript_hybrid_deepseek_deepseek-r1-0528', 'hybrid_file': 'evaluation_results_deepseek_deepseek-r1-0528_text-only_hybrid-rf-digital-twin.json'},
        'x-ai_grok-4-fast': {'dir': 'results_manuscript_x-ai_grok-4-fast', 'hybrid_dir': 'results_manuscript_hybrid_x-ai_grok-4-fast', 'hybrid_file': 'evaluation_results_x-ai_grok-4-fast_text-only_hybrid-rf-digital-twin.json'},
        'gemini-2.5-pro': {'dir': 'results_manuscript_gemini-2.5-pro', 'hybrid_dir': 'results_manuscript_hybrid_gemini-2.5-pro', 'hybrid_file': 'evaluation_results_gemini-2.5-pro_text-only_hybrid-rf-digital-twin.json'}
    }

    MODEL_DISPLAY_NAMES = {
        'gpt-4o-mini': 'GPT-4o-mini',
        'gpt-5': 'GPT-5',
        'deepseek_deepseek-r1-0528': 'DeepSeek-R1',
        'x-ai_grok-4-fast': 'Grok-4-Fast',
        'gemini-2.5-pro': 'Gemini-2.5-Pro'
    }

    # Define method groups: each row will have multiple methods plotted together
    method_groups = [
        {'name': 'True Score', 'methods': ['True Score']},
        {'name': 'Zero-shot / Few-shot', 'methods': ['Zero-shot', 'Few-shot']},
        {'name': 'Digital Twin / Hybrid', 'methods': ['Digital Twin', 'Hybrid']},
        {'name': 'RF / LR', 'methods': ['RF', 'LR']}
    ]

    model_ids = list(MODEL_CONFIGS_DENSITY.keys())

    # Get supervised baselines once
    supervised = run_supervised_baselines()

    # For each domain, create a 4x5 grid (4 method groups × 5 models)
    for domain in DOMAINS:
        fig, axes = plt.subplots(4, 5, figsize=(25, 16), sharex=True)

        # Pre-load all data
        all_model_data = {}
        for model_id in model_ids:
            model_cfg = MODEL_CONFIGS_DENSITY[model_id]
            model_dir = model_cfg['dir']

            methods_data = {}
            df_zs = load_results(os.path.join(model_dir, "generic_llm_1_zero_shot.json"))
            if df_zs is not None:
                methods_data['Zero-shot'] = df_zs

            df_fs = load_results(os.path.join(model_dir, "generic_llm_3_few_shot.json"))
            if df_fs is not None:
                methods_data['Few-shot'] = df_fs

            df_dt = load_results(os.path.join(model_dir, "digital_twin_4_cbtact_7030.json"))
            if df_dt is not None:
                methods_data['Digital Twin'] = df_dt

            hybrid_file = os.path.join(model_cfg['hybrid_dir'], model_cfg['hybrid_file'])
            df_hybrid = load_results(hybrid_file)
            if df_hybrid is not None:
                methods_data['Hybrid'] = df_hybrid

            all_model_data[model_id] = methods_data

        # Plot: rows = method groups, columns = models
        for row_idx, group in enumerate(method_groups):
            for col_idx, model_id in enumerate(model_ids):
                ax = axes[row_idx, col_idx]
                model_display = MODEL_DISPLAY_NAMES[model_id]
                methods_data = all_model_data[model_id]

                # Plot histogram for each method in the group
                bins = np.arange(0.5, 6.5, 1)

                for method in group['methods']:
                    color = COLORS.get(method, '#666666')

                    if method == 'True Score':
                        if 'Zero-shot' in methods_data:
                            gt_col = f'gt_{domain}_num'
                            if gt_col in methods_data['Zero-shot'].columns:
                                values = methods_data['Zero-shot'][gt_col].dropna().values
                                if len(values) > 0:
                                    ax.hist(values, bins=bins, alpha=0.7, color=color,
                                           label=method, edgecolor='black', linewidth=1.5)

                    elif method in ['RF', 'LR']:
                        if method in supervised and domain in supervised[method]:
                            values = supervised[method][domain]['predictions']
                            if len(values) > 0:
                                ax.hist(values, bins=bins, alpha=0.6, color=color,
                                       label=method, edgecolor='black', linewidth=1.5)

                    else:
                        if method in methods_data:
                            pred_col = f'pred_{domain}_num'
                            if pred_col in methods_data[method].columns:
                                values = methods_data[method][pred_col].dropna().values
                                if len(values) > 0:
                                    ax.hist(values, bins=bins, alpha=0.6, color=color,
                                           label=method, edgecolor='black', linewidth=1.5)

                # Formatting - BIGGER and BOLDER
                ax.set_xlim(0.5, 5.5)
                ax.set_xticks([1, 2, 3, 4, 5])
                ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
                ax.grid(axis='y', alpha=0.3, linestyle='--')

                # Make tick labels bold
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontweight('bold')

                # Add column titles (model names) at top - BIGGER
                if row_idx == 0:
                    ax.set_title(f'{model_display}', fontsize=18, fontweight='bold', pad=12)

                # Add row labels (method group names) on left - BIGGER
                if col_idx == 0:
                    ax.set_ylabel(f'{group["name"]}', fontsize=16, fontweight='bold', rotation=0,
                                 ha='right', va='center', labelpad=60)

                # Only show x-axis label on bottom row - BIGGER
                if row_idx == 3:
                    ax.set_xlabel('Score', fontsize=16, fontweight='bold')

                # Add legend only for rows with multiple methods - BIGGER
                if len(group['methods']) > 1 and col_idx == 4:  # Show on rightmost column
                    ax.legend(fontsize=13, loc='upper right', frameon=True, framealpha=0.9,
                             edgecolor='black', fancybox=False)

        plt.suptitle(f'Score Frequency Distributions: {domain.capitalize()} Domain\n(Rows = Method Groups, Columns = Models)',
                    fontsize=20, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        plot_path = os.path.join(output_dir, f'frequency_distributions_{domain}')
        save_figure(fig, plot_path)
        plt.close()


def main():
    print("="*80)
    print("CREATING ADDITIONAL FIGURES")
    print("="*80)
    print()

    output_dir = 'figures/hybrid'
    os.makedirs(output_dir, exist_ok=True)

    print("📊 Task 1: Creating confusion matrix (RF → Hybrid)...")
    create_hybrid_vs_rf_confusion_matrix(output_dir)

    print("\n📊 Task 2: Creating predicted vs true score distributions (all models)...")
    create_predicted_vs_true_distributions(output_dir)

    print("\n📊 Task 3: Creating score frequency distributions (all models)...")
    create_score_density_distributions(output_dir)

    print(f"\n{'='*80}")
    print("✓ Additional figures complete!")
    print(f"{'='*80}\n")
    print(f"Output directory: {output_dir}/")
    print("\nGenerated files:")
    print("  1. Confusion Matrix:")
    print("     - hybrid_vs_rf_confusion_matrix.png/pdf")
    print("  2. Predicted vs True Distributions (2x3 grid per model, confusion matrix style):")
    print("     - predicted_vs_true_{domain}_{model}.png/pdf (15 files)")
    print("  3. Score Frequency Distributions (4 method groups × 5 models):")
    print("     - frequency_distributions_content.png/pdf")
    print("     - frequency_distributions_coping.png/pdf")
    print("     - frequency_distributions_quitting.png/pdf")


if __name__ == '__main__':
    main()
