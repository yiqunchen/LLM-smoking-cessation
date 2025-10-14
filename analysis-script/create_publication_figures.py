"""
Create publication-ready figures with colorblind-friendly palette and improved layout.

Key improvements:
1. Colorblind-friendly palette (Okabe-Ito)
2. Single shared legend at bottom
3. Includes supervised learning baselines (Logistic Regression + Random Forest)
4. No radar plots
5. Meaningful scatter plots: Accuracy vs Spearman, Accuracy vs Kappa, etc.
6. Digital twin learning curves for all metrics
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from scipy.stats import spearmanr, kendalltau
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Okabe-Ito colorblind-friendly palette
COLORS = {
    'GPT-4o-mini': '#0173B2',      # Blue
    'GPT-5': '#DE8F05',            # Orange
    'DeepSeek-R1': '#029E73',      # Green
    'Grok-4-Fast': '#CC78BC',      # Purple
    'Gemini-2.5-Pro': '#CA9161',   # Brown
    'Logistic Regression': '#E02020',  # Red (distinct color)
    'Random Forest': '#7F7F7F'     # Neutral grey
}

plt.style.use('seaborn-v0_8-whitegrid')
DPI = 300
PDF_DPI = 400  # Higher DPI for publication-quality PDFs


def save_figure(fig, filepath_without_ext: str):
    """Save figure as both PNG and PDF with appropriate DPIs."""
    # Save PNG
    png_path = f"{filepath_without_ext}.png"
    fig.savefig(png_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved PNG: {png_path}")

    # Save PDF
    pdf_path = f"{filepath_without_ext}.pdf"
    fig.savefig(pdf_path, dpi=PDF_DPI, bbox_inches='tight', format='pdf')
    print(f"✓ Saved PDF: {pdf_path}")

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

# Directionality buckets: {1,2} -> low, {3} -> neutral, {4,5} -> high
DIRECTIONAL_BUCKET_MAP = {
    1: 0,
    2: 0,
    3: 1,
    4: 2,
    5: 2
}


def map_directionality(values: np.ndarray) -> np.ndarray:
    """Map 5-point ratings into directional buckets (low/neutral/high)."""
    as_int = values.astype(int)
    mapped = np.full(as_int.shape, fill_value=-1, dtype=int)
    for rating, bucket in DIRECTIONAL_BUCKET_MAP.items():
        mapped[as_int == rating] = bucket
    return mapped

# Model configurations
MODEL_CONFIGS = {
    'gpt-4o-mini': {'dir': 'results_manuscript_gpt-4o-mini', 'display': 'GPT-4o-mini'},
    'gpt-5': {'dir': 'results_manuscript_gpt-5', 'display': 'GPT-5'},
    'deepseek_deepseek-r1-0528': {'dir': 'results_manuscript_deepseek_deepseek-r1-0528', 'display': 'DeepSeek-R1'},
    'x-ai_grok-4-fast': {'dir': 'results_manuscript_x-ai_grok-4-fast', 'display': 'Grok-4-Fast'},
    'gemini-2.5-pro': {'dir': 'results_manuscript_gemini-2.5-pro', 'display': 'Gemini-2.5-Pro'}
}

# All methods to evaluate for finding MAX performance
ALL_METHOD_FILES = [
    'generic_llm_1_zero_shot.json',
    'generic_llm_2_zero_shot_select.json',
    'generic_llm_3_few_shot.json',
    'generic_llm_4_few_shot_select.json',
    'generic_llm_5_continuous.json',
    'digital_twin_4_cbtact_7030.json'
]


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


def calculate_metrics(df: pd.DataFrame, domains: list) -> dict:
    """Calculate comprehensive metrics for each domain."""
    metrics = {}
    
    for domain in domains:
        gt_col = f'gt_{domain}_num'
        pred_col = f'pred_{domain}_num'
        
        if gt_col not in df.columns or pred_col not in df.columns:
            continue
        
        # Filter valid rows
        valid_mask = df[gt_col].notna() & df[pred_col].notna()
        gt = df.loc[valid_mask, gt_col].values
        pred = df.loc[valid_mask, pred_col].values
        
        if len(gt) < 2:
            continue
        
        # Calculate metrics
        acc = accuracy_score(gt, pred)
        dir_gt = map_directionality(gt)
        dir_pred = map_directionality(pred)
        valid_dir_mask = (dir_gt != -1) & (dir_pred != -1)
        if not np.any(valid_dir_mask):
            directional_accuracy = np.nan
            directional_macro_f1 = np.nan
        else:
            dir_gt_valid = dir_gt[valid_dir_mask]
            dir_pred_valid = dir_pred[valid_dir_mask]
            directional_accuracy = np.mean(dir_gt_valid == dir_pred_valid)
            directional_macro_f1 = f1_score(dir_gt_valid, dir_pred_valid, average='macro', zero_division=0)
        kappa = cohen_kappa_score(gt, pred)
        tau, _ = kendalltau(gt, pred)
        
        # Per-participant Spearman's Rho
        participant_rhos = []
        if 'response_id' in df.columns:
            for participant_id, group in df.groupby('response_id'):
                if len(group) < 2:
                    continue
                group_gt = group[gt_col].dropna().values
                group_pred = group[pred_col].dropna().values
                if len(group_gt) > 1 and len(np.unique(group_gt)) > 1 and len(np.unique(group_pred)) > 1:
                    rho, _ = spearmanr(group_gt, group_pred)
                    if not np.isnan(rho):
                        participant_rhos.append(rho)
        
        avg_rho = np.mean(participant_rhos) if participant_rhos else np.nan
        
        metrics[domain] = {
            'accuracy': acc,
            'directional_accuracy': directional_accuracy,
            'directional_macro_f1': directional_macro_f1,
            'kappa': kappa,
            'kendall_tau': tau,
            'spearman_rho': avg_rho,
            'n_samples': len(gt)
        }
    
    return metrics


def run_supervised_baselines(data_path: str = 'data_splits/canonical') -> Dict:
    """Run supervised learning baselines (Logistic Regression + Random Forest)."""
    
    print("📊 Running supervised learning baselines...")
    
    # Load train and test data
    with open(os.path.join(data_path, 'train_participant_7030.json')) as f:
        train_data = json.load(f)
    with open(os.path.join(data_path, 'test_participant_7030.json')) as f:
        test_data = json.load(f)
    
    # Convert to DataFrames
    train_df = pd.DataFrame([item for item in train_data if isinstance(item, dict)])
    test_df = pd.DataFrame([item for item in test_data if isinstance(item, dict)])
    
    # Extract features (participant metadata) - USE CORRECT COLUMN NAMES
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
                # Convert to numeric
                if feat_name == 'age':
                    feat[feat_name] = float(val) if val and str(val).replace('.', '').isdigit() else 30.0
                elif feat_name == 'cigs_per_day':
                    # Try to convert to float
                    try:
                        feat[feat_name] = float(val) if val else 10.0
                    except:
                        feat[feat_name] = 10.0
                else:
                    # One-hot encode categorical
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
    
    results = {'Logistic Regression': {}, 'Random Forest': {}}
    
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
        
        y_test_array = y_test_clean.values.astype(int)

        acc_lr = accuracy_score(y_test_clean, pred_lr)
        f1_macro_lr = f1_score(y_test_clean, pred_lr, average='macro', zero_division=0)
        dir_gt = map_directionality(y_test_array)
        dir_pred_lr = map_directionality(np.asarray(pred_lr))
        valid_lr_mask = (dir_gt != -1) & (dir_pred_lr != -1)
        if np.any(valid_lr_mask):
            directional_accuracy_lr = np.mean(dir_gt[valid_lr_mask] == dir_pred_lr[valid_lr_mask])
            directional_macro_f1_lr = f1_score(dir_gt[valid_lr_mask], dir_pred_lr[valid_lr_mask], average='macro', zero_division=0)
        else:
            directional_accuracy_lr = np.nan
            directional_macro_f1_lr = np.nan
        kappa_lr = cohen_kappa_score(y_test_clean, pred_lr)

        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_clean, y_train_clean)
        pred_rf = rf.predict(X_test_clean)

        acc_rf = accuracy_score(y_test_clean, pred_rf)
        f1_macro_rf = f1_score(y_test_clean, pred_rf, average='macro', zero_division=0)
        dir_pred_rf = map_directionality(np.asarray(pred_rf))
        valid_rf_mask = (dir_gt != -1) & (dir_pred_rf != -1)
        if np.any(valid_rf_mask):
            directional_accuracy_rf = np.mean(dir_gt[valid_rf_mask] == dir_pred_rf[valid_rf_mask])
            directional_macro_f1_rf = f1_score(dir_gt[valid_rf_mask], dir_pred_rf[valid_rf_mask], average='macro', zero_division=0)
        else:
            directional_accuracy_rf = np.nan
            directional_macro_f1_rf = np.nan
        kappa_rf = cohen_kappa_score(y_test_clean, pred_rf)
        
        # Per-participant Spearman for baselines (using test data with participant IDs)
        test_df_clean = test_df[test_mask].copy()
        test_df_clean['pred_lr'] = pred_lr
        test_df_clean['pred_rf'] = pred_rf
        test_df_clean['gt'] = y_test_clean.values
        
        rhos_lr = []
        rhos_rf = []
        for pid, group in test_df_clean.groupby('response_id'):
            if len(group) < 2:
                continue
            gt = group['gt'].values
            pred_lr_group = group['pred_lr'].values
            pred_rf_group = group['pred_rf'].values
            
            if len(np.unique(gt)) > 1:
                if len(np.unique(pred_lr_group)) > 1:
                    rho_lr, _ = spearmanr(gt, pred_lr_group)
                    if not np.isnan(rho_lr):
                        rhos_lr.append(rho_lr)
                if len(np.unique(pred_rf_group)) > 1:
                    rho_rf, _ = spearmanr(gt, pred_rf_group)
                    if not np.isnan(rho_rf):
                        rhos_rf.append(rho_rf)
        
        results['Logistic Regression'][domain] = {
            'accuracy': acc_lr,
            'directional_accuracy': directional_accuracy_lr,
            'directional_macro_f1': directional_macro_f1_lr,
            'kappa': kappa_lr,
            'f1_macro': f1_macro_lr,
            'spearman_rho': np.mean(rhos_lr) if rhos_lr else 0.0,
            'n_samples': int(len(y_test_clean))
        }

        results['Random Forest'][domain] = {
            'accuracy': acc_rf,
            'directional_accuracy': directional_accuracy_rf,
            'directional_macro_f1': directional_macro_f1_rf,
            'kappa': kappa_rf,
            'f1_macro': f1_macro_rf,
            'spearman_rho': np.mean(rhos_rf) if rhos_rf else 0.0,
            'n_samples': int(len(y_test_clean))
        }
        
        print(f"  {domain.capitalize()}: LR={acc_lr:.3f}, RF={acc_rf:.3f}")
    
    return results


def collect_all_results(include_baselines: bool = True) -> pd.DataFrame:
    """Collect MAX results across ALL methods for each model and domain."""

    all_results = []

    # LLM models - evaluate ALL methods and take MAX for each metric
    for model_id, model_cfg in MODEL_CONFIGS.items():
        model_dir = model_cfg['dir']
        model_display = model_cfg['display']

        # Track best metrics across all methods for each domain
        best_metrics_by_domain = {domain: {} for domain in DOMAINS}

        # Load all methods
        for method_file in ALL_METHOD_FILES:
            filepath = os.path.join(model_dir, method_file)
            df = load_results(filepath)

            if df is None:
                continue

            metrics = calculate_metrics(df, DOMAINS)
            if not metrics:
                continue

            # Update best metrics for each domain
            for domain in DOMAINS:
                if domain not in metrics:
                    continue

                domain_metrics = metrics[domain]

                # Initialize if first method
                if not best_metrics_by_domain[domain]:
                    best_metrics_by_domain[domain] = domain_metrics.copy()
                else:
                    # Take MAX for each metric
                    for metric_name, metric_value in domain_metrics.items():
                        if metric_name == 'n_samples':  # Keep n_samples from any method
                            continue
                        if np.isnan(metric_value):
                            continue
                        current_best = best_metrics_by_domain[domain].get(metric_name, -np.inf)
                        if np.isnan(current_best) or metric_value > current_best:
                            best_metrics_by_domain[domain][metric_name] = metric_value

        # Add best results for this model
        found_any = False
        for domain in DOMAINS:
            if best_metrics_by_domain[domain]:
                all_results.append({
                    'Model': model_display,
                    'Type': 'LLM',
                    'Domain': domain.capitalize(),
                    **best_metrics_by_domain[domain]
                })
                found_any = True

        if found_any:
            print(f"✓ {model_display}: Loaded (MAX across all methods)")
        else:
            print(f"⚠️  {model_display}: No methods found")
    
    # Supervised learning baselines
    if include_baselines:
        baseline_results = run_supervised_baselines()
        
        for model_name, model_metrics in baseline_results.items():
            for domain in DOMAINS:
                if domain in model_metrics:
                    all_results.append({
                        'Model': model_name,
                        'Type': 'Supervised ML',
                        'Domain': domain.capitalize(),
                        **model_metrics[domain]
                    })
    
    if not all_results:
        return None
    
    return pd.DataFrame(all_results)


def create_scatter_plots(df: pd.DataFrame, output_dir: str):
    """Create separate scatter plots for each domain: Accuracy vs Spearman, 
    Accuracy vs Kappa, Directional vs Spearman, Directional vs Kappa."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    pairs = [
        ('accuracy', 'spearman_rho', 'Accuracy', 'Spearman\'s ρ'),
        ('accuracy', 'kappa', 'Accuracy', 'Cohen\'s κ'),
        ('directional_accuracy', 'spearman_rho', 'Directional Accuracy', 'Spearman\'s ρ'),
        ('directional_accuracy', 'kappa', 'Directional Accuracy', 'Cohen\'s κ')
    ]
    
    # Create SEPARATE figure for each domain
    for domain in ['Content', 'Coping', 'Quitting']:
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()
        
        domain_data = df[df['Domain'] == domain]
        
        for idx, (x_metric, y_metric, x_label, y_label) in enumerate(pairs):
            ax = axes[idx]
            
            for model in domain_data['Model'].unique():
                model_data = domain_data[domain_data['Model'] == model]
                
                if model_data.empty:
                    continue
                
                ax.scatter(
                    model_data[x_metric],
                    model_data[y_metric],
                    color=COLORS.get(model, '#999999'),
                    s=200,
                    alpha=0.85,
                    edgecolors='black',
                    linewidth=2
                )
            
            ax.set_xlabel(x_label, fontsize=13, fontweight='bold')
            ax.set_ylabel(y_label, fontsize=13, fontweight='bold')
            ax.set_title(f'{x_label} vs {y_label}', fontsize=14, fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=11)
        
        # Create single shared legend at bottom
        handles = []
        for model in sorted(df['Model'].unique()):
            from matplotlib.patches import Patch
            handles.append(Patch(facecolor=COLORS.get(model, '#999999'), 
                                edgecolor='black', linewidth=1.5, label=model))
        
        fig.legend(handles=handles, loc='lower center', ncol=4, 
                  bbox_to_anchor=(0.5, -0.03), fontsize=12, frameon=True, 
                  title='Models', title_fontsize=13)
        
        plt.suptitle(f'Performance Metrics Comparison: {domain} Domain',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0.05, 1, 0.99])

        plot_path = os.path.join(output_dir, f'scatter_{domain.lower()}')
        save_figure(fig, plot_path)
        plt.close()


def create_learning_curves(output_dir: str):
    """Create separate learning curves for each metric (10/30/70/90 splits)."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    splits = ['1090', '3070', '7030', '9010']
    train_pcts = [10, 30, 70, 90]
    
    # Collect data for each model
    learning_data = {}
    
    for model_id, model_cfg in MODEL_CONFIGS.items():
        model_display = model_cfg['display']
        model_dir = model_cfg['dir']
        
        learning_data[model_display] = {domain: {
            'accuracy': [],
            'directional_accuracy': [],
            'directional_macro_f1': [],
            'kappa': [],
            'spearman_rho': []
        } for domain in DOMAINS}
        
        for split in splits:
            filepath = os.path.join(model_dir, f'digital_twin_4_cbtact_{split}.json')
            df = load_results(filepath)
            
            if df is not None:
                metrics = calculate_metrics(df, DOMAINS)
                
                for domain in DOMAINS:
                    if domain in metrics:
                        learning_data[model_display][domain]['accuracy'].append(metrics[domain]['accuracy'])
                        learning_data[model_display][domain]['directional_accuracy'].append(metrics[domain]['directional_accuracy'])
                        learning_data[model_display][domain]['directional_macro_f1'].append(metrics[domain]['directional_macro_f1'])
                        learning_data[model_display][domain]['kappa'].append(metrics[domain]['kappa'])
                        learning_data[model_display][domain]['spearman_rho'].append(metrics[domain]['spearman_rho'])
    
    # Create SEPARATE figure for each metric: 1 row × 3 columns (one per domain)
    metrics_to_plot = ['accuracy', 'directional_accuracy', 'directional_macro_f1', 'kappa', 'spearman_rho']
    metric_labels = ['Accuracy', 'Directional Accuracy', 'Directional Macro-F1', 'Cohen\'s κ', 'Spearman\'s ρ']
    metric_filenames = ['accuracy', 'directional_accuracy', 'directional_macro_f1', 'kappa', 'spearman_rho']
    
    for metric, label, filename in zip(metrics_to_plot, metric_labels, metric_filenames):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Plot each domain in its own subplot
        for domain_idx, domain in enumerate(DOMAINS):
            ax = axes[domain_idx]
            
            for model_display in learning_data.keys():
                values = learning_data[model_display][domain][metric]
                
                if len(values) == len(train_pcts):
                    # Use ONLY COLOR (solid lines for all)
                    ax.plot(train_pcts, values, 
                           linestyle='-',  # All solid lines
                           linewidth=3, 
                           marker='o',
                           markersize=10,
                           color=COLORS.get(model_display, '#999999'),
                           alpha=0.9,
                           label=model_display if domain_idx == 0 else None)  # Only label in first plot
            
            ax.set_xlabel('Training Set Size (%)', fontsize=13, fontweight='bold')
            ax.set_ylabel(label if domain_idx == 0 else '', fontsize=13, fontweight='bold')
            ax.set_title(f'{domain.capitalize()}', fontsize=14, fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(train_pcts)
            ax.tick_params(labelsize=11)
        
        # Create single shared legend at bottom for all subplots
        handles = []
        for model in sorted(learning_data.keys()):
            from matplotlib.patches import Patch
            handles.append(Patch(facecolor=COLORS.get(model, '#999999'), 
                                edgecolor='black', linewidth=1, label=model))
        
        fig.legend(handles=handles, loc='lower center', ncol=len(learning_data), 
                  bbox_to_anchor=(0.5, -0.08), fontsize=12, frameon=True,
                  title='Models', title_fontsize=13)
        
        fig.suptitle(f'Digital Twin Learning Curves: {label}',
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout(rect=[0, 0.08, 1, 0.98])

        plot_path = os.path.join(output_dir, f'learning_curve_{filename}')
        save_figure(fig, plot_path)
        plt.close()


def main():
    print("="*80)
    print("PUBLICATION-READY FIGURES WITH IMPROVED DESIGN")
    print("="*80)
    print()
    
    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)
    
    print("📊 Collecting results from all models and baselines...")
    df = collect_all_results(include_baselines=True)
    
    if df is None or len(df) == 0:
        print("\n❌ No results found.")
        return
    
    print(f"\n✓ Loaded {len(df)} result entries")
    print(f"  - Models: {df['Model'].nunique()}")
    print(f"  - Domains: {df['Domain'].nunique()}")
    
    print(f"\n{'='*80}")
    print(f"GENERATING FIGURES")
    print(f"{'='*80}\n")
    
    print("📈 Creating scatter plots (Accuracy/Directional vs Spearman/Kappa)...")
    create_scatter_plots(df, output_dir)
    
    print("\n📈 Creating digital twin learning curves...")
    create_learning_curves(output_dir)
    
    # Save data table
    csv_path = os.path.join(output_dir, 'all_results_with_baselines.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results table saved: {csv_path}")
    
    print(f"\n{'='*80}")
    print("✓ Publication figures complete!")
    print(f"{'='*80}\n")
    print(f"Output directory: {output_dir}/")
    print("\nGenerated files:")
    print("  Scatter plots (separate by domain):")
    print("    - scatter_content.png")
    print("    - scatter_coping.png")
    print("    - scatter_quitting.png")
    print("  Learning curves (separate by metric):")
    print("    - learning_curve_accuracy.png")
    print("    - learning_curve_directional_accuracy.png")
    print("    - learning_curve_directional_macro_f1.png")
    print("    - learning_curve_kappa.png")
    print("    - learning_curve_spearman_rho.png")
    print("  Data:")
    print("    - all_results_with_baselines.csv")
    print("\n💡 Features:")
    print("  ✓ Colorblind-friendly Okabe-Ito palette")
    print("  ✓ Single shared legend at bottom on each figure")
    print("  ✓ Supervised learning baselines included")
    print("  ✓ Separate figures for each domain/metric (clean, not overloaded)")
    print("  ✓ No size encoding needed - each domain separate")


if __name__ == '__main__':
    main()
