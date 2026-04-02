"""
Create comprehensive cross-model comparison figures for manuscript.

This script analyzes ALL models across ALL methods:
- Zero-shot (all features)
- Zero-shot (selected features)
- Few-shot (all features)
- Few-shot (selected features)
- Continuous (natural language)
- Digital Twin 70/30 (CBT/ACT)

Saves all figures to a model-agnostic figures/ directory.
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
from scipy.stats import spearmanr, kendalltau
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import duplicate filter
from filter_duplicates import get_duplicate_signatures, is_duplicate
from create_publication_figures import run_supervised_baselines

# Set style
plt.style.use('seaborn-v0_8-paper')
DPI = 300
PDF_DPI = 400  # Higher DPI for publication-quality PDFs

# Load duplicate signatures once at module level
DUPLICATE_SIGS = get_duplicate_signatures()
print(f"Loaded {len(DUPLICATE_SIGS)} duplicate signatures for filtering")

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


def metric_to_filename(metric: str) -> str:
    """Create a filesystem-safe suffix from a metric label."""
    return (metric.lower()
            .replace('±', '_within_')
            .replace(' ', '_')
            .replace('-', '_'))


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


EMBEDDING_BASELINE_PATH = 'archive_results/embedding_results_test30/results.json'


def load_embedding_baseline_f1(filepath: str) -> Dict[str, Dict[str, float]]:
    """Load embedding baseline F1 scores for LR/RF by domain from archived results."""
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        by_cat = data.get('results_by_category', {})
        out: Dict[str, Dict[str, float]] = {}
        for domain in ['content', 'coping', 'quitting']:
            d = by_cat.get(domain, {})
            lr = d.get('LogisticRegression', {}).get('test_f1')
            rf = d.get('RandomForest', {}).get('test_f1')
            out[domain.capitalize()] = {
                'Logistic Regression + Embedding': float(lr) if lr is not None else np.nan,
                'Random Forest + Embedding': float(rf) if rf is not None else np.nan
            }
        return out
    except Exception as e:
        print(f"⚠️  Could not load embedding baselines from {filepath}: {e}")
        return {}


def load_embedding_baseline_metrics(filepath: str) -> Dict[str, Dict[str, float]]:
    """Load embedding baseline Accuracy/F1 for LR/RF by domain from archived results."""
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        by_cat = data.get('results_by_category', {})
        out: Dict[str, Dict[str, float]] = {}
        for domain in ['content', 'coping', 'quitting']:
            d = by_cat.get(domain, {})
            out[domain.capitalize()] = {
                'lr_acc': float(d.get('LogisticRegression', {}).get('test_accuracy', np.nan)),
                'rf_acc': float(d.get('RandomForest', {}).get('test_accuracy', np.nan)),
                'lr_f1': float(d.get('LogisticRegression', {}).get('test_f1', np.nan)),
                'rf_f1': float(d.get('RandomForest', {}).get('test_f1', np.nan)),
            }
        return out
    except Exception as e:
        print(f"⚠️  Could not load embedding baselines from {filepath}: {e}")
        return {}

# Model configurations with Okabe-Ito colorblind-friendly palette (SAME AS PUBLICATION FIGURES)
MODEL_CONFIGS = {
    'gpt-4o-mini': {'dir': 'results_manuscript_gpt-4o-mini', 'display': 'GPT-4o-mini', 'color': '#0173B2'},  # Blue
    'gpt-5': {'dir': 'results_manuscript_gpt-5', 'display': 'GPT-5', 'color': '#DE8F05'},  # Orange
    'deepseek_deepseek-r1-0528': {'dir': 'results_manuscript_deepseek_deepseek-r1-0528', 'display': 'DeepSeek-R1', 'color': '#029E73'},  # Green
    'x-ai_grok-4-fast': {'dir': 'results_manuscript_x-ai_grok-4-fast', 'display': 'Grok-4-Fast', 'color': '#CC78BC'},  # Purple
    'gemini-2.5-pro': {'dir': 'results_manuscript_gemini-2.5-pro', 'display': 'Gemini-2.5-Pro', 'color': '#CA9161'},  # Brown
    'logistic_regression': {'dir': None, 'display': 'Logistic Regression', 'color': '#E02020'},  # Red (distinct color)
    'random_forest': {'dir': None, 'display': 'Random Forest', 'color': '#7F7F7F'}  # Neutral grey
}

# Method configurations (all methods shown in main plots)
METHOD_CONFIGS = {
    'generic_llm_1_zero_shot.json': {'display': 'Zero-shot (all)', 'category': 'Generic LLM'},
    'generic_llm_2_zero_shot_select.json': {'display': 'Zero-shot (select)', 'category': 'Generic LLM'},
    'generic_llm_3_few_shot.json': {'display': 'Few-shot (all)', 'category': 'Generic LLM'},
    'generic_llm_4_few_shot_select.json': {'display': 'Few-shot (select)', 'category': 'Generic LLM'},
    'generic_llm_5_continuous.json': {'display': 'Zero-shot (w/ prob)', 'category': 'Generic LLM'},
    'digital_twin_4_cbtact_7030.json': {'display': 'Digital Twin', 'category': 'Digital Twin'},
    'hybrid': {'display': 'Hybrid RF+DT', 'category': 'Hybrid'}  # Special marker for hybrid (model-specific filenames)
}

def load_results(filepath: str, filter_duplicates: bool = True) -> pd.DataFrame:
    """Load evaluation results and convert to DataFrame.

    Args:
        filepath: Path to results JSON file
        filter_duplicates: If True, exclude the 16 known duplicate items (default: True)
    """
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r') as f:
        results = json.load(f)

    rows = []
    filtered_count = 0
    for item in results.values():
        if item.get("predicted_content") != "ERROR":
            # Filter duplicates if this is a digital twin or hybrid result
            if filter_duplicates and ('digital_twin' in filepath.lower() or 'hybrid' in filepath.lower()):
                if is_duplicate(item, DUPLICATE_SIGS):
                    filtered_count += 1
                    continue
            rows.append(item)

    if filtered_count > 0:
        print(f"  → Filtered {filtered_count} duplicate items from {os.path.basename(filepath)}")
    
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
        if np.any(valid_dir_mask):
            dir_gt_valid = dir_gt[valid_dir_mask]
            dir_pred_valid = dir_pred[valid_dir_mask]
            directional_accuracy = np.mean(dir_gt_valid == dir_pred_valid)
            directional_macro_f1 = f1_score(dir_gt_valid, dir_pred_valid, average='macro', zero_division=0)
        else:
            directional_accuracy = np.nan
            directional_macro_f1 = np.nan
        kappa = cohen_kappa_score(gt, pred)
        tau, _ = kendalltau(gt, pred)
        f1_macro = f1_score(gt, pred, average='macro', zero_division=0)
        
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
            'f1_macro': f1_macro,
            'spearman_rho': avg_rho,
            'n_samples': len(gt)
        }
    
    return metrics


def collect_all_results() -> pd.DataFrame:
    """Collect results from all models and methods."""
    
    all_results = []
    
    for model_id, model_cfg in MODEL_CONFIGS.items():
        model_dir = model_cfg['dir']
        model_display = model_cfg['display']
        
        # Skip baseline models (they don't have directories)
        if model_dir is None:
            continue
        
        # Load regular methods
        for method_file, method_cfg in METHOD_CONFIGS.items():
            # Skip hybrid marker (handled separately below)
            if method_file == 'hybrid':
                continue

            filepath = os.path.join(model_dir, method_file)

            df = load_results(filepath)
            if df is None:
                print(f"⚠️  {model_display} - {method_cfg['display']}: NOT FOUND")
                continue
            
            metrics = calculate_metrics(df, DOMAINS)
            if not metrics:
                print(f"⚠️  {model_display} - {method_cfg['display']}: No valid metrics")
                continue
            
            # Add results for each domain
            for domain in DOMAINS:
                if domain in metrics:
                    all_results.append({
                        'Model': model_display,
                        'Model_ID': model_id,
                        'Method': method_cfg['display'],
                        'Category': method_cfg['category'],
                        'Domain': domain.capitalize(),
                        'Accuracy': metrics[domain]['accuracy'],
                        'Directional Accuracy': metrics[domain]['directional_accuracy'],
                        'Directional Macro-F1': metrics[domain]['directional_macro_f1'],
                        'Kappa': metrics[domain]['kappa'],
                        'F1': metrics[domain]['f1_macro'],
                        'Kendall_Tau': metrics[domain]['kendall_tau'],
                        'Spearman_Rho': metrics[domain]['spearman_rho'],
                        'N': metrics[domain]['n_samples']
                    })
            
            print(f"✓ {model_display} - {method_cfg['display']}: Loaded ({len(df)} samples)")
        
        # Load hybrid method (if exists)
        # Convert model_id to safe filename (replace / with -, keep existing -)
        model_safe = model_id.replace('/', '-')
        hybrid_dir = f'results_manuscript_hybrid_{model_safe}'
        hybrid_file = f'evaluation_results_{model_safe}_text-only_hybrid-rf-digital-twin.json'
        hybrid_path = os.path.join(hybrid_dir, hybrid_file)

        df = load_results(hybrid_path)
        if df is not None:
            metrics = calculate_metrics(df, DOMAINS)
            if metrics:
                for domain in DOMAINS:
                    if domain in metrics:
                        all_results.append({
                            'Model': model_display,
                            'Model_ID': model_id,
                            'Method': METHOD_CONFIGS['hybrid']['display'],
                            'Category': METHOD_CONFIGS['hybrid']['category'],
                            'Domain': domain.capitalize(),
                            'Accuracy': metrics[domain]['accuracy'],
                            'Directional Accuracy': metrics[domain]['directional_accuracy'],
                            'Directional Macro-F1': metrics[domain]['directional_macro_f1'],
                            'Kappa': metrics[domain]['kappa'],
                            'F1': metrics[domain]['f1_macro'],
                            'Kendall_Tau': metrics[domain]['kendall_tau'],
                            'Spearman_Rho': metrics[domain]['spearman_rho'],
                            'N': metrics[domain]['n_samples']
                        })
                print(f"✓ {model_display} - {METHOD_CONFIGS['hybrid']['display']}: Loaded ({len(df)} samples)")
            else:
                print(f"⚠️  {model_display} - {METHOD_CONFIGS['hybrid']['display']}: No valid metrics")
        else:
            print(f"⚠️  {model_display} - {METHOD_CONFIGS['hybrid']['display']}: NOT FOUND")
    
    # Add supervised learning baselines (Logistic Regression & Random Forest)
    print("\n📊 Adding supervised learning baselines...")
    baseline_results = run_supervised_baselines()

    for model_name, model_metrics in baseline_results.items():
        for domain in DOMAINS:
            if domain in model_metrics:
                # Add as a single method entry for supervised ML
                all_results.append({
                    'Model': model_name,
                    'Model_ID': model_name.lower().replace(' ', '_'),
                    'Method': 'Supervised ML',
                    'Category': 'Baseline',
                    'Domain': domain.capitalize(),
                    'Accuracy': model_metrics[domain]['accuracy'],
                    'Directional Accuracy': model_metrics[domain]['directional_accuracy'],
                    'Directional Macro-F1': model_metrics[domain]['directional_macro_f1'],
                    'Kappa': model_metrics[domain]['kappa'],
                    'F1': model_metrics[domain]['f1_macro'],
                    'Kendall_Tau': 0.0,
                    'Spearman_Rho': model_metrics[domain]['spearman_rho'],
                    'N': model_metrics[domain].get('n_samples', 0)
                })
    
    print(f"✓ Added Logistic Regression and Random Forest baselines")
    
    if not all_results:
        return None
    
    return pd.DataFrame(all_results)


def create_radar_charts_all_methods(df: pd.DataFrame, output_dir: str):
    """Create radar charts for ALL methods across ALL models."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_to_plot = ['Accuracy', 'Kappa', 'Directional Accuracy', 'Directional Macro-F1', 'Spearman_Rho']
    metric_labels = ['Accuracy', 'Cohen\'s κ', 'Directional Accuracy', 'Directional Macro-F1', 'Spearman\'s ρ']
    
    # Create one figure per domain
    for domain in ['Content', 'Coping', 'Quitting']:
        domain_data = df[df['Domain'] == domain]
        
        if domain_data.empty:
            continue
        
        # Create subplots: one per model
        n_models = len(MODEL_CONFIGS)
        fig, axes = plt.subplots(2, 3, figsize=(20, 12), subplot_kw=dict(projection='polar'))
        axes = axes.flatten()
        
        for idx, (model_id, model_cfg) in enumerate(MODEL_CONFIGS.items()):
            ax = axes[idx]
            model_display = model_cfg['display']
            model_data = domain_data[domain_data['Model'] == model_display]
            
            if model_data.empty:
                ax.set_visible(False)
                continue
            
            angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
            angles += angles[:1]
            
            # Plot each method for this model
            for method in METHOD_CONFIGS.values():
                method_display = method['display']
                method_data = model_data[model_data['Method'] == method_display]
                
                if method_data.empty:
                    continue
                
                values = []
                for metric in metrics_to_plot:
                    val = method_data[metric].values[0] if len(method_data) > 0 else 0
                    # Normalize Spearman's Rho from [-1, 1] to [0, 1]
                    if metric == 'Spearman_Rho':
                        val = (val + 1) / 2 if not np.isnan(val) else 0.5
                    values.append(val)
                
                values += values[:1]
                
                ax.plot(angles, values, 'o-', linewidth=2, label=method_display, markersize=6)
                ax.fill(angles, values, alpha=0.1)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_labels, fontsize=10, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.set_title(f'{model_display}', fontsize=13, fontweight='bold', pad=15)
            ax.grid(True)
            ax.tick_params(labelsize=11, width=2, length=6)
            # Make tick labels bold
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')
            
            # Only show legend on first subplot
            if idx == 0:
                ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15), fontsize=8)
        
        # Hide extra subplot
        if n_models < len(axes):
            axes[-1].set_visible(False)
        
        plt.suptitle(f'Performance Radar Charts - {domain} Domain\n(All Models × All Methods)',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        plot_path = os.path.join(output_dir, f'radar_all_methods_{domain.lower()}')
        save_figure(fig, plot_path)
        plt.close()


def create_heatmap_all_methods(df: pd.DataFrame, output_dir: str):
    """Create heatmaps showing all models × all methods (LLMs only, no supervised baselines)."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out supervised learning baselines for heatmaps
    df_llm_only = df[~df['Model'].isin(['Logistic Regression', 'Random Forest'])].copy()

    metrics = ['Accuracy', 'Kappa', 'Directional Accuracy', 'Directional Macro-F1', 'F1', 'Spearman_Rho']
    
    for metric in metrics:
        fig, axes = plt.subplots(1, 3, figsize=(22, 8))
        
        for idx, domain in enumerate(['Content', 'Coping', 'Quitting']):
            ax = axes[idx]
            
            # Pivot: rows=models, columns=methods
            domain_data = df_llm_only[df_llm_only['Domain'] == domain]
            pivot = domain_data.pivot_table(
                index='Model',
                columns='Method',
                values=metric,
                aggfunc='first'
            )
            
            # Reorder columns to group by category
            col_order = [m['display'] for m in METHOD_CONFIGS.values()]
            pivot = pivot[[c for c in col_order if c in pivot.columns]]
            if pivot.empty:
                ax.set_visible(False)
                continue
            
            # Plot
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', 
                       vmin=0, vmax=1 if metric != 'Kappa' else 0.5,
                       cbar_kws={'label': metric}, ax=ax, linewidths=1.5, 
                       linecolor='white', annot_kws={'fontsize': 12, 'fontweight': 'bold'})
            
            ax.set_title(f'{domain} Domain', fontsize=17, fontweight='bold', pad=15)
            ax.set_xlabel('Method', fontsize=15, fontweight='bold')
            if idx == 0:
                ax.set_ylabel('Model (message-level)', fontsize=15, fontweight='bold')
            else:
                ax.set_ylabel('', fontsize=1)

            # Better tick labels - BOLD and MORE VISIBLE
            # Requested sizing: x ticks +1, y ticks +2, all bold
            ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right', fontsize=17, fontweight='bold')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=17, fontweight='bold')
            ax.tick_params(labelsize=17, width=2.2, length=7)

            if ax.collections and ax.collections[0].colorbar:
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(labelsize=17, width=2)
                cbar.set_label(f'{metric} (message-level)', fontsize=17, fontweight='bold')
        
        plt.suptitle(f'{metric} Across All Models and Methods',
                    fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f'heatmap_all_methods_{metric_to_filename(metric)}')
        save_figure(fig, plot_path)
        plt.close()


def create_grouped_bar_charts(df: pd.DataFrame, output_dir: str):
    """Create grouped bar charts comparing all models and methods."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # One figure per metric (F1 grouped with accuracies)
    metrics = ['Accuracy', 'Directional Accuracy', 'Directional Macro-F1', 'F1', 'Kappa', 'Spearman_Rho']
    
    # Separate supervised learning baselines from LLM models
    llm_models = ['GPT-4o-mini', 'GPT-5', 'DeepSeek-R1', 'Grok-4-Fast', 'Gemini-2.5-Pro']
    supervised_models = ['Logistic Regression', 'Random Forest']
    
    # New colors for supervised learning dashed lines (different from LLM colors)
    supervised_colors = {
        'Logistic Regression': '#E02020',  # Red (distinct color)
        'Random Forest': '#7F7F7F'  # Neutral grey to keep RF muted
    }
    embedding_baselines = load_embedding_baseline_metrics(EMBEDDING_BASELINE_PATH)
    embedding_colors = {
        'Logistic Regression + Embedding': '#9E1B1B',
        'Random Forest + Embedding': '#4D4D4D'
    }
    
    for metric in metrics:
        fig, axes = plt.subplots(1, 3, figsize=(22, 7))
        
        for idx, domain in enumerate(['Content', 'Coping', 'Quitting']):
            ax = axes[idx]
            
            domain_data = df[df['Domain'] == domain]
            
            # Split data: LLMs vs Supervised
            llm_data = domain_data[domain_data['Model'].isin(llm_models)]
            supervised_data = domain_data[domain_data['Model'].isin(supervised_models)]
            
            # Pivot for LLM models only
            pivot = llm_data.pivot_table(
                index='Method',
                columns='Model',
                values=metric,
                aggfunc='first'
            )
            
            # Reorder rows
            row_order = [m['display'] for m in METHOD_CONFIGS.values()]
            pivot = pivot.reindex([r for r in row_order if r in pivot.index])
            if pivot.empty:
                ax.set_visible(False)
                continue
            
            # Create color map for LLM models using Okabe-Ito colors
            model_color_map = {cfg['display']: cfg['color'] for cfg in MODEL_CONFIGS.values() 
                              if cfg['display'] in llm_models}
            colors = [model_color_map.get(col, '#999999') for col in pivot.columns]
            
            # Plot LLM models as bars
            bars = pivot.plot(kind='bar', ax=ax, width=0.8, rot=45, color=colors, 
                             edgecolor='black', linewidth=1.2, legend=False)
            
            # Add supervised learning as horizontal dashed lines
            for sup_model in supervised_models:
                sup_values = supervised_data[supervised_data['Model'] == sup_model]
                if not sup_values.empty:
                    sup_value = sup_values[metric].values[0]
                    ax.axhline(y=sup_value, color=supervised_colors[sup_model], 
                              linestyle='--', linewidth=2.5, alpha=0.8, label=sup_model)

            # Add embedding baselines where metric is available from archive_results.
            emb = embedding_baselines.get(domain, {})
            if metric == 'Accuracy':
                lr_emb = emb.get('lr_acc', np.nan)
                rf_emb = emb.get('rf_acc', np.nan)
            elif metric == 'F1':
                lr_emb = emb.get('lr_f1', np.nan)
                rf_emb = emb.get('rf_f1', np.nan)
            else:
                lr_emb = np.nan
                rf_emb = np.nan
            if lr_emb == lr_emb:
                ax.axhline(y=lr_emb, color=embedding_colors['Logistic Regression + Embedding'],
                           linestyle='-.', linewidth=2.5, alpha=0.9,
                           label='Logistic Reg. + Embedding')
            if rf_emb == rf_emb:
                ax.axhline(y=rf_emb, color=embedding_colors['Random Forest + Embedding'],
                           linestyle='-.', linewidth=2.5, alpha=0.9,
                           label='Random Forest + Embedding')
            
            ax.set_title(f'{domain} Domain', fontsize=17, fontweight='bold', pad=15)
            ax.set_xlabel('', fontsize=1)  # Remove xlabel, methods clear from ticks
            n_messages = int(domain_data['N'].dropna().max()) if domain_data['N'].notna().any() else 0
            if metric == 'F1':
                ylabel = f'Message-level F1 Macro (n={n_messages})'
            else:
                ylabel = f'{metric} (message-level)'
            ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            # Requested sizing: x ticks +1, y ticks +2, all bold
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=17, fontweight='bold')
            ax.tick_params(labelsize=17, width=2.2, length=7)
            # Make y-tick labels bold
            for label in ax.get_yticklabels():
                label.set_fontweight('bold')
                label.set_fontsize(17)

            # Increase y-axis upper limit for Quitting domain to prevent labels outside box
            if domain == 'Quitting':
                ylim = ax.get_ylim()
                ax.set_ylim(ylim[0], ylim[1] + 0.05)

            # Removed bar-value labels per user request (too small/cluttered).
        
        # Create ONE shared legend at the bottom for all subplots
        handles = []
        labels = []
        
        # LLM models (solid bars)
        for model_display, color in model_color_map.items():
            from matplotlib.patches import Patch
            handles.append(Patch(facecolor=color, edgecolor='black', linewidth=1.2))
            labels.append(model_display)
        
        # Supervised models (dashed lines)
        from matplotlib.lines import Line2D
        for sup_model in supervised_models:
            handles.append(Line2D([0], [0], color=supervised_colors[sup_model], 
                                linestyle='--', linewidth=2.5, alpha=0.8))
            if sup_model == 'Logistic Regression':
                labels.append('Logistic Reg. (No Embedding)')
            else:
                labels.append('Random Forest (No Embedding)')

        # Embedding baseline legend entries for metrics where they are plotted.
        if metric in ['Accuracy', 'F1']:
            handles.append(Line2D([0], [0], color=embedding_colors['Logistic Regression + Embedding'],
                                  linestyle='-.', linewidth=2.5, alpha=0.9))
            labels.append('Logistic Reg. + Embedding')
            handles.append(Line2D([0], [0], color=embedding_colors['Random Forest + Embedding'],
                                  linestyle='-.', linewidth=2.5, alpha=0.9))
            labels.append('Random Forest + Embedding')
        
        legend = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
                            ncol=len(labels), frameon=True, fontsize=17,
                            title='Models and Baselines', title_fontsize=17)
        legend.get_title().set_fontweight('bold')
        for txt in legend.get_texts():
            txt.set_fontweight('bold')
        
        plt.suptitle(f'{metric} Comparison: All Models × All Methods',
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])

        plot_path = os.path.join(output_dir, f'bars_all_methods_{metric_to_filename(metric)}')
        save_figure(fig, plot_path)
        plt.close()


def create_kappa_accuracy_scatter_all(df: pd.DataFrame, output_dir: str):
    """Create scatter plot of Kappa vs Accuracy for all models and methods."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    # One subplot per method
    for idx, (method_file, method_cfg) in enumerate(METHOD_CONFIGS.items()):
        ax = axes[idx]
        method_display = method_cfg['display']
        method_data = df[df['Method'] == method_display]
        
        if method_data.empty:
            ax.set_visible(False)
            continue
        
        # Color by domain
        domain_colors = {'Content': 'blue', 'Coping': 'green', 'Quitting': 'red'}
        
        for domain in ['Content', 'Coping', 'Quitting']:
            domain_method_data = method_data[method_data['Domain'] == domain]
            
            if domain_method_data.empty:
                continue
            
            ax.scatter(
                domain_method_data['Accuracy'],
                domain_method_data['Kappa'],
                label=domain,
                alpha=0.7,
                s=120,
                color=domain_colors[domain],
                edgecolors='black',
                linewidth=0.5
            )
        
        # Add diagonal reference line
        lim = max(method_data['Accuracy'].max(), method_data['Kappa'].max()) * 1.1
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cohen\'s Kappa', fontsize=12, fontweight='bold')
        ax.set_title(f'{method_display}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.6)
        ax.set_ylim(-0.1, 0.2)
        ax.tick_params(labelsize=11, width=2, length=6)
        # Make tick labels bold
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
    
    plt.suptitle('Kappa vs Accuracy: All Methods (All Models, All Domains)',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plot_path = os.path.join(output_dir, 'scatter_kappa_vs_accuracy_all_methods')
    save_figure(fig, plot_path)
    plt.close()


def create_method_focus_f1_figures(df: pd.DataFrame, output_dir: str):
    """Create NEW method-specific F1 figures with tabular baseline lines."""

    os.makedirs(output_dir, exist_ok=True)

    focus_methods = ['Zero-shot (all)', 'Few-shot (all)', 'Digital Twin']
    llm_models = ['GPT-4o-mini', 'GPT-5', 'DeepSeek-R1', 'Grok-4-Fast', 'Gemini-2.5-Pro']
    model_color_map = {cfg['display']: cfg['color'] for cfg in MODEL_CONFIGS.values() if cfg['display'] in llm_models}
    supervised_styles = {
        'Logistic Regression': {'color': '#E02020', 'linestyle': '--'},
        'Random Forest': {'color': '#7F7F7F', 'linestyle': '--'},
        'Logistic Regression + Embedding': {'color': '#9E1B1B', 'linestyle': '-.'},
        'Random Forest + Embedding': {'color': '#4D4D4D', 'linestyle': '-.'}
    }
    embedding_baselines = load_embedding_baseline_f1(EMBEDDING_BASELINE_PATH)

    for method_name in focus_methods:
        method_df = df[df['Method'] == method_name].copy()
        if method_df.empty:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        for idx, domain in enumerate(['Content', 'Coping', 'Quitting']):
            ax = axes[idx]
            domain_df = method_df[method_df['Domain'] == domain]
            llm_df = domain_df[domain_df['Model'].isin(llm_models)].copy()
            llm_df['Model'] = pd.Categorical(llm_df['Model'], categories=llm_models, ordered=True)
            llm_df = llm_df.sort_values('Model')

            bars = ax.bar(
                llm_df['Model'],
                llm_df['F1'],
                color=[model_color_map.get(m, '#999999') for m in llm_df['Model']],
                edgecolor='black',
                linewidth=1.2,
                width=0.75
            )

            # Baseline lines: no-embedding (from current supervised run).
            baseline_domain = df[(df['Method'] == 'Supervised ML') & (df['Domain'] == domain)]
            for baseline_name in ['Logistic Regression', 'Random Forest']:
                base_row = baseline_domain[baseline_domain['Model'] == baseline_name]
                if not base_row.empty:
                    base_f1 = float(base_row['F1'].iloc[0])
                    line_label = f'{baseline_name} (No Embedding)'
                    style = supervised_styles[baseline_name]
                    ax.axhline(base_f1, color=style['color'], linestyle=style['linestyle'],
                               linewidth=2.5, alpha=0.9, label=line_label)

            # Embedding baselines (archived tabular+embedding runs).
            domain_embed = embedding_baselines.get(domain, {})
            for emb_name in ['Logistic Regression + Embedding', 'Random Forest + Embedding']:
                emb_f1 = domain_embed.get(emb_name, np.nan)
                if emb_f1 == emb_f1:
                    style = supervised_styles[emb_name]
                    line_label = emb_name
                    ax.axhline(emb_f1, color=style['color'], linestyle=style['linestyle'],
                               linewidth=2.5, alpha=0.9, label=line_label)

            ax.set_title(f'{domain} Domain', fontsize=17, fontweight='bold', pad=15)
            ax.set_ylabel('Message-level F1 Macro', fontsize=16, fontweight='bold')
            ax.set_xlabel('LLM Model', fontsize=15, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.tick_params(axis='x', labelsize=14, width=2.2, length=7)
            ax.tick_params(axis='y', labelsize=15, width=2.2, length=7)
            for label in ax.get_xticklabels():
                label.set_fontweight('bold')
            for label in ax.get_yticklabels():
                label.set_fontweight('bold')
                label.set_fontsize(17)

            # Removed bar-value labels per user request (too small/cluttered).

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            legend = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.06),
                                ncol=4, frameon=True, fontsize=13, title='4 Baselines', title_fontsize=14)
            legend.get_title().set_fontweight('bold')
            for txt in legend.get_texts():
                txt.set_fontweight('bold')

        plt.suptitle(f'NEW: F1 Comparison for {method_name} (4 Baselines: w/ and w/o Embedding)',
                     fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.06, 1, 0.95])

        method_suffix = method_name.lower().replace('-', '_').replace(' ', '_').replace('(', '').replace(')', '')
        plot_path = os.path.join(output_dir, f'new_f1_method_focus_{method_suffix}')
        save_figure(fig, plot_path)
        plt.close()


def create_domain_focus_f1_across_methods(df: pd.DataFrame, output_dir: str):
    """
    Create NEW figures in the requested old-style layout:
    - One figure for Coping, one for Quitting
    - X-axis methods: Zero-shot (all), Few-shot (all), Digital Twin
    - Bar groups are LLM models
    - 4 baseline lines: LR/RF with and without embeddings
    """
    os.makedirs(output_dir, exist_ok=True)

    focus_domains = ['Coping', 'Quitting']
    focus_methods = ['Zero-shot (all)', 'Few-shot (all)', 'Digital Twin']
    llm_models = ['GPT-4o-mini', 'GPT-5', 'DeepSeek-R1', 'Grok-4-Fast', 'Gemini-2.5-Pro']
    model_color_map = {cfg['display']: cfg['color'] for cfg in MODEL_CONFIGS.values() if cfg['display'] in llm_models}
    embedding_baselines = load_embedding_baseline_f1(EMBEDDING_BASELINE_PATH)
    baseline_styles = {
        'Logistic Regression (No Embedding)': {'color': '#E02020', 'linestyle': '--'},
        'Random Forest (No Embedding)': {'color': '#7F7F7F', 'linestyle': '--'},
        'Logistic Regression + Embedding': {'color': '#9E1B1B', 'linestyle': '-.'},
        'Random Forest + Embedding': {'color': '#4D4D4D', 'linestyle': '-.'}
    }

    # Keep width same as before (14), combine two domains horizontally.
    fig, axes = plt.subplots(1, 2, figsize=(14, 8.6))
    method_label_map = {
        'Zero-shot (all)': 'Zero-shot',
        'Few-shot (all)': 'Few-shot',
        'Digital Twin': 'Digital Twin'
    }

    for idx, domain in enumerate(focus_domains):
        ax = axes[idx]
        domain_df = df[(df['Domain'] == domain) & (df['Method'].isin(focus_methods))].copy()
        llm_df = domain_df[domain_df['Model'].isin(llm_models)].copy()
        if llm_df.empty:
            ax.set_visible(False)
            continue

        pivot = llm_df.pivot_table(index='Method', columns='Model', values='F1', aggfunc='first')
        pivot = pivot.reindex([m for m in focus_methods if m in pivot.index])
        pivot = pivot[[m for m in llm_models if m in pivot.columns]]
        if pivot.empty:
            ax.set_visible(False)
            continue

        pivot.index = [method_label_map.get(i, i) for i in pivot.index]
        colors = [model_color_map.get(col, '#999999') for col in pivot.columns]
        pivot.plot(kind='bar', ax=ax, width=0.82, rot=15, color=colors,
                   edgecolor='black', linewidth=1.2, legend=False)

        # 4 baselines
        base_domain = df[(df['Method'] == 'Supervised ML') & (df['Domain'] == domain)]
        lr_no = base_domain[base_domain['Model'] == 'Logistic Regression']
        rf_no = base_domain[base_domain['Model'] == 'Random Forest']
        if not lr_no.empty:
            v = float(lr_no['F1'].iloc[0])
            s = baseline_styles['Logistic Regression (No Embedding)']
            ax.axhline(v, color=s['color'], linestyle=s['linestyle'], linewidth=2.5, alpha=0.9,
                       label='Logistic Reg. (No Embedding)')
        if not rf_no.empty:
            v = float(rf_no['F1'].iloc[0])
            s = baseline_styles['Random Forest (No Embedding)']
            ax.axhline(v, color=s['color'], linestyle=s['linestyle'], linewidth=2.5, alpha=0.9,
                       label='Random Forest (No Embedding)')

        emb = embedding_baselines.get(domain, {})
        lr_emb = emb.get('Logistic Regression + Embedding', np.nan)
        rf_emb = emb.get('Random Forest + Embedding', np.nan)
        if lr_emb == lr_emb:
            s = baseline_styles['Logistic Regression + Embedding']
            ax.axhline(lr_emb, color=s['color'], linestyle=s['linestyle'], linewidth=2.5, alpha=0.9,
                       label='Logistic Reg. + Embedding')
        if rf_emb == rf_emb:
            s = baseline_styles['Random Forest + Embedding']
            ax.axhline(rf_emb, color=s['color'], linestyle=s['linestyle'], linewidth=2.5, alpha=0.9,
                       label='Random Forest + Embedding')

        ax.set_title(f'{domain} F1 Across Models', fontsize=17, fontweight='bold', pad=12)
        ax.set_xlabel('Method', fontsize=17, fontweight='bold')
        ax.set_ylabel('Message-level F1 Macro', fontsize=17, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=17, fontweight='bold')
        ax.tick_params(axis='y', labelsize=17, width=2.2, length=7)
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

    # Single compact legend (model colors + 4 baselines)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Patch(facecolor=model_color_map[m], edgecolor='black', linewidth=1.2, label=m) for m in llm_models]
    handles += [
        Line2D([0], [0], color='#E02020', linestyle='--', linewidth=2.5, label='Logistic Reg. (No Embedding)'),
        Line2D([0], [0], color='#7F7F7F', linestyle='--', linewidth=2.5, label='Random Forest (No Embedding)'),
        Line2D([0], [0], color='#9E1B1B', linestyle='-.', linewidth=2.5, label='Logistic Reg. + Embedding'),
        Line2D([0], [0], color='#4D4D4D', linestyle='-.', linewidth=2.5, label='Random Forest + Embedding'),
    ]
    legend = fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.065), ncol=5,
                        frameon=True, fontsize=17, title='Models', title_fontsize=17,
                        columnspacing=1.0, handletextpad=0.5, borderaxespad=0.3)
    legend.get_title().set_fontweight('bold')
    for txt in legend.get_texts():
        txt.set_fontweight('bold')

    plt.tight_layout(rect=[0.02, 0.16, 0.98, 1])
    plot_path = os.path.join(output_dir, 'new_f1_coping_quitting_combined_zero_few_dt')
    save_figure(fig, plot_path)
    plt.close()


def create_summary_table(df: pd.DataFrame, output_dir: str):
    """Create comprehensive summary table."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full table
    csv_path = os.path.join(output_dir, 'comprehensive_results_all_methods.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Full results table saved: {csv_path}")
    
    # Create markdown version
    md_path = os.path.join(output_dir, 'comprehensive_results_all_methods.md')
    with open(md_path, 'w') as f:
        f.write("# Comprehensive Results: All Models × All Methods\n\n")
        try:
            f.write(df.to_markdown(index=False))
        except ImportError:
            # Fallback when optional dependency `tabulate` is unavailable.
            f.write("_`tabulate` not installed; showing CSV-style preview instead._\n\n")
            f.write("```csv\n")
            f.write(df.to_csv(index=False))
            f.write("```\n")
    print(f"✓ Markdown table saved: {md_path}")
    
    # Create summary statistics
    summary_stats = df.groupby(['Model', 'Method', 'Domain'])[
        ['Accuracy', 'Directional Accuracy', 'Directional Macro-F1', 'F1', 'Kappa']
    ].agg(['mean', 'std'])
    
    summary_path = os.path.join(output_dir, 'summary_statistics.csv')
    summary_stats.to_csv(summary_path)
    print(f"✓ Summary statistics saved: {summary_path}")


def main():
    print("="*80)
    print("COMPREHENSIVE CROSS-MODEL ANALYSIS")
    print("Analyzing ALL models × ALL methods")
    print("="*80)
    print()
    
    # Collect all results
    print("📊 Loading results from all models and methods...")
    df = collect_all_results()
    
    if df is None or len(df) == 0:
        print("\n❌ No results found. Run evaluations first:")
        print("   bash e2e_pipeline.sh")
        return
    
    print(f"\n✓ Loaded {len(df)} result entries")
    print(f"  - Models: {df['Model'].nunique()}")
    print(f"  - Methods: {df['Method'].nunique()}")
    print(f"  - Domains: {df['Domain'].nunique()}")
    
    # Create output directory (model-agnostic!)
    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"GENERATING COMPREHENSIVE FIGURES")
    print(f"Output directory: {output_dir}/ (model-agnostic)")
    print(f"{'='*80}\n")
    
    # Create all visualizations (RADAR CHARTS DELETED PER USER REQUEST)
    print("\n📈 Creating heatmaps (all methods, all models)...")
    create_heatmap_all_methods(df, output_dir)
    
    print("\n📈 Creating grouped bar charts (all methods, all models)...")
    create_grouped_bar_charts(df, output_dir)

    print("\n📈 Creating NEW method-specific F1 figures...")
    create_method_focus_f1_figures(df, output_dir)
    print("\n📈 Creating NEW domain-focused F1 figures (Coping + Quitting)...")
    create_domain_focus_f1_across_methods(df, output_dir)
    
    # Removed scatter plot per user request
    
    print("\n📈 Creating summary tables...")
    create_summary_table(df, output_dir)
    
    print(f"\n{'='*80}")
    print("✓ Comprehensive analysis complete!")
    print(f"{'='*80}\n")
    print(f"Output directory: {output_dir}/")
    print("\nGenerated files:")
    print("  Heatmaps (all methods):")
    print("    - heatmap_all_methods_accuracy.png/pdf")
    print("    - heatmap_all_methods_kappa.png/pdf")
    print("    - heatmap_all_methods_directional_accuracy.png/pdf")
    print("    - heatmap_all_methods_directional_macro_f1.png/pdf")
    print("    - heatmap_all_methods_f1.png/pdf")
    print("    - heatmap_all_methods_spearman_rho.png/pdf")
    print("  Bar Charts (message-level y-axis labels, supervised ML as dashed lines):")
    print("    - bars_all_methods_accuracy.png/pdf")
    print("    - bars_all_methods_directional_accuracy.png/pdf")
    print("    - bars_all_methods_directional_macro_f1.png/pdf")
    print("    - bars_all_methods_f1.png/pdf")
    print("    - bars_all_methods_kappa.png/pdf")
    print("    - bars_all_methods_spearman_rho.png/pdf")
    print("  Tables:")
    print("    - comprehensive_results_all_methods.csv")
    print("    - comprehensive_results_all_methods.md")
    print("    - summary_statistics.csv")
    print("\n💡 All figures include:")
    print("   • 7 models (5 LLMs + 2 Supervised ML baselines)")
    print("   • all methods in original (including Digital Twin/Hybrid)")
    print("   • 3 domains (Content, Coping, Quitting)")
    print("   • Colorblind-friendly Okabe-Ito palette")
    print("   • BOLD, LARGER value labels on bar charts")


if __name__ == '__main__':
    main()
    print("  NEW Method-Specific F1 Figures:")
    print("    - new_f1_method_focus_zero_shot_all.png/pdf")
    print("    - new_f1_method_focus_few_shot_all.png/pdf")
    print("    - new_f1_method_focus_digital_twin.png/pdf")
    print("  NEW Domain-Focused F1 Figure (Combined):")
    print("    - new_f1_coping_quitting_combined_zero_few_dt.png/pdf")
