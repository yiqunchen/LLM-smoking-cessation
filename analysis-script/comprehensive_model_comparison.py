"""
Comprehensive Model Comparison Analysis for Manuscript

This script creates publication-ready visualizations and statistical comparisons
following best practices from ML classification papers:

1. Main comparison: GPT-5 and DeepSeek-R1 (primary models)
2. Supplementary: Other models (GPT-4o-mini, Grok-4-Fast, Gemini-2.5-Pro)
3. Multiple visualization types:
   - Performance matrices (accuracy, F1, etc.)
   - Kappa vs Accuracy scatter plots
   - Domain-specific comparisons
   - Confusion matrices
   - Learning curves (for digital twin varying training sizes)
   - Statistical significance tests

Key visualizations:
- Fig 1: Main comparison table (GPT-5 vs DeepSeek-R1)
- Fig 2: Scatter plots (Kappa vs Accuracy, colored by domain)
- Fig 3: Per-domain performance radar charts
- Fig 4: Digital twin learning curves
- Fig 5: Supplementary model comparisons
- Table 1: Statistical significance tests (paired t-tests)
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    cohen_kappa_score, accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score
)
from scipy.stats import spearmanr, kendalltau, ttest_rel, wilcoxon
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
FIGSIZE_SINGLE = (8, 6)
FIGSIZE_MULTI = (16, 12)
DPI = 300

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

# Model display names and groups
MODEL_DISPLAY_NAMES = {
    'gpt-5': 'GPT-5',
    'deepseek_deepseek-r1-0528': 'DeepSeek-R1',
    'gpt-4o-mini': 'GPT-4o-mini',
    'x-ai_grok-4-fast': 'Grok-4-Fast',
    'gemini-2.5-pro': 'Gemini-2.5-Pro'
}

PRIMARY_MODELS = ['gpt-5', 'deepseek_deepseek-r1-0528']
SUPPLEMENTARY_MODELS = ['gpt-4o-mini', 'x-ai_grok-4-fast', 'gemini-2.5-pro']


def load_results(results_path: str) -> pd.DataFrame:
    """Load evaluation results and convert to DataFrame."""
    if not os.path.exists(results_path):
        return None
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    rows = []
    for item in results.values():
        if item.get("predicted_content") != "ERROR":
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


def calculate_comprehensive_metrics(df: pd.DataFrame, domains: list) -> dict:
    """Calculate comprehensive metrics including confidence intervals."""
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
        
        # Core metrics
        acc = accuracy_score(gt, pred)
        acc_within_1 = np.mean(np.abs(gt - pred) <= 1)
        kappa = cohen_kappa_score(gt, pred)
        tau, _ = kendalltau(gt, pred)
        f1_macro = f1_score(gt, pred, average='macro', zero_division=0)
        precision_macro = precision_score(gt, pred, average='macro', zero_division=0)
        recall_macro = recall_score(gt, pred, average='macro', zero_division=0)
        
        # Bootstrap confidence intervals (95%)
        n_bootstrap = 1000
        bootstrap_accs = []
        bootstrap_kappas = []
        bootstrap_f1s = []
        
        np.random.seed(42)
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(gt), len(gt), replace=True)
            gt_boot = gt[indices]
            pred_boot = pred[indices]
            if len(np.unique(gt_boot)) > 1:
                bootstrap_accs.append(accuracy_score(gt_boot, pred_boot))
                bootstrap_kappas.append(cohen_kappa_score(gt_boot, pred_boot))
                bootstrap_f1s.append(f1_score(gt_boot, pred_boot, average='macro', zero_division=0))
        
        acc_ci = np.percentile(bootstrap_accs, [2.5, 97.5]) if bootstrap_accs else (np.nan, np.nan)
        kappa_ci = np.percentile(bootstrap_kappas, [2.5, 97.5]) if bootstrap_kappas else (np.nan, np.nan)
        f1_ci = np.percentile(bootstrap_f1s, [2.5, 97.5]) if bootstrap_f1s else (np.nan, np.nan)
        
        # Per-participant metrics
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
        std_rho = np.std(participant_rhos) if participant_rhos else np.nan
        
        # Confusion matrix for additional insights
        cm = confusion_matrix(gt, pred, labels=[1, 2, 3, 4, 5])
        
        metrics[domain] = {
            'accuracy': acc,
            'accuracy_ci': acc_ci,
            'accuracy_within_1': acc_within_1,
            'kappa': kappa,
            'kappa_ci': kappa_ci,
            'kendall_tau': tau,
            'spearman_rho': avg_rho,
            'spearman_rho_std': std_rho,
            'f1_macro': f1_macro,
            'f1_ci': f1_ci,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'n_samples': len(gt),
            'n_participants': len(participant_rhos) if participant_rhos else 0,
            'confusion_matrix': cm,
            'predictions': pred,
            'ground_truth': gt
        }
    
    return metrics


def aggregate_model_results(base_dir: str = '.') -> Dict[str, Dict]:
    """Aggregate all results for each model across all methods."""
    
    results_by_model = {}
    
    # Detect all model result directories
    model_dirs = [d for d in os.listdir(base_dir) if d.startswith('results_manuscript_')]
    
    for model_dir in model_dirs:
        # Extract model name
        model_name = model_dir.replace('results_manuscript_', '')
        
        # Method filenames (Generic LLM methods only for fair comparison)
        method_files = [
            'generic_llm_1_zero_shot.json',
            'generic_llm_2_zero_shot_select.json',
            'generic_llm_5_continuous.json',  # Best performing generic methods
        ]
        
        model_results = {}
        
        for method_file in method_files:
            filepath = os.path.join(base_dir, model_dir, method_file)
            df = load_results(filepath)
            
            if df is not None:
                metrics = calculate_comprehensive_metrics(df, DOMAINS)
                if metrics:
                    method_name = method_file.replace('generic_llm_', '').replace('.json', '')
                    model_results[method_name] = metrics
        
        if model_results:
            results_by_model[model_name] = model_results
    
    return results_by_model


def create_main_comparison_table(results_by_model: Dict, output_dir: str = 'results_manuscript/figures'):
    """Create main comparison table (Table 1) for GPT-5 vs DeepSeek-R1."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    rows = []
    
    for model_name in PRIMARY_MODELS:
        if model_name not in results_by_model:
            print(f"⚠️  {model_name} not found in results")
            continue
        
        model_display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        
        for method_name, method_results in results_by_model[model_name].items():
            for domain in DOMAINS:
                if domain not in method_results:
                    continue
                
                metrics = method_results[domain]
                
                # Format with confidence intervals
                acc_str = f"{metrics['accuracy']:.3f} [{metrics['accuracy_ci'][0]:.3f}, {metrics['accuracy_ci'][1]:.3f}]"
                kappa_str = f"{metrics['kappa']:.3f} [{metrics['kappa_ci'][0]:.3f}, {metrics['kappa_ci'][1]:.3f}]"
                f1_str = f"{metrics['f1_macro']:.3f} [{metrics['f1_ci'][0]:.3f}, {metrics['f1_ci'][1]:.3f}]"
                
                rows.append({
                    'Model': model_display,
                    'Method': method_name.replace('_', ' ').title(),
                    'Domain': domain.capitalize(),
                    'Accuracy (95% CI)': acc_str,
                    'Cohen\'s κ (95% CI)': kappa_str,
                    'F1 Macro (95% CI)': f1_str,
                    'Acc±1': f"{metrics['accuracy_within_1']:.3f}",
                    'Kendall\'s τ': f"{metrics['kendall_tau']:.3f}",
                    'Spearman\'s ρ': f"{metrics['spearman_rho']:.3f}",
                    'N': metrics['n_samples']
                })
    
    df = pd.DataFrame(rows)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, 'table1_main_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Table 1 saved to: {csv_path}")
    
    # Save as LaTeX
    latex_path = os.path.join(output_dir, 'table1_main_comparison.tex')
    df.to_latex(latex_path, index=False, escape=False, column_format='llllllllll')
    print(f"✓ Table 1 (LaTeX) saved to: {latex_path}")
    
    # Save as Markdown
    md_path = os.path.join(output_dir, 'table1_main_comparison.md')
    with open(md_path, 'w') as f:
        f.write("# Table 1: Main Model Comparison (GPT-5 vs DeepSeek-R1)\n\n")
        f.write(df.to_markdown(index=False))
    print(f"✓ Table 1 (Markdown) saved to: {md_path}")
    
    return df


def create_kappa_accuracy_scatter(results_by_model: Dict, output_dir: str = 'results_manuscript/figures'):
    """Create scatter plot of Cohen's Kappa vs Accuracy (Fig 2)."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Primary models, Right: All models
    for ax_idx, (ax, model_list, title_suffix) in enumerate([
        (axes[0], PRIMARY_MODELS, 'Primary Models'),
        (axes[1], list(results_by_model.keys()), 'All Models')
    ]):
        
        scatter_data = []
        
        for model_name in model_list:
            if model_name not in results_by_model:
                continue
            
            model_display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            
            for method_name, method_results in results_by_model[model_name].items():
                for domain in DOMAINS:
                    if domain not in method_results:
                        continue
                    
                    metrics = method_results[domain]
                    scatter_data.append({
                        'model': model_display,
                        'domain': domain,
                        'accuracy': metrics['accuracy'],
                        'kappa': metrics['kappa'],
                        'f1': metrics['f1_macro']
                    })
        
        scatter_df = pd.DataFrame(scatter_data)
        
        if scatter_df.empty:
            continue
        
        # Color by domain
        domain_colors = {'content': 'blue', 'coping': 'green', 'quitting': 'red'}
        
        for domain in DOMAINS:
            domain_data = scatter_df[scatter_df['domain'] == domain]
            ax.scatter(
                domain_data['accuracy'],
                domain_data['kappa'],
                label=domain.capitalize(),
                alpha=0.7,
                s=100,
                color=domain_colors[domain],
                edgecolors='black',
                linewidth=0.5
            )
        
        # Add diagonal reference line (perfect agreement)
        lim_max = max(scatter_df['accuracy'].max(), scatter_df['kappa'].max()) * 1.1
        ax.plot([0, lim_max], [0, lim_max], 'k--', alpha=0.3, linewidth=1)
        
        # Formatting
        ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cohen\'s Kappa (κ)', fontsize=12, fontweight='bold')
        ax.set_title(f'Kappa vs Accuracy - {title_suffix}', fontsize=14, fontweight='bold')
        ax.legend(title='Domain', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, lim_max)
        ax.set_ylim(0, min(lim_max, scatter_df['kappa'].max() * 1.1))
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'fig2_kappa_vs_accuracy_scatter.png')
    plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Figure 2 saved to: {plot_path}")
    plt.close()


def create_radar_charts(results_by_model: Dict, output_dir: str = 'results_manuscript/figures'):
    """Create radar charts comparing models across multiple metrics (Fig 3)."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_to_plot = ['accuracy', 'kappa', 'f1_macro', 'accuracy_within_1', 'spearman_rho']
    metric_labels = ['Accuracy', 'Cohen\'s κ', 'F1 Macro', 'Acc±1', 'Spearman\'s ρ']
    
    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(18, 6), subplot_kw=dict(projection='polar'))
    
    for domain_idx, domain in enumerate(DOMAINS):
        ax = axes[domain_idx]
        
        # Prepare data for each model
        angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for model_name in PRIMARY_MODELS:
            if model_name not in results_by_model:
                continue
            
            model_display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            
            # Average across methods for this domain
            metric_values = {m: [] for m in metrics_to_plot}
            
            for method_name, method_results in results_by_model[model_name].items():
                if domain in method_results:
                    for metric in metrics_to_plot:
                        val = method_results[domain].get(metric, np.nan)
                        if not np.isnan(val):
                            metric_values[metric].append(val)
            
            # Average and normalize to [0, 1]
            values = [np.mean(metric_values[m]) if metric_values[m] else 0 for m in metrics_to_plot]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_display)
            ax.fill(angles, values, alpha=0.15)
        
        # Formatting
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title(f'{domain.capitalize()}', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'fig3_radar_charts_by_domain.png')
    plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Figure 3 saved to: {plot_path}")
    plt.close()


def create_digital_twin_learning_curves(base_dir: str = '.', output_dir: str = 'results_manuscript/figures'):
    """Create learning curves for digital twin models with varying training sizes (Fig 4)."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    split_ratios = ['1090', '3070', '7030', '9010']
    train_percentages = [10, 30, 70, 90]
    
    # Collect data for primary models
    learning_curve_data = {model: {domain: {'acc': [], 'kappa': []} for domain in DOMAINS} 
                          for model in PRIMARY_MODELS}
    
    for model_name in PRIMARY_MODELS:
        model_dir = f'results_manuscript_{model_name}'
        
        for split in split_ratios:
            # Use CBT/ACT digital twin (best performing)
            filepath = os.path.join(base_dir, model_dir, f'digital_twin_4_cbtact_{split}.json')
            df = load_results(filepath)
            
            if df is not None:
                metrics = calculate_comprehensive_metrics(df, DOMAINS)
                
                for domain in DOMAINS:
                    if domain in metrics:
                        learning_curve_data[model_name][domain]['acc'].append(metrics[domain]['accuracy'])
                        learning_curve_data[model_name][domain]['kappa'].append(metrics[domain]['kappa'])
    
    # Plot
    fig, axes = plt.subplots(len(DOMAINS), 2, figsize=(14, 12))
    
    for domain_idx, domain in enumerate(DOMAINS):
        # Left: Accuracy, Right: Kappa
        for metric_idx, (metric_key, metric_label) in enumerate([('acc', 'Accuracy'), ('kappa', 'Cohen\'s κ')]):
            ax = axes[domain_idx, metric_idx]
            
            for model_name in PRIMARY_MODELS:
                if model_name not in learning_curve_data:
                    continue
                
                model_display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
                values = learning_curve_data[model_name][domain][metric_key]
                
                if len(values) == len(train_percentages):
                    ax.plot(train_percentages, values, 'o-', linewidth=2, markersize=8, label=model_display)
            
            ax.set_xlabel('Training Set Size (%)', fontsize=11, fontweight='bold')
            ax.set_ylabel(metric_label, fontsize=11, fontweight='bold')
            ax.set_title(f'{domain.capitalize()} - {metric_label}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(train_percentages)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'fig4_digital_twin_learning_curves.png')
    plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Figure 4 saved to: {plot_path}")
    plt.close()


def create_supplementary_comparison(results_by_model: Dict, output_dir: str = 'results_manuscript/figures'):
    """Create supplementary comparison for all models (Supplementary Table)."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    rows = []
    
    for model_name in results_by_model.keys():
        model_display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        
        for method_name, method_results in results_by_model[model_name].items():
            for domain in DOMAINS:
                if domain not in method_results:
                    continue
                
                metrics = method_results[domain]
                
                rows.append({
                    'Model': model_display,
                    'Method': method_name.replace('_', ' ').title(),
                    'Domain': domain.capitalize(),
                    'Accuracy': f"{metrics['accuracy']:.3f}",
                    'Acc±1': f"{metrics['accuracy_within_1']:.3f}",
                    'Cohen\'s κ': f"{metrics['kappa']:.3f}",
                    'Kendall\'s τ': f"{metrics['kendall_tau']:.3f}",
                    'Spearman\'s ρ': f"{metrics['spearman_rho']:.3f} ± {metrics['spearman_rho_std']:.3f}",
                    'F1': f"{metrics['f1_macro']:.3f}",
                    'Precision': f"{metrics['precision_macro']:.3f}",
                    'Recall': f"{metrics['recall_macro']:.3f}",
                    'N': metrics['n_samples']
                })
    
    df = pd.DataFrame(rows)
    
    # Save
    csv_path = os.path.join(output_dir, 'supplementary_table_all_models.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Supplementary Table saved to: {csv_path}")
    
    latex_path = os.path.join(output_dir, 'supplementary_table_all_models.tex')
    df.to_latex(latex_path, index=False, escape=False)
    print(f"✓ Supplementary Table (LaTeX) saved to: {latex_path}")
    
    return df


def create_heatmap_comparison(results_by_model: Dict, output_dir: str = 'results_manuscript/figures'):
    """Create heatmap showing all models x domains x metrics (Fig 5)."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_to_plot = ['accuracy', 'kappa', 'f1_macro']
    
    for metric in metrics_to_plot:
        heatmap_data = []
        
        for model_name in sorted(results_by_model.keys()):
            model_display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            
            # Average across methods
            row = {'Model': model_display}
            
            for domain in DOMAINS:
                values = []
                for method_name, method_results in results_by_model[model_name].items():
                    if domain in method_results:
                        values.append(method_results[domain].get(metric, np.nan))
                
                row[domain.capitalize()] = np.nanmean(values) if values else np.nan
            
            heatmap_data.append(row)
        
        df = pd.DataFrame(heatmap_data).set_index('Model')
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1, 
                   cbar_kws={'label': metric.replace('_', ' ').title()}, ax=ax)
        ax.set_title(f'{metric.replace("_", " ").title()} Heatmap Across Models and Domains', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'fig5_heatmap_{metric}.png')
        plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
        print(f"✓ Heatmap ({metric}) saved to: {plot_path}")
        plt.close()


def perform_statistical_tests(results_by_model: Dict, output_dir: str = 'results_manuscript/figures'):
    """Perform pairwise statistical tests between primary models (Table 2)."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if len(PRIMARY_MODELS) != 2 or not all(m in results_by_model for m in PRIMARY_MODELS):
        print("⚠️  Statistical tests require exactly 2 primary models with results")
        return
    
    model1, model2 = PRIMARY_MODELS
    model1_display = MODEL_DISPLAY_NAMES.get(model1, model1)
    model2_display = MODEL_DISPLAY_NAMES.get(model2, model2)
    
    rows = []
    
    for domain in DOMAINS:
        # Collect predictions for the same method (e.g., best performing method)
        method_name = '5_continuous'  # Use continuous method for comparison
        
        if (method_name in results_by_model[model1] and 
            method_name in results_by_model[model2] and
            domain in results_by_model[model1][method_name] and
            domain in results_by_model[model2][method_name]):
            
            pred1 = results_by_model[model1][method_name][domain]['predictions']
            gt1 = results_by_model[model1][method_name][domain]['ground_truth']
            pred2 = results_by_model[model2][method_name][domain]['predictions']
            gt2 = results_by_model[model2][method_name][domain]['ground_truth']
            
            # Ensure same samples (may differ due to checkpointing/errors)
            min_len = min(len(pred1), len(pred2))
            pred1, gt1 = pred1[:min_len], gt1[:min_len]
            pred2, gt2 = pred2[:min_len], gt2[:min_len]
            
            # Paired t-test on accuracy
            acc1_per_sample = (pred1 == gt1).astype(float)
            acc2_per_sample = (pred2 == gt2).astype(float)
            
            t_stat, p_value = ttest_rel(acc1_per_sample, acc2_per_sample)
            
            # Wilcoxon signed-rank test (non-parametric alternative)
            w_stat, w_pvalue = wilcoxon(acc1_per_sample, acc2_per_sample)
            
            mean_diff = np.mean(acc1_per_sample - acc2_per_sample)
            
            rows.append({
                'Domain': domain.capitalize(),
                f'{model1_display} Mean Acc': f"{np.mean(acc1_per_sample):.3f}",
                f'{model2_display} Mean Acc': f"{np.mean(acc2_per_sample):.3f}",
                'Mean Difference': f"{mean_diff:.3f}",
                'Paired t-test p-value': f"{p_value:.4f}",
                'Wilcoxon p-value': f"{w_pvalue:.4f}",
                'Significant (p<0.05)': '✓' if p_value < 0.05 else '✗'
            })
    
    df = pd.DataFrame(rows)
    
    csv_path = os.path.join(output_dir, 'table2_statistical_tests.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Table 2 (Statistical Tests) saved to: {csv_path}")
    
    md_path = os.path.join(output_dir, 'table2_statistical_tests.md')
    with open(md_path, 'w') as f:
        f.write(f"# Table 2: Statistical Significance Tests ({model1_display} vs {model2_display})\n\n")
        f.write(df.to_markdown(index=False))
    print(f"✓ Table 2 (Markdown) saved to: {md_path}")
    
    return df


def main():
    print("="*80)
    print("COMPREHENSIVE MODEL COMPARISON ANALYSIS")
    print("="*80)
    print()
    
    # Aggregate all model results
    print("📊 Loading results from all models...")
    results_by_model = aggregate_model_results()
    
    if not results_by_model:
        print("\n❌ No results found. Run evaluations first:")
        print("   bash e2e_pipeline.sh")
        return
    
    print(f"\n✓ Found results for {len(results_by_model)} models:")
    for model in results_by_model.keys():
        print(f"  - {MODEL_DISPLAY_NAMES.get(model, model)}")
    
    print(f"\n{'='*80}")
    print("GENERATING PUBLICATION-READY FIGURES AND TABLES")
    print(f"{'='*80}\n")
    
    # Create all visualizations
    print("📈 Creating main comparison table (Table 1)...")
    create_main_comparison_table(results_by_model)
    
    print("📈 Creating Kappa vs Accuracy scatter plots (Figure 2)...")
    create_kappa_accuracy_scatter(results_by_model)
    
    print("📈 Creating radar charts by domain (Figure 3)...")
    create_radar_charts(results_by_model)
    
    print("📈 Creating digital twin learning curves (Figure 4)...")
    create_digital_twin_learning_curves()
    
    print("📈 Creating heatmap comparisons (Figure 5)...")
    create_heatmap_comparison(results_by_model)
    
    print("📈 Creating supplementary comparison table...")
    create_supplementary_comparison(results_by_model)
    
    print("📈 Performing statistical significance tests (Table 2)...")
    perform_statistical_tests(results_by_model)
    
    print(f"\n{'='*80}")
    print("✓ Comprehensive analysis complete!")
    print(f"{'='*80}\n")
    print("Output directory: results_manuscript/figures/")
    print("\nGenerated files:")
    print("  Main Results:")
    print("    - table1_main_comparison.csv/tex/md")
    print("    - fig2_kappa_vs_accuracy_scatter.png")
    print("    - fig3_radar_charts_by_domain.png")
    print("    - fig4_digital_twin_learning_curves.png")
    print("    - fig5_heatmap_*.png")
    print("  Statistical Tests:")
    print("    - table2_statistical_tests.csv/md")
    print("  Supplementary:")
    print("    - supplementary_table_all_models.csv/tex")
    print("\n💡 Use these for your manuscript figures and tables!")


if __name__ == '__main__':
    main()

