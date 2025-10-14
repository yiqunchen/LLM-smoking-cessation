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
from sklearn.metrics import cohen_kappa_score, accuracy_score
from scipy.stats import spearmanr, kendalltau
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-paper')
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

# Model configurations with Okabe-Ito colorblind-friendly palette (SAME AS PUBLICATION FIGURES)
MODEL_CONFIGS = {
    'gpt-4o-mini': {'dir': 'results_manuscript_gpt-4o-mini', 'display': 'GPT-4o-mini', 'color': '#0173B2'},  # Blue
    'gpt-5': {'dir': 'results_manuscript_gpt-5', 'display': 'GPT-5', 'color': '#DE8F05'},  # Orange
    'deepseek_deepseek-r1-0528': {'dir': 'results_manuscript_deepseek_deepseek-r1-0528', 'display': 'DeepSeek-R1', 'color': '#029E73'},  # Green
    'x-ai_grok-4-fast': {'dir': 'results_manuscript_x-ai_grok-4-fast', 'display': 'Grok-4-Fast', 'color': '#CC78BC'},  # Purple
    'gemini-2.5-pro': {'dir': 'results_manuscript_gemini-2.5-pro', 'display': 'Gemini-2.5-Pro', 'color': '#CA9161'}  # Brown
}

# Method configurations
METHOD_CONFIGS = {
    'generic_llm_1_zero_shot.json': {'display': 'Zero-shot (all)', 'category': 'Generic LLM'},
    'generic_llm_2_zero_shot_select.json': {'display': 'Zero-shot (select)', 'category': 'Generic LLM'},
    'generic_llm_3_few_shot.json': {'display': 'Few-shot (all)', 'category': 'Generic LLM'},
    'generic_llm_4_few_shot_select.json': {'display': 'Few-shot (select)', 'category': 'Generic LLM'},
    'generic_llm_5_continuous.json': {'display': 'Continuous (NL)', 'category': 'Generic LLM'},
    'digital_twin_4_cbtact_7030.json': {'display': 'Digital Twin (70/30)', 'category': 'Digital Twin'}
}


def load_results(filepath: str) -> pd.DataFrame:
    """Load evaluation results and convert to DataFrame."""
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
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
        acc_within_1 = np.mean(np.abs(gt - pred) <= 1)
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
            'accuracy_within_1': acc_within_1,
            'kappa': kappa,
            'kendall_tau': tau,
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
        
        for method_file, method_cfg in METHOD_CONFIGS.items():
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
                        'Acc±1': metrics[domain]['accuracy_within_1'],
                        'Kappa': metrics[domain]['kappa'],
                        'Kendall_Tau': metrics[domain]['kendall_tau'],
                        'Spearman_Rho': metrics[domain]['spearman_rho'],
                        'N': metrics[domain]['n_samples']
                    })
            
            print(f"✓ {model_display} - {method_cfg['display']}: Loaded ({len(df)} samples)")
    
    if not all_results:
        return None
    
    return pd.DataFrame(all_results)


def create_radar_charts_all_methods(df: pd.DataFrame, output_dir: str):
    """Create radar charts for ALL methods across ALL models."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_to_plot = ['Accuracy', 'Kappa', 'Acc±1', 'Spearman_Rho']
    metric_labels = ['Accuracy', 'Cohen\'s κ', 'Acc±1', 'Spearman\'s ρ']
    
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
            ax.set_xticklabels(metric_labels, fontsize=9)
            ax.set_ylim(0, 1)
            ax.set_title(f'{model_display}', fontsize=12, fontweight='bold', pad=15)
            ax.grid(True)
            
            # Only show legend on first subplot
            if idx == 0:
                ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15), fontsize=8)
        
        # Hide extra subplot
        if n_models < len(axes):
            axes[-1].set_visible(False)
        
        plt.suptitle(f'Performance Radar Charts - {domain} Domain\n(All Models × All Methods)', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        plot_path = os.path.join(output_dir, f'radar_all_methods_{domain.lower()}.png')
        plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
        print(f"✓ Radar chart saved: {plot_path}")
        plt.close()


def create_heatmap_all_methods(df: pd.DataFrame, output_dir: str):
    """Create heatmaps showing all models × all methods."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ['Accuracy', 'Kappa', 'Acc±1']
    
    for metric in metrics:
        fig, axes = plt.subplots(1, 3, figsize=(22, 8))
        
        for idx, domain in enumerate(['Content', 'Coping', 'Quitting']):
            ax = axes[idx]
            
            # Pivot: rows=models, columns=methods
            domain_data = df[df['Domain'] == domain]
            pivot = domain_data.pivot_table(
                index='Model',
                columns='Method',
                values=metric,
                aggfunc='first'
            )
            
            # Reorder columns to group by category
            col_order = [m['display'] for m in METHOD_CONFIGS.values()]
            pivot = pivot[[c for c in col_order if c in pivot.columns]]
            
            # Plot
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', 
                       vmin=0, vmax=1 if metric != 'Kappa' else 0.5,
                       cbar_kws={'label': metric}, ax=ax, linewidths=1.5, 
                       linecolor='white', annot_kws={'fontsize': 10, 'fontweight': 'bold'})
            
            ax.set_title(f'{domain} Domain', fontsize=15, fontweight='bold', pad=15)
            ax.set_xlabel('', fontsize=1)  # Remove xlabel, methods are clear from ticks
            ax.set_ylabel('', fontsize=1)   # Remove ylabel, clean look
            
            # Better tick labels
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11, fontweight='normal')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12, fontweight='normal')
        
        plt.suptitle(f'{metric} Across All Models and Methods', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f'heatmap_all_methods_{metric.lower().replace("±", "_within_")}.png')
        plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
        print(f"✓ Heatmap saved: {plot_path}")
        plt.close()


def create_grouped_bar_charts(df: pd.DataFrame, output_dir: str):
    """Create grouped bar charts comparing all models and methods."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # One figure per metric
    metrics = ['Accuracy', 'Acc±1', 'Kappa']
    
    for metric in metrics:
        fig, axes = plt.subplots(1, 3, figsize=(22, 7))
        
        for idx, domain in enumerate(['Content', 'Coping', 'Quitting']):
            ax = axes[idx]
            
            domain_data = df[df['Domain'] == domain]
            
            # Pivot for plotting
            pivot = domain_data.pivot_table(
                index='Method',
                columns='Model',
                values=metric,
                aggfunc='first'
            )
            
            # Reorder rows
            row_order = [m['display'] for m in METHOD_CONFIGS.values()]
            pivot = pivot.reindex([r for r in row_order if r in pivot.index])
            
            # Create color map for models using Okabe-Ito colors
            model_color_map = {cfg['display']: cfg['color'] for cfg in MODEL_CONFIGS.values()}
            colors = [model_color_map.get(col, '#999999') for col in pivot.columns]
            
            # Plot with consistent colors
            pivot.plot(kind='bar', ax=ax, width=0.8, rot=45, color=colors, edgecolor='black', linewidth=1.2)
            
            ax.set_title(f'{domain} Domain', fontsize=15, fontweight='bold', pad=15)
            ax.set_xlabel('', fontsize=1)  # Remove xlabel, methods clear from ticks
            ax.set_ylabel(metric, fontsize=13, fontweight='bold')
            ax.legend(title='Model', fontsize=10, title_fontsize=11, loc='upper right')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=10, fontweight='normal')
            
            # Add value labels on bars (smaller, less cluttered)
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', fontsize=6, padding=1)
        
        plt.suptitle(f'{metric} Comparison: All Models × All Methods', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f'bars_all_methods_{metric.lower().replace("±", "_within_")}.png')
        plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
        print(f"✓ Bar chart saved: {plot_path}")
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
        
        ax.set_xlabel('Accuracy', fontsize=11, fontweight='bold')
        ax.set_ylabel('Cohen\'s Kappa', fontsize=11, fontweight='bold')
        ax.set_title(f'{method_display}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.6)
        ax.set_ylim(-0.1, 0.2)
    
    plt.suptitle('Kappa vs Accuracy: All Methods (All Models, All Domains)', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_path = os.path.join(output_dir, 'scatter_kappa_vs_accuracy_all_methods.png')
    plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Scatter plot saved: {plot_path}")
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
        f.write(df.to_markdown(index=False))
    print(f"✓ Markdown table saved: {md_path}")
    
    # Create summary statistics
    summary_stats = df.groupby(['Model', 'Method', 'Domain'])[
        ['Accuracy', 'Acc±1', 'Kappa']
    ].agg(['mean', 'std'])
    
    summary_path = os.path.join(output_dir, 'summary_statistics.csv')
    summary_stats.to_csv(summary_path)
    print(f"✓ Summary statistics saved: {summary_path}")


def main():
    print("="*80)
    print("COMPREHENSIVE CROSS-MODEL ANALYSIS")
    print("Analyzing ALL models × ALL methods (including Digital Twin 70/30)")
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
    
    # Create all visualizations
    print("📈 Creating radar charts (all methods, all models)...")
    create_radar_charts_all_methods(df, output_dir)
    
    print("\n📈 Creating heatmaps (all methods, all models)...")
    create_heatmap_all_methods(df, output_dir)
    
    print("\n📈 Creating grouped bar charts (all methods, all models)...")
    create_grouped_bar_charts(df, output_dir)
    
    print("\n📈 Creating Kappa vs Accuracy scatter (all methods)...")
    create_kappa_accuracy_scatter_all(df, output_dir)
    
    print("\n📈 Creating summary tables...")
    create_summary_table(df, output_dir)
    
    print(f"\n{'='*80}")
    print("✓ Comprehensive analysis complete!")
    print(f"{'='*80}\n")
    print(f"Output directory: {output_dir}/")
    print("\nGenerated files:")
    print("  Radar Charts:")
    print("    - radar_all_methods_content.png")
    print("    - radar_all_methods_coping.png")
    print("    - radar_all_methods_quitting.png")
    print("  Heatmaps:")
    print("    - heatmap_all_methods_accuracy.png")
    print("    - heatmap_all_methods_kappa.png")
    print("    - heatmap_all_methods_acc_within_1.png")
    print("  Bar Charts:")
    print("    - bars_all_methods_accuracy.png")
    print("    - bars_all_methods_kappa.png")
    print("    - bars_all_methods_acc_within_1.png")
    print("  Scatter Plots:")
    print("    - scatter_kappa_vs_accuracy_all_methods.png")
    print("  Tables:")
    print("    - comprehensive_results_all_methods.csv")
    print("    - comprehensive_results_all_methods.md")
    print("    - summary_statistics.csv")
    print("\n💡 All figures include:")
    print("   • 5 models (GPT-4o-mini, GPT-5, DeepSeek-R1, Grok-4-Fast, Gemini-2.5-Pro)")
    print("   • 6 methods (Zero-shot all/select, Few-shot all/select, Continuous, Digital Twin 70/30)")
    print("   • 3 domains (Content, Coping, Quitting)")


if __name__ == '__main__':
    main()

