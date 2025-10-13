"""
Analyze results from all manuscript methods and generate comparison tables.

This script:
1. Loads results from all methods in results_manuscript/
2. Calculates accuracy, Cohen's Kappa, and Spearman's Rho for each method
3. Generates comparison tables for the manuscript
4. Creates visualizations
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    cohen_kappa_score, accuracy_score, 
    f1_score, precision_score, recall_score
)
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns

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

DOMAINS = ['content', 'coping', 'quitting']  # Excluding design per manuscript


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
        gt = df.loc[valid_mask, gt_col]
        pred = df.loc[valid_mask, pred_col]
        
        if len(gt) < 2:
            continue
        
        # 1. Accuracy (exact match)
        acc = accuracy_score(gt, pred)
        
        # 2. Accuracy within 1 (allow ±1 error)
        acc_within_1 = np.mean(np.abs(gt - pred) <= 1)
        
        # 3. Cohen's Kappa
        kappa = cohen_kappa_score(gt, pred)
        
        # 4. Kendall's Tau
        tau, _ = kendalltau(gt, pred)
        
        # 5. Macro-weighted F1, Precision, Recall
        f1_macro = f1_score(gt, pred, average='macro', zero_division=0)
        precision_macro = precision_score(gt, pred, average='macro', zero_division=0)
        recall_macro = recall_score(gt, pred, average='macro', zero_division=0)
        
        # 6. Per-participant Spearman's Rho
        participant_rhos = []
        if 'response_id' in df.columns:
            for participant_id, group in df.groupby('response_id'):
                if len(group) < 2:
                    continue
                group_gt = group[gt_col].dropna()
                group_pred = group[pred_col].dropna()
                if len(group_gt) > 1 and group_gt.nunique() > 1 and group_pred.nunique() > 1:
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
            'f1_macro': f1_macro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'n_samples': len(gt),
            'n_participants': len(participant_rhos) if participant_rhos else 0
        }
    
    return metrics


def analyze_all_results(results_dir: str = 'results_manuscript') -> pd.DataFrame:
    """Analyze all result files and create comparison table."""
    
    all_metrics = []
    
    # Method categories from manuscript - CORRECTED FILENAMES
    method_configs = {
        # 2.2.2 Generic LLM Models
        'generic_llm_1_zero_shot.json': {
            'category': '2.2.2 Generic LLM',
            'method': '1. Zero-shot + all features',
            'split_type': 'participant',
            'model': 'gpt-4o-mini'
        },
        'generic_llm_2_zero_shot_select.json': {
            'category': '2.2.2 Generic LLM',
            'method': '2. Zero-shot + selected features',
            'split_type': 'participant',
            'model': 'gpt-4o-mini'
        },
        'generic_llm_3_few_shot.json': {
            'category': '2.2.2 Generic LLM',
            'method': '3. Few-shot + all features',
            'split_type': 'participant',
            'model': 'gpt-4o-mini'
        },
        'generic_llm_4_few_shot_select.json': {
            'category': '2.2.2 Generic LLM',
            'method': '4. Few-shot + selected features',
            'split_type': 'participant',
            'model': 'gpt-4o-mini'
        },
        'generic_llm_5_continuous.json': {
            'category': '2.2.2 Generic LLM',
            'method': '5. Continuous + natural language',
            'split_type': 'participant',
            'model': 'gpt-4o-mini'
        },
        'generic_llm_6_continuous_select.json': {
            'category': '2.2.2 Generic LLM',
            'method': '6. Continuous + probabilities',
            'split_type': 'participant',
            'model': 'gpt-4o-mini'
        },
        # 2.2.4 Digital Twin Models
        'digital_twin_1_full_7030.json': {
            'category': '2.2.4 Digital Twin',
            'method': '1. Full-feature (70/30)',
            'split_type': 'digital_twin',
            'model': 'gpt-4o-mini'
        },
        'digital_twin_2_select_7030.json': {
            'category': '2.2.4 Digital Twin',
            'method': '2. Selected-feature (70/30)',
            'split_type': 'digital_twin',
            'model': 'gpt-4o-mini'
        },
        'digital_twin_3_feedback_7030.json': {
            'category': '2.2.4 Digital Twin',
            'method': '3. Full + feedback (70/30)',
            'split_type': 'digital_twin',
            'model': 'gpt-4o-mini'
        },
        'digital_twin_4_cbtact_1090.json': {
            'category': '2.2.4 Digital Twin',
            'method': '4. CBT/ACT (10/90)',
            'split_type': 'digital_twin',
            'model': 'gpt-4o-mini'
        },
        'digital_twin_4_cbtact_3070.json': {
            'category': '2.2.4 Digital Twin',
            'method': '4. CBT/ACT (30/70)',
            'split_type': 'digital_twin',
            'model': 'gpt-4o-mini'
        },
        'digital_twin_4_cbtact_7030.json': {
            'category': '2.2.4 Digital Twin',
            'method': '4. CBT/ACT (70/30)',
            'split_type': 'digital_twin',
            'model': 'gpt-4o-mini'
        },
        'digital_twin_4_cbtact_9010.json': {
            'category': '2.2.4 Digital Twin',
            'method': '4. CBT/ACT (90/10)',
            'split_type': 'digital_twin',
            'model': 'gpt-4o-mini'
        },
    }
    
    for filename, config in method_configs.items():
        filepath = os.path.join(results_dir, filename)
        df = load_results(filepath)
        
        if df is None:
            print(f"⚠️  {filename}: NOT FOUND")
            continue
        
        metrics = calculate_metrics(df, DOMAINS)
        
        if not metrics:
            print(f"⚠️  {filename}: No valid metrics")
            continue
        
        # Add row for each domain
        for domain, domain_metrics in metrics.items():
            row = {
                'Category': config['category'],
                'Method': config['method'],
                'Model': config['model'],
                'Domain': domain.capitalize(),
                'Accuracy': domain_metrics['accuracy'],
                'Acc±1': domain_metrics['accuracy_within_1'],
                'Cohen\'s κ': domain_metrics['kappa'],
                'Kendall\'s τ': domain_metrics['kendall_tau'],
                'Spearman\'s ρ': domain_metrics['spearman_rho'],
                'F1 (macro)': domain_metrics['f1_macro'],
                'Precision (macro)': domain_metrics['precision_macro'],
                'Recall (macro)': domain_metrics['recall_macro'],
                'N': domain_metrics['n_samples']
            }
            all_metrics.append(row)
        
        print(f"✓ {filename}: Loaded ({len(df)} samples)")
    
    if not all_metrics:
        print("No results found!")
        return None
    
    results_df = pd.DataFrame(all_metrics)
    return results_df


def create_summary_table(results_df: pd.DataFrame, output_path: str = 'results_manuscript/summary_table.csv'):
    """Create and save summary table."""
    
    # Save full table
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Full results table saved to: {output_path}")
    
    # Create markdown version
    md_path = output_path.replace('.csv', '.md')
    with open(md_path, 'w') as f:
        f.write("# Manuscript Results Summary\n\n")
        f.write("## All Metrics\n\n")
        f.write(results_df.to_markdown(index=False))
    print(f"✓ Markdown table saved to: {md_path}")
    
    # Create a pivot summary for key metrics
    pivot_metrics = ['Accuracy', 'Acc±1', 'Cohen\'s κ', 'Kendall\'s τ', 'Spearman\'s ρ', 'F1 (macro)']
    summary = results_df.pivot_table(
        index=['Category', 'Method'],
        columns='Domain',
        values=pivot_metrics,
        aggfunc='first'
    )
    
    summary_path = output_path.replace('.csv', '_pivot.csv')
    summary.to_csv(summary_path)
    print(f"✓ Pivot summary saved to: {summary_path}")
    
    return summary


def plot_comparison(results_df: pd.DataFrame, output_dir: str = 'results_manuscript'):
    """Create comparison visualizations."""
    
    # Plot 1: Key metrics comparison
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    metrics = ['Accuracy', 'Acc±1', 'Cohen\'s κ', 'Kendall\'s τ', 'Spearman\'s ρ', 'F1 (macro)']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Create grouped bar plot
        data_pivot = results_df.pivot_table(
            index='Method',
            columns='Domain',
            values=metric,
            aggfunc='first'
        )
        
        data_pivot.plot(kind='bar', ax=ax, rot=45, width=0.8)
        ax.set_title(f'{metric} by Method', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric)
        ax.set_xlabel('')
        ax.legend(title='Domain', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        
        # Add horizontal line at baseline for accuracy metrics
        if metric in ['Accuracy', 'Acc±1']:
            ax.axhline(y=0.2, color='r', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'method_comparison_all_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {plot_path}")
    plt.close()


def main():
    print("="*80)
    print("MANUSCRIPT RESULTS ANALYSIS")
    print("="*80)
    print()
    
    # Analyze all results
    results_df = analyze_all_results()
    
    if results_df is None or len(results_df) == 0:
        print("\n❌ No results to analyze. Run evaluations first:")
        print("   bash run_manuscript_evaluations.sh")
        return
    
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}\n")
    
    # Print summary statistics
    print(results_df.groupby(['Category', 'Domain'])[
        ['Accuracy', 'Acc±1', 'Cohen\'s κ', 'Kendall\'s τ', 'Spearman\'s ρ', 'F1 (macro)']
    ].agg(['mean', 'std']))
    
    # Create summary table
    summary = create_summary_table(results_df)
    
    # Create visualizations
    plot_comparison(results_df)
    
    print(f"\n{'='*80}")
    print("✓ Analysis complete!")
    print(f"{'='*80}\n")
    print("Output files:")
    print("  - results_manuscript/summary_table.csv")
    print("  - results_manuscript/summary_table.md")
    print("  - results_manuscript/method_comparison.png")


if __name__ == '__main__':
    main()

