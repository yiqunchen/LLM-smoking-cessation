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
from sklearn.metrics import cohen_kappa_score, accuracy_score
from scipy.stats import spearmanr
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
    """Calculate accuracy, Cohen's Kappa, and Spearman's Rho for each domain."""
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
        
        # Accuracy
        acc = accuracy_score(gt, pred)
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(gt, pred)
        
        # Per-participant Spearman's Rho
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
            'kappa': kappa,
            'spearman_rho': avg_rho,
            'n_samples': len(gt),
            'n_participants': len(participant_rhos) if participant_rhos else 0
        }
    
    return metrics


def analyze_all_results(results_dir: str = 'results_manuscript') -> pd.DataFrame:
    """Analyze all result files and create comparison table."""
    
    all_metrics = []
    
    # Method categories from manuscript
    method_configs = {
        # 2.2.2 Generic LLM Models
        'generic_llm_1_zero_shot_all.json': {
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
        'generic_llm_3_few_shot_all.json': {
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
        'generic_llm_5_continuous_all.json': {
            'category': '2.2.2 Generic LLM',
            'method': '5. Continuous + all features',
            'split_type': 'participant',
            'model': 'gpt-4o-mini'
        },
        'generic_llm_6_continuous_select.json': {
            'category': '2.2.2 Generic LLM',
            'method': '6. Continuous + selected features',
            'split_type': 'participant',
            'model': 'gpt-4o-mini'
        },
        # 2.2.4 Digital Twin Models
        'digital_twin_1_full_5050.json': {
            'category': '2.2.4 Digital Twin',
            'method': '1. Full-feature (50/50)',
            'split_type': 'digital_twin',
            'model': 'gpt-4o'
        },
        'digital_twin_2_select_5050.json': {
            'category': '2.2.4 Digital Twin',
            'method': '2. Selected-feature (50/50)',
            'split_type': 'digital_twin',
            'model': 'gpt-4o'
        },
        'digital_twin_3_feedback_5050.json': {
            'category': '2.2.4 Digital Twin',
            'method': '3. Full + feedback (50/50)',
            'split_type': 'digital_twin',
            'model': 'gpt-4o'
        },
        'digital_twin_4a_cbtact_5050.json': {
            'category': '2.2.4 Digital Twin',
            'method': '4a. CBT/ACT (50/50)',
            'split_type': 'digital_twin',
            'model': 'gpt-4o'
        },
        'digital_twin_4b_cbtact_7030.json': {
            'category': '2.2.4 Digital Twin',
            'method': '4b. CBT/ACT (70/30)',
            'split_type': 'digital_twin',
            'model': 'gpt-4o'
        },
        'digital_twin_4c_cbtact_9010.json': {
            'category': '2.2.4 Digital Twin',
            'method': '4c. CBT/ACT (90/10)',
            'split_type': 'digital_twin',
            'model': 'gpt-4o'
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
                'Cohen\'s Kappa': domain_metrics['kappa'],
                'Avg Spearman\'s ρ': domain_metrics['spearman_rho'],
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
    
    # Pivot to create a cleaner format
    summary = results_df.pivot_table(
        index=['Category', 'Method', 'Model'],
        columns='Domain',
        values=['Accuracy', 'Cohen\'s Kappa', 'Avg Spearman\'s ρ'],
        aggfunc='first'
    )
    
    summary.to_csv(output_path)
    print(f"\n✓ Summary table saved to: {output_path}")
    
    # Also create a markdown version
    md_path = output_path.replace('.csv', '.md')
    with open(md_path, 'w') as f:
        f.write("# Manuscript Results Summary\n\n")
        f.write(results_df.to_markdown(index=False))
    print(f"✓ Markdown table saved to: {md_path}")
    
    return summary


def plot_comparison(results_df: pd.DataFrame, output_dir: str = 'results_manuscript'):
    """Create comparison visualizations."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ['Accuracy', 'Cohen\'s Kappa', 'Avg Spearman\'s ρ']
    
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
        ax.set_title(f'{metric} by Method', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric)
        ax.set_xlabel('')
        ax.legend(title='Domain', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        
        # Add horizontal line at baseline
        if metric == 'Accuracy':
            ax.axhline(y=0.2, color='r', linestyle='--', alpha=0.5, label='Random (20%)')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'method_comparison.png')
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
    print(results_df.groupby(['Category', 'Domain'])[['Accuracy', 'Cohen\'s Kappa', 'Avg Spearman\'s ρ']].agg(['mean', 'std']))
    
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

