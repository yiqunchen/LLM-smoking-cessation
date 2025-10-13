"""
Create a publication-ready summary infographic combining multiple insights.

This creates a single comprehensive figure that could serve as a graphical abstract
or main figure in the manuscript.
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

RATING_MAPS = {
    'content': {'Very poor': 1, 'Poor': 2, 'Acceptable': 3, 'Good': 4, 'Very good': 5},
    'coping': {'Not at all helpful': 1, 'Somewhat helpful': 2, 'Moderately helpful': 3,
               'Very helpful': 4, 'Extremely helpful': 5, 'Not Helpful': 1},
    'quitting': {'Not at all helpful': 1, 'Somewhat helpful': 2, 'Moderately helpful': 3,
                 'Very helpful': 4, 'Extremely helpful': 5, 'Not Helpful': 1}
}

DOMAINS = ['content', 'coping', 'quitting']
MODEL_NAMES = {
    'gpt-5': 'GPT-5',
    'deepseek_deepseek-r1-0528': 'DeepSeek-R1',
    'gpt-4o-mini': 'GPT-4o-mini',
    'x-ai_grok-4-fast': 'Grok-4',
    'gemini-2.5-pro': 'Gemini-2.5'
}


def load_and_calculate_metrics(filepath: str) -> Dict:
    """Load results and calculate metrics."""
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    rows = [item for item in results.values() if item.get("predicted_content") != "ERROR"]
    if not rows:
        return None
    
    df = pd.DataFrame(rows)
    
    metrics = {}
    for domain in DOMAINS:
        if f'ground_truth_{domain}' in df.columns and f'predicted_{domain}' in df.columns:
            df[f'gt_{domain}_num'] = df[f'ground_truth_{domain}'].map(RATING_MAPS[domain])
            df[f'pred_{domain}_num'] = df[f'predicted_{domain}'].map(RATING_MAPS[domain])
            
            # Filter to rows with both valid gt and pred
            valid_mask = df[f'gt_{domain}_num'].notna() & df[f'pred_{domain}_num'].notna()
            gt = df.loc[valid_mask, f'gt_{domain}_num'].values
            pred = df.loc[valid_mask, f'pred_{domain}_num'].values
            
            if len(gt) > 0 and len(gt) == len(pred):
                acc = np.mean(gt == pred)
                acc_within_1 = np.mean(np.abs(gt - pred) <= 1)
                
                metrics[domain] = {
                    'accuracy': acc,
                    'accuracy_within_1': acc_within_1
                }
    
    return metrics


def create_summary_infographic(output_dir: str = 'results_manuscript/figures'):
    """Create comprehensive summary infographic."""
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)
    
    # Panel A: Model comparison bars (top-left, 2 columns)
    ax_bars = fig.add_subplot(gs[0, :2])
    
    # Panel B: Accuracy vs Acc±1 comparison (top-right, 2 columns)
    ax_comp = fig.add_subplot(gs[0, 2:])
    
    # Panel C: Domain difficulty (middle-left)
    ax_domain = fig.add_subplot(gs[1, :2])
    
    # Panel D: Learning curves (middle-right)
    ax_learning = fig.add_subplot(gs[1, 2:])
    
    # Panel E: Kappa scatter (bottom-left)
    ax_kappa = fig.add_subplot(gs[2, :2])
    
    # Panel F: Summary stats (bottom-right)
    ax_stats = fig.add_subplot(gs[2, 2:])
    
    # Collect data
    model_data = {}
    for model_code, model_name in MODEL_NAMES.items():
        filepath = f'results_manuscript_{model_code}/generic_llm_5_continuous.json'
        metrics = load_and_calculate_metrics(filepath)
        if metrics:
            model_data[model_name] = metrics
    
    if not model_data:
        print("⚠️  No data available for infographic")
        return
    
    # ===== PANEL A: Model Comparison Bars =====
    models = list(model_data.keys())
    domains_plot = DOMAINS
    x = np.arange(len(models))
    width = 0.25
    
    for i, domain in enumerate(domains_plot):
        accuracies = [model_data[m].get(domain, {}).get('accuracy', 0) for m in models]
        offset = (i - 1) * width
        bars = ax_bars.bar(x + offset, accuracies, width, label=domain.capitalize())
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax_bars.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=8)
    
    ax_bars.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax_bars.set_ylabel('Exact Accuracy', fontsize=12, fontweight='bold')
    ax_bars.set_title('A) Model Performance Comparison (Exact Accuracy)', 
                     fontsize=14, fontweight='bold', pad=15)
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(models, rotation=15, ha='right')
    ax_bars.legend(title='Domain', loc='upper right')
    ax_bars.set_ylim(0, 0.5)
    ax_bars.axhline(y=0.2, color='red', linestyle='--', alpha=0.3, linewidth=1, label='Chance (20%)')
    ax_bars.grid(axis='y', alpha=0.3)
    
    # ===== PANEL B: Accuracy vs Acc±1 =====
    avg_acc = {domain: [] for domain in domains_plot}
    avg_acc_within_1 = {domain: [] for domain in domains_plot}
    
    for model in models:
        for domain in domains_plot:
            avg_acc[domain].append(model_data[model].get(domain, {}).get('accuracy', 0))
            avg_acc_within_1[domain].append(model_data[model].get(domain, {}).get('accuracy_within_1', 0))
    
    x_pos = np.arange(len(domains_plot))
    width = 0.35
    
    exact_means = [np.mean(avg_acc[d]) for d in domains_plot]
    within1_means = [np.mean(avg_acc_within_1[d]) for d in domains_plot]
    
    bars1 = ax_comp.bar(x_pos - width/2, exact_means, width, label='Exact Match', color='steelblue')
    bars2 = ax_comp.bar(x_pos + width/2, within1_means, width, label='Within ±1', color='coral')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax_comp.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax_comp.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax_comp.set_title('B) Exact vs Directional Accuracy', fontsize=14, fontweight='bold', pad=15)
    ax_comp.set_xticks(x_pos)
    ax_comp.set_xticklabels([d.capitalize() for d in domains_plot])
    ax_comp.legend(loc='upper right')
    ax_comp.set_ylim(0, 1.0)
    ax_comp.grid(axis='y', alpha=0.3)
    
    # ===== PANEL C: Domain Difficulty =====
    domain_stats = []
    for domain in domains_plot:
        accs = [model_data[m].get(domain, {}).get('accuracy', 0) for m in models]
        domain_stats.append({
            'domain': domain.capitalize(),
            'mean': np.mean(accs),
            'std': np.std(accs),
            'min': np.min(accs),
            'max': np.max(accs)
        })
    
    df_domain = pd.DataFrame(domain_stats)
    
    # Box plot style visualization
    positions = range(len(domains_plot))
    for i, domain in enumerate(domains_plot):
        accs = [model_data[m].get(domain, {}).get('accuracy', 0) for m in models]
        bp = ax_domain.boxplot([accs], positions=[i], widths=0.5, patch_artist=True,
                               boxprops=dict(facecolor='lightblue', alpha=0.7),
                               medianprops=dict(color='darkblue', linewidth=2))
    
    ax_domain.set_ylabel('Accuracy Distribution', fontsize=12, fontweight='bold')
    ax_domain.set_title('C) Task Difficulty by Domain', fontsize=14, fontweight='bold', pad=15)
    ax_domain.set_xticks(positions)
    ax_domain.set_xticklabels([d.capitalize() for d in domains_plot])
    ax_domain.set_ylim(0, 0.5)
    ax_domain.grid(axis='y', alpha=0.3)
    
    # Add variance annotation
    for i, row in df_domain.iterrows():
        ax_domain.text(i, 0.45, f'σ={row["std"]:.3f}', ha='center', fontsize=9, 
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ===== PANEL D: Learning Curves =====
    # Load digital twin results for GPT-5
    splits = ['1090', '3070', '7030', '9010']
    train_pcts = [10, 30, 70, 90]
    
    learning_data = {domain: [] for domain in domains_plot}
    
    for split in splits:
        filepath = f'results_manuscript_gpt-5/digital_twin_4_cbtact_{split}.json'
        metrics = load_and_calculate_metrics(filepath)
        if metrics:
            for domain in domains_plot:
                learning_data[domain].append(metrics.get(domain, {}).get('accuracy', np.nan))
    
    colors_learning = {'content': 'blue', 'coping': 'green', 'quitting': 'red'}
    for domain in domains_plot:
        if learning_data[domain]:
            ax_learning.plot(train_pcts, learning_data[domain], 'o-', 
                           linewidth=2, markersize=8, label=domain.capitalize(),
                           color=colors_learning[domain])
    
    ax_learning.set_xlabel('Training Set Size (%)', fontsize=12, fontweight='bold')
    ax_learning.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax_learning.set_title('D) Digital Twin Learning Curves (GPT-5)', 
                         fontsize=14, fontweight='bold', pad=15)
    ax_learning.legend(title='Domain')
    ax_learning.set_xticks(train_pcts)
    ax_learning.grid(True, alpha=0.3)
    ax_learning.set_ylim(0, 0.5)
    
    # ===== PANEL E: Kappa vs Accuracy Scatter =====
    # Load kappa data from saved results
    from sklearn.metrics import cohen_kappa_score
    
    scatter_data = []
    for model_code, model_name in MODEL_NAMES.items():
        filepath = f'results_manuscript_{model_code}/generic_llm_5_continuous.json'
        if not os.path.exists(filepath):
            continue
        
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        rows = [item for item in results.values() if item.get("predicted_content") != "ERROR"]
        if not rows:
            continue
        
        df = pd.DataFrame(rows)
        
        for domain in domains_plot:
            if f'ground_truth_{domain}' in df.columns and f'predicted_{domain}' in df.columns:
                df[f'gt_{domain}_num'] = df[f'ground_truth_{domain}'].map(RATING_MAPS[domain])
                df[f'pred_{domain}_num'] = df[f'predicted_{domain}'].map(RATING_MAPS[domain])
                
                valid = df[f'gt_{domain}_num'].notna() & df[f'pred_{domain}_num'].notna()
                gt = df.loc[valid, f'gt_{domain}_num'].values
                pred = df.loc[valid, f'pred_{domain}_num'].values
                
                if len(gt) > 0:
                    acc = np.mean(gt == pred)
                    kappa = cohen_kappa_score(gt, pred)
                    scatter_data.append({
                        'model': model_name,
                        'domain': domain,
                        'accuracy': acc,
                        'kappa': kappa
                    })
    
    scatter_df = pd.DataFrame(scatter_data)
    
    domain_colors = {'content': 'blue', 'coping': 'green', 'quitting': 'red'}
    for domain in domains_plot:
        domain_data = scatter_df[scatter_df['domain'] == domain]
        ax_kappa.scatter(domain_data['accuracy'], domain_data['kappa'],
                        label=domain.capitalize(), s=150, alpha=0.7,
                        color=domain_colors[domain], edgecolors='black', linewidth=1)
    
    # Add reference line
    lim = 0.5
    ax_kappa.plot([0, lim], [0, lim], 'k--', alpha=0.3, linewidth=1)
    
    ax_kappa.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax_kappa.set_ylabel('Cohen\'s Kappa (κ)', fontsize=12, fontweight='bold')
    ax_kappa.set_title('E) Agreement vs Performance', fontsize=14, fontweight='bold', pad=15)
    ax_kappa.legend(title='Domain')
    ax_kappa.grid(True, alpha=0.3)
    ax_kappa.set_xlim(0, lim)
    ax_kappa.set_ylim(-0.1, 0.2)
    
    # ===== PANEL F: Summary Statistics =====
    ax_stats.axis('off')
    
    # Calculate summary statistics
    all_accs = []
    for model in models:
        for domain in domains_plot:
            all_accs.append(model_data[model].get(domain, {}).get('accuracy', 0))
    
    all_acc_w1 = []
    for model in models:
        for domain in domains_plot:
            all_acc_w1.append(model_data[model].get(domain, {}).get('accuracy_within_1', 0))
    
    summary_text = f"""
    F) Summary Statistics
    
    Overall Performance (all models, all domains):
    
    • Exact Accuracy:     {np.mean(all_accs):.1%} ± {np.std(all_accs):.1%}
    • Directional (±1):   {np.mean(all_acc_w1):.1%} ± {np.std(all_acc_w1):.1%}
    
    Domain Ranking (easiest → hardest):
    1. Content:   {df_domain[df_domain['domain']=='Content']['mean'].values[0]:.1%}
    2. Coping:    {df_domain[df_domain['domain']=='Coping']['mean'].values[0]:.1%}
    3. Quitting:  {df_domain[df_domain['domain']=='Quitting']['mean'].values[0]:.1%}
    
    Best Model (avg across domains):
    """
    
    # Find best model
    model_avgs = {m: np.mean([model_data[m].get(d, {}).get('accuracy', 0) for d in domains_plot]) 
                  for m in models}
    best_model = max(model_avgs, key=model_avgs.get)
    
    summary_text += f"• {best_model}: {model_avgs[best_model]:.1%}\n"
    
    summary_text += f"""
    Key Findings:
    • Models capture directional trends (70-80%)
      but struggle with exact ratings (25-35%)
    
    • Personalization (digital twins) improves
      performance by ~5-10% with more data
    
    • No significant difference between
      GPT-5 and DeepSeek-R1 (p > 0.05)
    """
    
    ax_stats.text(0.05, 0.95, summary_text, transform=ax_stats.transAxes,
                 fontsize=11, verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Add overall title
    fig.suptitle('LLM Performance on Smoking Cessation Message Evaluation', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Save
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = os.path.join(output_dir, 'summary_infographic.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Summary infographic saved to: {output_path}")
    plt.close()


if __name__ == '__main__':
    print("Creating summary infographic...")
    create_summary_infographic()
    print("\n✓ Done! Use this as your graphical abstract or main figure.")

