"""
Create publication-ready figures for bootstrap and practical analysis results.
Uses the same Okabe-Ito colorblind-friendly palette as other publication figures.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Okabe-Ito colorblind-friendly palette (same as create_publication_figures.py)
COLORS = {
    'GPT-4o-mini': '#0173B2',      # Blue
    'GPT-5': '#DE8F05',            # Orange
    'DeepSeek-R1': '#029E73',      # Green
    'Grok-4-Fast': '#CC78BC',      # Purple
    'Gemini-2.5-Pro': '#CA9161',   # Brown
}

# Domain colors
DOMAIN_COLORS = {
    'Content': '#0173B2',   # Blue
    'Coping': '#DE8F05',    # Orange
    'Quitting': '#029E73',  # Green
}

MODEL_ORDER = ['GPT-4o-mini', 'GPT-5', 'DeepSeek-R1', 'Grok-4-Fast', 'Gemini-2.5-Pro']
DOMAIN_ORDER = ['Content', 'Coping', 'Quitting']

plt.style.use('seaborn-v0_8-whitegrid')
DPI = 400
PDF_DPI = 400
BOOTSTRAP_SEED = 42
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.linewidth': 1.8,
    'grid.alpha': 0.12,
    'grid.color': '#4D4D4D',
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.width': 1.8,
    'ytick.major.width': 1.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Font sizes - LARGER and BOLD
TITLE_SIZE = 16
AXIS_LABEL_SIZE = 14
TICK_SIZE = 12
LEGEND_SIZE = 11


def save_figure(fig, filepath_without_ext: str):
    """Save figure as both PNG and PDF."""
    png_path = f"{filepath_without_ext}.png"
    fig.patch.set_facecolor('white')
    for ax in fig.axes:
        ax.set_facecolor('white')
        ax.xaxis.label.set_fontweight('bold')
        ax.yaxis.label.set_fontweight('bold')
        ax.title.set_fontweight('bold')
        ax.tick_params(axis='both', width=1.8, length=6, direction='out')
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)
    fig.savefig(png_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {png_path}")

    pdf_path = f"{filepath_without_ext}.pdf"
    fig.savefig(pdf_path, dpi=PDF_DPI, bbox_inches='tight', format='pdf', facecolor='white')
    print(f"  Saved: {pdf_path}")


def plot_bootstrap_2x2_grid(df: pd.DataFrame, output_dir: str):
    """
    Create 2x2 grid showing Accuracy, Directional Accuracy, F1-macro, and Kappa
    with random chance baselines for all metrics.
    """
    print("\nCreating 2x2 bootstrap grid with chance lines...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    # Metrics to plot with their chance levels
    # 5-class: chance = 0.2 for accuracy, ~0.2 for macro-F1
    # 3-class directional: chance = 0.33
    # Kappa: chance = 0
    metrics_config = [
        ('accuracy', 'Accuracy (5-class)', 0.2),
        ('directional_accuracy', 'Directional Accuracy (3-class)', 0.333),
        ('f1_macro', 'F1-macro (5-class)', 0.2),
        ('kappa', "Cohen's Kappa", 0.0),
    ]

    for idx, (metric, title, chance) in enumerate(metrics_config):
        ax = axes[idx]
        metric_df = df[df['Metric'] == metric].copy()

        if len(metric_df) == 0:
            ax.set_visible(False)
            continue

        x = np.arange(len(DOMAIN_ORDER))
        width = 0.15
        offsets = np.linspace(-2*width, 2*width, len(MODEL_ORDER))

        for i, model in enumerate(MODEL_ORDER):
            model_data = metric_df[metric_df['Model'] == model]

            values = []
            errors_lower = []
            errors_upper = []

            for domain in DOMAIN_ORDER:
                domain_data = model_data[model_data['Domain'] == domain]
                if len(domain_data) > 0:
                    row = domain_data.iloc[0]
                    pe = row['Point Estimate']
                    ci_l = row['CI Lower (95%)']
                    ci_u = row['CI Upper (95%)']
                    values.append(pe)
                    errors_lower.append(pe - ci_l)
                    errors_upper.append(ci_u - pe)
                else:
                    values.append(0)
                    errors_lower.append(0)
                    errors_upper.append(0)

            bars = ax.bar(x + offsets[i], values, width, label=model,
                         color=COLORS[model], yerr=[errors_lower, errors_upper],
                         capsize=3, error_kw={'linewidth': 1.5})

        # Add chance line
        ax.axhline(y=chance, color='red', linestyle='--', alpha=0.8, linewidth=2,
                   label=f'Random Chance ({chance:.0%})' if chance > 0 else 'Chance (κ=0)')

        # Formatting with LARGER BOLD fonts
        ax.set_xlabel('Domain', fontsize=AXIS_LABEL_SIZE, fontweight='bold')
        ax.set_ylabel(title, fontsize=AXIS_LABEL_SIZE, fontweight='bold')
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(DOMAIN_ORDER, fontsize=TICK_SIZE, fontweight='bold')
        ax.tick_params(axis='y', labelsize=TICK_SIZE)

        # Set y-axis limits
        if metric == 'kappa':
            ax.set_ylim(-0.3, 0.4)
        else:
            ax.set_ylim(0, 1.0)

    # Single legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02),
               fontsize=LEGEND_SIZE, frameon=True)

    plt.suptitle('Bootstrap 95% CI: Message-Level Metrics (n=107 messages)',
                fontsize=TITLE_SIZE + 2, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    save_figure(fig, os.path.join(output_dir, 'bootstrap_2x2_grid'))
    plt.close()


def plot_top_k_agreement(df: pd.DataFrame, output_dir: str):
    """
    Create line plots showing Top-K agreement across different K values with CI bands.
    """
    print("\nCreating Top-K agreement plots...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, domain in enumerate(DOMAIN_ORDER):
        ax = axes[idx]
        domain_df = df[df['Domain'] == domain]

        for model in MODEL_ORDER:
            model_df = domain_df[domain_df['Model'] == model].sort_values('K')
            if len(model_df) > 0:
                k_vals = model_df['K'].values
                overlap_vals = model_df['Overlap %'].values

                ax.plot(k_vals, overlap_vals, 'o-',
                       color=COLORS[model], label=model, linewidth=2.5, markersize=9)

                # Add CI band if available
                if 'CI Lower' in model_df.columns and 'CI Upper' in model_df.columns:
                    ci_lower = (model_df['CI Lower'] / model_df['K'] * 100).values
                    ci_upper = (model_df['CI Upper'] / model_df['K'] * 100).values
                    ax.fill_between(k_vals, ci_lower, ci_upper,
                                   color=COLORS[model], alpha=0.15)

        # Add chance line
        n_messages = 107
        k_values = df['K'].unique()
        chance = [(k / n_messages) * 100 for k in sorted(k_values)]
        ax.plot(sorted(k_values), chance, '--', color='red', alpha=0.8,
               linewidth=2.5, label='Random Chance')

        ax.set_xlabel('K (Top-K Messages)', fontsize=AXIS_LABEL_SIZE, fontweight='bold')
        ax.set_ylabel('Overlap %', fontsize=AXIS_LABEL_SIZE, fontweight='bold')
        ax.set_title(f'{domain}', fontsize=TITLE_SIZE, fontweight='bold')
        ax.set_ylim(0, 70)
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        ax.grid(True, alpha=0.3)

    # Single legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.02),
               fontsize=LEGEND_SIZE, frameon=True)

    plt.suptitle('Top-K Message Agreement: LLM vs Human Rankings', fontsize=TITLE_SIZE + 2, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    save_figure(fig, os.path.join(output_dir, 'top_k_agreement_line'))
    plt.close()


def _bootstrap_summary(values):
    """Return mean, SE, and percentile CI for a bootstrap sample."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            'mean': np.nan,
            'se': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
        }
    return {
        'mean': float(np.mean(arr)),
        'se': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        'ci_lower': float(np.percentile(arr, 2.5)),
        'ci_upper': float(np.percentile(arr, 97.5)),
    }


def compute_llm_selection_quality(base_dir: str, output_dir: str,
                                  n_bootstrap: int = 2000):
    """
    Key practical question: If we use LLM to select top-K messages,
    what is the average HUMAN rating of those messages?

    This directly answers: "Do LLM-selected messages perform well with humans?"

    Uses PP (CBT/ACT 70/30) results.
    """
    print(f"\nComputing LLM selection quality (PP CBT/ACT 70/30, bootstrap={n_bootstrap})...")

    RATING_MAPS = {
        'content': {'Very poor': 1, 'Poor': 2, 'Acceptable': 3, 'Good': 4, 'Very good': 5},
        'coping': {
            'Not at all helpful': 1, 'Somewhat helpful': 2, 'Moderately helpful': 3,
            'Very helpful': 4, 'Extremely helpful': 5, 'Not Helpful': 1
        },
        'quitting': {
            'Not at all helpful': 1, 'Somewhat helpful': 2, 'Moderately helpful': 3,
            'Very helpful': 4, 'Extremely helpful': 5, 'Not Helpful': 1
        }
    }

    # Use PP CBT/ACT 70/30 results
    MODELS = {
        'GPT-4o-mini': 'results_manuscript_gpt-4o-mini',
        'GPT-5': 'results_manuscript_gpt-5',
        'DeepSeek-R1': 'results_manuscript_deepseek_deepseek-r1-0528',
        'Grok-4-Fast': 'results_manuscript_x-ai_grok-4-fast',
        'Gemini-2.5-Pro': 'results_manuscript_gemini-2.5-pro'
    }

    results = []
    k_values = [5, 10, 15, 20, 25]
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for model_name, model_dir in MODELS.items():
        # Use PP CBT/ACT 70/30
        result_path = os.path.join(base_dir, model_dir, 'digital_twin_4_cbtact_7030.json')
        if not os.path.exists(result_path):
            print(f"  Skipping {model_name}: No PP results found")
            continue

        with open(result_path, 'r') as f:
            data = json.load(f)

        df = pd.DataFrame(data.values())

        for domain in ['content', 'coping', 'quitting']:
            gt_col = f'ground_truth_{domain}'
            pred_col = f'predicted_{domain}'

            if gt_col not in df.columns or pred_col not in df.columns:
                continue

            # Convert to numeric
            df[f'gt_num'] = df[gt_col].map(RATING_MAPS[domain])
            df[f'pred_num'] = df[pred_col].map(RATING_MAPS[domain])

            valid_mask = df['gt_num'].notna() & df['pred_num'].notna()
            valid_df = df[valid_mask].copy()

            # Aggregate to message level
            msg_df = valid_df.groupby('input_message').agg({
                'gt_num': 'mean',
                'pred_num': 'mean'
            }).reset_index()

            n_messages = len(msg_df)
            gt_vals = msg_df['gt_num'].to_numpy(dtype=float)
            pred_vals = msg_df['pred_num'].to_numpy(dtype=float)

            # Human oracle: top-K by human rating
            human_sorted = msg_df.sort_values('gt_num', ascending=False).reset_index(drop=True)

            # LLM selection: top-K by LLM prediction
            llm_sorted = msg_df.sort_values('pred_num', ascending=False).reset_index(drop=True)

            valid_k_values = [k for k in k_values if k <= n_messages]

            # Point estimates from the observed message set
            point_human = {}
            point_llm = {}
            for k in valid_k_values:
                point_human[k] = float(human_sorted.head(k)['gt_num'].mean())
                point_llm[k] = float(llm_sorted.head(k)['gt_num'].mean())

            # Bootstrap distributions from message resampling
            human_boot = {k: [] for k in valid_k_values}
            llm_boot = {k: [] for k in valid_k_values}
            random_boot = {k: [] for k in valid_k_values}

            for _ in range(n_bootstrap):
                boot_idx = rng.integers(0, n_messages, size=n_messages)
                boot_gt = gt_vals[boot_idx]
                boot_pred = pred_vals[boot_idx]

                human_order = np.argsort(-boot_gt, kind='mergesort')
                llm_order = np.argsort(-boot_pred, kind='mergesort')
                random_order = rng.permutation(n_messages)

                human_cumsum = np.cumsum(boot_gt[human_order])
                llm_cumsum = np.cumsum(boot_gt[llm_order])
                random_cumsum = np.cumsum(boot_gt[random_order])

                for k in valid_k_values:
                    human_boot[k].append(human_cumsum[k - 1] / k)
                    llm_boot[k].append(llm_cumsum[k - 1] / k)
                    random_boot[k].append(random_cumsum[k - 1] / k)

            for k in valid_k_values:
                human_stats = _bootstrap_summary(human_boot[k])
                llm_stats = _bootstrap_summary(llm_boot[k])
                random_stats = _bootstrap_summary(random_boot[k])

                results.append({
                    'Model': model_name,
                    'Domain': domain.capitalize(),
                    'K': k,
                    'LLM Selection (Human Rating)': point_llm[k],
                    'LLM SE': llm_stats['se'],
                    'LLM CI Lower': llm_stats['ci_lower'],
                    'LLM CI Upper': llm_stats['ci_upper'],
                    'Human Oracle (Human Rating)': point_human[k],
                    'Human Oracle SE': human_stats['se'],
                    'Human Oracle CI Lower': human_stats['ci_lower'],
                    'Human Oracle CI Upper': human_stats['ci_upper'],
                    'Random Mean': random_stats['mean'],
                    'Random SE': random_stats['se'],
                    'Random CI Lower': random_stats['ci_lower'],
                    'Random CI Upper': random_stats['ci_upper'],
                    'Bootstrap_N': n_bootstrap,
                    'N Messages': n_messages
                })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'llm_selection_quality.csv'), index=False)

    return results_df


def plot_llm_selection_quality(df: pd.DataFrame, output_dir: str,
                               band_mode: str = 'se'):
    """
    Visualize: Mean human rating of messages selected by LLM vs random vs human oracle.

    This answers: "If we use LLM to pick top messages, are they actually good?"
    Uses PP (CBT/ACT 70/30) results.
    """
    print(f"\nCreating LLM selection quality plot (PP, band_mode={band_mode})...")

    fig, axes = plt.subplots(3, 1, figsize=(11, 13.5), sharex=True, sharey=True)

    for idx, domain in enumerate(DOMAIN_ORDER):
        ax = axes[idx]
        domain_df = df[df['Domain'] == domain]

        if len(domain_df) == 0:
            continue

        k_vals = sorted(domain_df['K'].unique())

        # Get random selection data (from first model since it's the same for all)
        first_model_df = domain_df[domain_df['Model'] == MODEL_ORDER[0]].sort_values('K')
        random_means = first_model_df['Random Mean'].values
        random_ci_lower = first_model_df['Random CI Lower'].values
        random_ci_upper = first_model_df['Random CI Upper'].values

        if band_mode == 'se':
            random_lower = random_means - first_model_df['Random SE'].values
            random_upper = random_means + first_model_df['Random SE'].values
            random_label = 'Random Selection (±1 bootstrap SE)'
        else:
            random_lower = random_ci_lower
            random_upper = random_ci_upper
            random_label = 'Random Selection (95% CI)'

        ax.fill_between(k_vals, random_lower, random_upper,
                       color='gray', alpha=0.20)
        ax.plot(k_vals, random_means, ':', color='gray', linewidth=2.5, alpha=0.9, label=random_label)

        # Plot human oracle with uncertainty band
        human_df = domain_df.groupby('K', as_index=False).first().sort_values('K')
        human_vals = human_df['Human Oracle (Human Rating)'].values
        if band_mode == 'se':
            human_lower = human_vals - human_df['Human Oracle SE'].values
            human_upper = human_vals + human_df['Human Oracle SE'].values
            human_label = 'Human Oracle (±1 bootstrap SE)'
        else:
            human_lower = human_df['Human Oracle CI Lower'].values
            human_upper = human_df['Human Oracle CI Upper'].values
            human_label = 'Human Oracle (95% CI)'

        ax.fill_between(k_vals, human_lower, human_upper, color='black', alpha=0.08)
        ax.plot(k_vals, human_vals, 's--',
               color='black', linewidth=2.5, markersize=10, label=human_label, alpha=0.9)

        # Plot each model's LLM selection quality with uncertainty band
        for model in MODEL_ORDER:
            model_df = domain_df[domain_df['Model'] == model].sort_values('K')
            if len(model_df) > 0:
                x_vals = model_df['K'].values
                y_vals = model_df['LLM Selection (Human Rating)'].values
                if band_mode == 'se':
                    lower = y_vals - model_df['LLM SE'].values
                    upper = y_vals + model_df['LLM SE'].values
                else:
                    lower = model_df['LLM CI Lower'].values
                    upper = model_df['LLM CI Upper'].values

                ax.fill_between(x_vals, lower, upper, color=COLORS[model], alpha=0.12)
                ax.plot(x_vals, y_vals, 'o-',
                       color=COLORS[model], label=model, linewidth=2.5, markersize=9)

        ax.set_xlabel('Number of Messages Selected', fontsize=AXIS_LABEL_SIZE, fontweight='bold')
        ax.set_ylabel('Mean Human Rating', fontsize=AXIS_LABEL_SIZE, fontweight='bold')
        ax.set_title(f'{domain}', fontsize=TITLE_SIZE, fontweight='bold')
        ax.set_ylim(2.0, 5.05)
        ax.set_xticks(k_vals)  # Only integer ticks
        ax.tick_params(axis='both', labelsize=TICK_SIZE + 2)
        # Make tick labels bold
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        ax.grid(axis='y', alpha=0.12, linestyle='--', linewidth=0.6, color='#4D4D4D')
        ax.set_facecolor('white')

    axes[-1].set_xlabel('', fontsize=AXIS_LABEL_SIZE, fontweight='bold')

    # Single legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        ncol=3,
        bbox_to_anchor=(0.5, 0.995),
        prop={'weight': 'bold', 'size': LEGEND_SIZE + 2},
        frameon=True,
    )

    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.20)
    save_figure(fig, os.path.join(output_dir, 'llm_selection_quality'))
    plt.close()


def plot_selection_quality_gap(df: pd.DataFrame, output_dir: str):
    """
    Show the gap between LLM selection and optimal (human oracle) selection.
    Normalized: 0 = random, 1 = human oracle.
    """
    print("\nCreating selection quality gap plot...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, domain in enumerate(DOMAIN_ORDER):
        ax = axes[idx]
        domain_df = df[df['Domain'] == domain].copy()

        if len(domain_df) == 0:
            continue

        # Compute normalized quality: (LLM - Random) / (Oracle - Random)
        # 0 = random, 1 = optimal
        domain_df['Normalized Quality'] = (
            (domain_df['LLM Selection (Human Rating)'] - domain_df['Random Baseline']) /
            (domain_df['Human Oracle (Human Rating)'] - domain_df['Random Baseline'])
        )

        # Plot each model
        for model in MODEL_ORDER:
            model_df = domain_df[domain_df['Model'] == model].sort_values('K')
            if len(model_df) > 0:
                ax.plot(model_df['K'], model_df['Normalized Quality'], 'o-',
                       color=COLORS[model], label=model, linewidth=2.5, markersize=9)

        # Reference lines
        ax.axhline(y=0, color='gray', linestyle=':', linewidth=2, label='Random (0)', alpha=0.8)
        ax.axhline(y=1, color='black', linestyle='--', linewidth=2, label='Human Oracle (1)', alpha=0.8)

        ax.set_xlabel('K (Number of Messages Selected)', fontsize=AXIS_LABEL_SIZE, fontweight='bold')
        ax.set_ylabel('Normalized Quality\n(0=Random, 1=Oracle)', fontsize=AXIS_LABEL_SIZE, fontweight='bold')
        ax.set_title(f'{domain}', fontsize=TITLE_SIZE, fontweight='bold')
        ax.set_ylim(-0.5, 1.5)
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        ax.grid(True, alpha=0.3)

    # Single legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02),
               fontsize=LEGEND_SIZE, frameon=True)

    plt.suptitle('LLM Selection Quality: Normalized Score (0=Random, 1=Optimal)',
                fontsize=TITLE_SIZE + 2, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    save_figure(fig, os.path.join(output_dir, 'llm_selection_quality_normalized'))
    plt.close()


def plot_threshold_agreement(df: pd.DataFrame, output_dir: str):
    """
    Create bar charts showing threshold agreement metrics with error bars.
    """
    print("\nCreating threshold agreement plots...")

    for threshold in [4.0]:  # Focus on the main threshold
        thresh_df = df[df['Threshold'] == threshold].copy()

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        metrics_to_plot = ['Precision', 'Recall', 'F1']

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]

            x = np.arange(len(DOMAIN_ORDER))
            width = 0.15
            offsets = np.linspace(-2*width, 2*width, len(MODEL_ORDER))

            for i, model in enumerate(MODEL_ORDER):
                model_df = thresh_df[thresh_df['Model'] == model]
                values = []
                errors_lower = []
                errors_upper = []
                for domain in DOMAIN_ORDER:
                    domain_data = model_df[model_df['Domain'] == domain]
                    if len(domain_data) > 0:
                        val = domain_data.iloc[0][metric]
                        values.append(val)
                        ci_lower_col = f'{metric} CI Lower'
                        ci_upper_col = f'{metric} CI Upper'
                        if ci_lower_col in domain_data.columns:
                            ci_l = domain_data.iloc[0][ci_lower_col]
                            ci_u = domain_data.iloc[0][ci_upper_col]
                            errors_lower.append(val - ci_l)
                            errors_upper.append(ci_u - val)
                        else:
                            errors_lower.append(0)
                            errors_upper.append(0)
                    else:
                        values.append(0)
                        errors_lower.append(0)
                        errors_upper.append(0)

                if any(e > 0 for e in errors_lower + errors_upper):
                    ax.bar(x + offsets[i], values, width, label=model, color=COLORS[model],
                           yerr=[errors_lower, errors_upper], capsize=3, error_kw={'linewidth': 1.5})
                else:
                    ax.bar(x + offsets[i], values, width, label=model, color=COLORS[model])

            ax.set_xlabel('Domain', fontsize=AXIS_LABEL_SIZE, fontweight='bold')
            ax.set_ylabel(metric, fontsize=AXIS_LABEL_SIZE, fontweight='bold')
            ax.set_title(metric, fontsize=TITLE_SIZE, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(DOMAIN_ORDER, fontsize=TICK_SIZE, fontweight='bold')
            ax.tick_params(axis='y', labelsize=TICK_SIZE)
            ax.set_ylim(0, 1.1)
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.02),
                   fontsize=LEGEND_SIZE, frameon=True)

        plt.suptitle(f'"Good Enough" Threshold Agreement (Mean Score >= {threshold})',
                    fontsize=TITLE_SIZE + 2, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        save_figure(fig, os.path.join(output_dir, f'threshold_agreement_{threshold:.1f}'))
        plt.close()


def plot_rank_correlations(df: pd.DataFrame, output_dir: str):
    """
    Create heatmap showing message-level rank correlations.
    """
    print("\nCreating rank correlation heatmap...")

    fig, ax = plt.subplots(figsize=(10, 7))

    # Pivot for heatmap - use Spearman
    pivot = df.pivot_table(index='Model', columns='Domain', values='Spearman Rho')
    pivot = pivot.reindex(MODEL_ORDER)
    pivot = pivot[DOMAIN_ORDER]

    # Use diverging colormap centered at 0
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdBu_r',
                ax=ax, vmin=-0.3, vmax=0.3, center=0,
                cbar_kws={'label': "Spearman's ρ"},
                annot_kws={'fontsize': TICK_SIZE, 'fontweight': 'bold'})

    ax.set_title("Message-Level Rank Correlation: Spearman's ρ\n(n=107 messages per domain)",
                fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_xlabel('Domain', fontsize=AXIS_LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Model', fontsize=AXIS_LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_SIZE)

    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'rank_correlation_heatmap'))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Create practical analysis figures.")
    parser.add_argument(
        '--output-dir',
        default='figures',
        help='Output directory, relative to project root unless absolute.',
    )
    parser.add_argument(
        '--selection-only',
        action='store_true',
        help='Only recompute and plot the Figure 4 selection-quality analysis.',
    )
    parser.add_argument(
        '--n-bootstrap',
        type=int,
        default=2000,
        help='Number of bootstrap resamples for selection-quality uncertainty.',
    )
    parser.add_argument(
        '--band-mode',
        choices=['se', 'ci'],
        default='se',
        help='Uncertainty band to draw for selection-quality plot.',
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(base_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Creating Practical Analysis Figures (PP CBT/ACT 70/30)")
    print("=" * 70)

    if not args.selection_only:
        # Load data
        bootstrap_df = pd.read_csv(os.path.join(output_dir, 'bootstrap_confidence_intervals.csv'))
        top_k_df = pd.read_csv(os.path.join(output_dir, 'top_k_agreement.csv'))

        # Create essential figures only
        plot_bootstrap_2x2_grid(bootstrap_df, output_dir)
        plot_top_k_agreement(top_k_df, output_dir)

    # LLM Selection Quality Analysis - key practical figure
    selection_quality_df = compute_llm_selection_quality(
        base_dir, output_dir, n_bootstrap=args.n_bootstrap
    )
    plot_llm_selection_quality(
        selection_quality_df, output_dir, band_mode=args.band_mode
    )

    print("\n" + "=" * 70)
    print("All figures created successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
