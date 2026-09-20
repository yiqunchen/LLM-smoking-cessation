"""
Pairwise statistical significance tests between PP LLMs.

Performs pairwise comparisons for all C(5,2)=10 model pairs using:
  1. Paired bootstrap test (N=2,000) for Accuracy, F1, Kappa, Directional Accuracy
  2. McNemar's test (chi-squared) for Accuracy only

The saved CSV keeps the original two-sided bootstrap p-values for omnibus
reporting, and also stores directional one-sided p-values used by the lower-
triangle heatmap (row model > column model).

Applies Bonferroni correction (alpha=0.05/10=0.005).

Outputs:
  - revision/figures/pairwise_significance_tests.csv
  - revision/figures/pairwise_significance_heatmap.png/pdf

Usage:
    uv run python analysis-script/pairwise_significance_tests.py
    uv run python analysis-script/pairwise_significance_tests.py --plot-only
"""

import argparse
import sys
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import chi2 as chi2_dist
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'analysis-script')
from revision_utils import (
    load_results_aligned, MODEL_CONFIGS, DOMAINS,
    map_directionality, save_figure, figures_path, COLORS,
)
from filter_duplicates import get_duplicate_signatures, is_duplicate

# -------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

METHOD_FILE = 'digital_twin_4_cbtact_7030.json'
N_BOOT = 2_000
ALPHA = 0.05
N_COMPARISONS = 10  # C(5,2)
ALPHA_CORRECTED = ALPHA / N_COMPARISONS  # 0.005

METRICS = {
    'Accuracy': lambda gt, pred: accuracy_score(gt, pred),
    'F1': lambda gt, pred: f1_score(gt, pred, average='macro', zero_division=0),
    'Kappa': lambda gt, pred: cohen_kappa_score(gt, pred),
    'Directional Accuracy': None,  # handled separately because of mapping
}

rng = np.random.default_rng(42)
DUPLICATE_SIGS = get_duplicate_signatures()


# -----------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------

def directional_accuracy(gt, pred):
    """Compute directional accuracy after mapping to 3 buckets."""
    dir_gt = map_directionality(gt)
    dir_pred = map_directionality(pred)
    valid = (dir_gt != -1) & (dir_pred != -1)
    if not np.any(valid):
        return np.nan
    return np.mean(dir_gt[valid] == dir_pred[valid])


def compute_metric(name, gt, pred):
    """Compute a single metric by name."""
    if name == 'Directional Accuracy':
        return directional_accuracy(gt, pred)
    return METRICS[name](gt, pred)


def bh_adjust(p_values):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not np.any(valid):
        return adjusted

    p_valid = p[valid]
    order = np.argsort(p_valid)
    ranked = p_valid[order]
    n = len(ranked)

    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)

    adjusted_valid = np.empty_like(ranked)
    adjusted_valid[order] = q
    adjusted[valid] = adjusted_valid
    return adjusted


def paired_bootstrap_tests(gt, pred_a, pred_b, n_boot=N_BOOT):
    """Run one shared paired-bootstrap pass for all tracked metrics."""
    metric_names = list(METRICS.keys())
    n = len(gt)
    observed_diffs = {}
    diffs = {metric_name: np.empty(n_boot) for metric_name in metric_names}

    for metric_name in metric_names:
        observed_a = compute_metric(metric_name, gt, pred_a)
        observed_b = compute_metric(metric_name, gt, pred_b)
        observed_diffs[metric_name] = observed_a - observed_b

    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        gt_b = gt[idx]
        pred_a_b = pred_a[idx]
        pred_b_b = pred_b[idx]
        for metric_name in metric_names:
            m_a = compute_metric(metric_name, gt_b, pred_a_b)
            m_b = compute_metric(metric_name, gt_b, pred_b_b)
            diffs[metric_name][b] = m_a - m_b

    results = {}
    for metric_name in metric_names:
        observed_diff = observed_diffs[metric_name]
        diff_samples = diffs[metric_name]
        p_less = float(np.mean(diff_samples < 0))
        p_equal = float(np.mean(diff_samples == 0))
        p_a_gt_b = p_less + 0.5 * p_equal
        p_b_gt_a = 1.0 - p_a_gt_b
        p_val = min(2 * min(p_a_gt_b, p_b_gt_a), 1.0)
        results[metric_name] = {
            'Diff': observed_diff,
            'p_bootstrap': p_val,
            'p_bootstrap_a_gt_b': p_a_gt_b,
            'p_bootstrap_b_gt_a': p_b_gt_a,
        }
    return results


def mcnemar_test(gt, pred_a, pred_b):
    """
    McNemar's test on correctness (for accuracy).

    Returns (b_val, c_val, chi2_stat, p_value).
    b_val = A correct & B wrong; c_val = A wrong & B correct.
    """
    correct_a = (gt == pred_a)
    correct_b = (gt == pred_b)
    b_val = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
    c_val = int(np.sum(~correct_a & correct_b))   # A wrong, B right

    if b_val + c_val == 0:
        return b_val, c_val, 0.0, 1.0

    # McNemar chi-squared with continuity correction
    chi2_stat = (abs(b_val - c_val) - 1) ** 2 / (b_val + c_val)
    p_val = chi2_dist.sf(chi2_stat, df=1)
    return b_val, c_val, chi2_stat, p_val


# -------------------------------------------------------------------------
# Load & align data across all models
# -------------------------------------------------------------------------

def load_and_align_all_models():
    """
    Load predictions for all models using the chosen method.
    Align on (response_id, input_message) so all models share the same items.

    Returns:
        aligned: dict of {model_display: DataFrame} with shared index
        model_order: list of display names
    """
    model_dfs = {}
    for model_id, cfg in MODEL_CONFIGS.items():
        df = load_results_aligned(model_id, METHOD_FILE)
        if df is None:
            print(f"  WARNING: No results for {cfg['display']} with {METHOD_FILE}")
            continue
        keep = ~df.apply(lambda row: is_duplicate(row.to_dict(), DUPLICATE_SIGS), axis=1)
        df = df.loc[keep].reset_index(drop=True)
        # Keep only the columns we need
        keep_cols = ['response_id', 'input_message']
        for domain in DOMAINS:
            keep_cols += [f'gt_{domain}_num', f'pred_{domain}_num']
        available = [c for c in keep_cols if c in df.columns]
        df = df[available].copy()
        model_dfs[cfg['display']] = df

    if len(model_dfs) < 2:
        raise RuntimeError("Need at least 2 models to do pairwise tests.")

    # Find common items across ALL models via inner merge on (response_id, input_message)
    model_names = list(model_dfs.keys())
    common_keys = model_dfs[model_names[0]][['response_id', 'input_message']].copy()
    for name in model_names[1:]:
        other = model_dfs[name][['response_id', 'input_message']].copy()
        common_keys = common_keys.merge(other, on=['response_id', 'input_message'], how='inner')
    common_keys = common_keys.drop_duplicates()

    print(f"  Common items across all {len(model_names)} models: {len(common_keys)}")

    # Re-align each model to common keys
    aligned = {}
    for name, df in model_dfs.items():
        merged = common_keys.merge(df, on=['response_id', 'input_message'], how='left')
        # Drop any duplicate rows (in case of duplicates in original data)
        merged = merged.drop_duplicates(subset=['response_id', 'input_message'])
        # Sort consistently
        merged = merged.sort_values(['response_id', 'input_message']).reset_index(drop=True)
        aligned[name] = merged

    model_order = list(aligned.keys())
    return aligned, model_order


# -------------------------------------------------------------------------
# Run all pairwise tests
# -----------------------------------------------------------------------

def run_pairwise_tests(aligned, model_order):
    """
    Run pairwise bootstrap and McNemar tests for all model pairs x domains x metrics.

    Returns a DataFrame of results.
    """
    rows = []
    pairs = list(itertools.combinations(model_order, 2))
    print(f"\n  Running pairwise tests for {len(pairs)} pairs x {len(DOMAINS)} domains x {len(METRICS)} metrics...")

    for pair_idx, (name_a, name_b) in enumerate(pairs):
        df_a = aligned[name_a]
        df_b = aligned[name_b]

        for domain in DOMAINS:
            gt_col = f'gt_{domain}_num'
            pred_col = f'pred_{domain}_num'

            # Ground truth should be the same for both; take from model A
            gt = df_a[gt_col].values.astype(float)
            pred_a = df_a[pred_col].values.astype(float)
            pred_b = df_b[pred_col].values.astype(float)

            # Drop rows where any value is NaN
            valid = ~(np.isnan(gt) | np.isnan(pred_a) | np.isnan(pred_b))
            gt = gt[valid].astype(int)
            pred_a = pred_a[valid].astype(int)
            pred_b = pred_b[valid].astype(int)

            if len(gt) < 10:
                continue

            bootstrap_results = paired_bootstrap_tests(gt, pred_a, pred_b)
            for metric_name in METRICS:
                diff = bootstrap_results[metric_name]['Diff']
                p_boot = bootstrap_results[metric_name]['p_bootstrap']
                p_boot_a_gt_b = bootstrap_results[metric_name]['p_bootstrap_a_gt_b']
                p_boot_b_gt_a = bootstrap_results[metric_name]['p_bootstrap_b_gt_a']
                p_corrected = min(p_boot * N_COMPARISONS, 1.0)  # Bonferroni
                p_corrected_a_gt_b = min(p_boot_a_gt_b * N_COMPARISONS, 1.0)
                p_corrected_b_gt_a = min(p_boot_b_gt_a * N_COMPARISONS, 1.0)

                # McNemar test (only meaningful for accuracy)
                p_mcnemar = np.nan
                if metric_name == 'Accuracy':
                    _, _, _, p_mcnemar = mcnemar_test(gt, pred_a, pred_b)

                significant = p_corrected < ALPHA

                rows.append({
                    'Model_A': name_a,
                    'Model_B': name_b,
                    'Domain': domain.capitalize(),
                    'Metric': metric_name,
                    'Score_A': compute_metric(metric_name, gt, pred_a),
                    'Score_B': compute_metric(metric_name, gt, pred_b),
                    'Diff': diff,
                    'p_bootstrap': p_boot,
                    'p_bootstrap_a_gt_b': p_boot_a_gt_b,
                    'p_bootstrap_b_gt_a': p_boot_b_gt_a,
                    'p_mcnemar': p_mcnemar,
                    'p_corrected': p_corrected,
                    'p_corrected_a_gt_b': p_corrected_a_gt_b,
                    'p_corrected_b_gt_a': p_corrected_b_gt_a,
                    'Significant': significant,
                    'N': len(gt),
                })

        print(f"    Pair {pair_idx + 1}/{len(pairs)}: {name_a} vs {name_b} done")

    results_df = pd.DataFrame(rows)

    for domain in results_df['Domain'].unique():
        for metric_name in results_df['Metric'].unique():
            mask = (
                (results_df['Domain'] == domain) &
                (results_df['Metric'] == metric_name)
            )
            results_df.loc[mask, 'p_bh'] = bh_adjust(
                results_df.loc[mask, 'p_bootstrap'].to_numpy()
            )
            results_df.loc[mask, 'p_bh_a_gt_b'] = bh_adjust(
                results_df.loc[mask, 'p_bootstrap_a_gt_b'].to_numpy()
            )
            results_df.loc[mask, 'p_bh_b_gt_a'] = bh_adjust(
                results_df.loc[mask, 'p_bootstrap_b_gt_a'].to_numpy()
            )

    return results_df


# -------------------------------------------------------------------------
# Heatmap visualization
# -------------------------------------------------------------------------

def create_significance_heatmap(results_df, model_order):
    """
    Create a 1x3 full off-diagonal heatmap for directional one-sided accuracy tests.

    The direction with the higher observed point estimate shows the BH-adjusted
    one-sided p-value. The opposite cell keeps the raw one-sided value so the
    full matrix remains readable without creating clipped 0/1 artifacts from
    correction.
    """
    acc_df = results_df[results_df['Metric'] == 'Accuracy'].copy()

    fig, axes = plt.subplots(1, 3, figsize=(23.5, 9.0))
    cmap = ListedColormap(["#FBFBFA", "#E6F2EC"])
    cmap.set_bad(color="white")

    for ax_idx, domain in enumerate(DOMAINS):
        domain_cap = domain.capitalize()
        dom_df = acc_df[acc_df['Domain'] == domain_cap]

        n_models = len(model_order)
        p_display_matrix = np.full((n_models, n_models), np.nan)
        sig_matrix = np.full((n_models, n_models), np.nan)

        for _, row in dom_df.iterrows():
            idx_a = model_order.index(row['Model_A'])
            idx_b = model_order.index(row['Model_B'])
            diff = row['Diff']

            if diff > 0:
                p_favored = float(row['p_bh_a_gt_b'])
                p_display_matrix[idx_a, idx_b] = p_favored
                p_display_matrix[idx_b, idx_a] = float(row['p_bootstrap_b_gt_a'])
                sig_matrix[idx_a, idx_b] = 1.0 if p_favored < ALPHA else 0.0
                sig_matrix[idx_b, idx_a] = 0.0
            elif diff < 0:
                p_favored = float(row['p_bh_b_gt_a'])
                p_display_matrix[idx_b, idx_a] = p_favored
                p_display_matrix[idx_a, idx_b] = float(row['p_bootstrap_a_gt_b'])
                sig_matrix[idx_b, idx_a] = 1.0 if p_favored < ALPHA else 0.0
                sig_matrix[idx_a, idx_b] = 0.0
            else:
                p_display_matrix[idx_a, idx_b] = 0.5
                p_display_matrix[idx_b, idx_a] = 0.5
                sig_matrix[idx_a, idx_b] = 0.0
                sig_matrix[idx_b, idx_a] = 0.0

        np.fill_diagonal(p_display_matrix, np.nan)
        np.fill_diagonal(sig_matrix, np.nan)

        ax = axes[ax_idx]
        masked = np.ma.masked_invalid(sig_matrix)
        ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect='equal')

        ax.set_xticks(np.arange(-0.5, n_models, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_models, 1), minor=True)
        ax.grid(which='minor', color='#D9D9D9', linestyle='-', linewidth=1.0)
        ax.tick_params(which='minor', bottom=False, left=False)
        ax.grid(False)

        # Annotate cells
        for i in range(n_models):
            for j in range(n_models):
                if i == j:
                    continue
                p_val = p_display_matrix[i, j]
                if np.isnan(p_val):
                    continue
                if p_val < 0.001:
                    p_text = '<0.001'
                else:
                    p_text = f'{p_val:.3f}'
                ax.text(
                    j, i, p_text,
                    ha='center', va='center',
                    fontsize=13, color='black',
                    fontweight='bold' if sig_matrix[i, j] == 1.0 else 'normal',
                )

        ax.set_xticks(range(n_models))
        ax.set_yticks(range(n_models))
        ax.set_xticklabels(model_order, rotation=38, ha='right', fontsize=12, fontweight='bold')
        ax.set_yticklabels(model_order, fontsize=12, fontweight='bold')
        ax.set_title(f'{domain_cap}', fontsize=16, fontweight='bold', pad=12)
        ax.set_xlim(-0.5, n_models - 0.5)
        ax.set_ylim(n_models - 0.5, -0.5)
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color(COLORS.get(label.get_text(), 'black'))

    fig.patch.set_facecolor('white')
    fig.suptitle(
        'PP LLMs: One-Sided Pairwise Bootstrap Tests (Accuracy)',
        fontsize=17, fontweight='bold', y=0.985
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Pairwise PP LLM significance tests.")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Reload pairwise_significance_tests.csv and regenerate the heatmap only.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70, flush=True)
    print("Pairwise Statistical Significance Tests Between Models", flush=True)
    print("=" * 70, flush=True)
    print(f"  Method: {METHOD_FILE}", flush=True)
    print(f"  Bootstrap iterations: {N_BOOT:,}", flush=True)
    print(f"  Bonferroni alpha: {ALPHA_CORRECTED}", flush=True)

    csv_path = figures_path('pairwise_significance_tests') + '.csv'
    model_order = [cfg['display'] for cfg in MODEL_CONFIGS.values()]
    if args.plot_only:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing cached results CSV: {csv_path}")
        print(f"\n[plot-only] Loading cached results: {csv_path}", flush=True)
        results_df = pd.read_csv(csv_path)
        fig = create_significance_heatmap(results_df, model_order)
        save_figure(fig, figures_path('pairwise_significance_heatmap'))
        print("\nDone.", flush=True)
        return

    # 1. Load and align
    print("\n[1/3] Loading and aligning model predictions...", flush=True)
    aligned, model_order = load_and_align_all_models()
    print(f"  Model order: {model_order}", flush=True)

    # 2. Run tests
    print("\n[2/3] Running pairwise significance tests...", flush=True)
    results_df = run_pairwise_tests(aligned, model_order)

    # Save CSV
    results_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\n  Saved: {csv_path}", flush=True)

    # Print summary
    print("\n  --- Summary of significant differences (Bonferroni-corrected) ---")
    sig = results_df[results_df['Significant']]
    if len(sig) > 0:
        for _, row in sig.iterrows():
            if row['Diff'] > 0:
                direction = f"{row['Model_A']} > {row['Model_B']}"
            else:
                direction = f"{row['Model_B']} > {row['Model_A']}"
            print(f"    {row['Domain']:>10s} | {row['Metric']:<25s} | "
                  f"{direction:<35s} | diff={row['Diff']:+.4f} | "
                  f"p_boot={row['p_bootstrap']:.4f} | "
                  f"p_corr={row['p_corrected']:.4f}", flush=True)
    else:
        print("    No significant differences found after Bonferroni correction.", flush=True)

    not_sig = results_df[~results_df['Significant']]
    print(f"\n  Total tests: {len(results_df)}", flush=True)
    print(f"  Significant: {len(sig)}  |  Not significant: {len(not_sig)}", flush=True)

    # 3. Heatmap
    print("\n[3/3] Creating significance heatmap...", flush=True)
    fig = create_significance_heatmap(results_df, model_order)
    heatmap_path = figures_path('pairwise_significance_heatmap')
    save_figure(fig, heatmap_path)

    print("\n" + "=" * 70, flush=True)
    print("Done.", flush=True)
    print("=" * 70, flush=True)


if __name__ == '__main__':
    main()
