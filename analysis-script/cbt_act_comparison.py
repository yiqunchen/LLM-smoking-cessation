"""
CBT vs ACT Message Performance Comparison (Reviewer R4)

Compares model performance on CBT-based vs ACT-based messages.
Messages are identified by the Image ID prefix in metadata:
  - 'A' prefix -> ACT messages
  - 'D' prefix -> CBT messages

For each model x method x domain, splits data into ACT/CBT subsets,
computes metrics with bootstrap 95% CIs, and produces:
  - revision/figures/cbt_act_comparison.csv
  - revision/figures/cbt_vs_act_performance.png/pdf

Usage:
    uv run python analysis-script/cbt_act_comparison.py
    uv run python analysis-script/cbt_act_comparison.py --plot-only
    uv run python analysis-script/cbt_act_comparison.py --models gpt-5 \
        --methods digital_twin_4_cbtact_7030.json --n-bootstrap 50 \
        --output-tag smoke
"""

import argparse
import sys
import os

sys.path.insert(0, 'analysis-script')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from revision_utils import (
    load_results_aligned, compute_all_metrics,
    load_canonical_data, extract_labels,
    MODEL_CONFIGS, METHOD_CONFIGS, DOMAINS, COLORS,
    apply_repo_plot_style, save_figure, figures_path,
)
from text_baselines import (
    baseline_tfidf_model,
    baseline_embedding_model,
    baseline_demographics_model,
    _load_embeddings,
    _match_embeddings,
)
from history_supervised_baselines import _make_item_key_from_record

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Best methods per category (keeps figure manageable)
SELECTED_METHODS = [
    'generic_llm_2_zero_shot_select.json',   # best zero-shot
    'generic_llm_4_few_shot_select.json',     # best few-shot
    'digital_twin_4_cbtact_7030.json',        # PP
    'hybrid',                                  # hybrid RF + PP
]

FOCUS_BASELINE_METHOD = 'Best Text Baseline'
SUPERVISED_PREDICTIONS_PATH = figures_path('history_supervised_predictions') + '.csv'
SUPERVISED_CBT_ACT_SYSTEMS = [
    {
        'System': 'RF - Demographics',
        'Feature_Set': 'Demographics',
        'Classifier': 'RF',
        'Label': 'RF\nDemo',
    },
    {
        'System': 'LR - Demographics',
        'Feature_Set': 'Demographics',
        'Classifier': 'LR',
        'Label': 'LR\nDemo',
    },
    {
        'System': 'LR - Demo+History+Embedding',
        'Feature_Set': 'Demographics + History + Message Embedding',
        'Classifier': 'LR',
        'Label': 'LR\nDemo+Hist\n+ Embed',
    },
    {
        'System': 'RF - Demo+History+Embedding',
        'Feature_Set': 'Demographics + History + Message Embedding',
        'Classifier': 'RF',
        'Label': 'RF\nDemo+Hist\n+ Embed',
    },
]
TEXT_BASELINE_CANDIDATES = [
    'TF-IDF + LR',
    'TF-IDF + RF',
    'Embedding + LR',
    'Embedding + RF',
]
METHOD_FILE_BY_DISPLAY = {
    cfg['display']: method_file for method_file, cfg in METHOD_CONFIGS.items()
}
SHORT_MODEL_LABELS = {
    'GPT-4o-mini': '4o-mini',
    'GPT-5': 'GPT-5',
    'DeepSeek-R1': 'R1',
    'Grok-4-Fast': 'Grok',
    'Gemini-2.5-Pro': 'Gemini',
}
TEXT_BASELINE_SHORT_LABELS = {
    'TF-IDF + LR': 'TF-IDF\n+ LR',
    'TF-IDF + RF': 'TF-IDF\n+ RF',
    'Embedding + LR': 'Embed.\n+ LR',
    'Embedding + RF': 'Embed.\n+ RF',
}
THERAPY_COLORS = {'ACT': COLORS['GPT-4o-mini'], 'CBT': COLORS['GPT-5']}
LEFT_PANEL_ORDER = [spec['System'] for spec in SUPERVISED_CBT_ACT_SYSTEMS]
LEFT_PANEL_LABELS = {spec['System']: spec['Label'] for spec in SUPERVISED_CBT_ACT_SYSTEMS}
SUPERVISED_SYSTEM_LOOKUP = {
    spec['System']: (spec['Feature_Set'], spec['Classifier'])
    for spec in SUPERVISED_CBT_ACT_SYSTEMS
}

# Keep the revision deliverable bootstrap-based with a stable CI estimate.
N_BOOTSTRAP = 2000
RANDOM_SEED = 42
CI_LOWER_PCT = 2.5
CI_UPPER_PCT = 97.5

METRICS_TO_COMPUTE = [
    'Accuracy', 'F1', 'Kappa', 'QWK',
    'Directional Accuracy', 'Directional Macro-F1', 'Spearman_Rho',
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_therapy_category(df):
    """Return a Series with ACT, CBT, or None for each row based on Image ID prefix."""
    def _get_category(meta):
        if not isinstance(meta, dict):
            return None
        image_id = meta.get('Image ID', '')
        if isinstance(image_id, str):
            if image_id.startswith('A'):
                return 'ACT'
            elif image_id.startswith('D'):
                return 'CBT'
        return None
    return df['metadata'].apply(_get_category)


def bootstrap_all_metrics(gt, pred, response_ids,
                          n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    """Compute point estimates, bootstrap SEs, and bootstrap CIs.

    This avoids recomputing the full metric bundle once per metric and is
    substantially faster than separate bootstrap passes.
    """
    rng = np.random.RandomState(seed)
    point_estimates = compute_all_metrics(gt, pred, response_ids)
    boot_values = {metric_name: [] for metric_name in METRICS_TO_COMPUTE}

    # Bootstrap resamples
    n = len(gt)
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        gt_b = gt[idx]
        pred_b = pred[idx]
        rids_b = response_ids[idx] if response_ids is not None else None
        m = compute_all_metrics(gt_b, pred_b, rids_b)
        for metric_name in METRICS_TO_COMPUTE:
            val = m.get(metric_name, np.nan)
            if not np.isnan(val):
                boot_values[metric_name].append(val)

    results = {}
    for metric_name in METRICS_TO_COMPUTE:
        vals = boot_values[metric_name]
        if vals:
            se = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            ci_lower = float(np.percentile(vals, CI_LOWER_PCT))
            ci_upper = float(np.percentile(vals, CI_UPPER_PCT))
        else:
            se = np.nan
            ci_lower = np.nan
            ci_upper = np.nan
        results[metric_name] = {
            'Value': point_estimates.get(metric_name, np.nan),
            'SE': se,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
        }

    return results


def bootstrap_accuracy_difference(
    gt_act,
    pred_act,
    gt_cbt,
    pred_cbt,
    n_boot=N_BOOTSTRAP,
    seed=RANDOM_SEED,
):
    """Bootstrap two-sided ACT vs CBT accuracy difference."""
    rng = np.random.RandomState(seed)
    n_act = len(gt_act)
    n_cbt = len(gt_cbt)
    boot_diffs = np.empty(n_boot, dtype=float)

    for boot_idx in range(n_boot):
        act_idx = rng.randint(0, n_act, size=n_act)
        cbt_idx = rng.randint(0, n_cbt, size=n_cbt)
        act_acc = np.mean(gt_act[act_idx] == pred_act[act_idx])
        cbt_acc = np.mean(gt_cbt[cbt_idx] == pred_cbt[cbt_idx])
        boot_diffs[boot_idx] = cbt_acc - act_acc

    observed_diff = float(np.mean(gt_cbt == pred_cbt) - np.mean(gt_act == pred_act))
    p_two_sided = 2 * min(
        np.mean(boot_diffs <= 0),
        np.mean(boot_diffs >= 0),
    )
    return {
        'Diff_CBT_minus_ACT': observed_diff,
        'p_raw': min(1.0, float(p_two_sided)),
    }


def bootstrap_accuracy_summary(gt, pred, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    """Bootstrap accuracy with 95% CI."""
    gt = np.asarray(gt, dtype=float)
    pred = np.asarray(pred, dtype=float)
    rng = np.random.RandomState(seed)
    n = len(gt)
    boot_acc = np.empty(n_boot, dtype=float)
    for boot_idx in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_acc[boot_idx] = np.mean(gt[idx] == pred[idx])
    return {
        'Value': float(np.mean(gt == pred)),
        'CI_Lower': float(np.percentile(boot_acc, CI_LOWER_PCT)),
        'CI_Upper': float(np.percentile(boot_acc, CI_UPPER_PCT)),
    }


def benjamini_hochberg(p_values):
    """Return BH-adjusted p-values in original order."""
    p_values = np.asarray(p_values, dtype=float)
    if p_values.size == 0:
        return np.array([], dtype=float)

    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    output = np.empty_like(adjusted)
    output[order] = adjusted
    return output


def _get_text_baseline_outputs(
    train_data,
    test_data,
    domain,
    baseline_name,
    emb_matrix=None,
    emb_lookup=None,
):
    """Return aligned ACT/CBT labels, truths, and predictions for one text baseline."""
    test_df = pd.DataFrame(test_data).copy()
    test_df['therapy'] = extract_therapy_category(test_df)
    y_test = extract_labels(test_data, domain)

    if baseline_name == 'TF-IDF + LR':
        gt, pred, rids = baseline_tfidf_model(train_data, test_data, domain, 'lr')
        valid_test = ~np.isnan(y_test)
        meta_df = test_df.loc[valid_test].copy().reset_index(drop=True)
    elif baseline_name == 'TF-IDF + RF':
        gt, pred, rids = baseline_tfidf_model(train_data, test_data, domain, 'rf')
        valid_test = ~np.isnan(y_test)
        meta_df = test_df.loc[valid_test].copy().reset_index(drop=True)
    elif baseline_name == 'Embedding + LR':
        gt, pred, rids = baseline_embedding_model(
            train_data, test_data, domain, emb_matrix, emb_lookup, 'lr'
        )
        _, valid_test_idx = _match_embeddings(test_data, emb_matrix, emb_lookup)
        matched_df = test_df.iloc[valid_test_idx].copy().reset_index(drop=True)
        y_te_all = y_test[valid_test_idx]
        valid_label_test = ~np.isnan(y_te_all)
        meta_df = matched_df.loc[valid_label_test].copy().reset_index(drop=True)
    elif baseline_name == 'Embedding + RF':
        gt, pred, rids = baseline_embedding_model(
            train_data, test_data, domain, emb_matrix, emb_lookup, 'rf'
        )
        _, valid_test_idx = _match_embeddings(test_data, emb_matrix, emb_lookup)
        matched_df = test_df.iloc[valid_test_idx].copy().reset_index(drop=True)
        y_te_all = y_test[valid_test_idx]
        valid_label_test = ~np.isnan(y_te_all)
        meta_df = matched_df.loc[valid_label_test].copy().reset_index(drop=True)
    else:
        raise ValueError(f'Unknown text baseline: {baseline_name}')

    if len(meta_df) != len(gt):
        raise ValueError(
            f'Baseline alignment mismatch for {baseline_name} / {domain}: '
            f'{len(meta_df)} rows vs {len(gt)} predictions'
        )

    out_df = meta_df[['therapy']].copy()
    out_df['gt'] = gt.astype(float)
    out_df['pred'] = pred.astype(float)
    out_df['response_id'] = np.asarray(rids)
    return out_df


def _get_non_llm_accuracy_outputs(
    train_data,
    test_data,
    domain,
    system_name,
    emb_matrix=None,
    emb_lookup=None,
):
    """Return therapy labels, truths, and predictions for one non-LLM baseline."""
    test_df = pd.DataFrame(test_data).copy()
    test_df['therapy'] = extract_therapy_category(test_df)
    y_test = extract_labels(test_data, domain)

    if system_name == 'Demographics LR':
        gt, pred, _ = baseline_demographics_model(train_data, test_data, domain, 'lr')
        valid_test = ~np.isnan(y_test)
        out_df = test_df.loc[valid_test, ['therapy']].copy().reset_index(drop=True)
        out_df['gt'] = gt.astype(float)
        out_df['pred'] = pred.astype(float)
        return out_df

    if system_name == 'Demographics RF':
        gt, pred, _ = baseline_demographics_model(train_data, test_data, domain, 'rf')
        valid_test = ~np.isnan(y_test)
        out_df = test_df.loc[valid_test, ['therapy']].copy().reset_index(drop=True)
        out_df['gt'] = gt.astype(float)
        out_df['pred'] = pred.astype(float)
        return out_df

    if system_name == 'Embedding + LR':
        baseline_name = 'Embedding + LR'
    elif system_name == 'Embedding + RF':
        baseline_name = 'Embedding + RF'
    else:
        raise ValueError(f'Unknown non-LLM system: {system_name}')

    baseline_df = _get_text_baseline_outputs(
        train_data,
        test_data,
        domain,
        baseline_name,
        emb_matrix=emb_matrix,
        emb_lookup=emb_lookup,
    )
    return baseline_df[['therapy', 'gt', 'pred']].copy()


def _load_existing_supervised_predictions_with_therapy():
    """Load committed supervised predictions and attach ACT/CBT labels.

    The source is `history_supervised_predictions.csv`, produced by
    `history_supervised_baselines.py` on the cleaned canonical digital-twin
    70/30 split. This keeps the CBT/ACT supervised panel on the same
    supervised lineage used by Figure 2.
    """
    if not os.path.exists(SUPERVISED_PREDICTIONS_PATH):
        raise FileNotFoundError(
            f'Missing supervised prediction source: {SUPERVISED_PREDICTIONS_PATH}'
        )

    pred_df = pd.read_csv(SUPERVISED_PREDICTIONS_PATH)
    _, test_data = load_canonical_data('7030', 'digital_twin')
    test_df = pd.DataFrame(test_data).copy()
    test_df['therapy'] = extract_therapy_category(test_df)

    therapy_by_key = {
        _make_item_key_from_record(record): therapy
        for record, therapy in zip(test_data, test_df['therapy'])
    }
    pred_df['therapy'] = pred_df['Item_Key'].map(therapy_by_key)

    missing = int(pred_df['therapy'].isna().sum())
    if missing:
        print(
            f'WARNING: {missing} supervised prediction rows lacked ACT/CBT labels '
            f'from digital-twin test metadata.',
            flush=True,
        )
    return pred_df[pred_df['therapy'].isin(['ACT', 'CBT'])].copy()


def compute_non_llm_accuracy_summary(domains=None, categories=None, n_boot=N_BOOTSTRAP):
    """Compute ACT/CBT accuracy summaries from existing supervised predictions."""
    domains = domains or list(DOMAINS)
    categories = categories or ['ACT', 'CBT']

    pred_df = _load_existing_supervised_predictions_with_therapy()
    rows = []

    for domain_idx, domain in enumerate(domains):
        for system_idx, system_name in enumerate(LEFT_PANEL_ORDER):
            feature_set, classifier = SUPERVISED_SYSTEM_LOOKUP[system_name]
            output_df = pred_df[
                (pred_df['Domain'] == domain.capitalize()) &
                (pred_df['Feature_Set'] == feature_set) &
                (pred_df['Classifier'] == classifier)
            ].copy()

            for category_idx, category in enumerate(categories):
                sub = output_df[output_df['therapy'] == category]
                if len(sub) < 5:
                    continue
                summary = bootstrap_accuracy_summary(
                    sub['Ground_Truth_Num'].values.astype(float),
                    sub['Predicted_Num'].values.astype(float),
                    n_boot=n_boot,
                    seed=RANDOM_SEED + 1000 + (100 * domain_idx) + (10 * system_idx) + category_idx,
                )
                rows.append({
                    'System': system_name,
                    'Feature_Set': feature_set,
                    'Classifier': classifier,
                    'Domain': domain.capitalize(),
                    'Category': category,
                    'Value': summary['Value'],
                    'CI_Lower': summary['CI_Lower'],
                    'CI_Upper': summary['CI_Upper'],
                    'Bootstrap_N': n_boot,
                    'N': len(sub),
                })

    return pd.DataFrame(rows)


def compute_non_llm_accuracy_significance(domains=None, n_boot=N_BOOTSTRAP):
    """Compute ACT vs CBT accuracy tests for existing supervised predictions."""
    domains = domains or list(DOMAINS)

    pred_df = _load_existing_supervised_predictions_with_therapy()
    rows = []

    for domain_idx, domain in enumerate(domains):
        family_rows = []
        for system_idx, system_name in enumerate(LEFT_PANEL_ORDER):
            feature_set, classifier = SUPERVISED_SYSTEM_LOOKUP[system_name]
            output_df = pred_df[
                (pred_df['Domain'] == domain.capitalize()) &
                (pred_df['Feature_Set'] == feature_set) &
                (pred_df['Classifier'] == classifier)
            ].copy()
            act_df = output_df[output_df['therapy'] == 'ACT']
            cbt_df = output_df[output_df['therapy'] == 'CBT']
            if len(act_df) < 5 or len(cbt_df) < 5:
                continue

            diff_result = bootstrap_accuracy_difference(
                act_df['Ground_Truth_Num'].values.astype(float),
                act_df['Predicted_Num'].values.astype(float),
                cbt_df['Ground_Truth_Num'].values.astype(float),
                cbt_df['Predicted_Num'].values.astype(float),
                n_boot=n_boot,
                seed=RANDOM_SEED + 2000 + (100 * domain_idx) + system_idx,
            )
            family_rows.append({
                'Panel': 'Supervised Baseline Accuracy',
                'System': system_name,
                'Feature_Set': feature_set,
                'Classifier': classifier,
                'Domain': domain.capitalize(),
                'ACT_Accuracy': float(np.mean(
                    act_df['Ground_Truth_Num'].values == act_df['Predicted_Num'].values
                )),
                'CBT_Accuracy': float(np.mean(
                    cbt_df['Ground_Truth_Num'].values == cbt_df['Predicted_Num'].values
                )),
                'Diff_CBT_minus_ACT': diff_result['Diff_CBT_minus_ACT'],
                'p_raw': diff_result['p_raw'],
                'N_ACT': len(act_df),
                'N_CBT': len(cbt_df),
                'Bootstrap_N': n_boot,
            })

        adjusted = benjamini_hochberg([row['p_raw'] for row in family_rows])
        for row, p_bh in zip(family_rows, adjusted):
            row['p_bh'] = float(p_bh)
            row['Significant_BH_0_05'] = bool(p_bh < 0.05)
            rows.append(row)

    return pd.DataFrame(rows)


def choose_best_text_baseline(domains=None):
    """Pick the strongest pure-text baseline among TF-IDF/embedding candidates."""
    domains = domains or list(DOMAINS)
    train_data, test_data = load_canonical_data('7030', 'participant')
    emb_matrix, emb_lookup = _load_embeddings()

    rows = []
    for baseline_name in TEXT_BASELINE_CANDIDATES:
        domain_acc = []
        therapy_acc = []
        for domain in domains:
            output_df = _get_text_baseline_outputs(
                train_data,
                test_data,
                domain,
                baseline_name,
                emb_matrix=emb_matrix,
                emb_lookup=emb_lookup,
            )
            domain_acc.append(float(np.mean(output_df['gt'] == output_df['pred'])))
            for therapy in ['ACT', 'CBT']:
                sub = output_df[output_df['therapy'] == therapy]
                if len(sub):
                    therapy_acc.append(float(np.mean(sub['gt'] == sub['pred'])))

        rows.append({
            'Baseline': baseline_name,
            'Mean_Accuracy': float(np.mean(domain_acc)),
            'Mean_ACT_CBT_Cell_Accuracy': float(np.mean(therapy_acc)),
        })

    summary_df = pd.DataFrame(rows).sort_values(
        ['Mean_Accuracy', 'Mean_ACT_CBT_Cell_Accuracy', 'Baseline'],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return summary_df.loc[0, 'Baseline'], summary_df


def run_text_baseline_analysis(best_baseline_name, domains=None,
                               categories=None, n_boot=N_BOOTSTRAP):
    """Compute bootstrap metrics for the selected text baseline."""
    domains = domains or list(DOMAINS)
    categories = categories or ['ACT', 'CBT']

    train_data, test_data = load_canonical_data('7030', 'participant')
    emb_matrix, emb_lookup = _load_embeddings()
    rows = []

    for domain in domains:
        output_df = _get_text_baseline_outputs(
            train_data,
            test_data,
            domain,
            best_baseline_name,
            emb_matrix=emb_matrix,
            emb_lookup=emb_lookup,
        )

        for category in categories:
            sub = output_df[output_df['therapy'] == category].copy()
            if len(sub) < 5:
                continue

            gt = sub['gt'].values.astype(float)
            pred = sub['pred'].values.astype(float)
            rids = sub['response_id'].values
            metric_results = bootstrap_all_metrics(gt, pred, rids, n_boot=n_boot)
            for metric_name in METRICS_TO_COMPUTE:
                result = metric_results[metric_name]
                rows.append({
                    'Model': best_baseline_name,
                    'Method': FOCUS_BASELINE_METHOD,
                    'Domain': domain.capitalize(),
                    'Category': category,
                    'Metric': metric_name,
                    'Value': result['Value'],
                    'SE': result['SE'],
                    'CI_Lower': result['CI_Lower'],
                    'CI_Upper': result['CI_Upper'],
                    'Bootstrap_N': n_boot,
                    'N': len(gt),
                })

    return pd.DataFrame(rows)


def compute_text_baseline_significance(best_baseline_name, domains=None, n_boot=N_BOOTSTRAP):
    """Compute ACT vs CBT significance for the selected text baseline."""
    domains = domains or list(DOMAINS)
    train_data, test_data = load_canonical_data('7030', 'participant')
    emb_matrix, emb_lookup = _load_embeddings()

    rows = []
    for domain_idx, domain in enumerate(domains):
        output_df = _get_text_baseline_outputs(
            train_data,
            test_data,
            domain,
            best_baseline_name,
            emb_matrix=emb_matrix,
            emb_lookup=emb_lookup,
        )
        act_df = output_df[output_df['therapy'] == 'ACT']
        cbt_df = output_df[output_df['therapy'] == 'CBT']
        if len(act_df) < 5 or len(cbt_df) < 5:
            continue

        gt_act = act_df['gt'].values.astype(float)
        pred_act = act_df['pred'].values.astype(float)
        gt_cbt = cbt_df['gt'].values.astype(float)
        pred_cbt = cbt_df['pred'].values.astype(float)
        diff_result = bootstrap_accuracy_difference(
            gt_act,
            pred_act,
            gt_cbt,
            pred_cbt,
            n_boot=n_boot,
            seed=RANDOM_SEED + 900 + domain_idx,
        )
        p_bh = diff_result['p_raw']  # one test in this panel family
        rows.append({
            'Model': best_baseline_name,
            'Method': FOCUS_BASELINE_METHOD,
            'Domain': domain.capitalize(),
            'ACT_Accuracy': float(np.mean(gt_act == pred_act)),
            'CBT_Accuracy': float(np.mean(gt_cbt == pred_cbt)),
            'Diff_CBT_minus_ACT': diff_result['Diff_CBT_minus_ACT'],
            'p_raw': diff_result['p_raw'],
            'N_ACT': len(gt_act),
            'N_CBT': len(gt_cbt),
            'Bootstrap_N': n_boot,
            'p_bh': float(p_bh),
            'Significant_BH_0_05': bool(p_bh < 0.05),
        })

    return pd.DataFrame(rows)


def compute_actual_rating_summary():
    """Summarize observed human rating differences for participant and digital-twin splits."""
    rows = []
    for method_label, split_type in [
        (FOCUS_BASELINE_METHOD, 'participant'),
        ('PP', 'digital_twin'),
    ]:
        _, test_data = load_canonical_data('7030', split_type)
        test_df = pd.DataFrame(test_data).copy()
        test_df['therapy'] = extract_therapy_category(test_df)
        for domain in DOMAINS:
            labels = extract_labels(test_data, domain)
            valid = ~np.isnan(labels)
            sub_df = test_df.loc[valid, ['therapy']].copy().reset_index(drop=True)
            sub_df['rating'] = labels[valid].astype(float)
            act_df = sub_df[sub_df['therapy'] == 'ACT']
            cbt_df = sub_df[sub_df['therapy'] == 'CBT']
            if len(act_df) == 0 or len(cbt_df) == 0:
                continue
            rows.append({
                'Method': method_label,
                'Domain': domain.capitalize(),
                'ACT_Mean_Rating': float(act_df['rating'].mean()),
                'CBT_Mean_Rating': float(cbt_df['rating'].mean()),
                'Diff_CBT_minus_ACT': float(cbt_df['rating'].mean() - act_df['rating'].mean()),
                'N_ACT': len(act_df),
                'N_CBT': len(cbt_df),
            })
    return pd.DataFrame(rows)


def compute_focus_significance(model_ids, focus_method_displays, domains, n_boot=N_BOOTSTRAP):
    """Compute ACT vs CBT bootstrap tests for the plotted methods.

    BH correction is applied separately within each method-domain family
    across the five model comparisons.
    """
    rows = []
    for method_idx, method_display in enumerate(focus_method_displays):
        method_file = METHOD_FILE_BY_DISPLAY.get(method_display)
        if method_file is None:
            continue

        for domain_idx, domain in enumerate(domains):
            family_rows = []
            for model_idx, model_id in enumerate(model_ids):
                model_cfg = MODEL_CONFIGS[model_id]
                df = load_results_aligned(model_id, method_file)
                if df is None:
                    continue

                df = df.copy()
                df['therapy'] = extract_therapy_category(df)
                gt_col = f'gt_{domain}_num'
                pred_col = f'pred_{domain}_num'

                act_df = df[df['therapy'] == 'ACT']
                cbt_df = df[df['therapy'] == 'CBT']
                act_valid = act_df[act_df[gt_col].notna() & act_df[pred_col].notna()]
                cbt_valid = cbt_df[cbt_df[gt_col].notna() & cbt_df[pred_col].notna()]

                if len(act_valid) < 5 or len(cbt_valid) < 5:
                    continue

                gt_act = act_valid[gt_col].values.astype(float)
                pred_act = act_valid[pred_col].values.astype(float)
                gt_cbt = cbt_valid[gt_col].values.astype(float)
                pred_cbt = cbt_valid[pred_col].values.astype(float)
                diff_result = bootstrap_accuracy_difference(
                    gt_act,
                    pred_act,
                    gt_cbt,
                    pred_cbt,
                    n_boot=n_boot,
                    seed=RANDOM_SEED + (100 * method_idx) + (10 * domain_idx) + model_idx,
                )

                family_rows.append({
                    'Panel': 'PP LLM Accuracy',
                    'Model': model_cfg['display'],
                    'Method': method_display,
                    'Domain': domain.capitalize(),
                    'ACT_Accuracy': float(np.mean(gt_act == pred_act)),
                    'CBT_Accuracy': float(np.mean(gt_cbt == pred_cbt)),
                    'Diff_CBT_minus_ACT': diff_result['Diff_CBT_minus_ACT'],
                    'p_raw': diff_result['p_raw'],
                    'N_ACT': len(gt_act),
                    'N_CBT': len(gt_cbt),
                    'Bootstrap_N': n_boot,
                })

            if not family_rows:
                continue

            adjusted = benjamini_hochberg([row['p_raw'] for row in family_rows])
            for row, p_bh in zip(family_rows, adjusted):
                row['p_bh'] = float(p_bh)
                row['Significant_BH_0_05'] = bool(p_bh < 0.05)
                rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def normalize_output_stem(default_stem: str, tag: str) -> str:
    """Append a user-provided tag to the default output stem."""
    if not tag:
        return default_stem
    safe_tag = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in tag)
    return f'{default_stem}_{safe_tag}'


def resolve_subset(requested, mapping, label: str):
    """Resolve CLI subset values against config keys and display names."""
    if not requested:
        return list(mapping.keys())

    lookup = {}
    for key, cfg in mapping.items():
        lookup[key.lower()] = key
        display = cfg.get('display')
        if display:
            lookup[display.lower()] = key

    resolved = []
    for raw in requested:
        key = lookup.get(raw.lower())
        if key is None:
            valid = ', '.join(mapping.keys())
            raise ValueError(f'Unknown {label}: {raw}. Valid keys: {valid}')
        resolved.append(key)

    # Preserve request order while dropping duplicates.
    return list(dict.fromkeys(resolved))


def resolve_simple_subset(requested, valid_values, label: str):
    """Resolve CLI subset values for plain string lists."""
    if not requested:
        return list(valid_values)

    lookup = {value.lower(): value for value in valid_values}
    resolved = []
    for raw in requested:
        value = lookup.get(raw.lower())
        if value is None:
            valid = ', '.join(valid_values)
            raise ValueError(f'Unknown {label}: {raw}. Valid values: {valid}')
        resolved.append(value)
    return list(dict.fromkeys(resolved))


def run_analysis(model_ids=None, method_files=None, domains=None,
                 categories=None, n_boot=N_BOOTSTRAP):
    """Run CBT vs ACT comparison for all model x method x domain combos."""
    rows = []

    model_ids = model_ids or list(MODEL_CONFIGS.keys())
    method_files = method_files or list(SELECTED_METHODS)
    domains = domains or list(DOMAINS)
    categories = categories or ['ACT', 'CBT']

    for model_id in model_ids:
        model_cfg = MODEL_CONFIGS[model_id]
        for method_file in method_files:
            method_cfg = METHOD_CONFIGS[method_file]
            print(f"  RUN: {model_cfg['display']} / {method_cfg['display']}", flush=True)
            df = load_results_aligned(model_id, method_file)
            if df is None:
                print(f"  SKIP: {model_cfg['display']} / {method_cfg['display']} -- no data", flush=True)
                continue

            # Determine therapy category
            df['therapy'] = extract_therapy_category(df)
            for category in categories:
                sub = df[df['therapy'] == category]
                if len(sub) == 0:
                    continue

                for domain in domains:
                    gt_col = f'gt_{domain}_num'
                    pred_col = f'pred_{domain}_num'
                    if gt_col not in sub.columns or pred_col not in sub.columns:
                        continue

                    valid = sub[gt_col].notna() & sub[pred_col].notna()
                    gt = sub.loc[valid, gt_col].values.astype(float)
                    pred = sub.loc[valid, pred_col].values.astype(float)
                    rids = (sub.loc[valid, 'response_id'].values
                            if 'response_id' in sub.columns else None)

                    if len(gt) < 5:
                        continue

                    metric_results = bootstrap_all_metrics(gt, pred, rids, n_boot=n_boot)
                    for metric_name in METRICS_TO_COMPUTE:
                        result = metric_results[metric_name]
                        rows.append({
                            'Model': model_cfg['display'],
                            'Method': method_cfg['display'],
                            'Domain': domain.capitalize(),
                            'Category': category,
                            'Metric': metric_name,
                            'Value': result['Value'],
                            'SE': result['SE'],
                            'CI_Lower': result['CI_Lower'],
                            'CI_Upper': result['CI_Upper'],
                            'Bootstrap_N': n_boot,
                            'N': len(gt),
                        })

            n_act = (df['therapy'] == 'ACT').sum()
            n_cbt = (df['therapy'] == 'CBT').sum()
            n_other = df['therapy'].isna().sum()
            print(f"  {model_cfg['display']:15s} / {method_cfg['display']:22s} -- "
                  f"ACT={n_act}, CBT={n_cbt}, other={n_other}", flush=True)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _add_significance_bracket(ax, x_left, x_right, y, height=0.008):
    """Draw a small significance bracket with a star."""
    ax.plot(
        [x_left, x_left, x_right, x_right],
        [y - height, y, y, y - height],
        color='black',
        linewidth=1.2,
        clip_on=False,
        zorder=5,
    )
    ax.text(
        (x_left + x_right) / 2,
        y + 0.003,
        '*',
        ha='center',
        va='bottom',
        fontsize=15,
        fontweight='bold',
        color='black',
        clip_on=False,
        zorder=6,
    )


def plot_cbt_vs_act(results_df, llm_significance_df=None, non_llm_accuracy_df=None,
                    non_llm_significance_df=None, figure_stem='cbt_vs_act_performance'):
    """Create paired panels for non-LLM and PP LLM accuracy."""
    right_df = results_df[
        (results_df['Metric'] == 'Accuracy') &
        (results_df['Method'].isin(['Digital Twin', 'Personalized Prompt', 'PP']))
    ].copy()
    right_df['Method'] = 'PP'

    if right_df.empty or non_llm_accuracy_df is None or non_llm_accuracy_df.empty:
        print('WARNING: Missing data for CBT vs ACT figure.')
        return

    domains = ['Content', 'Coping', 'Quitting']
    categories = ['ACT', 'CBT']
    left_x = np.arange(len(LEFT_PANEL_ORDER))
    right_models = [MODEL_CONFIGS[m]['display'] for m in MODEL_CONFIGS]
    right_x = np.arange(len(right_models))

    left_ymin = max(0.0, float(non_llm_accuracy_df['CI_Lower'].min()) - 0.04)
    left_ymax = min(1.0, float(non_llm_accuracy_df['CI_Upper'].max()) + 0.10)
    right_ymin = max(0.0, float(right_df['CI_Lower'].min()) - 0.04)
    right_ymax = min(1.0, float(right_df['CI_Upper'].max()) + 0.10)

    fig, axes = plt.subplots(
        len(domains),
        2,
        figsize=(14.5, 11.5),
        sharex=False,
        sharey=False,
    )
    axes = np.array(axes, ndmin=2).reshape(len(domains), 2)
    apply_repo_plot_style(fig, axes)

    has_any_sig = bool(
        (llm_significance_df is not None and not llm_significance_df.empty and llm_significance_df['Significant_BH_0_05'].any()) or
        (non_llm_significance_df is not None and not non_llm_significance_df.empty and non_llm_significance_df['Significant_BH_0_05'].any())
    )

    for row_idx, domain in enumerate(domains):
        left_ax = axes[row_idx, 0]
        right_ax = axes[row_idx, 1]
        left_ax.tick_params(axis='both', labelsize=13)
        right_ax.tick_params(axis='both', labelsize=13)

        left_panel = non_llm_accuracy_df[non_llm_accuracy_df['Domain'] == domain]
        right_panel = right_df[right_df['Domain'] == domain]

        for category in categories:
            left_xs, left_ys, left_low, left_high = [], [], [], []
            for position, system_name in enumerate(LEFT_PANEL_ORDER):
                row = left_panel[
                    (left_panel['System'] == system_name) &
                    (left_panel['Category'] == category)
                ]
                if len(row) == 1:
                    value = float(row['Value'].iloc[0])
                    left_xs.append(position + (-0.12 if category == 'ACT' else 0.12))
                    left_ys.append(value)
                    left_low.append(value - float(row['CI_Lower'].iloc[0]))
                    left_high.append(float(row['CI_Upper'].iloc[0]) - value)

            left_ax.errorbar(
                left_xs,
                left_ys,
                yerr=[left_low, left_high],
                fmt='o',
                linestyle='none',
                label=category if row_idx == 0 else None,
                color=THERAPY_COLORS[category],
                ecolor=THERAPY_COLORS[category],
                elinewidth=1.6,
                capsize=3.2,
                markersize=7.5,
                markeredgecolor='black',
                markeredgewidth=0.9,
                alpha=0.96,
                zorder=3,
            )

            right_xs, right_ys, right_low, right_high = [], [], [], []
            for position, model_name in enumerate(right_models):
                row = right_panel[
                    (right_panel['Model'] == model_name) &
                    (right_panel['Category'] == category)
                ]
                if len(row) == 1:
                    value = float(row['Value'].iloc[0])
                    right_xs.append(position + (-0.12 if category == 'ACT' else 0.12))
                    right_ys.append(value)
                    right_low.append(value - float(row['CI_Lower'].iloc[0]))
                    right_high.append(float(row['CI_Upper'].iloc[0]) - value)

            right_ax.errorbar(
                right_xs,
                right_ys,
                yerr=[right_low, right_high],
                fmt='o',
                linestyle='none',
                color=THERAPY_COLORS[category],
                ecolor=THERAPY_COLORS[category],
                elinewidth=1.6,
                capsize=3.2,
                markersize=7.5,
                markeredgecolor='black',
                markeredgewidth=0.9,
                alpha=0.96,
                zorder=3,
            )

        if non_llm_significance_df is not None and not non_llm_significance_df.empty:
            sig_panel = non_llm_significance_df[non_llm_significance_df['Domain'] == domain]
            for position, system_name in enumerate(LEFT_PANEL_ORDER):
                sig_row = sig_panel[sig_panel['System'] == system_name]
                if len(sig_row) != 1 or not bool(sig_row['Significant_BH_0_05'].iloc[0]):
                    continue
                pair_rows = left_panel[left_panel['System'] == system_name]
                y = float(pair_rows['CI_Upper'].max()) + 0.06
                _add_significance_bracket(left_ax, position - 0.12, position + 0.12, y, height=0.015)

        if llm_significance_df is not None and not llm_significance_df.empty:
            sig_panel = llm_significance_df[llm_significance_df['Domain'] == domain]
            for position, model_name in enumerate(right_models):
                sig_row = sig_panel[sig_panel['Model'] == model_name]
                if len(sig_row) != 1 or not bool(sig_row['Significant_BH_0_05'].iloc[0]):
                    continue
                pair_rows = right_panel[right_panel['Model'] == model_name]
                y = float(pair_rows['CI_Upper'].max()) + 0.03
                _add_significance_bracket(right_ax, position - 0.12, position + 0.12, y)

        if row_idx == 0:
            left_ax.set_title('Supervised\nBaseline Accuracy', fontsize=15, fontweight='bold')
            right_ax.set_title('PP LLM\nAccuracy', fontsize=15, fontweight='bold')

        left_ax.set_xlim(-0.55, len(LEFT_PANEL_ORDER) - 0.45)
        right_ax.set_xlim(-0.55, len(right_models) - 0.45)
        left_ax.set_ylim(left_ymin, left_ymax)
        right_ax.set_ylim(right_ymin, right_ymax)

        left_ax.set_ylabel(f'{domain}\nAccuracy', fontsize=13, fontweight='bold')
        right_ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')

        if row_idx < len(domains) - 1:
            left_ax.tick_params(axis='x', which='both', length=0, labelbottom=False)
            right_ax.tick_params(axis='x', which='both', length=0, labelbottom=False)
        else:
            left_ax.set_xticks(left_x)
            left_ax.set_xticklabels(
                [LEFT_PANEL_LABELS[name] for name in LEFT_PANEL_ORDER],
                rotation=0,
                ha='center',
                fontsize=13,
                fontweight='bold',
            )
            right_ax.set_xticks(right_x)
            right_ax.set_xticklabels(
                [SHORT_MODEL_LABELS[name] for name in right_models],
                rotation=0,
                ha='center',
                fontsize=13,
                fontweight='bold',
            )

    handles = [
        plt.Line2D([0], [0], marker='o', linestyle='none', markersize=8,
                   markerfacecolor=THERAPY_COLORS['ACT'], markeredgecolor='black', label='ACT'),
        plt.Line2D([0], [0], marker='o', linestyle='none', markersize=8,
                   markerfacecolor=THERAPY_COLORS['CBT'], markeredgecolor='black', label='CBT'),
    ]
    labels = ['ACT', 'CBT']
    if has_any_sig:
        handles.append(
            plt.Line2D([0], [0], marker='*', linestyle='none', markersize=11,
                       markerfacecolor='black', markeredgecolor='black',
                       label='BH-adjusted p < 0.05')
        )
        labels.append('BH-adjusted p < 0.05')
    fig.legend(
        handles,
        labels,
        loc='upper center',
        ncol=len(labels),
        prop={'weight': 'bold', 'size': 13},
        frameon=False,
        bbox_to_anchor=(0.5, 0.975),
    )

    fig.suptitle(
        'ACT vs CBT Message Accuracy',
        fontsize=17,
        fontweight='bold',
        y=0.992,
    )
    fig.text(
        0.5,
        0.04,
        (
            '* BH correction applied within each domain panel'
            if has_any_sig else
            'No ACT vs CBT accuracy differences survived BH correction in any panel'
        ),
        ha='center',
        va='center',
        fontsize=11,
        fontweight='bold',
    )
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.09, right=0.98, hspace=0.18, wspace=0.08)

    out_path = figures_path(figure_stem)
    save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='CBT vs ACT message comparison')
    parser.add_argument('--models', nargs='+',
                        help='Subset of model ids or display names')
    parser.add_argument('--methods', nargs='+',
                        help='Subset of method filenames or display names')
    parser.add_argument('--domains', nargs='+',
                        help='Subset of domains: content coping quitting')
    parser.add_argument('--categories', nargs='+',
                        help='Subset of categories: ACT CBT')
    parser.add_argument('--n-bootstrap', type=int, default=N_BOOTSTRAP,
                        help=f'Number of bootstrap resamples (default: {N_BOOTSTRAP})')
    parser.add_argument('--output-tag', default='',
                        help='Optional suffix to avoid overwriting canonical outputs')
    parser.add_argument('--skip-figure', action='store_true',
                        help='Skip figure generation')
    parser.add_argument('--plot-only', action='store_true',
                        help='Reload saved CSVs and regenerate the figure without rerunning bootstrap analyses')
    return parser.parse_args()


def main():
    args = parse_args()

    model_ids = resolve_subset(args.models, MODEL_CONFIGS, 'model')
    if args.methods:
        method_files = resolve_subset(args.methods, METHOD_CONFIGS, 'method')
    else:
        method_files = list(SELECTED_METHODS)
    domains = resolve_simple_subset(args.domains, DOMAINS, 'domain')
    categories = resolve_simple_subset(args.categories, ['ACT', 'CBT'], 'category')
    csv_stem = normalize_output_stem('cbt_act_comparison', args.output_tag)
    figure_stem = normalize_output_stem('cbt_vs_act_performance', args.output_tag)
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'revision', 'figures'
    )
    csv_path = os.path.join(output_dir, f'{csv_stem}.csv')
    sig_stem = normalize_output_stem('cbt_act_significance', args.output_tag)
    score_stem = normalize_output_stem('cbt_act_nonllm_accuracy_summary', args.output_tag)
    score_sig_stem = normalize_output_stem('cbt_act_nonllm_accuracy_significance', args.output_tag)
    sig_path = os.path.join(output_dir, f'{sig_stem}.csv')
    score_path = os.path.join(output_dir, f'{score_stem}.csv')
    score_sig_path = os.path.join(output_dir, f'{score_sig_stem}.csv')

    print('=' * 70, flush=True)
    print('CBT vs ACT Message Performance Comparison', flush=True)
    print('=' * 70, flush=True)

    if args.plot_only:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f'Missing cached results CSV: {csv_path}')
        print(f'Reloading cached CBT/ACT results: {csv_path}', flush=True)
        results_df = pd.read_csv(csv_path)
        results_df['Method'] = results_df['Method'].replace({
            'Digital Twin': 'PP',
            'Personalized Prompt': 'PP',
            'Hybrid RF+DT': 'Hybrid RF+PP',
        })
        llm_significance_df = pd.read_csv(sig_path) if os.path.exists(sig_path) else pd.DataFrame()
        if not llm_significance_df.empty and 'Method' in llm_significance_df.columns:
            llm_significance_df['Method'] = llm_significance_df['Method'].replace({
                'Digital Twin': 'PP',
                'Personalized Prompt': 'PP',
            })
        if not llm_significance_df.empty and 'Panel' in llm_significance_df.columns:
            llm_significance_df['Panel'] = llm_significance_df['Panel'].replace({
                'Digital-Twin LLM Accuracy': 'PP LLM Accuracy',
            })
        non_llm_accuracy_df = pd.read_csv(score_path) if os.path.exists(score_path) else pd.DataFrame()
        if (
            non_llm_accuracy_df.empty or
            not set(LEFT_PANEL_ORDER).issubset(set(non_llm_accuracy_df.get('System', [])))
        ):
            print('Recomputing supervised CBT/ACT summary from history_supervised_predictions.csv', flush=True)
            non_llm_accuracy_df = compute_non_llm_accuracy_summary(
                domains=domains,
                categories=categories,
                n_boot=args.n_bootstrap,
            )
            if not non_llm_accuracy_df.empty:
                non_llm_accuracy_df.to_csv(score_path, index=False, float_format='%.4f')
                print(f'       Supervised accuracy summary CSV: {score_path}', flush=True)

        non_llm_significance_df = pd.read_csv(score_sig_path) if os.path.exists(score_sig_path) else pd.DataFrame()
        if (
            non_llm_significance_df.empty or
            not set(LEFT_PANEL_ORDER).issubset(set(non_llm_significance_df.get('System', [])))
        ):
            non_llm_significance_df = compute_non_llm_accuracy_significance(
                domains=domains,
                n_boot=args.n_bootstrap,
            )
            if not non_llm_significance_df.empty:
                non_llm_significance_df.to_csv(score_sig_path, index=False, float_format='%.4f')
                print(f'       Supervised accuracy significance CSV: {score_sig_path}', flush=True)

        if args.skip_figure:
            print('Skipping figure generation (--skip-figure).', flush=True)
        else:
            plot_cbt_vs_act(
                results_df,
                llm_significance_df=llm_significance_df,
                non_llm_accuracy_df=non_llm_accuracy_df,
                non_llm_significance_df=non_llm_significance_df,
                figure_stem=figure_stem,
            )
        return

    # Run analysis
    print(flush=True)
    print('[1/3] Computing metrics with bootstrap CIs ...', flush=True)
    print(f'       Models: {len(model_ids)}  |  Methods: {len(method_files)}  |  '
          f'Domains: {len(domains)}  |  Categories: {len(categories)}  |  '
          f'Bootstraps: {args.n_bootstrap}', flush=True)
    results_df = run_analysis(
        model_ids=model_ids,
        method_files=method_files,
        domains=domains,
        categories=categories,
        n_boot=args.n_bootstrap,
    )
    results_df['Method'] = results_df['Method'].replace({
        'Digital Twin': 'PP',
        'Personalized Prompt': 'PP',
        'Hybrid RF+DT': 'Hybrid RF+PP',
    })

    if results_df.empty:
        print('ERROR: No results produced. Check data paths.', flush=True)
        return

    # Save CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    results_df.to_csv(csv_path, index=False, float_format='%.4f')
    print(flush=True)
    print(f'[2/3] Saved CSV: {csv_path}', flush=True)
    print(f"       Rows: {len(results_df)}  |  "
          f"Models: {results_df['Model'].nunique()}  |  "
          f"Methods: {results_df['Method'].nunique()}", flush=True)

    # Print summary table for Accuracy
    print(flush=True)
    print('--- Accuracy Summary ---', flush=True)
    acc_df = results_df[results_df['Metric'] == 'Accuracy'][
        ['Model', 'Method', 'Domain', 'Category', 'Value', 'CI_Lower', 'CI_Upper', 'Bootstrap_N', 'N']
    ].copy()
    acc_df['Value'] = acc_df['Value'].apply(lambda x: f'{x:.3f}')
    acc_df['CI'] = acc_df.apply(
        lambda r: f"[{float(r['CI_Lower']):.3f}, {float(r['CI_Upper']):.3f}]", axis=1
    )
    print(acc_df[['Model', 'Method', 'Domain', 'Category', 'Value', 'CI', 'Bootstrap_N', 'N']].to_string(
        index=False
    ), flush=True)

    significance_llm_df = compute_focus_significance(
        model_ids=model_ids,
        focus_method_displays=['PP'],
        domains=domains,
        n_boot=args.n_bootstrap,
    )
    if not significance_llm_df.empty and 'Method' in significance_llm_df.columns:
        significance_llm_df['Method'] = significance_llm_df['Method'].replace({
            'Digital Twin': 'PP',
            'Personalized Prompt': 'PP',
        })
    non_llm_accuracy_df = compute_non_llm_accuracy_summary(
        domains=domains,
        categories=categories,
        n_boot=args.n_bootstrap,
    )
    non_llm_significance_df = compute_non_llm_accuracy_significance(
        domains=domains,
        n_boot=args.n_bootstrap,
    )

    if not significance_llm_df.empty:
        significance_llm_df.to_csv(sig_path, index=False, float_format='%.4f')
        print(flush=True)
        print(f'       LLM significance CSV: {sig_path}', flush=True)

    if not non_llm_accuracy_df.empty:
        non_llm_accuracy_df.to_csv(score_path, index=False, float_format='%.4f')
        print(f'       Non-LLM accuracy summary CSV: {score_path}', flush=True)

    if not non_llm_significance_df.empty:
        non_llm_significance_df.to_csv(score_sig_path, index=False, float_format='%.4f')
        print(f'       Non-LLM accuracy significance CSV: {score_sig_path}', flush=True)

    # Plot
    if args.skip_figure:
        print(flush=True)
        print('[3/3] Skipping figure generation (--skip-figure).', flush=True)
    else:
        print(flush=True)
        print('[3/3] Creating figure ...', flush=True)
        plot_cbt_vs_act(
            results_df,
            llm_significance_df=significance_llm_df,
            non_llm_accuracy_df=non_llm_accuracy_df,
            non_llm_significance_df=non_llm_significance_df,
            figure_stem=figure_stem,
        )

    print(flush=True)
    print('Done.', flush=True)


if __name__ == '__main__':
    main()
