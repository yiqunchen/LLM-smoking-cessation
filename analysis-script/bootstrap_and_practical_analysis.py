"""
Bootstrap Analysis and Practical Evaluation for LLM Message Rating

This script performs:
1. Bootstrap analysis with 95% CI at the MESSAGE level (proper aggregation)
2. Top-K message agreement analysis: Do LLMs identify the same top messages as humans?
3. "Good enough" threshold analysis: Do LLMs identify messages above a quality threshold?

Key fix: Bootstrap resamples MESSAGES, not individual predictions, to account for
the fact that each message is rated by multiple participants.

Usage:
    conda activate mai-ds
    python analysis-script/bootstrap_and_practical_analysis.py
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
from scipy.stats import spearmanr, kendalltau
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

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

# Models and their result directories
MODELS = {
    'GPT-4o-mini': 'results_manuscript_gpt-4o-mini',
    'GPT-5': 'results_manuscript_gpt-5',
    'DeepSeek-R1': 'results_manuscript_deepseek_deepseek-r1-0528',
    'Grok-4-Fast': 'results_manuscript_x-ai_grok-4-fast',
    'Gemini-2.5-Pro': 'results_manuscript_gemini-2.5-pro'
}


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


def map_to_directional(value):
    """Map 1-5 rating to directional: 0=low (1,2), 1=neutral (3), 2=high (4,5)"""
    if value <= 2:
        return 0
    elif value == 3:
        return 1
    else:
        return 2


def aggregate_to_message_level(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    """
    Aggregate predictions to message level by taking mean ratings.

    Returns DataFrame with one row per message containing:
    - mean_gt: mean human rating
    - mean_pred: mean LLM prediction
    - n_ratings: number of ratings for this message
    """
    gt_col = f'gt_{domain}_num'
    pred_col = f'pred_{domain}_num'

    if gt_col not in df.columns or pred_col not in df.columns:
        return None

    valid_mask = df[gt_col].notna() & df[pred_col].notna()
    valid_df = df[valid_mask].copy()

    # Aggregate by message
    message_df = valid_df.groupby('input_message').agg({
        gt_col: ['mean', 'count'],
        pred_col: 'mean'
    }).reset_index()

    message_df.columns = ['message', 'mean_gt', 'n_ratings', 'mean_pred']

    # Round to nearest integer for classification metrics
    message_df['gt_rounded'] = message_df['mean_gt'].round().astype(int)
    message_df['pred_rounded'] = message_df['mean_pred'].round().astype(int)

    # Directional buckets (low/neutral/high)
    message_df['gt_directional'] = message_df['gt_rounded'].apply(map_to_directional)
    message_df['pred_directional'] = message_df['pred_rounded'].apply(map_to_directional)

    return message_df


def bootstrap_message_level_metrics(message_df: pd.DataFrame, n_bootstrap: int = 1000,
                                    confidence_level: float = 0.95) -> Dict[str, Dict[str, float]]:
    """
    Calculate bootstrap confidence intervals at the MESSAGE level.

    This resamples MESSAGES (not individual predictions) to properly account
    for the fact that each message may be rated by multiple participants.

    Returns dict with {metric: {point_estimate, ci_lower, ci_upper, std}}
    """
    n_messages = len(message_df)

    gt = message_df['gt_rounded'].values
    pred = message_df['pred_rounded'].values
    gt_dir = message_df['gt_directional'].values
    pred_dir = message_df['pred_directional'].values
    gt_mean = message_df['mean_gt'].values
    pred_mean = message_df['mean_pred'].values

    # Point estimates
    accuracy_point = accuracy_score(gt, pred)
    dir_accuracy_point = accuracy_score(gt_dir, pred_dir)
    f1_point = f1_score(gt, pred, average='macro', zero_division=0)
    dir_f1_point = f1_score(gt_dir, pred_dir, average='macro', zero_division=0)
    kappa_point = cohen_kappa_score(gt, pred)

    # Rank correlations on continuous values
    spearman_point, _ = spearmanr(gt_mean, pred_mean)
    kendall_point, _ = kendalltau(gt_mean, pred_mean)

    # Bootstrap by resampling messages
    accuracy_boots = []
    dir_accuracy_boots = []
    f1_boots = []
    dir_f1_boots = []
    kappa_boots = []
    spearman_boots = []
    kendall_boots = []

    np.random.seed(42)
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_messages, size=n_messages, replace=True)
        gt_boot = gt[indices]
        pred_boot = pred[indices]
        gt_dir_boot = gt_dir[indices]
        pred_dir_boot = pred_dir[indices]
        gt_mean_boot = gt_mean[indices]
        pred_mean_boot = pred_mean[indices]

        accuracy_boots.append(accuracy_score(gt_boot, pred_boot))
        dir_accuracy_boots.append(accuracy_score(gt_dir_boot, pred_dir_boot))
        f1_boots.append(f1_score(gt_boot, pred_boot, average='macro', zero_division=0))
        dir_f1_boots.append(f1_score(gt_dir_boot, pred_dir_boot, average='macro', zero_division=0))
        kappa_boots.append(cohen_kappa_score(gt_boot, pred_boot))

        # Rank correlations
        if np.std(gt_mean_boot) > 0 and np.std(pred_mean_boot) > 0:
            rho, _ = spearmanr(gt_mean_boot, pred_mean_boot)
            tau, _ = kendalltau(gt_mean_boot, pred_mean_boot)
            if not np.isnan(rho):
                spearman_boots.append(rho)
            if not np.isnan(tau):
                kendall_boots.append(tau)

    alpha = 1 - confidence_level

    results = {}
    for metric_name, point_est, boots in [
        ('accuracy', accuracy_point, accuracy_boots),
        ('directional_accuracy', dir_accuracy_point, dir_accuracy_boots),
        ('f1_macro', f1_point, f1_boots),
        ('directional_f1', dir_f1_point, dir_f1_boots),
        ('kappa', kappa_point, kappa_boots),
        ('spearman_rho', spearman_point, spearman_boots),
        ('kendall_tau', kendall_point, kendall_boots)
    ]:
        boots = np.array(boots)
        if len(boots) > 0:
            ci_lower = np.percentile(boots, (alpha/2) * 100)
            ci_upper = np.percentile(boots, (1 - alpha/2) * 100)
            results[metric_name] = {
                'point_estimate': point_est,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'std': np.std(boots)
            }

    return results


def run_bootstrap_analysis(base_dir: str) -> pd.DataFrame:
    """Run MESSAGE-LEVEL bootstrap analysis for all top models and save results."""
    print("=" * 80)
    print("BOOTSTRAP ANALYSIS (MESSAGE-LEVEL): Accuracy, F1, Kappa, Correlations")
    print("=" * 80)
    print("\nNOTE: Bootstrap resamples MESSAGES (n~107), not individual predictions.")
    print("      Metrics computed on mean ratings per message.\n")

    all_results = []

    for model_name, model_dir in MODELS.items():
        # Use Digital Twin CBT/ACT 70/30 results
        result_path = os.path.join(base_dir, model_dir, 'digital_twin_4_cbtact_7030.json')
        df = load_results(result_path)

        if df is None:
            print(f"  Skipping {model_name}: No digital twin results found")
            continue

        print(f"\n{model_name}:")

        for domain in DOMAINS:
            message_df = aggregate_to_message_level(df, domain)

            if message_df is None or len(message_df) < 10:
                continue

            n_messages = len(message_df)
            metrics = bootstrap_message_level_metrics(message_df, n_bootstrap=1000)

            print(f"  {domain.capitalize()} (n={n_messages} messages):")
            for metric_name, vals in metrics.items():
                ci_str = f"[{vals['ci_lower']:.3f}, {vals['ci_upper']:.3f}]"
                print(f"    {metric_name}: {vals['point_estimate']:.3f} 95% CI {ci_str}")

                all_results.append({
                    'Model': model_name,
                    'Domain': domain.capitalize(),
                    'Metric': metric_name,
                    'Point Estimate': vals['point_estimate'],
                    'CI Lower (95%)': vals['ci_lower'],
                    'CI Upper (95%)': vals['ci_upper'],
                    'Std': vals['std'],
                    'N Messages': n_messages
                })

    results_df = pd.DataFrame(all_results)
    return results_df


def compute_message_rankings(df: pd.DataFrame, domain: str,
                             score_col: str, message_col: str = 'input_message') -> pd.DataFrame:
    """
    Compute message rankings by averaging scores across all participants.

    Returns DataFrame with message_text, mean_score, and rank.
    """
    if score_col not in df.columns:
        return None

    message_scores = df.groupby(message_col)[score_col].agg(['mean', 'count']).reset_index()
    message_scores.columns = [message_col, 'mean_score', 'n_ratings']
    message_scores = message_scores.sort_values('mean_score', ascending=False)
    message_scores['rank'] = range(1, len(message_scores) + 1)

    return message_scores


def bootstrap_top_k_agreement(human_ranking: pd.DataFrame, llm_ranking: pd.DataFrame,
                               k: int, n_bootstrap: int = 1000,
                               message_col: str = 'input_message') -> Dict:
    """
    Bootstrap confidence interval for Top-K overlap.

    Resamples messages and recomputes rankings for each bootstrap sample.
    """
    n_messages = len(human_ranking)
    messages = human_ranking[message_col].values
    human_scores = human_ranking.set_index(message_col)['mean_score']
    llm_scores = llm_ranking.set_index(message_col)['mean_score']

    # Point estimate
    human_top_k = set(human_ranking.head(k)[message_col].tolist())
    llm_top_k = set(llm_ranking.head(k)[message_col].tolist())
    overlap_point = len(human_top_k & llm_top_k)

    # Bootstrap
    overlaps = []
    np.random.seed(42)
    for _ in range(n_bootstrap):
        # Resample messages
        boot_messages = np.random.choice(messages, size=n_messages, replace=True)

        # Get scores for bootstrap sample
        boot_human = pd.Series([human_scores[m] for m in boot_messages], index=boot_messages)
        boot_llm = pd.Series([llm_scores[m] for m in boot_messages], index=boot_messages)

        # Rank and find top-K
        human_top_k_boot = set(boot_human.nlargest(k).index)
        llm_top_k_boot = set(boot_llm.nlargest(k).index)

        overlaps.append(len(human_top_k_boot & llm_top_k_boot))

    overlaps = np.array(overlaps)

    return {
        'overlap_count': overlap_point,
        'overlap_pct': overlap_point / k * 100,
        'ci_lower': np.percentile(overlaps, 2.5),
        'ci_upper': np.percentile(overlaps, 97.5),
        'std': np.std(overlaps)
    }


def compute_top_k_agreement(human_ranking: pd.DataFrame, llm_ranking: pd.DataFrame,
                            k_values: List[int], message_col: str = 'input_message') -> Dict[int, Dict]:
    """
    Compute agreement between human and LLM top-K message selections.
    """
    results = {}

    for k in k_values:
        human_top_k = set(human_ranking.head(k)[message_col].tolist())
        llm_top_k = set(llm_ranking.head(k)[message_col].tolist())

        overlap = len(human_top_k & llm_top_k)
        union = len(human_top_k | llm_top_k)
        jaccard = overlap / union if union > 0 else 0

        results[k] = {
            'overlap_count': overlap,
            'overlap_pct': overlap / k * 100,
            'jaccard': jaccard,
            'human_top_k': human_top_k,
            'llm_top_k': llm_top_k
        }

    return results


def bootstrap_threshold_agreement(human_df: pd.DataFrame, llm_df: pd.DataFrame,
                                   threshold: float, n_bootstrap: int = 1000,
                                   message_col: str = 'input_message') -> Dict:
    """
    Bootstrap confidence interval for threshold agreement metrics.
    """
    messages = human_df[message_col].values
    n_messages = len(messages)
    human_scores = human_df.set_index(message_col)['mean_score']
    llm_scores = llm_df.set_index(message_col)['mean_score']

    # Point estimates
    human_good = set(human_df[human_df['mean_score'] >= threshold][message_col].tolist())
    llm_good = set(llm_df[llm_df['mean_score'] >= threshold][message_col].tolist())

    tp = len(human_good & llm_good)
    precision_point = tp / len(llm_good) if len(llm_good) > 0 else 0
    recall_point = tp / len(human_good) if len(human_good) > 0 else 0
    f1_point = 2 * precision_point * recall_point / (precision_point + recall_point) if (precision_point + recall_point) > 0 else 0

    # Bootstrap
    precisions, recalls, f1s = [], [], []
    np.random.seed(42)
    for _ in range(n_bootstrap):
        boot_messages = np.random.choice(messages, size=n_messages, replace=True)

        boot_human = pd.Series([human_scores[m] for m in boot_messages], index=boot_messages)
        boot_llm = pd.Series([llm_scores[m] for m in boot_messages], index=boot_messages)

        human_good_boot = set(boot_human[boot_human >= threshold].index)
        llm_good_boot = set(boot_llm[boot_llm >= threshold].index)

        tp_boot = len(human_good_boot & llm_good_boot)
        p = tp_boot / len(llm_good_boot) if len(llm_good_boot) > 0 else 0
        r = tp_boot / len(human_good_boot) if len(human_good_boot) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    return {
        'n_human_good': len(human_good),
        'n_llm_good': len(llm_good),
        'true_positives': tp,
        'precision': precision_point,
        'precision_ci': (np.percentile(precisions, 2.5), np.percentile(precisions, 97.5)),
        'recall': recall_point,
        'recall_ci': (np.percentile(recalls, 2.5), np.percentile(recalls, 97.5)),
        'f1': f1_point,
        'f1_ci': (np.percentile(f1s, 2.5), np.percentile(f1s, 97.5))
    }


def compute_threshold_agreement(human_df: pd.DataFrame, llm_df: pd.DataFrame,
                                threshold: float, message_col: str = 'input_message') -> Dict:
    """
    Compute agreement on messages above a "good enough" threshold.
    """
    human_good = set(human_df[human_df['mean_score'] >= threshold][message_col].tolist())
    llm_good = set(llm_df[llm_df['mean_score'] >= threshold][message_col].tolist())

    if len(llm_good) == 0:
        precision = 0.0
    else:
        precision = len(human_good & llm_good) / len(llm_good)

    if len(human_good) == 0:
        recall = 0.0
    else:
        recall = len(human_good & llm_good) / len(human_good)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        'n_human_good': len(human_good),
        'n_llm_good': len(llm_good),
        'true_positives': len(human_good & llm_good),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'human_good_messages': human_good,
        'llm_good_messages': llm_good
    }


def run_practical_analysis(base_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run practical analyses with bootstrap CIs:
    1. Top-K message agreement
    2. Good-enough threshold agreement
    """
    print("\n" + "=" * 80)
    print("PRACTICAL ANALYSIS 1: Top-K Message Agreement (with Bootstrap CI)")
    print("=" * 80)
    print("\nDo LLMs identify the same TOP messages as human participants?")
    print()

    k_values = [5, 10, 15, 20, 25]
    thresholds = [3.5, 4.0, 4.5]

    top_k_results = []
    threshold_results = []

    for model_name, model_dir in MODELS.items():
        result_path = os.path.join(base_dir, model_dir, 'digital_twin_4_cbtact_7030.json')
        df = load_results(result_path)

        if df is None:
            continue

        print(f"\n{model_name}:")

        for domain in DOMAINS:
            gt_col = f'gt_{domain}_num'
            pred_col = f'pred_{domain}_num'

            if gt_col not in df.columns or pred_col not in df.columns:
                continue

            valid_mask = df[gt_col].notna() & df[pred_col].notna()
            valid_df = df[valid_mask].copy()

            human_ranking = compute_message_rankings(valid_df, domain, gt_col)
            llm_ranking = compute_message_rankings(valid_df, domain, pred_col)

            if human_ranking is None or llm_ranking is None:
                continue

            n_messages = len(human_ranking)

            # Top-K agreement with bootstrap
            print(f"  {domain.capitalize()} ({n_messages} messages):")
            for k in k_values:
                if k <= n_messages:
                    boot_result = bootstrap_top_k_agreement(human_ranking, llm_ranking, k)
                    print(f"    Top-{k}: {boot_result['overlap_count']}/{k} "
                          f"({boot_result['overlap_pct']:.1f}%) "
                          f"95% CI [{boot_result['ci_lower']:.0f}, {boot_result['ci_upper']:.0f}]")

                    top_k_results.append({
                        'Model': model_name,
                        'Domain': domain.capitalize(),
                        'K': k,
                        'Overlap Count': boot_result['overlap_count'],
                        'Overlap %': boot_result['overlap_pct'],
                        'CI Lower': boot_result['ci_lower'],
                        'CI Upper': boot_result['ci_upper'],
                        'N Messages': n_messages
                    })

            # Threshold agreement with bootstrap
            for threshold in thresholds:
                thresh_result = bootstrap_threshold_agreement(human_ranking, llm_ranking, threshold)

                threshold_results.append({
                    'Model': model_name,
                    'Domain': domain.capitalize(),
                    'Threshold': threshold,
                    'N Human Good': thresh_result['n_human_good'],
                    'N LLM Good': thresh_result['n_llm_good'],
                    'True Positives': thresh_result['true_positives'],
                    'Precision': thresh_result['precision'],
                    'Precision CI Lower': thresh_result['precision_ci'][0],
                    'Precision CI Upper': thresh_result['precision_ci'][1],
                    'Recall': thresh_result['recall'],
                    'Recall CI Lower': thresh_result['recall_ci'][0],
                    'Recall CI Upper': thresh_result['recall_ci'][1],
                    'F1': thresh_result['f1'],
                    'F1 CI Lower': thresh_result['f1_ci'][0],
                    'F1 CI Upper': thresh_result['f1_ci'][1],
                    'N Messages': n_messages
                })

    print("\n" + "=" * 80)
    print("PRACTICAL ANALYSIS 2: 'Good Enough' Threshold Agreement (with Bootstrap CI)")
    print("=" * 80)

    for model_name in MODELS.keys():
        model_results = [r for r in threshold_results if r['Model'] == model_name]
        if not model_results:
            continue

        print(f"\n{model_name}:")
        for domain in DOMAINS:
            domain_results = [r for r in model_results if r['Domain'] == domain.capitalize()]
            if not domain_results:
                continue

            print(f"  {domain.capitalize()}:")
            for r in domain_results:
                print(f"    Threshold >= {r['Threshold']}: "
                      f"P={r['Precision']:.3f} [{r['Precision CI Lower']:.3f}, {r['Precision CI Upper']:.3f}], "
                      f"R={r['Recall']:.3f} [{r['Recall CI Lower']:.3f}, {r['Recall CI Upper']:.3f}], "
                      f"F1={r['F1']:.3f} [{r['F1 CI Lower']:.3f}, {r['F1 CI Upper']:.3f}]")

    top_k_df = pd.DataFrame(top_k_results)
    threshold_df = pd.DataFrame(threshold_results)

    return top_k_df, threshold_df


def compute_rank_correlation_at_message_level(base_dir: str) -> pd.DataFrame:
    """
    Compute Spearman and Kendall rank correlations at the message level.
    (This is now also included in the main bootstrap analysis)
    """
    print("\n" + "=" * 80)
    print("MESSAGE-LEVEL RANK CORRELATIONS")
    print("=" * 80)

    results = []

    for model_name, model_dir in MODELS.items():
        result_path = os.path.join(base_dir, model_dir, 'digital_twin_4_cbtact_7030.json')
        df = load_results(result_path)

        if df is None:
            continue

        print(f"\n{model_name}:")

        for domain in DOMAINS:
            message_df = aggregate_to_message_level(df, domain)

            if message_df is None or len(message_df) < 3:
                continue

            spearman_rho, spearman_p = spearmanr(message_df['mean_gt'], message_df['mean_pred'])
            kendall_tau, kendall_p = kendalltau(message_df['mean_gt'], message_df['mean_pred'])

            print(f"  {domain.capitalize()}: Spearman rho={spearman_rho:.3f} (p={spearman_p:.4f}), "
                  f"Kendall tau={kendall_tau:.3f} (p={kendall_p:.4f}), n_messages={len(message_df)}")

            results.append({
                'Model': model_name,
                'Domain': domain.capitalize(),
                'Spearman Rho': spearman_rho,
                'Spearman p-value': spearman_p,
                'Kendall Tau': kendall_tau,
                'Kendall p-value': kendall_p,
                'N Messages': len(message_df)
            })

    return pd.DataFrame(results)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'figures')
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("LLM MESSAGE RATING: Bootstrap and Practical Analysis")
    print("(MESSAGE-LEVEL BOOTSTRAP - Correct Aggregation)")
    print("=" * 80 + "\n")

    # 1. Bootstrap Analysis (MESSAGE LEVEL)
    bootstrap_df = run_bootstrap_analysis(base_dir)
    bootstrap_path = os.path.join(output_dir, 'bootstrap_confidence_intervals.csv')
    bootstrap_df.to_csv(bootstrap_path, index=False)
    print(f"\nBootstrap results saved to: {bootstrap_path}")

    # 2. Practical Analyses with Bootstrap CI
    top_k_df, threshold_df = run_practical_analysis(base_dir)

    top_k_path = os.path.join(output_dir, 'top_k_agreement.csv')
    top_k_df.to_csv(top_k_path, index=False)
    print(f"\nTop-K agreement results saved to: {top_k_path}")

    threshold_path = os.path.join(output_dir, 'threshold_agreement.csv')
    threshold_df.to_csv(threshold_path, index=False)
    print(f"Threshold agreement results saved to: {threshold_path}")

    # 3. Message-level rank correlations
    rank_corr_df = compute_rank_correlation_at_message_level(base_dir)
    rank_corr_path = os.path.join(output_dir, 'message_level_rank_correlations.csv')
    rank_corr_df.to_csv(rank_corr_path, index=False)
    print(f"Message-level rank correlations saved to: {rank_corr_path}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY (MESSAGE-LEVEL METRICS)")
    print("=" * 80)

    print("\n1. Bootstrap 95% CI Summary:")
    for metric in ['accuracy', 'f1_macro', 'kappa', 'spearman_rho', 'kendall_tau']:
        metric_df = bootstrap_df[bootstrap_df['Metric'] == metric]
        if len(metric_df) > 0:
            print(f"\n   {metric}:")
            for _, row in metric_df.iterrows():
                print(f"     {row['Model']} - {row['Domain']}: "
                      f"{row['Point Estimate']:.3f} [{row['CI Lower (95%)']:.3f}, {row['CI Upper (95%)']: .3f}]")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
