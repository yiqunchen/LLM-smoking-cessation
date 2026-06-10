"""
Shared utilities for all revision analysis scripts.

Provides canonical data loading, metric computation, figure saving,
and feature extraction used across all revision tasks.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score
)
from scipy.stats import kendalltau, spearmanr
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CANONICAL_DIR = os.path.join(PROJECT_ROOT, 'data_splits', 'canonical')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'revision', 'figures')

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

REVERSE_RATING_MAPS = {
    'content': {v: k for k, v in RATING_MAPS['content'].items()},
    'coping': {1: 'Not at all helpful', 2: 'Somewhat helpful', 3: 'Moderately helpful',
               4: 'Very helpful', 5: 'Extremely helpful'},
    'quitting': {1: 'Not at all helpful', 2: 'Somewhat helpful', 3: 'Moderately helpful',
                 4: 'Very helpful', 5: 'Extremely helpful'},
}

DOMAINS = ['content', 'coping', 'quitting']

DIRECTIONAL_BUCKET_MAP = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}

# Okabe-Ito colorblind-friendly palette
COLORS = {
    'GPT-4o-mini': '#0173B2',
    'GPT-5': '#DE8F05',
    'DeepSeek-R1': '#029E73',
    'Grok-4-Fast': '#CC78BC',
    'Gemini-2.5-Pro': '#CA9161',
    'Logistic Regression': '#E02020',
    'Random Forest': '#7F7F7F',
}

GRID_COLOR = '#4D4D4D'
NEUTRAL_REFERENCE_COLOR = '#4D4D4D'

MODEL_CONFIGS = {
    'gpt-4o-mini': {'dir': 'results_manuscript_gpt-4o-mini', 'display': 'GPT-4o-mini'},
    'gpt-5': {'dir': 'results_manuscript_gpt-5', 'display': 'GPT-5'},
    'deepseek_deepseek-r1-0528': {'dir': 'results_manuscript_deepseek_deepseek-r1-0528', 'display': 'DeepSeek-R1'},
    'x-ai_grok-4-fast': {'dir': 'results_manuscript_x-ai_grok-4-fast', 'display': 'Grok-4-Fast'},
    'gemini-2.5-pro': {'dir': 'results_manuscript_gemini-2.5-pro', 'display': 'Gemini-2.5-Pro'},
}

METHOD_CONFIGS = {
    'generic_llm_1_zero_shot.json': {'display': 'Zero-shot (all)', 'category': 'Generic LLM'},
    'generic_llm_2_zero_shot_select.json': {'display': 'Zero-shot (select)', 'category': 'Generic LLM'},
    'generic_llm_3_few_shot.json': {'display': 'Few-shot (all)', 'category': 'Generic LLM'},
    'generic_llm_4_few_shot_select.json': {'display': 'Few-shot (select)', 'category': 'Generic LLM'},
    'generic_llm_5_continuous.json': {'display': 'Zero-shot (w/ prob)', 'category': 'Generic LLM'},
    'digital_twin_4_cbtact_7030.json': {'display': 'PP', 'category': 'Personalized Prompt'},
    'hybrid': {'display': 'Hybrid RF+PP', 'category': 'Hybrid'},
}

DPI = 400
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
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


def apply_publication_style(fig=None, axes=None, with_grid: bool = True):
    """Apply repo-consistent white-background, bold-label styling."""
    if fig is not None:
        fig.patch.set_facecolor('white')

    if axes is None:
        return

    axes_array = np.atleast_1d(axes).ravel()
    for ax in axes_array:
        if ax is None:
            continue
        ax.set_facecolor('white')
        if with_grid:
            ax.grid(axis='y', alpha=0.12, color=GRID_COLOR,
                    linestyle='--', linewidth=0.6)
        else:
            ax.grid(False)
        ax.tick_params(width=2.0, length=6, direction='out')
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_canonical_data(split: str = '7030', split_type: str = 'participant') -> Tuple[list, list]:
    """Load canonical train/test JSON data.

    Args:
        split: e.g. '7030', '3070', '1090', '9010'
        split_type: 'participant' or 'digital_twin'

    Returns:
        (train_data, test_data) as raw lists of dicts
    """
    train_path = os.path.join(CANONICAL_DIR, f'train_{split_type}_{split}.json')
    test_path = os.path.join(CANONICAL_DIR, f'test_{split_type}_{split}.json')
    with open(train_path) as f:
        train = json.load(f)
    with open(test_path) as f:
        test = json.load(f)
    return train, test


def load_results_file(filepath: str) -> Optional[pd.DataFrame]:
    """Load a single results JSON file into a DataFrame with numeric columns."""
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        results = json.load(f)
    rows = [v for v in results.values()
            if isinstance(v, dict) and v.get('predicted_content') != 'ERROR']
    if not rows:
        return None
    df = pd.DataFrame(rows)
    for domain in DOMAINS:
        gt_col = f'ground_truth_{domain}'
        pred_col = f'predicted_{domain}'
        if gt_col in df.columns and pred_col in df.columns:
            df[f'gt_{domain}_num'] = df[gt_col].map(RATING_MAPS[domain])
            df[f'pred_{domain}_num'] = df[pred_col].map(RATING_MAPS[domain])
    return df


def load_results_aligned(model_id: str, method_file: str) -> Optional[pd.DataFrame]:
    """Load results for a given model + method, returning aligned DataFrame."""
    cfg = MODEL_CONFIGS.get(model_id)
    if cfg is None:
        return None

    if method_file == 'hybrid':
        model_safe = model_id.replace('/', '-')
        hybrid_dir = f'results_manuscript_hybrid_{model_safe}'
        hybrid_file = f'evaluation_results_{model_safe}_text-only_hybrid-rf-digital-twin.json'
        filepath = os.path.join(PROJECT_ROOT, hybrid_dir, hybrid_file)
    else:
        filepath = os.path.join(PROJECT_ROOT, cfg['dir'], method_file)

    return load_results_file(filepath)


def collect_all_results() -> pd.DataFrame:
    """Collect metrics for all models x methods x domains. Returns long-form DataFrame."""
    rows = []
    for model_id, model_cfg in MODEL_CONFIGS.items():
        for method_file, method_cfg in METHOD_CONFIGS.items():
            df = load_results_aligned(model_id, method_file)
            if df is None:
                continue
            metrics = compute_all_metrics_df(df)
            for domain, m in metrics.items():
                rows.append({
                    'Model': model_cfg['display'],
                    'Model_ID': model_id,
                    'Method': method_cfg['display'],
                    'Category': method_cfg['category'],
                    'Domain': domain.capitalize(),
                    **{k: m[k] for k in m}
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Directionality helpers
# ---------------------------------------------------------------------------

def map_directionality(values: np.ndarray) -> np.ndarray:
    as_int = values.astype(int)
    mapped = np.full(as_int.shape, fill_value=-1, dtype=int)
    for rating, bucket in DIRECTIONAL_BUCKET_MAP.items():
        mapped[as_int == rating] = bucket
    return mapped


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_all_metrics(gt: np.ndarray, pred: np.ndarray,
                        response_ids: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute full metric suite on numeric arrays.

    Returns dict with: accuracy, f1, kappa, qwk, directional_accuracy,
    directional_macro_f1, kendall_tau, spearman_rho.
    """
    if len(gt) < 2:
        return {}

    acc = accuracy_score(gt, pred)
    f1 = f1_score(gt, pred, average='macro', zero_division=0)
    kappa = cohen_kappa_score(gt, pred)
    qwk = cohen_kappa_score(gt, pred, weights='quadratic')
    tau, _ = kendalltau(gt, pred)

    dir_gt = map_directionality(gt)
    dir_pred = map_directionality(pred)
    valid_dir = (dir_gt != -1) & (dir_pred != -1)
    if np.any(valid_dir):
        directional_accuracy = np.mean(dir_gt[valid_dir] == dir_pred[valid_dir])
        directional_macro_f1 = f1_score(dir_gt[valid_dir], dir_pred[valid_dir],
                                        average='macro', zero_division=0)
    else:
        directional_accuracy = np.nan
        directional_macro_f1 = np.nan

    # Per-participant Spearman
    avg_rho = np.nan
    if response_ids is not None:
        rhos = []
        for pid in np.unique(response_ids):
            mask = response_ids == pid
            g, p = gt[mask], pred[mask]
            if len(g) > 1 and len(np.unique(g)) > 1 and len(np.unique(p)) > 1:
                rho, _ = spearmanr(g, p)
                if not np.isnan(rho):
                    rhos.append(rho)
        if rhos:
            avg_rho = float(np.mean(rhos))

    return {
        'Accuracy': acc,
        'F1': f1,
        'Kappa': kappa,
        'QWK': qwk,
        'Directional Accuracy': directional_accuracy,
        'Directional Macro-F1': directional_macro_f1,
        'Kendall_Tau': tau,
        'Spearman_Rho': avg_rho,
        'N': len(gt),
    }


def compute_all_metrics_df(df: pd.DataFrame) -> Dict[str, dict]:
    """Compute metrics for all domains from a results DataFrame."""
    results = {}
    for domain in DOMAINS:
        gt_col = f'gt_{domain}_num'
        pred_col = f'pred_{domain}_num'
        if gt_col not in df.columns or pred_col not in df.columns:
            continue
        valid = df[gt_col].notna() & df[pred_col].notna()
        gt = df.loc[valid, gt_col].values
        pred = df.loc[valid, pred_col].values
        rids = df.loc[valid, 'response_id'].values if 'response_id' in df.columns else None
        results[domain] = compute_all_metrics(gt, pred, rids)
    return results


# ---------------------------------------------------------------------------
# Feature extraction (reused from run_hybrid_rf_digital_twin.py)
# ---------------------------------------------------------------------------

def extract_features(records: List[dict]) -> pd.DataFrame:
    """Extract demographic/metadata features from data records for ML models."""
    df = pd.DataFrame(records) if not isinstance(records, pd.DataFrame) else records.copy()
    features = []
    for _, row in df.iterrows():
        metadata = row.get('metadata', {})
        if not isinstance(metadata, dict):
            metadata = {}
        feat = {}

        # Numeric features
        for col in ['age_years', 'days_smoked_past_30d', 'cigs_per_day',
                     'quit_attempts_count']:
            val = metadata.get(col)
            try:
                feat[col] = float(val) if val is not None else 0.0
            except (ValueError, TypeError):
                feat[col] = 0.0

        # Categorical features (one-hot)
        for col in ['gender_identity', 'race_ethnicity', 'is_hispanic_latino',
                     'quit_intention', 'sexual_orientation_identity', 'education_level',
                     'household_income', 'time_to_first_cig', 'household_smokers',
                     'friends_smoke_level', 'quit_attempt_past_year',
                     'smoking_status', 'quit_motivation_level', 'social_support_to_quit']:
            val = metadata.get(col)
            if val is not None and str(val) != 'nan':
                feat[f'{col}_{val}'] = 1.0

        # Ordinal-ish psychological items (encode as numeric)
        ordinal_map = {
            'Never true': 1, 'Very rarely true': 2, 'Seldom true': 3,
            'Sometimes true': 4, 'Frequently true': 5, 'Almost always true': 6,
            'Always true': 7,
        }
        for col in ['pain_blocks_valued_life', 'fear_of_feelings', 'worry_about_control',
                     'memories_block_fulfillment', 'emotions_cause_problems',
                     'others_handle_life_better', 'worry_blocks_success']:
            val = metadata.get(col)
            feat[col] = ordinal_map.get(val, 0)

        features.append(feat)

    feat_df = pd.DataFrame(features).fillna(0)
    return feat_df


def extract_demographic_features(records: List[dict]) -> pd.DataFrame:
    """Extract demographics/sociodemographics only.

    This intentionally excludes smoking behavior, quit-readiness/support, and
    psychosocial variables that are present in the broader metadata extractor.
    """
    df = pd.DataFrame(records) if not isinstance(records, pd.DataFrame) else records.copy()
    features = []
    for _, row in df.iterrows():
        metadata = row.get('metadata', {})
        if not isinstance(metadata, dict):
            metadata = {}
        feat = {}

        val = metadata.get('age_years')
        try:
            feat['age_years'] = float(val) if val is not None else 0.0
        except (ValueError, TypeError):
            feat['age_years'] = 0.0

        for col in [
            'gender_identity',
            'race_ethnicity',
            'is_hispanic_latino',
            'sexual_orientation_identity',
            'education_level',
            'household_income',
        ]:
            val = metadata.get(col)
            if val is not None and str(val) != 'nan':
                feat[f'{col}_{val}'] = 1.0

        features.append(feat)

    feat_df = pd.DataFrame(features).fillna(0)
    return feat_df


def extract_labels(records: List[dict], domain: str) -> np.ndarray:
    """Extract numeric labels for a domain from data records."""
    labels = []
    for r in records:
        ratings = r.get('ratings', {})
        if not isinstance(ratings, dict):
            labels.append(np.nan)
            continue
        text = ratings.get(domain)
        labels.append(RATING_MAPS[domain].get(text, np.nan) if text else np.nan)
    return np.array(labels, dtype=float)


def align_features_labels(X_train, X_test):
    """Align columns of train/test feature DataFrames."""
    all_cols = sorted(set(X_train.columns) | set(X_test.columns))
    for col in all_cols:
        if col not in X_train.columns:
            X_train[col] = 0
        if col not in X_test.columns:
            X_test[col] = 0
    return X_train[all_cols], X_test[all_cols]


# ---------------------------------------------------------------------------
# Figure saving
# ---------------------------------------------------------------------------

def save_figure(fig, path: str, dpi: int = DPI):
    """Save figure as PNG + PDF. `path` should be without extension."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    fig.savefig(f'{path}.png', dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    fig.savefig(f'{path}.pdf', dpi=dpi, bbox_inches='tight', format='pdf', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}.png + .pdf")


def figures_path(name: str) -> str:
    """Return full path (without extension) for a figure in revision/figures/."""
    return os.path.join(FIGURES_DIR, name)


def apply_repo_plot_style(fig, axes):
    """Apply the repo's publication style to a figure/axes collection."""
    fig.patch.set_facecolor('white')

    if isinstance(axes, np.ndarray):
        axes_iter = axes.ravel().tolist()
    elif isinstance(axes, (list, tuple)):
        axes_iter = list(axes)
    else:
        axes_iter = [axes]

    for ax in axes_iter:
        if ax is None:
            continue
        ax.set_facecolor('white')
        ax.grid(axis='y', alpha=0.12, color='#4D4D4D', linestyle='--', linewidth=0.6)
        ax.tick_params(labelsize=11, width=1.8, length=6)
        ax.xaxis.label.set_fontweight('bold')
        ax.yaxis.label.set_fontweight('bold')
        ax.title.set_fontweight('bold')
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)
