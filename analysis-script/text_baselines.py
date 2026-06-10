#!/usr/bin/env python3
"""
Text-enhanced baselines for revision.

Addresses reviewer R2/R3/R4 concerns about unfair baseline comparison by
implementing text-aware baselines (TF-IDF, embeddings) alongside demographic-only
and simple mean baselines.

Baselines:
  1. Participant Mean         - global train mean (participant-split: no overlap)
  2. Message Mean             - per-message average from training data
  3. TF-IDF + LR             - TF-IDF on message text -> Logistic Regression
  4. TF-IDF + RF             - TF-IDF on message text -> Random Forest
  5. Demographics-only LR    - demographic features -> LR
  6. Demographics-only RF    - demographic features -> RF
  7. Embedding + LR          - 1024-dim embeddings -> LR
  8. Embedding + RF          - 1024-dim embeddings -> RF
  9. TF-IDF + Demographics + LR   - concatenated features -> LR
 10. TF-IDF + Demographics + RF   - concatenated features -> RF
 11. Embedding + Demographics + LR - concatenated features -> LR
 12. Embedding + Demographics + RF - concatenated features -> RF

Outputs:
  - revision/figures/text_baseline_results.csv
  - revision/figures/all_baselines_comparison.png/pdf
  - revision/figures/baseline_feature_ablation.png/pdf

Usage:
  uv run python analysis-script/text_baselines.py
"""

import os
import sys
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.sparse import hstack as sparse_hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------------------------
# Import revision utilities
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from revision_utils import (
    load_canonical_data,
    extract_features,
    extract_labels,
    align_features_labels,
    compute_all_metrics,
    collect_all_results,
    save_figure,
    figures_path,
    DOMAINS,
    RATING_MAPS,
    COLORS,
    PROJECT_ROOT,
)

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
EMBEDDING_PATH = os.path.join(PROJECT_ROOT, 'archive', 'data', 'message_embeddings.pkl')


# ===================================================================
# Helper functions
# ===================================================================

def _clip_round(arr: np.ndarray) -> np.ndarray:
    """Round predictions to nearest integer and clip to [1, 5]."""
    return np.clip(np.round(arr).astype(int), 1, 5)


def _get_response_ids(records):
    """Extract response_id array from record list."""
    return np.array([r['response_id'] for r in records])


def _load_embeddings():
    """Load the precomputed embeddings pkl and build a lookup dict."""
    with open(EMBEDDING_PATH, 'rb') as f:
        emb_data = pickle.load(f)
    emb_lookup = {}
    for i, (rid, msg) in enumerate(zip(emb_data['response_ids'],
                                        emb_data['input_messages'])):
        emb_lookup[(rid, msg)] = i
    return emb_data['embeddings'], emb_lookup


def _match_embeddings(records, emb_matrix, emb_lookup):
    """Match canonical records to embedding rows.

    Returns:
        emb_array: ndarray (n_matched, 1024)
        valid_indices: list of indices into records that were matched
    """
    indices_emb = []
    valid_indices = []
    for idx, item in enumerate(records):
        key = (item['response_id'], item['input_message'])
        if key in emb_lookup:
            indices_emb.append(emb_lookup[key])
            valid_indices.append(idx)
    emb_array = emb_matrix[indices_emb]
    return emb_array, valid_indices


# ===================================================================
# Baseline implementations
# ===================================================================

def baseline_participant_mean(train_data, test_data, domain):
    """Global train mean baseline.

    Since this is a participant-split, no test participants appear in
    training data.  Every test item is predicted as the global training mean.
    """
    y_train = extract_labels(train_data, domain)
    y_test = extract_labels(test_data, domain)

    valid_train = ~np.isnan(y_train)
    valid_test = ~np.isnan(y_test)

    global_mean = np.nanmean(y_train[valid_train])
    pred = np.full(len(y_test), global_mean)
    pred = _clip_round(pred)

    gt = y_test[valid_test].astype(int)
    pred = pred[valid_test]
    rids = _get_response_ids(test_data)[valid_test]
    return gt, pred, rids


def baseline_message_mean(train_data, test_data, domain):
    """Per-message average from training data.

    Many messages overlap between train/test because the split is by
    participant, not by message.  For each test item we look up the
    average rating that other participants gave the same message in
    the training set.  If a message has no training ratings, fall back
    to the global training mean.
    """
    y_train = extract_labels(train_data, domain)
    y_test = extract_labels(test_data, domain)

    # Build message -> list of ratings from training data
    msg_ratings = {}
    for item, label in zip(train_data, y_train):
        if np.isnan(label):
            continue
        msg = item['input_message']
        msg_ratings.setdefault(msg, []).append(label)

    msg_means = {msg: np.mean(vals) for msg, vals in msg_ratings.items()}
    global_mean = np.nanmean(y_train[~np.isnan(y_train)])

    valid_test = ~np.isnan(y_test)
    pred = []
    for item in test_data:
        msg = item['input_message']
        pred.append(msg_means.get(msg, global_mean))
    pred = _clip_round(np.array(pred))

    gt = y_test[valid_test].astype(int)
    pred = pred[valid_test]
    rids = _get_response_ids(test_data)[valid_test]
    return gt, pred, rids


def baseline_tfidf_model(train_data, test_data, domain, model_type='lr'):
    """TF-IDF on message text -> classifier (LR or RF)."""
    y_train = extract_labels(train_data, domain)
    y_test = extract_labels(test_data, domain)

    train_msgs = [item['input_message'] for item in train_data]
    test_msgs = [item['input_message'] for item in test_data]

    valid_train = ~np.isnan(y_train)
    valid_test = ~np.isnan(y_test)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(np.array(train_msgs)[valid_train])
    X_test_tfidf = vectorizer.transform(np.array(test_msgs)[valid_test])

    y_tr = y_train[valid_train].astype(int)

    if model_type == 'lr':
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    else:
        clf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                     random_state=RANDOM_STATE)

    clf.fit(X_train_tfidf, y_tr)
    pred = clf.predict(X_test_tfidf)
    pred = _clip_round(pred.astype(float))

    gt = y_test[valid_test].astype(int)
    rids = _get_response_ids(test_data)[valid_test]
    return gt, pred, rids


def baseline_demographics_model(train_data, test_data, domain, model_type='lr'):
    """Demographics-only features -> classifier (LR or RF)."""
    y_train = extract_labels(train_data, domain)
    y_test = extract_labels(test_data, domain)

    X_train_demo = extract_features(train_data)
    X_test_demo = extract_features(test_data)
    X_train_demo, X_test_demo = align_features_labels(X_train_demo, X_test_demo)

    valid_train = ~np.isnan(y_train)
    valid_test = ~np.isnan(y_test)

    X_tr = X_train_demo.values[valid_train]
    X_te = X_test_demo.values[valid_test]
    y_tr = y_train[valid_train].astype(int)

    if model_type == 'lr':
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    else:
        clf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                     random_state=RANDOM_STATE)

    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    pred = _clip_round(pred.astype(float))

    gt = y_test[valid_test].astype(int)
    rids = _get_response_ids(test_data)[valid_test]
    return gt, pred, rids


def baseline_embedding_model(train_data, test_data, domain, emb_matrix, emb_lookup,
                              model_type='lr'):
    """Embedding features (1024-d) -> classifier (LR or RF)."""
    y_train = extract_labels(train_data, domain)
    y_test = extract_labels(test_data, domain)

    emb_train, valid_train_idx = _match_embeddings(train_data, emb_matrix, emb_lookup)
    emb_test, valid_test_idx = _match_embeddings(test_data, emb_matrix, emb_lookup)

    # Filter to items with both valid embeddings and valid labels
    y_tr_all = y_train[valid_train_idx]
    y_te_all = y_test[valid_test_idx]

    valid_label_train = ~np.isnan(y_tr_all)
    valid_label_test = ~np.isnan(y_te_all)

    X_tr = emb_train[valid_label_train]
    X_te = emb_test[valid_label_test]
    y_tr = y_tr_all[valid_label_train].astype(int)
    gt = y_te_all[valid_label_test].astype(int)

    # Get response_ids for the valid test items
    test_rids = _get_response_ids(test_data)
    rids = test_rids[np.array(valid_test_idx)][valid_label_test]

    if model_type == 'lr':
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    else:
        clf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                     random_state=RANDOM_STATE)

    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    pred = _clip_round(pred.astype(float))
    return gt, pred, rids


def baseline_tfidf_demographics_model(train_data, test_data, domain, model_type='lr'):
    """TF-IDF + Demographics concatenated -> classifier (LR or RF)."""
    y_train = extract_labels(train_data, domain)
    y_test = extract_labels(test_data, domain)

    train_msgs = [item['input_message'] for item in train_data]
    test_msgs = [item['input_message'] for item in test_data]

    X_train_demo = extract_features(train_data)
    X_test_demo = extract_features(test_data)
    X_train_demo, X_test_demo = align_features_labels(X_train_demo, X_test_demo)

    valid_train = ~np.isnan(y_train)
    valid_test = ~np.isnan(y_test)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(np.array(train_msgs)[valid_train])
    X_test_tfidf = vectorizer.transform(np.array(test_msgs)[valid_test])

    # Concatenate sparse TF-IDF with dense demographics
    X_tr_demo = csr_matrix(X_train_demo.values[valid_train])
    X_te_demo = csr_matrix(X_test_demo.values[valid_test])
    X_tr = sparse_hstack([X_train_tfidf, X_tr_demo])
    X_te = sparse_hstack([X_test_tfidf, X_te_demo])

    y_tr = y_train[valid_train].astype(int)

    if model_type == 'lr':
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    else:
        clf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                     random_state=RANDOM_STATE)

    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    pred = _clip_round(pred.astype(float))

    gt = y_test[valid_test].astype(int)
    rids = _get_response_ids(test_data)[valid_test]
    return gt, pred, rids


def baseline_embedding_demographics_model(train_data, test_data, domain,
                                           emb_matrix, emb_lookup, model_type='lr'):
    """Embedding + Demographics concatenated -> classifier (LR or RF)."""
    y_train = extract_labels(train_data, domain)
    y_test = extract_labels(test_data, domain)

    emb_train, valid_train_idx = _match_embeddings(train_data, emb_matrix, emb_lookup)
    emb_test, valid_test_idx = _match_embeddings(test_data, emb_matrix, emb_lookup)

    # Demographics for matched items only
    train_matched = [train_data[i] for i in valid_train_idx]
    test_matched = [test_data[i] for i in valid_test_idx]
    X_train_demo = extract_features(train_matched)
    X_test_demo = extract_features(test_matched)
    X_train_demo, X_test_demo = align_features_labels(X_train_demo, X_test_demo)

    y_tr_all = y_train[valid_train_idx]
    y_te_all = y_test[valid_test_idx]

    valid_label_train = ~np.isnan(y_tr_all)
    valid_label_test = ~np.isnan(y_te_all)

    # Concatenate embedding + demographics
    X_tr = np.hstack([emb_train[valid_label_train],
                      X_train_demo.values[valid_label_train]])
    X_te = np.hstack([emb_test[valid_label_test],
                      X_test_demo.values[valid_label_test]])
    y_tr = y_tr_all[valid_label_train].astype(int)
    gt = y_te_all[valid_label_test].astype(int)

    test_rids = _get_response_ids(test_data)
    rids = test_rids[np.array(valid_test_idx)][valid_label_test]

    if model_type == 'lr':
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    else:
        clf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                     random_state=RANDOM_STATE)

    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    pred = _clip_round(pred.astype(float))
    return gt, pred, rids


# ===================================================================
# Run all baselines
# ===================================================================

def run_all_baselines():
    """Run all 12 baselines across all 3 domains and collect results."""
    print("Loading canonical data...")
    train_data, test_data = load_canonical_data('7030', 'participant')
    print(f"  Train: {len(train_data)} items, Test: {len(test_data)} items")

    print("Loading embeddings...")
    emb_matrix, emb_lookup = _load_embeddings()
    print(f"  Embeddings: {emb_matrix.shape}")

    # Define baselines -- use default-argument binding to capture loop variables
    baselines = [
        ('Participant Mean',
         lambda td, ted, dom: baseline_participant_mean(td, ted, dom)),
        ('Message Mean',
         lambda td, ted, dom: baseline_message_mean(td, ted, dom)),
        ('TF-IDF + LR',
         lambda td, ted, dom: baseline_tfidf_model(td, ted, dom, 'lr')),
        ('TF-IDF + RF',
         lambda td, ted, dom: baseline_tfidf_model(td, ted, dom, 'rf')),
        ('Demographics LR',
         lambda td, ted, dom: baseline_demographics_model(td, ted, dom, 'lr')),
        ('Demographics RF',
         lambda td, ted, dom: baseline_demographics_model(td, ted, dom, 'rf')),
        ('Embedding + LR',
         lambda td, ted, dom, em=emb_matrix, el=emb_lookup:
             baseline_embedding_model(td, ted, dom, em, el, 'lr')),
        ('Embedding + RF',
         lambda td, ted, dom, em=emb_matrix, el=emb_lookup:
             baseline_embedding_model(td, ted, dom, em, el, 'rf')),
        ('TF-IDF + Demo + LR',
         lambda td, ted, dom: baseline_tfidf_demographics_model(td, ted, dom, 'lr')),
        ('TF-IDF + Demo + RF',
         lambda td, ted, dom: baseline_tfidf_demographics_model(td, ted, dom, 'rf')),
        ('Embedding + Demo + LR',
         lambda td, ted, dom, em=emb_matrix, el=emb_lookup:
             baseline_embedding_demographics_model(td, ted, dom, em, el, 'lr')),
        ('Embedding + Demo + RF',
         lambda td, ted, dom, em=emb_matrix, el=emb_lookup:
             baseline_embedding_demographics_model(td, ted, dom, em, el, 'rf')),
    ]

    all_results = []
    for bname, bfunc in baselines:
        for domain in DOMAINS:
            print(f"  Running: {bname} / {domain}...")
            gt, pred, rids = bfunc(train_data, test_data, domain)
            metrics = compute_all_metrics(gt, pred, rids)
            row = {
                'Baseline': bname,
                'Domain': domain.capitalize(),
                **metrics,
            }
            all_results.append(row)
            print(f"    Accuracy={metrics.get('Accuracy', 0):.3f}  "
                  f"F1={metrics.get('F1', 0):.3f}  "
                  f"Kappa={metrics.get('Kappa', 0):.3f}  "
                  f"QWK={metrics.get('QWK', 0):.3f}  "
                  f"Dir.Acc={metrics.get('Directional Accuracy', 0):.3f}")

    results_df = pd.DataFrame(all_results)
    return results_df


# ===================================================================
# Visualization
# ===================================================================

def plot_all_baselines_comparison(baseline_df, llm_df):
    """Grouped bar chart comparing all baselines + best LLM methods.

    figsize 20x8, domains as subplots (1x3), y-axis = Accuracy.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=True)

    # Identify best 3 LLM model+method combos per domain by accuracy
    best_llm_rows = []
    for domain in ['Content', 'Coping', 'Quitting']:
        dom_df = llm_df[llm_df['Domain'] == domain]
        if len(dom_df) == 0:
            continue
        dom_sorted = dom_df.sort_values('Accuracy', ascending=False)
        seen = set()
        for _, row in dom_sorted.iterrows():
            label = f"{row['Model']} ({row['Method']})"
            if label not in seen:
                best_llm_rows.append({
                    'Baseline': label,
                    'Domain': domain,
                    'Accuracy': row['Accuracy'],
                    'is_llm': True,
                })
                seen.add(label)
                if len(seen) >= 3:
                    break

    best_llm_df = pd.DataFrame(best_llm_rows)

    # Combine baseline + LLM rows for plotting
    baseline_plot = baseline_df[['Baseline', 'Domain', 'Accuracy']].copy()
    baseline_plot['is_llm'] = False
    combined = pd.concat([baseline_plot, best_llm_df], ignore_index=True)

    baseline_color = '#A6A6A6'
    model_legend_handles = []
    seen_models = set()

    for ax_idx, domain in enumerate(['Content', 'Coping', 'Quitting']):
        ax = axes[ax_idx]
        dom_data = combined[combined['Domain'] == domain].copy()
        dom_data = dom_data.sort_values('Accuracy', ascending=True).reset_index(drop=True)

        colors = []
        for _, row in dom_data.iterrows():
            if not row['is_llm']:
                colors.append(baseline_color)
                continue
            model_name = row['Baseline'].split(' (', 1)[0]
            colors.append(COLORS.get(model_name, '#4D4D4D'))
            if model_name in COLORS and model_name not in seen_models:
                model_legend_handles.append(Patch(facecolor=COLORS[model_name], label=model_name))
                seen_models.add(model_name)
        bars = ax.barh(range(len(dom_data)), dom_data['Accuracy'], color=colors,
                       edgecolor='black', linewidth=0.4)

        ax.set_yticks(range(len(dom_data)))
        ax.set_yticklabels(dom_data['Baseline'], fontsize=9, fontweight='bold')
        ax.set_xlabel('Accuracy', fontsize=11, fontweight='bold')
        ax.set_title(domain, fontsize=13, fontweight='bold')
        ax.set_xlim(0, max(dom_data['Accuracy'].max() * 1.15, 0.5))

        # Add value labels
        for bar_idx, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.005, bar.get_y() + bar.get_height() / 2,
                    f'{width:.3f}', va='center', fontsize=8, fontweight='bold')

    legend_elements = [Patch(facecolor=baseline_color, label='Non-LLM baselines')]
    legend_elements.extend(model_legend_handles)
    axes[2].legend(handles=legend_elements, loc='lower right', fontsize=9, frameon=True)

    fig.suptitle('All Baselines vs. Best LLM Methods',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_feature_ablation(baseline_df):
    """Feature ablation chart showing demographics-only, text-only (TF-IDF),
    text-only (embedding), demographics+TF-IDF, demographics+embedding.

    figsize 18x7, one subplot per domain.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)

    # Map each baseline name to its ablation group
    ablation_groups = {
        'Demographics LR':       'Demographics only',
        'Demographics RF':       'Demographics only',
        'TF-IDF + LR':          'Text only (TF-IDF)',
        'TF-IDF + RF':          'Text only (TF-IDF)',
        'Embedding + LR':       'Text only (Embedding)',
        'Embedding + RF':       'Text only (Embedding)',
        'TF-IDF + Demo + LR':   'Demographics + TF-IDF',
        'TF-IDF + Demo + RF':   'Demographics + TF-IDF',
        'Embedding + Demo + LR': 'Demographics + Embedding',
        'Embedding + Demo + RF': 'Demographics + Embedding',
    }

    model_type_map = {
        'Demographics LR':       'LR',
        'Demographics RF':       'RF',
        'TF-IDF + LR':          'LR',
        'TF-IDF + RF':          'RF',
        'Embedding + LR':       'LR',
        'Embedding + RF':       'RF',
        'TF-IDF + Demo + LR':   'LR',
        'TF-IDF + Demo + RF':   'RF',
        'Embedding + Demo + LR': 'LR',
        'Embedding + Demo + RF': 'RF',
    }

    ablation_order = [
        'Demographics only',
        'Text only (TF-IDF)',
        'Text only (Embedding)',
        'Demographics + TF-IDF',
        'Demographics + Embedding',
    ]

    color_lr = COLORS['Logistic Regression']
    color_rf = COLORS['Random Forest']

    for ax_idx, domain in enumerate(['Content', 'Coping', 'Quitting']):
        ax = axes[ax_idx]
        dom_data = baseline_df[baseline_df['Domain'] == domain].copy()

        # Filter to ablation baselines only
        dom_data = dom_data[dom_data['Baseline'].isin(ablation_groups.keys())].copy()
        dom_data['Feature Group'] = dom_data['Baseline'].map(ablation_groups)
        dom_data['Model Type'] = dom_data['Baseline'].map(model_type_map)

        x = np.arange(len(ablation_order))
        width = 0.35

        lr_vals = []
        rf_vals = []
        for group in ablation_order:
            lr_row = dom_data[(dom_data['Feature Group'] == group) &
                              (dom_data['Model Type'] == 'LR')]
            rf_row = dom_data[(dom_data['Feature Group'] == group) &
                              (dom_data['Model Type'] == 'RF')]
            lr_vals.append(lr_row['Accuracy'].values[0] if len(lr_row) > 0 else 0)
            rf_vals.append(rf_row['Accuracy'].values[0] if len(rf_row) > 0 else 0)

        bars_lr = ax.bar(x - width / 2, lr_vals, width, label='Logistic Regression',
                         color=color_lr, edgecolor='white', linewidth=0.5)
        bars_rf = ax.bar(x + width / 2, rf_vals, width, label='Random Forest',
                         color=color_rf, edgecolor='white', linewidth=0.5)

        # Value labels on bars
        for bars in [bars_lr, bars_rf]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.005,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(ablation_order, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Accuracy' if ax_idx == 0 else '', fontsize=11)
        ax.set_title(domain, fontsize=13, fontweight='bold')
        ax.set_ylim(0, max(max(lr_vals), max(rf_vals)) * 1.15)

    axes[0].legend(fontsize=10, loc='upper left')

    fig.suptitle('Feature Ablation: Impact of Text and Demographic Features',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 70)
    print("TEXT-ENHANCED BASELINES FOR REVISION")
    print("=" * 70)

    # Run all baselines
    results_df = run_all_baselines()

    # Save results CSV
    csv_path = figures_path('text_baseline_results') + '.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results to: {csv_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    pivot = results_df.pivot_table(
        index='Baseline', columns='Domain',
        values=['Accuracy', 'F1', 'Kappa', 'QWK', 'Directional Accuracy'],
        aggfunc='first'
    )
    print(pivot.to_string())

    # Collect LLM results for comparison plot
    print("\nCollecting LLM results for comparison...")
    llm_df = collect_all_results()
    print(f"  LLM results: {len(llm_df)} rows")

    # Plot 1: All baselines vs best LLM
    print("\nGenerating all baselines comparison plot...")
    fig1 = plot_all_baselines_comparison(results_df, llm_df)
    save_figure(fig1, figures_path('all_baselines_comparison'))

    # Plot 2: Feature ablation
    print("Generating feature ablation plot...")
    fig2 = plot_feature_ablation(results_df)
    save_figure(fig2, figures_path('baseline_feature_ablation'))

    print("\nDone!")


if __name__ == '__main__':
    main()
