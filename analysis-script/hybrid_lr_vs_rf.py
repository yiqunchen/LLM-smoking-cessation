#!/usr/bin/env python3
"""
Compare Logistic Regression vs Random Forest on demographic features.

Addresses reviewer R4s question about why Random Forest was chosen over
Logistic Regression for the hybrid pipeline.  Both models are trained on
identical one-hot/numeric demographic features extracted from participant
metadata and evaluated with 5-fold cross-validation and held-out test set.

Outputs
-------
- revision/figures/hybrid_lr_vs_rf_comparison.csv
- revision/figures/hybrid_lr_vs_rf.png / .pdf
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, 'analysis-script')

from revision_utils import (
    load_canonical_data,
    extract_features,
    extract_labels,
    align_features_labels,
    compute_all_metrics,
    DOMAINS,
    COLORS,
    save_figure,
    figures_path,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.base import clone


# -- 1. Load canonical 70/30 participant split -------------------------------
print("Loading canonical 70/30 participant split ...")
train_data, test_data = load_canonical_data('7030', 'participant')
print(f"  Train: {len(train_data)} records, Test: {len(test_data)} records")

# -- 2. Extract features -----------------------------------------------------
print("Extracting features ...")
X_train_raw = extract_features(train_data)
X_test_raw = extract_features(test_data)

# -- 3. Align columns --------------------------------------------------------
X_train, X_test = align_features_labels(X_train_raw, X_test_raw)
print(f"  Feature dimensionality: {X_train.shape[1]}")

# -- 4. Extract response IDs for Spearman computation ------------------------
test_response_ids = np.array(
    [r.get('response_id', i) for i, r in enumerate(test_data)]
)
train_response_ids = np.array(
    [r.get('response_id', i) for i, r in enumerate(train_data)]
)

# -- 5. Define models ---------------------------------------------------------
MODELS = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    ),
}

# -- 6. Train, CV, and test for each domain ----------------------------------
all_rows = []

for domain in DOMAINS:
    print(f"\n{'='*60}")
    print(f"Domain: {domain.capitalize()}")
    print(f"{'='*60}")

    y_train_full = extract_labels(train_data, domain)
    y_test_full = extract_labels(test_data, domain)

    # Filter NaN labels -- train
    train_valid = ~np.isnan(y_train_full)
    X_tr = X_train.loc[train_valid].reset_index(drop=True)
    y_tr = y_train_full[train_valid].astype(int)
    rid_tr = train_response_ids[train_valid]

    # Filter NaN labels -- test
    test_valid = ~np.isnan(y_test_full)
    X_te = X_test.loc[test_valid].reset_index(drop=True)
    y_te = y_test_full[test_valid].astype(int)
    rid_te = test_response_ids[test_valid]

    print(f"  Train samples: {len(y_tr)}, Test samples: {len(y_te)}")

    for model_name, model_template in MODELS.items():
        print(f"\n  --- {model_name} ---")

        # 5-fold cross-validation on train set
        model_cv = clone(model_template)
        cv_preds = cross_val_predict(model_cv, X_tr, y_tr, cv=5)
        cv_metrics = compute_all_metrics(y_tr, cv_preds, rid_tr)
        print(
            f"    CV  Accuracy={cv_metrics['Accuracy']:.3f}  "
            f"F1={cv_metrics['F1']:.3f}  "
            f"Kappa={cv_metrics['Kappa']:.3f}  "
            f"QWK={cv_metrics['QWK']:.3f}  "
            f"DirAcc={cv_metrics['Directional Accuracy']:.3f}  "
            f"DirF1={cv_metrics['Directional Macro-F1']:.3f}"
        )

        all_rows.append({
            'Model': model_name,
            'Domain': domain.capitalize(),
            'Split': 'CV',
            'Accuracy': cv_metrics['Accuracy'],
            'F1': cv_metrics['F1'],
            'Kappa': cv_metrics['Kappa'],
            'QWK': cv_metrics['QWK'],
            'Directional_Accuracy': cv_metrics['Directional Accuracy'],
            'Directional_Macro_F1': cv_metrics['Directional Macro-F1'],
            'Spearman_Rho': cv_metrics.get('Spearman_Rho', np.nan),
            'N': cv_metrics['N'],
        })

        # Retrain on full train set, evaluate on test
        model_final = clone(model_template)
        model_final.fit(X_tr, y_tr)
        test_preds = model_final.predict(X_te)
        test_metrics = compute_all_metrics(y_te, test_preds, rid_te)
        print(
            f"    Test Accuracy={test_metrics['Accuracy']:.3f}  "
            f"F1={test_metrics['F1']:.3f}  "
            f"Kappa={test_metrics['Kappa']:.3f}  "
            f"QWK={test_metrics['QWK']:.3f}  "
            f"DirAcc={test_metrics['Directional Accuracy']:.3f}  "
            f"DirF1={test_metrics['Directional Macro-F1']:.3f}"
        )

        all_rows.append({
            'Model': model_name,
            'Domain': domain.capitalize(),
            'Split': 'Test',
            'Accuracy': test_metrics['Accuracy'],
            'F1': test_metrics['F1'],
            'Kappa': test_metrics['Kappa'],
            'QWK': test_metrics['QWK'],
            'Directional_Accuracy': test_metrics['Directional Accuracy'],
            'Directional_Macro_F1': test_metrics['Directional Macro-F1'],
            'Spearman_Rho': test_metrics.get('Spearman_Rho', np.nan),
            'N': test_metrics['N'],
        })

# -- 7. Save CSV --------------------------------------------------------------
results_df = pd.DataFrame(all_rows)
csv_path = os.path.join(
    os.path.dirname(__file__), '..', 'revision', 'figures',
    'hybrid_lr_vs_rf_comparison.csv'
)
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
results_df.to_csv(csv_path, index=False, float_format='%.4f')
print(f"\nSaved CSV: {csv_path}")

# -- 8. Create figure: side-by-side grouped bars (test set only) ---------------
test_df = results_df[results_df['Split'] == 'Test'].copy()

METRICS_TO_PLOT = [
    'Accuracy', 'F1', 'Kappa', 'QWK',
    'Directional_Accuracy', 'Directional_Macro_F1',
]
METRIC_LABELS = {
    'Accuracy': 'Accuracy',
    'F1': 'Macro F1',
    'Kappa': "Cohen's κ",
    'QWK': 'QWK',
    'Directional_Accuracy': 'Dir. Accuracy',
    'Directional_Macro_F1': 'Dir. Macro F1',
}

model_names = ['Logistic Regression', 'Random Forest']
domain_labels = [d.capitalize() for d in DOMAINS]

fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=False)

bar_width = 0.35
x = np.arange(len(METRICS_TO_PLOT))

for ax_idx, domain_label in enumerate(domain_labels):
    ax = axes[ax_idx]
    for m_idx, model_name in enumerate(model_names):
        subset = test_df[
            (test_df['Domain'] == domain_label)
            & (test_df['Model'] == model_name)
        ]
        if subset.empty:
            continue
        row = subset.iloc[0]
        values = [row[m] for m in METRICS_TO_PLOT]
        offset = (m_idx - 0.5) * bar_width
        color = COLORS.get(model_name, '#333333')
        bars = ax.bar(
            x + offset, values, bar_width, label=model_name,
            color=color, edgecolor='white', linewidth=0.5,
        )
        # Value labels on bars
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{val:.2f}',
                    ha='center', va='bottom', fontsize=7,
                )

    ax.set_title(f'{domain_label}', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(
        [METRIC_LABELS[m] for m in METRICS_TO_PLOT],
        rotation=35, ha='right', fontsize=9,
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score' if ax_idx == 0 else '')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Single legend at top
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels, loc='upper center', ncol=2, fontsize=11,
    frameon=True, bbox_to_anchor=(0.5, 1.02),
)

fig.suptitle(
    'Logistic Regression vs Random Forest — Test Set Performance',
    fontsize=14, fontweight='bold', y=1.07,
)
fig.tight_layout()

fig_path = figures_path('hybrid_lr_vs_rf')
save_figure(fig, fig_path)

print("\nDone.")
