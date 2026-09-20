#!/usr/bin/env python3
"""
Class imbalance analysis for revision.

Tabulates rating distributions per domain on the 1-5 scale (train vs cleaned
test), and writes QWK (+ Accuracy, F1, Kappa) from the same consistent
PP 70/30 source table used by the current bars_all_methods figures.

Usage:
    uv run python analysis-script/class_imbalance_analysis.py
"""

import sys
import os

sys.path.insert(0, 'analysis-script')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

import revision_utils as ru
from create_comprehensive_figures import (
    CONSISTENT_SPLIT,
    collect_consistent_split_results,
)
from filter_duplicates import get_duplicate_signatures

# -- Styling -------------------------------------------------------------------
DPI = 400
FONT_CHAIN = ['Arial', 'Helvetica', 'Helvetica Neue', 'Avenir',
              'Avenir Next', 'DejaVu Sans']
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': FONT_CHAIN,
    'font.size': 15,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.8,
    'grid.alpha': 0.12,
    'grid.color': '#4D4D4D',
    'grid.linestyle': '--',
    'grid.linewidth': 0.7,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.width': 1.8,
    'ytick.major.width': 1.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

def is_raw_duplicate(row: dict, duplicate_sigs: set) -> bool:
    if not isinstance(row, dict):
        return False
    sig = (
        row.get('response_id'),
        str(row.get('input_message', '')).strip().lower(),
        tuple(sorted(row.get('ratings', {}).items())),
    )
    return sig in duplicate_sigs


# -- 1. Load canonical data ----------------------------------------------------
print('Loading canonical PP 70/30 split...')
train_data, test_data_raw = ru.load_canonical_data(CONSISTENT_SPLIT, 'digital_twin')
duplicate_sigs = get_duplicate_signatures()
test_data = [row for row in test_data_raw if not is_raw_duplicate(row, duplicate_sigs)]
print(
    f'  Train: {len(train_data)} records, '
    f'Test: {len(test_data)} cleaned records '
    f'({len(test_data_raw) - len(test_data)} duplicate test records removed)'
)

# -- 2. Tabulate rating distributions per domain -------------------------------
print()
print('Tabulating rating distributions...')

dist_rows = []
for domain in ru.DOMAINS:
    train_labels = ru.extract_labels(train_data, domain)
    test_labels = ru.extract_labels(test_data, domain)

    # Drop NaN
    train_labels = train_labels[~np.isnan(train_labels)].astype(int)
    test_labels = test_labels[~np.isnan(test_labels)].astype(int)

    reverse_map = ru.REVERSE_RATING_MAPS[domain]

    for rating in range(1, 6):
        train_n = int(np.sum(train_labels == rating))
        test_n = int(np.sum(test_labels == rating))
        train_pct = train_n / len(train_labels) * 100 if len(train_labels) > 0 else 0.0
        test_pct = test_n / len(test_labels) * 100 if len(test_labels) > 0 else 0.0

        dist_rows.append({
            'Split': f'digital_twin_{CONSISTENT_SPLIT}',
            'Test_Filter': 'known_dt7030_duplicates_removed',
            'Removed_Duplicate_Test_N': len(test_data_raw) - len(test_data),
            'Domain': domain.capitalize(),
            'Rating': rating,
            'Rating_Label': reverse_map.get(rating, str(rating)),
            'Train_N': train_n,
            'Test_N': test_n,
            'Train_Pct': round(train_pct, 2),
            'Test_Pct': round(test_pct, 2),
        })

dist_df = pd.DataFrame(dist_rows)

# Save distribution CSV
dist_csv_path = ru.figures_path('class_distribution') + '.csv'
os.makedirs(os.path.dirname(dist_csv_path), exist_ok=True)
dist_df.to_csv(dist_csv_path, index=False)
print(f'  Saved: {dist_csv_path}')

# -- 3. Plot 1x3 bar chart of class distributions -----------------------------
print()
print('Creating class distribution figure...')

fig, axes = plt.subplots(1, 3, figsize=(20, 7.2), constrained_layout=True)
fig.patch.set_facecolor('white')

bar_width = 0.35

for idx, domain in enumerate(ru.DOMAINS):
    ax = axes[idx]
    dom_df = dist_df[dist_df['Domain'] == domain.capitalize()]

    ratings = dom_df['Rating'].values
    train_pcts = dom_df['Train_Pct'].values
    test_pcts = dom_df['Test_Pct'].values
    labels = dom_df['Rating_Label'].values

    x = np.arange(len(ratings))
    bars_train = ax.bar(x - bar_width / 2, train_pcts, bar_width,
                        label='Train', color='#0173B2', alpha=0.88,
                        edgecolor='#0173B2', linewidth=1.6)
    bars_test = ax.bar(x + bar_width / 2, test_pcts, bar_width,
                       label='Test', color='#DE8F05', alpha=0.88,
                       edgecolor='#DE8F05', linewidth=1.6)

    # Add count annotations on top of bars
    for bar, n in zip(bars_train, dom_df['Train_N'].values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(n), ha='center', va='bottom', fontsize=10,
                fontweight='bold', color='#0173B2')
    for bar, n in zip(bars_test, dom_df['Test_N'].values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(n), ha='center', va='bottom', fontsize=10,
                fontweight='bold', color='#DE8F05')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha='right',
                       fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=16, fontweight='bold')
    ax.set_title(f'{domain.capitalize()}', fontsize=18, fontweight='bold', pad=12)
    ax.legend(frameon=False, prop={'weight': 'bold', 'size': 13})
    ax.set_ylim(0, max(max(train_pcts), max(test_pcts)) * 1.25)
    ax.grid(axis='y', alpha=0.12, color='#4D4D4D',
            linestyle='--', linewidth=0.7)
    ax.set_facecolor('white')
    ax.tick_params(axis='both', labelsize=13, width=1.8,
                   length=6, direction='out')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

fig.suptitle('Rating Distribution by Domain (Train vs Test)',
             fontsize=21, fontweight='bold')

ru.save_figure(fig, ru.figures_path('class_distribution_by_domain'), dpi=DPI)

# -- 4. Compute QWK + metrics for all models x methods x domains ---------------
print()
print('Writing QWK and metrics from the consistent all-model source table...')

consistent_df, audit_df = collect_consistent_split_results(split=CONSISTENT_SPLIT)
if consistent_df.empty:
    raise RuntimeError('No consistent all-model rows were available for qwk_results.csv')

qwk_cols = [
    'Model', 'Model_ID', 'Method', 'Category', 'Domain',
    'Accuracy', 'F1', 'Kappa', 'QWK', 'N', 'Split',
    'Source_File', 'Feature_Set', 'Classifier',
]
qwk_df = consistent_df[qwk_cols].copy()
for col in ['Accuracy', 'F1', 'Kappa', 'QWK']:
    qwk_df[col] = qwk_df[col].round(4)

# Save QWK results CSV
qwk_csv_path = ru.figures_path('qwk_results') + '.csv'
qwk_df.to_csv(qwk_csv_path, index=False)
print(f'  Saved: {qwk_csv_path}')

# -- Summary -------------------------------------------------------------------
print()
print('=== Summary ===')
print(f'  Class distribution CSV:    {dist_csv_path}')
print(f"  Class distribution figure: {ru.figures_path('class_distribution_by_domain')}.png/.pdf")
print(f'  QWK results CSV:           {qwk_csv_path}')
print(f'  Total model x method x domain entries: {len(qwk_df)}')

if not qwk_df.empty:
    print(f"  QWK range: {qwk_df['QWK'].min():.4f} - {qwk_df['QWK'].max():.4f}")
    print(f"  Mean QWK:  {qwk_df['QWK'].mean():.4f}")
    print()
    print('  QWK by domain (mean):')
    for domain_name in ['Content', 'Coping', 'Quitting']:
        subset = qwk_df[qwk_df['Domain'] == domain_name]
        if not subset.empty:
            print(f"    {domain_name}: {subset['QWK'].mean():.4f}")

print()
print('Done.')
