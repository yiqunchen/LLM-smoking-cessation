#!/usr/bin/env python3
"""RF learning curve on the new dt10 splits (k = 1 / 3 / 5 / 7 prior msgs/pp).

Trains Random Forest on each (split, feature_set, domain) cell using the
new ``train_dt10_k{k}.json`` / ``test_dt10_k{k}.json`` splits and saves
metrics + a per-row predictions CSV.

Outputs:
    revision/figures/lc_dt10_rf.csv             (long-form metrics)
    revision/figures/lc_dt10_rf_predictions.csv  (per-row RF predictions)

Usage:
    uv run python analysis-script/lc_dt10_rf.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from revision_utils import (  # noqa: E402
    DOMAINS,
    align_features_labels,
    compute_all_metrics,
    extract_demographic_features,
    extract_features,
    extract_labels,
    figures_path,
)
from history_supervised_baselines import (  # noqa: E402
    _build_history_lookup,
    extract_history_features,
)
from text_baselines import _load_embeddings, _match_embeddings  # noqa: E402

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CANONICAL_DIR = os.path.join(PROJECT_ROOT, "data_splits", "canonical")

K_VALUES = [1, 3, 5, 7]
RANDOM_STATE = 42

FEATURE_SETS = {
    "Demographics":     {"demo": True,  "history": False, "embedding": False},
    "Avg-History":      {"demo": False, "history": True,  "embedding": False},
    "Demo+History":     {"demo": True,  "history": True,  "embedding": False},
    "Embedding":        {"demo": False, "history": False, "embedding": True},
    "Embedding+Demo":   {"demo": True,  "history": False, "embedding": True},
    "Demographics + History + Message Embedding": {"demo": True, "history": True, "embedding": True},
}


def _load_split(k: int) -> tuple[list, list]:
    with open(os.path.join(CANONICAL_DIR, f"train_dt10_k{k}.json")) as f:
        train = json.load(f)
    with open(os.path.join(CANONICAL_DIR, f"test_dt10_k{k}.json")) as f:
        test = json.load(f)
    return train, test


def build_features(train: list, test: list, spec: dict
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, idx_train, X_test, idx_test).

    `idx_*` are the indices into the original train/test lists that survived
    the embedding lookup (full coverage when embedding is False).
    """
    demo_tr = extract_demographic_features(train)
    demo_te = extract_demographic_features(test)
    demo_tr, demo_te = align_features_labels(demo_tr, demo_te)

    if spec["history"]:
        lookup = _build_history_lookup(train)
        hist_tr = extract_history_features(train, lookup, exclude_self=True)
        hist_te = extract_history_features(test, lookup, exclude_self=False)
    else:
        hist_tr = hist_te = None

    if spec["embedding"]:
        emb_matrix, emb_lookup = _load_embeddings()
        emb_tr_arr, idx_tr = _match_embeddings(train, emb_matrix, emb_lookup)
        emb_te_arr, idx_te = _match_embeddings(test, emb_matrix, emb_lookup)
        idx_tr = np.asarray(idx_tr, dtype=int)
        idx_te = np.asarray(idx_te, dtype=int)
    else:
        idx_tr = np.arange(len(train), dtype=int)
        idx_te = np.arange(len(test), dtype=int)

    pieces_tr, pieces_te = [], []
    if spec["embedding"]:
        pieces_tr.append(np.asarray(emb_tr_arr, dtype=float))
        pieces_te.append(np.asarray(emb_te_arr, dtype=float))
    if spec["demo"]:
        pieces_tr.append(np.asarray(demo_tr.iloc[idx_tr].values, dtype=float))
        pieces_te.append(np.asarray(demo_te.iloc[idx_te].values, dtype=float))
    if spec["history"]:
        pieces_tr.append(np.asarray(hist_tr.iloc[idx_tr].values, dtype=float))
        pieces_te.append(np.asarray(hist_te.iloc[idx_te].values, dtype=float))

    X_tr = pieces_tr[0] if len(pieces_tr) == 1 else np.hstack(pieces_tr)
    X_te = pieces_te[0] if len(pieces_te) == 1 else np.hstack(pieces_te)
    return X_tr, idx_tr, X_te, idx_te


def fit_one_split(k: int) -> tuple[list[dict], list[dict]]:
    train, test = _load_split(k)
    print(f"\n[k={k}]  train={len(train)}  test={len(test)}")

    metric_rows: list[dict] = []
    pred_rows: list[dict] = []

    for fs_name, spec in FEATURE_SETS.items():
        X_tr, idx_tr, X_te, idx_te = build_features(train, test, spec)
        for domain in DOMAINS:
            y_tr_full = extract_labels(train, domain)[idx_tr]
            y_te_full = extract_labels(test, domain)[idx_te]
            valid_tr = ~np.isnan(y_tr_full)
            valid_te = ~np.isnan(y_te_full)
            X_tr_v, y_tr_v = X_tr[valid_tr], y_tr_full[valid_tr].astype(int)
            X_te_v, y_te_v = X_te[valid_te], y_te_full[valid_te].astype(int)
            rids_te = np.array([test[int(i)]["response_id"]
                                 for i in idx_te[valid_te]])

            if len(np.unique(y_tr_v)) < 2:
                metric_rows.append({
                    "k_train": k, "feature_set": fs_name, "domain": domain,
                    "train_n": int(len(y_tr_v)), "test_n": int(len(y_te_v)),
                    "Accuracy": np.nan, "F1": np.nan, "QWK": np.nan,
                    "Kappa": np.nan, "Spearman_Rho": np.nan,
                    "Directional Accuracy": np.nan,
                    "Directional Macro-F1": np.nan, "N": int(len(y_te_v)),
                })
                continue

            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
            )
            rf.fit(X_tr_v, y_tr_v)
            pred = np.clip(np.round(rf.predict(X_te_v)).astype(int), 1, 5)
            metrics = compute_all_metrics(y_te_v, pred, rids_te)
            metric_rows.append({
                "k_train": k, "feature_set": fs_name, "domain": domain,
                "train_n": int(len(y_tr_v)), "test_n": int(len(y_te_v)),
                **metrics,
            })
            print(f"    {fs_name:23s}  {domain:9s}  acc={metrics['Accuracy']:.3f}  "
                  f"f1={metrics['F1']:.3f}  qwk={metrics['QWK']:.3f}  "
                  f"rho={metrics.get('Spearman_Rho', float('nan')):.3f}")

            valid_record_idx = idx_te[valid_te]
            for ridx, p_val, gt_val in zip(valid_record_idx, pred, y_te_v):
                rec = test[int(ridx)]
                pred_rows.append({
                    "k_train": k, "feature_set": fs_name, "domain": domain,
                    "response_id": rec["response_id"],
                    "input_message": rec["input_message"],
                    "ground_truth_num": int(gt_val),
                    "predicted_num": int(p_val),
                })

    return metric_rows, pred_rows


def main():
    metric_rows: list[dict] = []
    pred_rows: list[dict] = []
    for k in K_VALUES:
        m, p = fit_one_split(k)
        metric_rows.extend(m)
        pred_rows.extend(p)

    df = pd.DataFrame(metric_rows)
    out_csv = figures_path("lc_dt10_rf") + ".csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved metrics: {out_csv}  ({len(df)} rows)")

    pdf = pd.DataFrame(pred_rows)
    pred_path = figures_path("lc_dt10_rf_predictions") + ".csv"
    pdf.to_csv(pred_path, index=False)
    print(f"Saved RF predictions: {pred_path}  ({len(pdf)} rows)")


if __name__ == "__main__":
    main()
