#!/usr/bin/env python3
"""k=0 baselines for the dt10 learning curve.

Two no-in-participant-data conditions, both evaluated on every rated row in
the dt10 dataset (msg 1–10 per participant):

  Majority-class:
      Predict the dataset-wide modal rating for each domain. True
      zero-information floor.

  LOPO Demographics:
      For each participant P, train Random Forest on the OTHER 300
      participants' (demographics → rating) pairs (~3,000 rows), predict
      for P's 10 messages. RF never sees any of P's ratings during
      training but inherits cross-participant demographic structure.

Appends results to revision/figures/lc_dt10_rf.csv as k=0 rows.

Usage:
    uv run python analysis-script/lc_dt10_rf_k0.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from revision_utils import (  # noqa: E402
    DOMAINS,
    align_features_labels,
    compute_all_metrics,
    extract_features,
    extract_labels,
    figures_path,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CANONICAL_DIR = os.path.join(PROJECT_ROOT, "data_splits", "canonical")
RANDOM_STATE = 42


def _load_full_dt10() -> list:
    """Load the full 3,005-row dt10 dataset by unioning train+test of any
    k-split (they cover the same row universe; k=1 is convenient)."""
    paths = [
        os.path.join(CANONICAL_DIR, "train_dt10_k1.json"),
        os.path.join(CANONICAL_DIR, "test_dt10_k1.json"),
    ]
    out = []
    for p in paths:
        with open(p) as f:
            out.extend(json.load(f))
    return out


def _modal_class_baseline(records: list) -> list[dict]:
    rows: list[dict] = []
    for domain in DOMAINS:
        y = extract_labels(records, domain)
        valid = ~np.isnan(y)
        y_valid = y[valid].astype(int)
        if len(y_valid) == 0:
            continue
        modal = Counter(y_valid.tolist()).most_common(1)[0][0]
        pred = np.full_like(y_valid, modal)
        rids = np.array([records[i]["response_id"] for i in np.where(valid)[0]])
        m = compute_all_metrics(y_valid, pred, rids)
        rows.append({
            "k_train": 0, "feature_set": "Majority class",
            "domain": domain, "train_n": 0, "test_n": int(len(y_valid)),
            **m,
        })
        print(f"  Majority-class  {domain:9s}  modal={modal}  "
              f"acc={m['Accuracy']:.3f}  f1={m['F1']:.3f}  qwk={m['QWK']:.3f}")
    return rows


def _lopo_demographics(records: list) -> list[dict]:
    """For each participant, train Demographics-RF on the other 300
    participants' rated rows; predict for the held-out participant.
    """
    by_pid: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        by_pid.setdefault(r["response_id"], []).append(i)
    pids = list(by_pid.keys())
    print(f"  LOPO across {len(pids)} participants…")

    demo_df = extract_features(records)
    demo_arr = np.asarray(demo_df.values, dtype=float)

    rows: list[dict] = []
    for domain in DOMAINS:
        y_full = extract_labels(records, domain)
        preds = np.full_like(y_full, np.nan, dtype=float)
        for pid in pids:
            test_idx = np.array(by_pid[pid])
            train_idx = np.array(
                [i for i in range(len(records)) if i not in set(by_pid[pid])]
            )
            y_tr = y_full[train_idx]
            valid_tr = ~np.isnan(y_tr)
            X_tr = demo_arr[train_idx][valid_tr]
            y_tr_v = y_tr[valid_tr].astype(int)
            if len(np.unique(y_tr_v)) < 2:
                continue
            rf = RandomForestClassifier(
                n_estimators=200, max_depth=10,
                random_state=RANDOM_STATE, n_jobs=-1,
            )
            rf.fit(X_tr, y_tr_v)
            X_te = demo_arr[test_idx]
            pred = np.clip(np.round(rf.predict(X_te)).astype(int), 1, 5)
            preds[test_idx] = pred

        valid = ~np.isnan(y_full) & ~np.isnan(preds)
        y_valid = y_full[valid].astype(int)
        p_valid = preds[valid].astype(int)
        rids = np.array([records[i]["response_id"] for i in np.where(valid)[0]])
        m = compute_all_metrics(y_valid, p_valid, rids)
        rows.append({
            "k_train": 0, "feature_set": "Demographics (LOPO)",
            "domain": domain, "train_n": 0,  # 0 from THIS participant
            "test_n": int(len(y_valid)), **m,
        })
        print(f"    Demographics-LOPO  {domain:9s}  acc={m['Accuracy']:.3f}  "
              f"f1={m['F1']:.3f}  qwk={m['QWK']:.3f}")
    return rows


def main():
    print("Loading dt10 records (full 3,005-row universe)…")
    records = _load_full_dt10()
    print(f"  {len(records)} rated rows")

    print("\nMajority-class baseline:")
    rows_majority = _modal_class_baseline(records)

    print("\nLOPO Demographics-RF baseline:")
    rows_lopo = _lopo_demographics(records)

    # Append to lc_dt10_rf.csv (deduplicate by k_train+feature_set+domain)
    csv_path = figures_path("lc_dt10_rf") + ".csv"
    new_df = pd.DataFrame(rows_majority + rows_lopo)
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        existing = existing[
            ~((existing["k_train"] == 0)
              & (existing["feature_set"].isin(new_df["feature_set"])))
        ]
        out = pd.concat([existing, new_df], ignore_index=True)
    else:
        out = new_df
    out = out.sort_values(["k_train", "feature_set", "domain"])
    out.to_csv(csv_path, index=False)
    print(f"\nUpdated {csv_path}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
