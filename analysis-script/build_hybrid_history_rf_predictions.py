#!/usr/bin/env python3
"""Generate the RF-predictions JSON for the Hybrid v2 (history-augmented RF) eval.

Reuses the per-domain best Sup-RF from history_supervised_baselines.py:
  Content : Demographics + RF
  Coping  : Demographics + RF
  Quitting: Avg History Score + RF

Output JSON has the same shape as the original
  results_manuscript_hybrid_rf_grok4/rf_predictions_all_features.json
i.e. {response_id: {rf_pred_content: X, rf_pred_coping: Y, rf_pred_quitting: Z}}
so the existing `generate_hybrid_rf_digital_twin_prompt` can consume it via the
new HYBRID_RF_PREDICTIONS env var override.

Saves to: results_manuscript_hybrid_rf_history/rf_predictions_history_features.json
"""
from __future__ import annotations
import json, os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

BASE = "/Users/yiqun/Desktop/chen-lab/LLM-smoking-cessation"
sys.path.insert(0, os.path.join(BASE, "analysis-script"))

from revision_utils import (  # noqa: E402
    load_canonical_data, extract_features, extract_labels, align_features_labels, DOMAINS,
)
from history_supervised_baselines import (  # noqa: E402
    extract_history_features, _build_history_lookup,
    _make_item_key_from_record, _duplicate_item_keys,
)

SUP_FS = {"content": "Demographics", "coping": "Demographics", "quitting": "Avg History Score"}
RANDOM_STATE = 42
OUTPUT_DIR = os.path.join(BASE, "results_manuscript_hybrid_rf_history")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "rf_predictions_history_features.json")


def build_features(records_train, records_test, fs_name, all_train_records):
    if fs_name == "Demographics":
        Xt = extract_features(records_train); Xe = extract_features(records_test)
        Xt, Xe = align_features_labels(Xt, Xe)
        return Xt.values.astype(float), Xe.values.astype(float)
    elif fs_name == "Avg History Score":
        lookup = _build_history_lookup(all_train_records)
        Ht = extract_history_features(records_train, lookup, exclude_self=True)
        He = extract_history_features(records_test, lookup, exclude_self=False)
        return Ht.values.astype(float), He.values.astype(float)
    raise ValueError(fs_name)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_records, test_records_raw = load_canonical_data("7030", "digital_twin")

    # do NOT drop duplicate test items here; the hybrid prompt JSON should
    # cover EVERY response_id in the test set, otherwise the prompt falls back
    # silently. Duplicates get removed downstream in evaluation, not here.
    print(f"train: {len(train_records)} items  ·  test: {len(test_records_raw)} items")

    predictions_by_id: dict[str, dict[str, int]] = {}

    for domain in DOMAINS:
        fs = SUP_FS[domain]
        print(f"\n[{domain}] feature_set={fs}, classifier=RF")
        X_tr, X_te = build_features(train_records, test_records_raw, fs, train_records)
        y_tr = extract_labels(train_records, domain)
        valid_tr = ~np.isnan(y_tr)
        X_tr, y_tr = X_tr[valid_tr], y_tr[valid_tr].astype(int)

        rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                     random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        pred = np.clip(np.round(rf.predict(X_te)).astype(int), 1, 5)

        for rec, p in zip(test_records_raw, pred):
            rid = rec["response_id"]
            entry = predictions_by_id.setdefault(rid, {})
            # If a participant has multiple test items, the RF predicts the same
            # value (features don't depend on test message text); keep the first.
            entry.setdefault(f"rf_pred_{domain}", int(p))

        print(f"  → {len(predictions_by_id)} unique participants covered")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(predictions_by_id, f, indent=2)
    print(f"\nSaved RF v2 predictions to: {OUTPUT_PATH}")
    print(f"  {len(predictions_by_id)} participants × 3 domain predictions each")


if __name__ == "__main__":
    main()
