#!/usr/bin/env python3
"""
Apples-to-apples learning curve: Random Forest vs LLM Digital Twin
across the canonical digital-twin splits.

Within each split, RF and LLM are evaluated on the IDENTICAL test rows
(the split's canonical test partition). Across splits, the train fraction
varies, giving a learning-curve x-axis.

  Split   Train_n   Test_n   RF?   LLM?
  -----   -------   ------   ---   ----
  1090    0         916       no   yes  (zero-shot DT)
  3070    22        894      yes   yes
  5050    323       593      yes   no
  7030    593       323      yes   yes
  9010    615       301      yes   yes

LLMs: GPT-5, GPT-4o-mini, Gemini-2.5-Pro, Grok-4-Fast (4 splits each);
DeepSeek-R1 only ran 7030 so it is shown as a single anchor point.

RF feature sets (fit at every RF-eligible split):
    - Demographics
    - Avg-History (avg prior rating across the participant's other messages)
    - Embedding (1024-d OpenAI text-embedding-3-large of the message)
    - Embedding+Demo

Outputs:
    revision/figures/lc_rf_vs_llm.csv            -- long-form metrics
    revision/figures/lc_rf_vs_llm_predictions.csv -- per-row RF predictions

Usage:
    uv run python analysis-script/lc_rf_vs_llm.py
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
    RATING_MAPS,
    align_features_labels,
    compute_all_metrics,
    extract_features,
    extract_labels,
    figures_path,
    load_canonical_data,
)
from history_supervised_baselines import (  # noqa: E402
    _build_history_lookup,
    extract_history_features,
)
from text_baselines import _load_embeddings, _match_embeddings  # noqa: E402

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RANDOM_STATE = 42

ALL_SPLITS = ["1090", "3070", "5050", "7030", "9010"]
RF_SPLITS = ["3070", "5050", "7030", "9010"]   # 1090 has 0 train items
LLM_SPLITS = ["1090", "3070", "7030", "9010"]  # 5050 not run

FEATURE_SETS = {
    "Demographics":    {"demo": True,  "history": False, "embedding": False},
    "Avg-History":     {"demo": False, "history": True,  "embedding": False},
    "Demo+History":    {"demo": True,  "history": True,  "embedding": False},
    "Embedding":       {"demo": False, "history": False, "embedding": True},
    "Embedding+Demo":  {"demo": True,  "history": False, "embedding": True},
}

LLM_DIRS = {
    "GPT-4o-mini":     "results_manuscript_gpt-4o-mini",
    "GPT-5":           "results_manuscript_gpt-5",
    "Gemini-2.5-Pro":  "results_manuscript_gemini-2.5-pro",
    "Grok-4-Fast":     "results_manuscript_x-ai_grok-4-fast",
    "DeepSeek-R1":     "results_manuscript_deepseek_deepseek-r1-0528",
}


# ---------------------------------------------------------------------------
# RF
# ---------------------------------------------------------------------------

def build_features(records_train, records_test, fs_spec, all_train_records):
    """Construct (X_train, idx_train, X_test, idx_test) for the requested feature set.

    idx_* index back into the original records list (used to align labels).
    """
    demo_tr = extract_features(records_train)
    demo_te = extract_features(records_test)
    demo_tr, demo_te = align_features_labels(demo_tr, demo_te)

    if fs_spec["history"]:
        hist_lookup = _build_history_lookup(all_train_records)
        hist_tr = extract_history_features(records_train, hist_lookup, exclude_self=True)
        hist_te = extract_history_features(records_test, hist_lookup, exclude_self=False)
    else:
        hist_tr = hist_te = None

    if fs_spec["embedding"]:
        emb_matrix, emb_lookup = _load_embeddings()
        emb_tr_arr, idx_tr = _match_embeddings(records_train, emb_matrix, emb_lookup)
        emb_te_arr, idx_te = _match_embeddings(records_test, emb_matrix, emb_lookup)
        idx_tr = np.asarray(idx_tr, dtype=int)
        idx_te = np.asarray(idx_te, dtype=int)
    else:
        idx_tr = np.arange(len(records_train), dtype=int)
        idx_te = np.arange(len(records_test), dtype=int)

    pieces_tr, pieces_te = [], []
    if fs_spec["embedding"]:
        pieces_tr.append(np.asarray(emb_tr_arr, dtype=float))
        pieces_te.append(np.asarray(emb_te_arr, dtype=float))
    if fs_spec["demo"]:
        pieces_tr.append(np.asarray(demo_tr.iloc[idx_tr].values, dtype=float))
        pieces_te.append(np.asarray(demo_te.iloc[idx_te].values, dtype=float))
    if fs_spec["history"]:
        pieces_tr.append(np.asarray(hist_tr.iloc[idx_tr].values, dtype=float))
        pieces_te.append(np.asarray(hist_te.iloc[idx_te].values, dtype=float))

    X_tr = pieces_tr[0] if len(pieces_tr) == 1 else np.hstack(pieces_tr)
    X_te = pieces_te[0] if len(pieces_te) == 1 else np.hstack(pieces_te)
    return X_tr, idx_tr, X_te, idx_te


def fit_rf_one_split(split: str):
    train, test = load_canonical_data(split, "digital_twin")
    if len(train) == 0:
        return [], []
    print(f"  [{split}] train={len(train)}  test={len(test)}")

    metric_rows: list[dict] = []
    pred_rows: list[dict] = []

    for fs_name, fs_spec in FEATURE_SETS.items():
        X_tr, idx_tr, X_te, idx_te = build_features(train, test, fs_spec, train)

        for domain in DOMAINS:
            y_tr_full = extract_labels(train, domain)[idx_tr]
            y_te_full = extract_labels(test, domain)[idx_te]
            valid_tr = ~np.isnan(y_tr_full)
            valid_te = ~np.isnan(y_te_full)

            X_tr_v, y_tr_v = X_tr[valid_tr], y_tr_full[valid_tr].astype(int)
            X_te_v, y_te_v = X_te[valid_te], y_te_full[valid_te].astype(int)
            rids_te_v = np.array(
                [test[int(i)]["response_id"] for i in idx_te[valid_te]]
            )

            if len(np.unique(y_tr_v)) < 2:
                metric_rows.append({
                    "split": split, "method": "RF", "feature_set": fs_name,
                    "domain": domain, "train_n": int(len(y_tr_v)),
                    "test_n": int(len(y_te_v)), "Accuracy": np.nan,
                    "F1": np.nan, "Spearman_Rho": np.nan, "QWK": np.nan,
                    "Kappa": np.nan, "Directional Accuracy": np.nan,
                    "Directional Macro-F1": np.nan, "N": int(len(y_te_v)),
                })
                continue

            rf = RandomForestClassifier(
                n_estimators=200, max_depth=10,
                random_state=RANDOM_STATE, n_jobs=-1,
            )
            rf.fit(X_tr_v, y_tr_v)
            pred = np.clip(np.round(rf.predict(X_te_v)).astype(int), 1, 5)
            metrics = compute_all_metrics(y_te_v, pred, rids_te_v)
            metric_rows.append({
                "split": split, "method": "RF", "feature_set": fs_name,
                "domain": domain, "train_n": int(len(y_tr_v)),
                "test_n": int(len(y_te_v)), **metrics,
            })

            valid_idx = idx_te[valid_te]
            for k, ridx in enumerate(valid_idx):
                rec = test[int(ridx)]
                pred_rows.append({
                    "split": split, "feature_set": fs_name, "domain": domain,
                    "response_id": rec["response_id"],
                    "input_message": rec["input_message"],
                    "ground_truth_num": int(y_te_v[k]),
                    "predicted_num": int(pred[k]),
                })

    return metric_rows, pred_rows


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def _train_n_for_split(split: str) -> int:
    path = os.path.join(
        PROJECT_ROOT, "data_splits", "canonical",
        f"train_digital_twin_{split}.json",
    )
    with open(path) as f:
        return len(json.load(f))


def llm_metrics_one_split(split: str) -> list[dict]:
    train_n = _train_n_for_split(split)
    rows: list[dict] = []
    for display, mdir in LLM_DIRS.items():
        path = os.path.join(
            PROJECT_ROOT, mdir, f"digital_twin_4_cbtact_{split}.json"
        )
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        recs = [
            v for v in data.values()
            if isinstance(v, dict) and v.get("predicted_content") != "ERROR"
        ]
        for domain in DOMAINS:
            gt, pred, rids = [], [], []
            for r in recs:
                gt_t = r.get(f"ground_truth_{domain}")
                pr_t = r.get(f"predicted_{domain}")
                gt_n = RATING_MAPS[domain].get(gt_t)
                pr_n = RATING_MAPS[domain].get(pr_t)
                if gt_n is None or pr_n is None:
                    continue
                gt.append(gt_n)
                pred.append(pr_n)
                rids.append(r["response_id"])
            metrics = compute_all_metrics(np.array(gt), np.array(pred),
                                          np.array(rids))
            rows.append({
                "split": split, "method": "LLM-DT", "feature_set": display,
                "domain": domain, "train_n": train_n,
                "test_n": len(gt), **metrics,
            })
    return rows


# ---------------------------------------------------------------------------

def main():
    metric_rows: list[dict] = []
    pred_rows: list[dict] = []

    print("=== Random Forest ===")
    for split in RF_SPLITS:
        m, p = fit_rf_one_split(split)
        metric_rows.extend(m)
        pred_rows.extend(p)

    print("=== LLM Digital Twin ===")
    for split in LLM_SPLITS:
        rows = llm_metrics_one_split(split)
        metric_rows.extend(rows)
        for r in rows:
            print(f"  [{split}] {r['feature_set']:14s} {r['domain']:9s}"
                  f"  acc={r.get('Accuracy', float('nan')):.3f}"
                  f"  f1={r.get('F1', float('nan')):.3f}"
                  f"  rho={r.get('Spearman_Rho', float('nan')):.3f}")

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = figures_path("lc_rf_vs_llm") + ".csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics: {metrics_path}  ({len(metrics_df)} rows)")

    pred_df = pd.DataFrame(pred_rows)
    pred_path = figures_path("lc_rf_vs_llm_predictions") + ".csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved RF predictions: {pred_path}  ({len(pred_df)} rows)")


if __name__ == "__main__":
    main()
