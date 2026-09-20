#!/usr/bin/env python3
"""Simple late-fusion ensembles: RF + LLM-DT canonical Grok at k=1.

We have per-row predictions for:
  - LLM-DT canonical (Grok, dt10 k=1)  -> digital_twin_dt10_k1.json
  - RF (multiple feature sets, dt10 k=1) -> lc_dt10_rf_predictions.csv

For each (RF feature set × test row, domain) we compute:
  Mean ensemble:    round( (rf_pred + llm_pred) / 2 ), clipped to [1, 5]
  RF-anchor + LLM nudge: round( 0.7 * rf + 0.3 * llm )
  Majority of two:  if equal -> that, else split 50/50

The simple mean ensemble is the standard 'soft hybrid'.

Outputs:
    revision/figures/lc_dt10_ensemble_k1.csv

Usage:
    uv run python analysis-script/lc_dt10_ensemble_k1.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_utils import (  # noqa: E402
    DOMAINS,
    RATING_MAPS,
    compute_all_metrics,
    figures_path,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RF_PRED_CSV = figures_path("lc_dt10_rf_predictions") + ".csv"


def grok_path(k: int) -> str:
    return os.path.join(
        PROJECT_ROOT, "results_manuscript_x-ai_grok-4-fast",
        f"digital_twin_dt10_k{k}.json",
    )


def test_path(k: int) -> str:
    return os.path.join(
        PROJECT_ROOT, "data_splits", "canonical", f"test_dt10_k{k}.json"
    )


# Default to the latest k argument, but main() can iterate
K = 1
FEATURE_SETS = [
    "Demographics",
    "Avg-History",
    "Demo+History",
    "Embedding",
    "Embedding+Demo",
    "Demographics + History + Message Embedding",
]

ENSEMBLE_WEIGHTS = {
    "RF only":        (1.0, 0.0),
    "LLM only":       (0.0, 1.0),
    "Mean (50/50)":   (0.5, 0.5),
    "RF-anchor 70/30": (0.7, 0.3),
    "LLM-anchor 30/70": (0.3, 0.7),
}


def _load_llm_preds(k: int) -> dict[tuple[str, str], dict[str, int]]:
    """(rid, msg) -> domain -> predicted_int."""
    with open(grok_path(k)) as f:
        data = json.load(f)
    out: dict[tuple[str, str], dict[str, int]] = {}
    for v in data.values():
        if not isinstance(v, dict):
            continue
        key = (v.get("response_id"), v.get("input_message"))
        if not key:
            continue
        d = {}
        for domain in DOMAINS:
            pr = RATING_MAPS[domain].get(v.get(f"predicted_{domain}"))
            if pr is not None:
                d[domain] = int(pr)
        if d:
            out[key] = d
    return out


def _load_rf_preds(k: int) -> dict[str, dict[tuple[str, str], dict[str, int]]]:
    """feature_set -> (rid, msg) -> domain -> predicted_int."""
    df = pd.read_csv(RF_PRED_CSV)
    df = df[df["k_train"] == k]
    out: dict[str, dict[tuple[str, str], dict[str, int]]] = {}
    for fs in df["feature_set"].unique():
        sub = df[df.feature_set == fs]
        per_key: dict[tuple[str, str], dict[str, int]] = {}
        for _, r in sub.iterrows():
            key = (r["response_id"], r["input_message"])
            per_key.setdefault(key, {})[r["domain"]] = int(r["predicted_num"])
        out[fs] = per_key
    return out


def _ground_truth(k: int) -> dict[tuple[str, str], dict[str, int]]:
    with open(test_path(k)) as f:
        test = json.load(f)
    out: dict[tuple[str, str], dict[str, int]] = {}
    for r in test:
        key = (r["response_id"], r["input_message"])
        d = {}
        for domain in DOMAINS:
            t = r.get("ratings", {}).get(domain)
            n = RATING_MAPS[domain].get(t)
            if n is not None:
                d[domain] = int(n)
        if d:
            out[key] = d
    return out


def _ensemble(rf_pred: int, llm_pred: int, w_rf: float, w_llm: float) -> int:
    blended = w_rf * rf_pred + w_llm * llm_pred
    return int(max(1, min(5, round(blended))))


def main():
    import sys
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(f"Loading predictions for k={k}…")
    gt = _ground_truth(k)
    llm = _load_llm_preds(k)
    rf_all = _load_rf_preds(k)
    print(f"  ground truth keys: {len(gt)}")
    print(f"  LLM keys:          {len(llm)}")
    for fs, kv in rf_all.items():
        print(f"  RF[{fs}] keys: {len(kv)}")

    rows: list[dict] = []
    for fs in FEATURE_SETS:
        rf = rf_all.get(fs)
        if rf is None:
            continue
        keys = sorted(set(gt.keys()) & set(llm.keys()) & set(rf.keys()))
        if not keys:
            continue
        for ens_name, (w_rf, w_llm) in ENSEMBLE_WEIGHTS.items():
            for domain in DOMAINS:
                preds, gts, rids = [], [], []
                for key in keys:
                    if domain not in gt[key] or domain not in rf[key] or domain not in llm[key]:
                        continue
                    p = _ensemble(rf[key][domain], llm[key][domain], w_rf, w_llm)
                    preds.append(p)
                    gts.append(gt[key][domain])
                    rids.append(key[0])
                if not preds:
                    continue
                m = compute_all_metrics(np.array(gts), np.array(preds), np.array(rids))
                rows.append({
                    "k_train": k, "feature_set": fs,
                    "ensemble": ens_name, "domain": domain,
                    "N": m["N"], **m,
                })

    df = pd.DataFrame(rows)
    out_path = figures_path(f"lc_dt10_ensemble_k{k}") + ".csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(df)} rows)")

    # Compact summary table
    print("\n=== Best of {RF only, LLM only, Mean, RF70/30, LLM70/30} per (feature_set, domain) — Accuracy ===")
    summary = (
        df.pivot_table(index=["feature_set", "domain"], columns="ensemble",
                        values="Accuracy", aggfunc="first").round(3)
    )
    print(summary.to_string())

    print("\n=== Same — Macro-F1 ===")
    summary = (
        df.pivot_table(index=["feature_set", "domain"], columns="ensemble",
                        values="F1", aggfunc="first").round(3)
    )
    print(summary.to_string())

    print("\n=== Same — within-participant Spearman ρ (NaN = degenerate constant predictor) ===")
    summary = (
        df.pivot_table(index=["feature_set", "domain"], columns="ensemble",
                        values="Spearman_Rho", aggfunc="first").round(3)
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
