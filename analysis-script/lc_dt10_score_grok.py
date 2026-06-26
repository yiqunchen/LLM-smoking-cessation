#!/usr/bin/env python3
"""Score the dt10 LLM-DT (Grok-4-Fast) runs and append to lc_dt10_llmdt.csv.

For every k for which a digital_twin_dt10_k{k}.json exists in
results_manuscript_x-ai_grok-4-fast/, this script:
  1. Loads the predictions
  2. Filters to (response_id, input_message) keys present in
     test_dt10_k{k}.json (drops any stray rows)
  3. Computes Accuracy / Macro-F1 / QWK / Cohen's kappa per domain
  4. Appends rows to revision/figures/lc_dt10_llmdt.csv with a
     method_group of "LLM-DT canonical (Grok, dt10)" so they don't
     collide with the existing cbtact rows.

Then prints a side-by-side comparison vs the RF curve from
lc_dt10_rf.csv at the same k values.

Usage:
    uv run python analysis-script/lc_dt10_score_grok.py
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
CANONICAL_DIR = os.path.join(PROJECT_ROOT, "data_splits", "canonical")
GROK_DIR = os.path.join(PROJECT_ROOT, "results_manuscript_x-ai_grok-4-fast")
LLMDT_CSV = figures_path("lc_dt10_llmdt") + ".csv"
RF_CSV = figures_path("lc_dt10_rf") + ".csv"
K_VALUES = [1, 3, 5, 7]


def _test_keys(k: int) -> set[tuple[str, str]]:
    p = os.path.join(CANONICAL_DIR, f"test_dt10_k{k}.json")
    with open(p) as f:
        test = json.load(f)
    return {(r["response_id"], r["input_message"]) for r in test}


def score_grok_k(k: int) -> list[dict]:
    p = os.path.join(GROK_DIR, f"digital_twin_dt10_k{k}.json")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        data = json.load(f)
    test_keys = _test_keys(k)

    rows: list[dict] = []
    for domain in DOMAINS:
        gt, pred, rids = [], [], []
        for v in data.values():
            if not isinstance(v, dict):
                continue
            key = (v.get("response_id"), v.get("input_message"))
            if key not in test_keys:
                continue
            gt_t = v.get(f"ground_truth_{domain}")
            pr_t = v.get(f"predicted_{domain}")
            gt_n = RATING_MAPS[domain].get(gt_t)
            pr_n = RATING_MAPS[domain].get(pr_t)
            if gt_n is None or pr_n is None:
                continue
            gt.append(gt_n); pred.append(pr_n); rids.append(key[0])
        if not gt:
            continue
        m = compute_all_metrics(np.array(gt), np.array(pred), np.array(rids))
        rows.append({
            "k_train": k,
            "method_group": "LLM-DT canonical (Grok, dt10)",
            "model": "Grok-4-Fast",
            "domain": domain,
            "covered_n": len(gt),
            **m,
        })
    return rows


def main():
    new_rows: list[dict] = []
    for k in K_VALUES:
        new_rows.extend(score_grok_k(k))

    if not new_rows:
        print("No dt10 Grok files found.")
        return

    new_df = pd.DataFrame(new_rows)

    # Append to lc_dt10_llmdt.csv (drop any prior rows from same method_group)
    if os.path.exists(LLMDT_CSV):
        existing = pd.read_csv(LLMDT_CSV)
        existing = existing[existing["method_group"] != "LLM-DT canonical (Grok, dt10)"]
        out = pd.concat([existing, new_df], ignore_index=True)
    else:
        out = new_df
    out.to_csv(LLMDT_CSV, index=False)
    print(f"Saved {LLMDT_CSV}  ({len(new_df)} new rows, {len(out)} total)")

    # Print scored summary
    print("\n=== LLM-DT canonical (Grok, dt10) ===")
    for k in K_VALUES:
        sub = new_df[new_df.k_train == k]
        if sub.empty:
            continue
        for domain in DOMAINS:
            row = sub[sub.domain == domain].iloc[0] if len(sub[sub.domain == domain]) else None
            if row is None:
                continue
            print(f"  k={k}  {domain:9s}  N={int(row['covered_n']):4d}  "
                  f"acc={row['Accuracy']:.3f}  f1={row['F1']:.3f}  "
                  f"qwk={row['QWK']:.3f}  kappa={row['Kappa']:.3f}")

    # Side-by-side: RF vs LLM-DT canonical Grok (only k values we have)
    rf = pd.read_csv(RF_CSV)
    rf["k_train"] = rf["k_train"].astype(int)
    print("\n=== RF best (across feature sets) vs LLM-DT canonical Grok dt10 ===")
    print(f"{'k':>3} {'domain':10s}  {'RF best':18s} {'RF acc':>7} {'LLM acc':>7} {'Δ':>6}  {'RF F1':>6} {'LLM F1':>6}  {'RF QWK':>7} {'LLM QWK':>7}")
    for k in K_VALUES:
        if k not in new_df.k_train.unique():
            continue
        for domain in DOMAINS:
            rf_panel = rf[(rf.k_train == k) & (rf.domain == domain)
                           & (rf.feature_set != "Majority class")]
            if rf_panel.empty:
                continue
            best = rf_panel.loc[rf_panel["Accuracy"].astype(float).idxmax()]
            llm_row = new_df[(new_df.k_train == k) & (new_df.domain == domain)]
            if llm_row.empty:
                continue
            llm = llm_row.iloc[0]
            d_acc = float(llm["Accuracy"]) - float(best["Accuracy"])
            print(f"{k:>3} {domain:10s}  RF/{best['feature_set']:<14s} "
                  f"{best['Accuracy']:>7.3f} {llm['Accuracy']:>7.3f} {d_acc:>+6.3f}  "
                  f"{best['F1']:>6.3f} {llm['F1']:>6.3f}  "
                  f"{best['QWK']:>7.3f} {llm['QWK']:>7.3f}")


if __name__ == "__main__":
    main()
