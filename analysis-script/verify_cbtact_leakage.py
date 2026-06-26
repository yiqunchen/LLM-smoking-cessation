#!/usr/bin/env python3
"""Verify whether the cbtact digital-twin prompt leaks the test rating.

Reproduces the prompt-builder logic from
prompt_config.py:generate_digital_twin_cbtact_prompt and asks, for each
canonical test row at every split:

  1. Does the participant's Excel row contain a `message_*` column whose
     text exactly matches the test row's `input_message`?
     (= "rating shown in the prompt for the very item we're predicting")
  2. After whitespace normalization?
  3. With Levenshtein distance ≤ 5?

Reports leakage rate per split + the average number of in-prompt
message_* slots per participant (i.e. how many other rated messages the
LLM sees for context, leakage aside).

Usage:
    uv run python analysis-script/verify_cbtact_leakage.py
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXCEL_PATH = os.path.join(PROJECT_ROOT, "archive", "data", "digitalTwin_msg.xlsx")
SPLIT_ORDER = ["1090", "3070", "5050", "7030", "9010"]


def normalize(s: str) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().lower().split())


def lev(a: str, b: str, cutoff: int = 50) -> int:
    """Cheap iterative Levenshtein with early exit."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > cutoff:
        return cutoff + 1
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * lb
        row_min = i
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur[j] = min(ins, dele, sub)
            if cur[j] < row_min:
                row_min = cur[j]
        prev = cur
        if row_min > cutoff:
            return cutoff + 1
    return prev[lb]


def main():
    print(f"Reading Excel: {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH)
    df["response_id"] = df["response_id"].astype(str)
    msg_cols = [c for c in df.columns if c.startswith("message_")]
    print(f"  shape={df.shape}, message_* columns={len(msg_cols)}")

    # Per-participant Excel history with non-empty messages
    excel_history: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        rid = str(row["response_id"])
        items = []
        for c in msg_cols:
            txt = str(row[c]).strip() if pd.notna(row[c]) else ""
            if txt and txt.lower() != "nan":
                items.append(txt)
        excel_history[rid] = items

    avg_excel_n = sum(len(v) for v in excel_history.values()) / max(1, len(excel_history))
    print(f"  avg non-empty message_* per participant in Excel: {avg_excel_n:.2f}")
    print()

    print(f"{'Split':<6} {'TestN':>6} {'in-prompt slots':>17} {'exact':>8} {'norm':>8} {'≤5 lev':>9} {'no-overlap':>11}")
    print("-" * 80)

    summary_rows = []
    for split in SPLIT_ORDER:
        path = os.path.join(
            PROJECT_ROOT, "data_splits", "canonical",
            f"test_digital_twin_{split}.json",
        )
        with open(path) as f:
            test = json.load(f)

        n = len(test)
        exact_hits = 0
        norm_hits = 0
        fuzzy_hits = 0
        no_overlap = 0  # participant has no Excel history at all
        slots_total = 0

        for r in test:
            rid = str(r["response_id"])
            test_msg = r.get("input_message", "")
            history = excel_history.get(rid, [])
            slots_total += len(history)
            if not history:
                no_overlap += 1
                continue
            if test_msg in history:
                exact_hits += 1
                norm_hits += 1
                fuzzy_hits += 1
                continue
            test_norm = normalize(test_msg)
            hist_norm = [normalize(h) for h in history]
            if test_norm in hist_norm:
                norm_hits += 1
                fuzzy_hits += 1
                continue
            if any(lev(test_norm, h, cutoff=5) <= 5 for h in hist_norm):
                fuzzy_hits += 1

        avg_slots = slots_total / max(1, n)
        print(f"{split:<6} {n:>6} {avg_slots:>17.2f} "
              f"{exact_hits:>4} ({100*exact_hits/n:>4.1f}%) "
              f"{norm_hits:>4} ({100*norm_hits/n:>4.1f}%) "
              f"{fuzzy_hits:>4} ({100*fuzzy_hits/n:>4.1f}%) "
              f"{no_overlap:>4} ({100*no_overlap/n:>4.1f}%)")
        summary_rows.append({
            "split": split,
            "test_n": n,
            "avg_excel_slots_per_test_row": avg_slots,
            "exact_match": exact_hits,
            "exact_pct": 100 * exact_hits / n,
            "normalized_match": norm_hits,
            "normalized_pct": 100 * norm_hits / n,
            "fuzzy_match_lev5": fuzzy_hits,
            "fuzzy_pct": 100 * fuzzy_hits / n,
            "no_excel_history": no_overlap,
        })

    out_path = os.path.join(PROJECT_ROOT, "revision", "figures",
                             "cbtact_leakage_audit.csv")
    pd.DataFrame(summary_rows).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Optional dump: a few example test rows that exhibited exact leakage
    print("\nExample leakage cases (first 5 from dt 7030):")
    path = os.path.join(PROJECT_ROOT, "data_splits", "canonical",
                         "test_digital_twin_7030.json")
    with open(path) as f:
        test = json.load(f)
    shown = 0
    for r in test:
        rid = str(r["response_id"])
        test_msg = r.get("input_message", "")
        history = excel_history.get(rid, [])
        if test_msg and test_msg in history:
            shown += 1
            print(f"  rid={rid[:24]}…  test_msg starts: {test_msg[:90]!r}")
            if shown >= 5:
                break


if __name__ == "__main__":
    main()
