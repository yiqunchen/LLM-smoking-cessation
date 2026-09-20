#!/usr/bin/env python3
"""LLM-PP (Excel-7 profile) reference rows for the dt10 curve.

The original cbtact LLM-PP runs (``digital_twin_4_cbtact_*.json``) use
each participant's 7 Excel-history messages as the in-context profile —
i.e. messages 1–7 of 10. Those 7 history messages are *fully disjoint*
from the canonical "evaluation" pool (messages 8–10), as verified by
``verify_cbtact_leakage.py``.

For an apples-to-apples comparison with the dt10 RF curve we therefore
restrict the LLM-PP cbtact predictions to dt10 test rows whose message
text matches a row in processed_llm_data.json — i.e. the "evaluation
half" of the dataset (no Excel-7 leakage). At each k, we further
restrict to rows that are in dt10's ``test_dt10_k{k}.json`` test
partition.

The LLM-PP cbtact effectively sits at "k_excel = 7" — it always sees the
full 7 Excel history messages as profile, so its predictions don't vary
with k. We report the same number at every k so the reader can read it
as a horizontal reference line on the curve.

Outputs:
    revision/figures/lc_dt10_llmdt.csv

Usage:
    uv run python analysis-script/lc_dt10_llmdt.py
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
PROCESSED = os.path.join(PROJECT_ROOT, "archive", "data", "processed_llm_data.json")

LLM_DIRS = {
    "GPT-4o-mini":     "results_manuscript_gpt-4o-mini",
    "GPT-5":           "results_manuscript_gpt-5",
    "Gemini-2.5-Pro":  "results_manuscript_gemini-2.5-pro",
    "Grok-4-Fast":     "results_manuscript_x-ai_grok-4-fast",
    "DeepSeek-R1":     "results_manuscript_deepseek_deepseek-r1-0528",
}
CBTACT_FILES = ["digital_twin_4_cbtact_1090.json",
                "digital_twin_4_cbtact_3070.json",
                "digital_twin_4_cbtact_7030.json",
                "digital_twin_4_cbtact_9010.json"]
K_VALUES = [0, 1, 3, 5, 7]


def _processed_keys() -> set[tuple[str, str]]:
    """The (rid, msg) universe of the canonical evaluation pool — dt10 rows
    NOT in Excel-7. Restricting to this universe avoids leakage."""
    with open(PROCESSED) as f:
        recs = json.load(f)
    return {(r["response_id"], r["input_message"]) for r in recs}


def _dt10_test_keys(k: int) -> set[tuple[str, str]]:
    if k == 0:
        # whole dt10 universe (train + test of any k)
        with open(os.path.join(CANONICAL_DIR, "test_dt10_k1.json")) as f:
            test = json.load(f)
        with open(os.path.join(CANONICAL_DIR, "train_dt10_k1.json")) as f:
            train = json.load(f)
        return ({(r["response_id"], r["input_message"]) for r in test}
                | {(r["response_id"], r["input_message"]) for r in train})
    with open(os.path.join(CANONICAL_DIR, f"test_dt10_k{k}.json")) as f:
        test = json.load(f)
    return {(r["response_id"], r["input_message"]) for r in test}


def _gather_llmdt_predictions(model: str) -> dict[tuple[str, str], dict]:
    """Merge the four cbtact JSONs for one model into a single dict
    keyed by (rid, input_message)."""
    out: dict[tuple[str, str], dict] = {}
    for fname in CBTACT_FILES:
        path = os.path.join(PROJECT_ROOT, LLM_DIRS[model], fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for r in data.values():
            if not isinstance(r, dict):
                continue
            key = (r.get("response_id"), r.get("input_message"))
            if key in out:
                continue
            out[key] = r
    return out


def main():
    proc_keys = _processed_keys()
    print(f"Canonical eval-pool universe (msgs 8-10): {len(proc_keys)} rows")

    rows: list[dict] = []
    for model in LLM_DIRS:
        preds = _gather_llmdt_predictions(model)
        if not preds:
            continue
        coverage = len(preds.keys() & proc_keys)
        print(f"\n{model}: total pred rows = {len(preds)}; "
              f"intersect canonical eval-pool = {coverage}")

        for k in K_VALUES:
            test_keys = _dt10_test_keys(k) & proc_keys
            for domain in DOMAINS:
                gt, pred, rids = [], [], []
                for key in test_keys:
                    r = preds.get(key)
                    if r is None:
                        continue
                    gt_t = r.get(f"ground_truth_{domain}")
                    pr_t = r.get(f"predicted_{domain}")
                    gt_n = RATING_MAPS[domain].get(gt_t)
                    pr_n = RATING_MAPS[domain].get(pr_t)
                    if gt_n is None or pr_n is None:
                        continue
                    gt.append(gt_n)
                    pred.append(pr_n)
                    rids.append(key[0])
                if not gt:
                    continue
                m = compute_all_metrics(np.array(gt), np.array(pred), np.array(rids))
                rows.append({
                    "k_train": k, "method_group": "LLM-PP (cbtact, Excel-7)",
                    "model": model, "domain": domain,
                    "covered_n": len(gt), **m,
                })

    df = pd.DataFrame(rows)
    out = figures_path("lc_dt10_llmdt") + ".csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}  ({len(df)} rows)")

    # Summary on the canonical eval-pool universe (k=0)
    sub = df[df.k_train == 0]
    for metric in ["Accuracy", "F1", "QWK"]:
        wide = sub.pivot_table(
            index="model", columns="domain", values=metric, aggfunc="first"
        ).round(3)
        print(f"\n=== LLM-PP cbtact (Excel-7 profile), {metric} ===")
        print(wide.to_string())


if __name__ == "__main__":
    main()
