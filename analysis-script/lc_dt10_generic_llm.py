#!/usr/bin/env python3
"""Generic-LLM (zero-shot / few-shot) reference rows for the dt10 learning curve.

Generic-LLM predictions don't depend on training data, so they live at
"k=0" of the dt10 axis. For every (LLM, prompt, dt10 test split, domain)
cell we filter the existing Generic-LLM result JSON to whatever rows
overlap with the dt10 test set, then compute the same metrics as the
RF runs.

Both result files cover the same 319 (response_id, input_message) pairs:
    generic_llm_1_zero_shot_dt7030.json   ← zero-shot
    generic_llm_3_few_shot_dt7030.json    ← few-shot

Outputs:
    revision/figures/lc_dt10_generic_llm.csv
    (long form: split=k_train, llm, prompt, domain, accuracy, f1, qwk, kappa, …)

Usage:
    uv run python analysis-script/lc_dt10_generic_llm.py
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
K_VALUES = [0, 1, 3, 5, 7]

LLM_DIRS = {
    "GPT-4o-mini":     "results_manuscript_gpt-4o-mini",
    "GPT-5":           "results_manuscript_gpt-5",
    "Gemini-2.5-Pro":  "results_manuscript_gemini-2.5-pro",
    "Grok-4-Fast":     "results_manuscript_x-ai_grok-4-fast",
    "DeepSeek-R1":     "results_manuscript_deepseek_deepseek-r1-0528",
}

GENERIC_FILES = {
    "Zero-shot": "generic_llm_1_zero_shot_dt7030.json",
    "Few-shot":  "generic_llm_3_few_shot_dt7030.json",
}


def _load_dt10_test_keys(k: int) -> set[tuple[str, str]]:
    """Load (response_id, input_message) pairs for dt10 test partition at k."""
    if k == 0:
        # Whole universe — train and test of any k cover the same rows.
        path = os.path.join(CANONICAL_DIR, "test_dt10_k1.json")
        with open(path) as f:
            test = json.load(f)
        keys = {(r["response_id"], r["input_message"]) for r in test}
        path2 = os.path.join(CANONICAL_DIR, "train_dt10_k1.json")
        with open(path2) as f:
            train = json.load(f)
        keys |= {(r["response_id"], r["input_message"]) for r in train}
        return keys
    path = os.path.join(CANONICAL_DIR, f"test_dt10_k{k}.json")
    with open(path) as f:
        test = json.load(f)
    return {(r["response_id"], r["input_message"]) for r in test}


def _generic_llm_metrics(model: str, prompt: str, k: int) -> list[dict]:
    file_name = GENERIC_FILES[prompt]
    path = os.path.join(PROJECT_ROOT, LLM_DIRS[model], file_name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    test_keys = _load_dt10_test_keys(k)
    rows: list[dict] = []
    for domain in DOMAINS:
        gt, pred, rids = [], [], []
        for r in data.values():
            if not isinstance(r, dict):
                continue
            key = (r.get("response_id"), r.get("input_message"))
            if key not in test_keys:
                continue
            gt_t = r.get(f"ground_truth_{domain}")
            pr_t = r.get(f"predicted_{domain}")
            gt_n = RATING_MAPS[domain].get(gt_t)
            pr_n = RATING_MAPS[domain].get(pr_t)
            if gt_n is None or pr_n is None:
                continue
            gt.append(gt_n)
            pred.append(pr_n)
            rids.append(r["response_id"])
        if not gt:
            continue
        m = compute_all_metrics(np.array(gt), np.array(pred), np.array(rids))
        rows.append({
            "k_train": k, "method_group": "Generic LLM",
            "feature_set": prompt, "model": model, "domain": domain,
            "covered_n": len(gt), **m,
        })
    return rows


def main():
    rows: list[dict] = []
    for k in K_VALUES:
        for model in LLM_DIRS:
            for prompt in GENERIC_FILES:
                rows.extend(_generic_llm_metrics(model, prompt, k))

    df = pd.DataFrame(rows)
    out = figures_path("lc_dt10_generic_llm") + ".csv"
    df.to_csv(out, index=False)
    print(f"Saved: {out}  ({len(df)} rows)")

    # Quick console summary at k=0 (the "no training" anchor)
    print("\n=== Generic LLM at k=0 (whole dt10 universe), Accuracy ===")
    sub = df[df.k_train == 0]
    pivoted = sub.pivot_table(
        index=["feature_set", "model"], columns="domain",
        values="Accuracy", aggfunc="first",
    ).round(3)
    print(pivoted.to_string())
    print("\n=== Generic LLM at k=0, Macro-F1 ===")
    pivoted = sub.pivot_table(
        index=["feature_set", "model"], columns="domain",
        values="F1", aggfunc="first",
    ).round(3)
    print(pivoted.to_string())
    print("\n=== Generic LLM at k=0, QWK ===")
    pivoted = sub.pivot_table(
        index=["feature_set", "model"], columns="domain",
        values="QWK", aggfunc="first",
    ).round(3)
    print(pivoted.to_string())
    print("\nNote: each cell averaged over the 319-item (rid, message) overlap "
          "with dt10. Generic LLM predictions are split-invariant — values "
          "differ across k only because the test partition (denominator) differs.")


if __name__ == "__main__":
    main()
