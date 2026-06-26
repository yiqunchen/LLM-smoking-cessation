#!/usr/bin/env python3
"""Complete head-to-head tables across method groups, ratios, and domains.

EVERY (split × method_group × variant × model) gets its own row — not just
the best variant per group. Three tables share the same row structure but
display different metrics:

    T1:  Accuracy
    T2:  Macro-F1
    T3:  Quadratic-Weighted Kappa  (companion: Cohen's Kappa)

Method groups and the variants enumerated within each:
    Generic LLM
        Zero-shot × {GPT-4o-mini, GPT-5, Gemini-2.5-Pro, Grok-4-Fast,
                     DeepSeek-R1}
        Few-shot  × same 5 LLMs
    Supervised RF
        feature sets: Demographics, Avg-History, Demo+History,
                      Embedding, Embedding+Demo
    LLM Digital Twin
        × {GPT-4o-mini, GPT-5, Gemini-2.5-Pro, Grok-4-Fast, DeepSeek-R1}

Outputs:
    revision/figures/lc_full_methods.csv
    revision/figures/lc_full_methods_t1_accuracy.md
    revision/figures/lc_full_methods_t2_f1.md
    revision/figures/lc_full_methods_t3_qwk_kappa.md

Usage:
    uv run python analysis-script/lc_full_method_tables.py
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
LC_CSV = figures_path("lc_rf_vs_llm") + ".csv"

DOMAIN_ORDER = ["content", "coping", "quitting"]
DOMAIN_HEADER = {"content": "Content", "coping": "Coping",
                 "quitting": "Quitting"}
SPLIT_ORDER = ["1090", "3070", "5050", "7030", "9010"]
LLM_DIRS = {
    "GPT-4o-mini":     "results_manuscript_gpt-4o-mini",
    "GPT-5":           "results_manuscript_gpt-5",
    "Gemini-2.5-Pro":  "results_manuscript_gemini-2.5-pro",
    "Grok-4-Fast":     "results_manuscript_x-ai_grok-4-fast",
    "DeepSeek-R1":     "results_manuscript_deepseek_deepseek-r1-0528",
}
LLM_ORDER = list(LLM_DIRS.keys())
GENERIC_FILES = {
    "Zero-shot": "generic_llm_1_zero_shot_dt7030.json",
    "Few-shot":  "generic_llm_3_few_shot_dt7030.json",
}
RF_FEATURE_ORDER = [
    "Demographics", "Avg-History", "Demo+History",
    "Embedding", "Embedding+Demo",
]


# ---------------------------------------------------------------------------
# Per-split training-data summary
# ---------------------------------------------------------------------------

def split_train_stats(split: str) -> tuple[int, int]:
    base = os.path.join(PROJECT_ROOT, "data_splits", "canonical")
    with open(os.path.join(base, f"train_digital_twin_{split}.json")) as f:
        train = json.load(f)
    pids = {r["response_id"] for r in train}
    return len(pids), len(train)


def split_test_keys(split: str) -> set[tuple[str, str]]:
    base = os.path.join(PROJECT_ROOT, "data_splits", "canonical")
    with open(os.path.join(base, f"test_digital_twin_{split}.json")) as f:
        test = json.load(f)
    return {(r["response_id"], r["input_message"]) for r in test}


# ---------------------------------------------------------------------------
# Generic LLM evaluation
# ---------------------------------------------------------------------------

def generic_llm_metrics(model: str, method: str, split: str
                         ) -> dict[str, dict] | None:
    file_name = GENERIC_FILES[method]
    path = os.path.join(PROJECT_ROOT, LLM_DIRS[model], file_name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    test_keys = split_test_keys(split)
    out: dict[str, dict] = {}
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
        m["covered_n"] = len(gt)
        out[domain] = m
    return out or None


# ---------------------------------------------------------------------------
# Long-form assembly
# ---------------------------------------------------------------------------

def build_long_df() -> pd.DataFrame:
    rows: list[dict] = []

    # Generic LLM × splits
    for split in SPLIT_ORDER:
        for method in GENERIC_FILES:
            for model in LLM_ORDER:
                m = generic_llm_metrics(model, method, split)
                if m is None:
                    continue
                for domain in DOMAINS:
                    if domain not in m:
                        continue
                    rows.append({
                        "split": split,
                        "method_group": "Generic LLM",
                        "variant": method,
                        "model": model,
                        "domain": domain,
                        "test_n": int(m[domain].get("covered_n", 0)),
                        "Accuracy": m[domain].get("Accuracy", np.nan),
                        "F1": m[domain].get("F1", np.nan),
                        "QWK": m[domain].get("QWK", np.nan),
                        "Kappa": m[domain].get("Kappa", np.nan),
                    })

    # RF and LLM-DT × splits — from the existing CSV
    df = pd.read_csv(LC_CSV)
    df["split"] = df["split"].astype(str)
    for _, r in df.iterrows():
        if r["method"] == "RF":
            mg = "Supervised RF"; variant = r["feature_set"]; model = ""
        elif r["method"] == "LLM-DT":
            mg = "LLM Digital Twin"; variant = "Digital Twin"
            model = r["feature_set"]
        else:
            continue
        rows.append({
            "split": r["split"],
            "method_group": mg,
            "variant": variant,
            "model": model,
            "domain": r["domain"],
            "test_n": int(r["test_n"]),
            "Accuracy": r.get("Accuracy", np.nan),
            "F1": r.get("F1", np.nan),
            "QWK": r.get("QWK", np.nan),
            "Kappa": r.get("Kappa", np.nan),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.3f}"


def _row_label(method_group: str, variant: str, model: str) -> str:
    if method_group == "Generic LLM":
        return f"{variant} · {model}"
    if method_group == "Supervised RF":
        return f"RF · {variant}"
    if method_group == "LLM Digital Twin":
        return f"DT · {model}"
    return f"{method_group} · {variant} · {model}"


def _variant_sort_key(method_group: str, variant: str, model: str) -> tuple:
    # Generic LLM rows: order by variant (Zero-shot, Few-shot), then model
    if method_group == "Generic LLM":
        v_order = ["Zero-shot", "Few-shot"].index(variant) if variant in ["Zero-shot", "Few-shot"] else 99
        m_order = LLM_ORDER.index(model) if model in LLM_ORDER else 99
        return (0, v_order, m_order)
    if method_group == "Supervised RF":
        v_order = (RF_FEATURE_ORDER.index(variant)
                   if variant in RF_FEATURE_ORDER else 99)
        return (1, v_order, 0)
    if method_group == "LLM Digital Twin":
        m_order = LLM_ORDER.index(model) if model in LLM_ORDER else 99
        return (2, 0, m_order)
    return (3, 0, 0)


def render_table(df_long: pd.DataFrame, metric_keys: list[str],
                  title: str, source_path: str) -> str:
    """One markdown document containing one table per metric.

    Rows: (split, method_group, variant, model). Columns: train pp / msgs,
    test n, then metric values for Content / Coping / Quitting.
    """
    out: list[str] = [f"# {title}\n"]
    out.append(
        "Every variant of every method group is reported at every split. "
        "Train n is shown as **participants × messages**. Generic LLM "
        "rows have no training (zero-shot has none; few-shot uses a "
        "fixed pool of in-prompt exemplars).\n"
    )

    # Pivot once: index = (split, method_group, variant, model), cols = domain
    long_with_metric = df_long.copy()
    pivots = {}
    for metric in metric_keys:
        wide = (
            long_with_metric.pivot_table(
                index=["split", "method_group", "variant", "model"],
                columns="domain", values=metric, aggfunc="first",
            )
            .reset_index()
        )
        pivots[metric] = wide

    headers_per_metric = {
        "Accuracy": ["Content acc", "Coping acc", "Quitting acc"],
        "F1":       ["Content F1", "Coping F1", "Quitting F1"],
        "QWK":      ["Content QWK", "Coping QWK", "Quitting QWK"],
        "Kappa":    ["Content κ", "Coping κ", "Quitting κ"],
    }

    for metric in metric_keys:
        wide = pivots[metric]
        out.append(f"\n## Metric: {metric}\n")

        col_headers = ["Split", "Train pp × msgs", "Method · variant"]
        col_headers += headers_per_metric[metric]
        out.append("| " + " | ".join(col_headers) + " |")
        out.append("|" + "|".join(["---"] * len(col_headers)) + "|")

        for split in SPLIT_ORDER:
            n_pp, n_msg = split_train_stats(split)
            train_label_with = (
                f"{n_pp} × {n_msg}" if n_pp > 0 else "0 × 0  (no profile)"
            )

            sub = wide[wide.split == split].copy()
            sub["sort_key"] = sub.apply(
                lambda r: _variant_sort_key(r.method_group, r.variant, r.model),
                axis=1,
            )
            sub = sub.sort_values("sort_key").drop(columns="sort_key")

            for _, r in sub.iterrows():
                if r["method_group"] == "Generic LLM":
                    train_disp = "0 × 0  (no training)"
                else:
                    train_disp = train_label_with
                label = _row_label(r["method_group"], r["variant"], r["model"])
                cells = [
                    _fmt(r.get("content")),
                    _fmt(r.get("coping")),
                    _fmt(r.get("quitting")),
                ]
                out.append(
                    f"| {split} | {train_disp} | {label} | "
                    + " | ".join(cells) + " |"
                )
            # blank separator row between splits
            out.append("|     |     |     |     |     |     |")

    out.append("\n## Coverage and caveats")
    out.append(
        "- **Generic LLM** rows are computed by filtering the dt7030-aligned "
        "zero-shot / few-shot result files to the response_ids in each "
        "split's test partition. Coverage is the file's 319 unique items, "
        "which is the FULL dt 7030 / dt 9010 test set but a 319-item subset "
        "of dt 1090 / dt 3070 / dt 5050 (those have larger test partitions). "
        "Within each split, RF and LLM-DT cover the *full* test partition "
        "(916 / 894 / 593 / 323 / 301 items at 1090 / 3070 / 5050 / 7030 / "
        "9010). Generic LLM numbers should therefore be read on a slightly "
        "different denominator at the smaller-train splits — accuracy "
        "differences across splits within a Generic LLM row reflect that.\n"
        "- **Supervised RF** is absent at split 1090 because the canonical "
        "1090 split has 0 training items.\n"
        "- **DeepSeek-R1** only ran the 7030 LLM-DT split, so it is "
        "absent from LLM-DT rows at 1090, 3070, 9010 (and from 5050 "
        "where no LLM-DT was run for any model).\n"
        f"- Source CSVs: `revision/figures/lc_rf_vs_llm.csv`, "
        f"`{os.path.basename(source_path)}`."
    )
    return "\n".join(out)


def main():
    print("Building long-form table…")
    long_df = build_long_df()
    print(f"  {len(long_df)} rows")

    out_csv = figures_path("lc_full_methods") + ".csv"
    long_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    md_t1 = render_table(long_df, ["Accuracy"],
                          "Table 1 — Accuracy by domain (every variant)",
                          out_csv)
    md_t2 = render_table(long_df, ["F1"],
                          "Table 2 — Macro-F1 by domain (every variant)",
                          out_csv)
    md_t3 = render_table(long_df, ["QWK", "Kappa"],
                          "Table 3 — Quadratic-Weighted Kappa & Cohen's "
                          "Kappa by domain (every variant)",
                          out_csv)

    for name, body in [
        ("lc_full_methods_t1_accuracy", md_t1),
        ("lc_full_methods_t2_f1", md_t2),
        ("lc_full_methods_t3_qwk_kappa", md_t3),
    ]:
        path = figures_path(name) + ".md"
        with open(path, "w") as f:
            f.write(body)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
