#!/usr/bin/env python3
"""Three head-to-head tables: Generic LLM vs Supervised RF vs LLM Digital Twin.

For each canonical digital-twin split (1090 / 3070 / 7030 / 9010) and each
method group, pick the best-performing variant by the table's metric and
report the per-domain value alongside the training-data exposure
(participants × messages).

Method groups:
    GenericLLM  — Zero-shot or Few-shot (no per-participant profile);
                  available only at the dt7030-aligned predictions but
                  evaluated on every split's test rows that the file covers
    RF          — Random Forest on Demographics / Avg-History / Embedding /
                  Embedding+Demo features
    LLM-DT      — LLM Digital Twin (5 LLMs)

Metrics → tables:
    T1: Accuracy             revision/figures/lc_three_methods_t1_accuracy.{md,csv}
    T2: Macro-F1             revision/figures/lc_three_methods_t2_f1.{md,csv}
    T3: QWK + Kappa          revision/figures/lc_three_methods_t3_qwk_kappa.{md,csv}

Usage:
    uv run python analysis-script/lc_three_method_tables.py
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
DOMAIN_TITLE = {"content": "Content", "coping": "Coping", "quitting": "Quitting"}
SPLIT_ORDER = ["1090", "3070", "5050", "7030", "9010"]
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_train_stats(split: str) -> tuple[int, int, float]:
    """(participants, messages, avg messages per participant) for the given
    digital-twin split's training partition."""
    base = os.path.join(PROJECT_ROOT, "data_splits", "canonical")
    with open(os.path.join(base, f"train_digital_twin_{split}.json")) as f:
        train = json.load(f)
    pids = {r["response_id"] for r in train}
    n_pp = len(pids)
    n_msg = len(train)
    avg = n_msg / n_pp if n_pp else 0.0
    return n_pp, n_msg, avg


def _split_test_keys(split: str) -> set[tuple[str, str]]:
    base = os.path.join(PROJECT_ROOT, "data_splits", "canonical")
    with open(os.path.join(base, f"test_digital_twin_{split}.json")) as f:
        test = json.load(f)
    return {(r["response_id"], r["input_message"]) for r in test}


def _generic_llm_metrics(model: str, method: str, split: str) -> dict | None:
    """Compute per-domain metrics for a Generic LLM (zero/few-shot) on the
    intersection of its predictions and the given dt split's test rows."""
    file_name = GENERIC_FILES[method]
    path = os.path.join(PROJECT_ROOT, LLM_DIRS[model], file_name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    test_keys = _split_test_keys(split)
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


def _select_best_in_group(rows: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    """For each (split, method_group), keep the variant whose AVERAGE
    `metric_key` across content/coping/quitting is the highest.

    Returns the full per-domain rows for that winning variant so the
    table cell for each domain comes from one consistent model+detail.
    """
    if rows.empty:
        return rows
    valid = rows.dropna(subset=[metric_key]).copy()
    if valid.empty:
        return valid
    agg = (
        valid.groupby(
            ["split", "method_group", "method_detail", "model"],
            as_index=False, dropna=False,
        )[metric_key]
        .mean()
        .rename(columns={metric_key: "_avg"})
    )
    winners = (
        agg.sort_values("_avg", ascending=False)
        .groupby(["split", "method_group"], as_index=False, dropna=False)
        .first()
        [["split", "method_group", "method_detail", "model"]]
    )
    keep = valid.merge(
        winners,
        on=["split", "method_group", "method_detail", "model"],
        how="inner",
    )
    return keep


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def gather_generic_rows() -> pd.DataFrame:
    rows: list[dict] = []
    for split in SPLIT_ORDER:
        for method in GENERIC_FILES:
            for model in LLM_DIRS:
                metrics = _generic_llm_metrics(model, method, split)
                if metrics is None:
                    continue
                for domain, m in metrics.items():
                    rows.append({
                        "split": split,
                        "method_group": "Generic LLM",
                        "method_detail": method,
                        "model": model,
                        "domain": domain,
                        "test_n": int(m.get("covered_n", 0)),
                        "Accuracy": m.get("Accuracy", np.nan),
                        "F1": m.get("F1", np.nan),
                        "QWK": m.get("QWK", np.nan),
                        "Kappa": m.get("Kappa", np.nan),
                    })
    return pd.DataFrame(rows)


def gather_rf_and_llmdt_rows() -> pd.DataFrame:
    df = pd.read_csv(LC_CSV)
    df["split"] = df["split"].astype(str)
    rows: list[dict] = []
    for _, r in df.iterrows():
        if r["method"] == "RF":
            mg, md = "Supervised RF", r["feature_set"]
            model = ""
        elif r["method"] == "LLM-DT":
            mg, md = "LLM Digital Twin", "Digital Twin"
            model = r["feature_set"]
        else:
            continue
        rows.append({
            "split": r["split"],
            "method_group": mg,
            "method_detail": md,
            "model": model,
            "domain": r["domain"],
            "test_n": int(r["test_n"]),
            "Accuracy": r.get("Accuracy", np.nan),
            "F1": r.get("F1", np.nan),
            "QWK": r.get("QWK", np.nan),
            "Kappa": r.get("Kappa", np.nan),
        })
    return pd.DataFrame(rows)


def build_long(df_all: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    """Per (split, method_group, domain), pick the best variant by metric_key."""
    keep_rows: list[dict] = []
    for split in SPLIT_ORDER:
        for mg in ["Generic LLM", "Supervised RF", "LLM Digital Twin"]:
            sub = df_all[(df_all.split == split) & (df_all.method_group == mg)]
            if sub.empty:
                continue
            best = _select_best_in_group(sub, metric_key)
            for _, r in best.iterrows():
                keep_rows.append(dict(r))
    return pd.DataFrame(keep_rows)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.3f}"


def _row_label(method_group: str, detail: str, model: str) -> str:
    if method_group == "Generic LLM":
        return f"Generic LLM · {detail} · {model}"
    if method_group == "Supervised RF":
        return f"Supervised RF · {detail}"
    if method_group == "LLM Digital Twin":
        return f"LLM Digital Twin · {model}"
    return f"{method_group} · {detail} · {model}"


def render_metric_table(df: pd.DataFrame, metric_keys: list[str],
                         title: str, source_path: str) -> str:
    """Produce a markdown table with rows = (split, method_group winner)
    and columns = Train (pp × msgs) | Content | Coping | Quitting | (per metric)."""
    out: list[str] = [f"# {title}\n"]
    out.append(
        "Within each canonical digital-twin split, the entry shown is the "
        "best variant of that method group by the table's metric. Train n "
        "is reported as **participants × messages** (and avg messages per "
        "participant). Generic LLM rows have no training (zero-shot or a "
        "fixed pool of in-prompt few-shot exemplars).\n"
    )

    headers_per_metric = {
        "Accuracy": ["Content acc", "Coping acc", "Quitting acc"],
        "F1":       ["Content F1", "Coping F1", "Quitting F1"],
        "QWK":      ["Content QWK", "Coping QWK", "Quitting QWK"],
        "Kappa":    ["Content κ", "Coping κ", "Quitting κ"],
    }

    for metric_key in metric_keys:
        out.append(f"\n## Metric: {metric_key}\n")

        col_headers = ["Split", "Train pp × msgs", "Method (best variant)"]
        col_headers += headers_per_metric[metric_key]

        out.append("| " + " | ".join(col_headers) + " |")
        out.append("|" + "|".join(["---"] * len(col_headers)) + "|")

        wide = (
            df.pivot_table(
                index=["split", "method_group", "method_detail", "model"],
                columns="domain", values=metric_key, aggfunc="first",
            )
            .reset_index()
        )
        for split in SPLIT_ORDER:
            n_pp, n_msg, avg = _split_train_stats(split)
            if n_pp == 0:
                train_label = "0 × 0  (no profile)"
            else:
                train_label = f"{n_pp} × {n_msg}  ({avg:.2f} msg/pp)"
            for mg in ["Generic LLM", "Supervised RF", "LLM Digital Twin"]:
                rows_mg = wide[(wide.split == split) & (wide.method_group == mg)]
                if rows_mg.empty:
                    continue
                # If multiple "best" rows across domains, summarize them as one
                # row per (detail, model) and pick the one with the best mean
                # across domains (already filtered upstream).
                row = rows_mg.iloc[0]
                if mg == "Generic LLM":
                    train_disp = "0 × 0  (no training)"
                else:
                    train_disp = train_label
                label = _row_label(mg, row["method_detail"], row["model"])
                cells = [
                    _fmt(row.get("content")),
                    _fmt(row.get("coping")),
                    _fmt(row.get("quitting")),
                ]
                out.append(
                    f"| {split} | {train_disp} | {label} | "
                    + " | ".join(cells) + " |"
                )
        out.append("")

    out.append("\n## Notes")
    out.append(
        "- Generic LLM rows are computed by filtering the dt7030-aligned "
        "zero-shot / few-shot result files to the response_ids in each "
        "split's test partition. At splits other than 7030 this covers a "
        "subset of the test set, not all of it.\n"
        "- 1090 has 0 training items by canonical-split design, so RF rows "
        "are absent at that split.\n"
        "- DeepSeek-R1 only ran the 7030 LLM-DT split.\n"
        "- Source CSVs: `revision/figures/lc_rf_vs_llm.csv`, "
        f"`{os.path.basename(source_path)}`."
    )
    return "\n".join(out)


def main():
    print("Gathering Generic LLM rows…")
    gen_df = gather_generic_rows()
    print(f"  {len(gen_df)} rows")
    print("Gathering RF and LLM-DT rows from lc_rf_vs_llm.csv…")
    rf_dt_df = gather_rf_and_llmdt_rows()
    print(f"  {len(rf_dt_df)} rows")
    df_all = pd.concat([gen_df, rf_dt_df], ignore_index=True)

    out_csv = figures_path("lc_three_methods") + ".csv"
    df_all.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}  ({len(df_all)} rows)")

    # Per-metric "best variant per (split, group, domain)"
    best_acc = build_long(df_all, "Accuracy")
    best_f1  = build_long(df_all, "F1")
    best_qwk = build_long(df_all, "QWK")
    best_kap = build_long(df_all, "Kappa")

    md_t1 = render_metric_table(
        best_acc, ["Accuracy"],
        "Table 1 — Accuracy by domain (best variant per method group)",
        out_csv,
    )
    md_t2 = render_metric_table(
        best_f1, ["F1"],
        "Table 2 — Macro-F1 by domain (best variant per method group)",
        out_csv,
    )
    md_t3 = render_metric_table(
        pd.concat([best_qwk, best_kap]), ["QWK", "Kappa"],
        "Table 3 — Quadratic Weighted Kappa & Cohen's Kappa by domain "
        "(best variant per method group)",
        out_csv,
    )

    paths = [
        ("lc_three_methods_t1_accuracy.md", md_t1),
        ("lc_three_methods_t2_f1.md", md_t2),
        ("lc_three_methods_t3_qwk_kappa.md", md_t3),
    ]
    for name, body in paths:
        out_path = figures_path(name.replace(".md", "")) + ".md"
        with open(out_path, "w") as f:
            f.write(body)
        print(f"Saved: {out_path}")

    # Console preview of T1
    print("\n" + md_t1)


if __name__ == "__main__":
    main()
