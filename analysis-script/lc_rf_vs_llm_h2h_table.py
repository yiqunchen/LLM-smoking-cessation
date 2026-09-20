#!/usr/bin/env python3
"""Head-to-head tables: best Random Forest vs best LLM Digital Twin.

For each (split, domain) cell we pick the BEST RF (across feature sets)
and the BEST LLM-DT (across LLMs) by the chosen metric, then report
both side-by-side. Each metric gets its own table so the "winner"
column is unambiguous.

Outputs:
    revision/figures/lc_rf_vs_llm_h2h.md          (human-readable tables)
    revision/figures/lc_rf_vs_llm_h2h.csv         (long-form, all metrics)

Usage:
    uv run python analysis-script/lc_rf_vs_llm_h2h_table.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_utils import figures_path  # noqa: E402

CSV_IN = figures_path("lc_rf_vs_llm") + ".csv"
DOMAIN_ORDER = ["content", "coping", "quitting"]
DOMAIN_TITLE = {"content": "Content", "coping": "Coping",
                "quitting": "Quitting"}
SPLIT_ORDER = ["1090", "3070", "5050", "7030", "9010"]
SPLIT_LABEL = {"1090": "1090 (10 / 90)", "3070": "3070 (30 / 70)",
               "5050": "5050 (50 / 50)", "7030": "7030 (70 / 30)",
               "9010": "9010 (90 / 10)"}
TRAIN_N = {"1090": 0, "3070": 22, "5050": 323, "7030": 593, "9010": 615}

METRICS = ["Accuracy", "F1", "Spearman_Rho"]
METRIC_LABEL = {"Accuracy": "Accuracy", "F1": "Macro-F1",
                "Spearman_Rho": "Within-participant Spearman ρ"}


def best_row(panel: pd.DataFrame, metric: str) -> pd.Series | None:
    """Return the row in `panel` with the highest value of `metric`,
    skipping NaNs. Returns None if every value is NaN or the panel is empty."""
    if panel.empty:
        return None
    valid = panel.dropna(subset=[metric])
    if valid.empty:
        return None
    idx = valid[metric].astype(float).idxmax()
    return valid.loc[idx]


def build_long_table(df: pd.DataFrame) -> pd.DataFrame:
    """Long-form: one row per (split, domain, metric) with best RF + best LLM."""
    rows: list[dict] = []
    for split in SPLIT_ORDER:
        for domain in DOMAIN_ORDER:
            for metric in METRICS:
                rf_panel = df[(df.split == split) & (df.domain == domain)
                              & (df.method == "RF")]
                llm_panel = df[(df.split == split) & (df.domain == domain)
                               & (df.method == "LLM-DT")]
                rf_best = best_row(rf_panel, metric)
                llm_best = best_row(llm_panel, metric)

                rf_val = float(rf_best[metric]) if rf_best is not None else np.nan
                llm_val = float(llm_best[metric]) if llm_best is not None else np.nan
                rf_name = rf_best["feature_set"] if rf_best is not None else None
                llm_name = llm_best["feature_set"] if llm_best is not None else None
                if pd.notna(rf_val) and pd.notna(llm_val):
                    delta = llm_val - rf_val
                    winner = "LLM" if delta > 0 else ("RF" if delta < 0 else "tie")
                else:
                    delta, winner = np.nan, ""

                rows.append({
                    "split": split,
                    "train_n": TRAIN_N[split],
                    "domain": DOMAIN_TITLE[domain],
                    "metric": METRIC_LABEL[metric],
                    "best_rf_feature_set": rf_name,
                    "best_rf_value": rf_val,
                    "best_llm_model": llm_name,
                    "best_llm_value": llm_val,
                    "delta_llm_minus_rf": delta,
                    "winner": winner,
                })
    return pd.DataFrame(rows)


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.3f}"


def _fmt_delta(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.3f}"


def render_md(long_df: pd.DataFrame) -> str:
    out = []
    out.append("# Head-to-head: best RF vs best LLM Digital Twin\n")
    out.append("Within each split RF and LLM are evaluated on the IDENTICAL "
               "test rows. For every (split, domain, metric) cell we pick the "
               "best RF (across feature sets: Demographics, Avg-History, "
               "Embedding, Embedding+Demo) and the best LLM-DT "
               "(across GPT-5 / GPT-4o-mini / Gemini-2.5-Pro / Grok-4-Fast / "
               "DeepSeek-R1).\n")
    out.append("Δ = LLM − RF; positive Δ means the LLM wins.\n")

    for metric_label in [METRIC_LABEL[m] for m in METRICS]:
        out.append(f"\n## {metric_label}\n")
        out.append(
            "| Split (train n) | Domain | Best RF (feature set) | RF | "
            "Best LLM-DT (model) | LLM | Δ (LLM − RF) | Winner |"
        )
        out.append(
            "|---|---|---|---|---|---|---|---|"
        )
        sub = long_df[long_df["metric"] == metric_label]
        for split in SPLIT_ORDER:
            for domain in [DOMAIN_TITLE[d] for d in DOMAIN_ORDER]:
                row = sub[(sub.split == split) & (sub.domain == domain)]
                if row.empty:
                    continue
                r = row.iloc[0]
                rf_name = r['best_rf_feature_set'] if pd.notna(r['best_rf_feature_set']) else '—'
                llm_name = r['best_llm_model'] if pd.notna(r['best_llm_model']) else '—'
                out.append(
                    f"| {SPLIT_LABEL[split]} (n={r['train_n']}) "
                    f"| {domain} "
                    f"| {rf_name} "
                    f"| {_fmt(r['best_rf_value'])} "
                    f"| {llm_name} "
                    f"| {_fmt(r['best_llm_value'])} "
                    f"| {_fmt_delta(r['delta_llm_minus_rf'])} "
                    f"| {r['winner']} |"
                )

    out.append("\n## Source\n")
    out.append("- Per-split per-method per-domain metrics: "
               "`revision/figures/lc_rf_vs_llm.csv`\n"
               "- Generation: `analysis-script/lc_rf_vs_llm.py` "
               "(RF training + LLM aggregation), "
               "`analysis-script/lc_rf_vs_llm_h2h_table.py` (this table).\n")
    out.append(
        "Notes:\n"
        "- Spearman ρ is the per-participant rank correlation across that "
        "participant's test items, averaged across participants. At splits "
        "with ≈1 test message per participant (7030, 9010) the metric is "
        "computed only on the few participants that retain ≥2 test items "
        "and is therefore noisy.\n"
        "- DeepSeek-R1 only ran the 7030 split, so it never appears as the "
        "best LLM at 1090 / 3070 / 9010.\n"
        "- 1090 has zero RF training items by canonical-split design, so "
        "no RF column is reported.\n"
    )
    return "\n".join(out)


def _print_console_table(long_df: pd.DataFrame, metric_label: str) -> None:
    sub = long_df[long_df["metric"] == metric_label].copy()
    if sub.empty:
        return
    print(f"\n=== {metric_label} ===")
    rows = []
    rows.append(["Split", "Train n", "Domain",
                 "Best RF feat", "RF", "Best LLM", "LLM", "Δ", "Winner"])
    for _, r in sub.iterrows():
        rf_name = r["best_rf_feature_set"] if pd.notna(r["best_rf_feature_set"]) else "—"
        llm_name = r["best_llm_model"] if pd.notna(r["best_llm_model"]) else "—"
        rows.append([
            r["split"], str(r["train_n"]), r["domain"],
            rf_name[:14],
            _fmt(r["best_rf_value"]),
            llm_name,
            _fmt(r["best_llm_value"]),
            _fmt_delta(r["delta_llm_minus_rf"]),
            r["winner"],
        ])
    widths = [max(len(str(row[c])) for row in rows) for c in range(len(rows[0]))]
    for i, row in enumerate(rows):
        line = "  ".join(str(row[c]).ljust(widths[c]) for c in range(len(row)))
        print(line)
        if i == 0:
            print("-" * len(line))


def main():
    df = pd.read_csv(CSV_IN)
    df["split"] = df["split"].astype(str)

    long_df = build_long_table(df)
    out_csv = figures_path("lc_rf_vs_llm_h2h") + ".csv"
    long_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}  ({len(long_df)} rows)")

    md = render_md(long_df)
    out_md = figures_path("lc_rf_vs_llm_h2h") + ".md"
    with open(out_md, "w") as f:
        f.write(md)
    print(f"Saved: {out_md}")

    for metric in METRICS:
        _print_console_table(long_df, METRIC_LABEL[metric])


if __name__ == "__main__":
    main()
