#!/usr/bin/env python3
"""
Create a clean revision-folder artifact for accuracy confidence intervals.

This script bootstraps individual-level accuracy CIs for the core manuscript
methods on the same cleaned canonical PP 70/30 source used by the
current all-model bar figures, and writes revision-native outputs:

  - revision/figures/accuracy_confidence_intervals.csv
  - revision/figures/accuracy_confidence_intervals.png/pdf
  - revision/figures/accuracy_confidence_intervals.md

The intent is to answer the reviewer request for confidence intervals of the
reported accuracy rates with a direct artifact under revision/figures.

Usage:
    uv run python analysis-script/accuracy_ci_summary.py
    uv run python analysis-script/accuracy_ci_summary.py --plot-only
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from revision_utils import (
    COLORS,
    MODEL_CONFIGS,
    apply_repo_plot_style,
    figures_path,
    load_results_aligned,
    save_figure,
)
from filter_duplicates import get_duplicate_signatures, is_duplicate


N_BOOTSTRAP = 2_000
SEED = 42
DUPLICATE_SIGS = get_duplicate_signatures()

METHODS = [
    ("generic_llm_1_zero_shot_dt7030.json", "Zero-shot (all)"),
    ("generic_llm_3_few_shot_dt7030.json", "Few-shot (all)"),
    ("digital_twin_4_cbtact_7030.json", "PP"),
    ("hybrid", "Hybrid RF+PP"),
]
DOMAINS = ["Content", "Coping", "Quitting"]
MODEL_ORDER = ["GPT-4o-mini", "GPT-5", "DeepSeek-R1", "Grok-4-Fast", "Gemini-2.5-Pro"]
PLOT_METHOD_LABELS = {
    "Zero-shot (all)": "Zero-shot (all)",
    "Few-shot (all)": "Few-shot (all)",
    "PP": "PP",
    "Hybrid RF+PP": "Hybrid\nRF+PP",
}


def bootstrap_accuracy(gt: np.ndarray, pred: np.ndarray, rng: np.random.Generator) -> dict:
    n = len(gt)
    point = float(np.mean(gt == pred))
    boots = np.empty(N_BOOTSTRAP, dtype=float)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        boots[b] = np.mean(gt[idx] == pred[idx])
    return {
        "Accuracy": point,
        "Accuracy_CI_Lower": float(np.percentile(boots, 2.5)),
        "Accuracy_CI_Upper": float(np.percentile(boots, 97.5)),
        "Accuracy_SE": float(np.std(boots, ddof=1)),
    }


def load_clean_results(model_id: str, method_file: str) -> pd.DataFrame | None:
    """Load one method and apply the shared PP 70/30 duplicate filter."""
    df = load_results_aligned(model_id, method_file)
    if df is None:
        return None
    keep = ~df.apply(lambda row: is_duplicate(row.to_dict(), DUPLICATE_SIGS), axis=1)
    return df.loc[keep].reset_index(drop=True)


def collect_results() -> pd.DataFrame:
    rows = []
    master_rng = np.random.default_rng(SEED)

    for model_id, cfg in MODEL_CONFIGS.items():
        for method_file, method_name in METHODS:
            df = load_clean_results(model_id, method_file)
            if df is None:
                continue

            for domain in ["content", "coping", "quitting"]:
                gt_col = f"gt_{domain}_num"
                pred_col = f"pred_{domain}_num"
                valid = df[gt_col].notna() & df[pred_col].notna()
                gt = df.loc[valid, gt_col].astype(int).to_numpy()
                pred = df.loc[valid, pred_col].astype(int).to_numpy()
                if len(gt) < 10:
                    continue

                rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
                stats = bootstrap_accuracy(gt, pred, rng)
                rows.append({
                    "Model": cfg["display"],
                    "Method": method_name,
                    "Domain": domain.capitalize(),
                    "N": len(gt),
                    "Split": "digital_twin_7030_cleaned",
                    "Source_File": method_file,
                    "Bootstrap_N": N_BOOTSTRAP,
                    **stats,
                })

    out = pd.DataFrame(rows)
    out["Model"] = pd.Categorical(out["Model"], categories=MODEL_ORDER, ordered=True)
    out["Domain"] = pd.Categorical(out["Domain"], categories=DOMAINS, ordered=True)
    out["Method"] = pd.Categorical(
        out["Method"],
        categories=[m[1] for m in METHODS],
        ordered=True,
    )
    return out.sort_values(["Domain", "Method", "Model"]).reset_index(drop=True)


def plot_results(df: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(12, 12.5), sharex=True, sharey=True)
    method_order = [m[1] for m in METHODS]
    method_ticklabels = [PLOT_METHOD_LABELS.get(method, method) for method in method_order]
    offsets = np.linspace(-0.24, 0.24, len(MODEL_ORDER))

    for ax_idx, (ax, domain) in enumerate(zip(axes, DOMAINS)):
        dom = df[df["Domain"] == domain].copy()
        x = np.arange(len(method_order))

        for i, model in enumerate(MODEL_ORDER):
            model_df = dom[dom["Model"] == model].set_index("Method").reindex(method_order)
            y = model_df["Accuracy"].to_numpy(dtype=float)
            yerr_low = y - model_df["Accuracy_CI_Lower"].to_numpy(dtype=float)
            yerr_high = model_df["Accuracy_CI_Upper"].to_numpy(dtype=float) - y

            ax.errorbar(
                x + offsets[i],
                y,
                yerr=np.vstack([yerr_low, yerr_high]),
                fmt="o",
                color=COLORS[model],
                ecolor=COLORS[model],
                elinewidth=1.4,
                capsize=3,
                markersize=6.5,
                linewidth=0,
                markeredgecolor="black",
                markeredgewidth=0.7,
                label=model if ax_idx == 0 else None,
            )

        ax.set_title(domain, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        if ax_idx < len(DOMAINS) - 1:
            ax.tick_params(axis="x", which="both", labelbottom=False, length=0)
        else:
            ax.set_xticklabels(method_ticklabels, rotation=18, ha="right", fontsize=11, fontweight="bold")
        ax.set_ylim(0.1, 0.75)
        ax.set_ylabel("Accuracy (95% bootstrap CI)", fontsize=12, fontweight="bold")

    apply_repo_plot_style(fig, axes)
    fig.suptitle(
        "Accuracy Confidence Intervals Across Core Methods",
        fontsize=15,
        fontweight="bold",
        y=0.992,
    )
    axes[-1].set_xlabel("", fontsize=12, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.972), fontsize=10, frameon=True)
    fig.tight_layout()
    fig.subplots_adjust(top=0.885, bottom=0.12, hspace=0.18)
    save_figure(fig, figures_path("accuracy_confidence_intervals"))


def write_markdown(df: pd.DataFrame):
    md_path = figures_path("accuracy_confidence_intervals") + ".md"
    display = df.copy()
    for col in ["Accuracy", "Accuracy_CI_Lower", "Accuracy_CI_Upper", "Accuracy_SE"]:
        display[col] = display[col].map(lambda x: f"{x:.3f}")
    display["N"] = display["N"].astype(int).astype(str)
    display["Bootstrap_N"] = display["Bootstrap_N"].astype(int).astype(str)

    header = "| Model | Method | Domain | N | Split | Accuracy | Accuracy_CI_Lower | Accuracy_CI_Upper | Accuracy_SE | Bootstrap_N |"
    divider = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    rows = [
        "| " + " | ".join(
            row[col] for col in [
                "Model", "Method", "Domain", "N", "Split", "Accuracy",
                "Accuracy_CI_Lower", "Accuracy_CI_Upper", "Accuracy_SE", "Bootstrap_N"
            ]
        ) + " |"
        for _, row in display.iterrows()
    ]

    lines = [
        "# Accuracy Confidence Intervals",
        "",
        f"Individual-level bootstrap accuracy confidence intervals with `{N_BOOTSTRAP}` resamples.",
        "",
        "All rows use the cleaned canonical PP 70/30 source; known train/test duplicate items are removed from every method.",
        "",
        header,
        divider,
        *rows,
        "",
        "Artifacts:",
        "",
        "- [accuracy_confidence_intervals.csv](accuracy_confidence_intervals.csv)",
        "- [accuracy_confidence_intervals.png](accuracy_confidence_intervals.png)",
        "- [accuracy_confidence_intervals.pdf](accuracy_confidence_intervals.pdf)",
    ]

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")
    print(f"  Saved: {md_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Accuracy confidence interval artifacts.")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Reload accuracy_confidence_intervals.csv and regenerate the figure/markdown only.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("Accuracy Confidence Intervals")
    print("=" * 70)

    csv_path = figures_path("accuracy_confidence_intervals") + ".csv"
    if args.plot_only:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing cached results CSV: {csv_path}")
        print(f"\n[plot-only] Loading cached results: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print(f"\n[1/3] Bootstrapping accuracy CIs with N={N_BOOTSTRAP} ...")
        df = collect_results()
        df.to_csv(csv_path, index=False)
        print(f"[2/3] Saved CSV: {csv_path}")
        print(f"       Rows: {len(df)}")

    print("[3/3] Creating figure and markdown ...")
    plot_results(df)
    write_markdown(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
