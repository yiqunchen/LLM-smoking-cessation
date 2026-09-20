#!/usr/bin/env python3
"""Plot the RF-vs-LLM learning curve produced by lc_rf_vs_llm.py.

Apples-to-apples: within each canonical split (1090, 3070, 7030, 9010)
RF and LLM-PP are evaluated on the IDENTICAL test partition;
the x-axis varies the training fraction (and thus the LLM's
in-context profile length per participant).

Outputs:
    revision/figures/lc_rf_vs_llm_main.{png,pdf}
        2 rows (Accuracy, Macro-F1) x 3 cols (Content, Coping, Quitting)
    revision/figures/lc_rf_vs_llm_spearman.{png,pdf}
        Within-participant Spearman, only at splits where >=2 test
        items per participant make the metric well-defined.

Usage:
    uv run python analysis-script/lc_rf_vs_llm_plot.py
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from revision_utils import COLORS, figures_path, save_figure  # noqa: E402

CSV_PATH = figures_path("lc_rf_vs_llm") + ".csv"

DOMAIN_ORDER = ["content", "coping", "quitting"]
DOMAIN_TITLE = {"content": "Content", "coping": "Coping", "quitting": "Quitting"}

SPLIT_TRAIN_PCT = {"1090": 10, "3070": 30, "5050": 50, "7030": 70, "9010": 90}
SPLIT_TRAIN_N = {"1090": 0, "3070": 22, "5050": 323, "7030": 593, "9010": 615}
SPLITS_ALL = ["1090", "3070", "5050", "7030", "9010"]

LLM_ORDER = ["GPT-4o-mini", "GPT-5", "Gemini-2.5-Pro", "Grok-4-Fast", "DeepSeek-R1"]

RF_FEATURE_ORDER = ["Demographics", "Avg-History", "Embedding", "Embedding+Demo"]
RF_FEATURE_COLORS = {
    "Demographics":   "#262626",
    "Avg-History":    "#5C5C5C",
    "Embedding":      "#9A9A9A",
    "Embedding+Demo": "#BCBCBC",
}
RF_FEATURE_MARKERS = {
    "Demographics":   "s",
    "Avg-History":    "D",
    "Embedding":      "v",
    "Embedding+Demo": "^",
}


def _draw_curve(ax, sub, color, marker, label, linestyle="-", lw=2.6, ms=10,
                alpha=0.95, zorder=3):
    sub = sub.sort_values("train_n")
    xs = [SPLIT_TRAIN_PCT[s] for s in sub["split"]]
    ys = sub.iloc[:, sub.columns.get_loc("metric_value")].values
    if len(xs) == 0:
        return
    if len(xs) == 1:
        ax.scatter(xs, ys, color=color, marker=marker, s=130,
                   edgecolor="#1F1F1F", linewidth=1.0, zorder=zorder + 1,
                   label=label)
        return
    ax.plot(xs, ys, color=color, marker=marker, markersize=ms,
            linewidth=lw, linestyle=linestyle, alpha=alpha,
            label=label, zorder=zorder)


def _ylim_for(metric_key):
    return {
        "Accuracy": (0.20, 0.65),
        "F1": (0.05, 0.50),
    }[metric_key]


def plot_main(df: pd.DataFrame) -> None:
    metrics = [("Accuracy", "Accuracy"), ("F1", "Macro-F1")]
    xs_pct_all = [SPLIT_TRAIN_PCT[s] for s in SPLITS_ALL]

    fig, axes = plt.subplots(
        len(metrics), len(DOMAIN_ORDER),
        figsize=(17.0, 9.5), sharex=True,
    )
    fig.patch.set_facecolor("white")

    for row, (metric_key, metric_label) in enumerate(metrics):
        for col, domain in enumerate(DOMAIN_ORDER):
            ax = axes[row, col]
            ax.set_facecolor("white")

            ax.axvspan(8, 32, color="#FFD580", alpha=0.16, zorder=0)

            panel_df = df[df.domain == domain].copy()
            panel_df["metric_value"] = panel_df[metric_key]

            # RF first so LLMs paint on top
            rf_panel = panel_df[panel_df.method == "RF"]
            for fs in RF_FEATURE_ORDER:
                _draw_curve(
                    ax, rf_panel[rf_panel.feature_set == fs],
                    color=RF_FEATURE_COLORS[fs],
                    marker=RF_FEATURE_MARKERS[fs],
                    label=f"RF: {fs}",
                    linestyle="--", lw=2.0, ms=9, alpha=0.95, zorder=2,
                )

            llm_panel = panel_df[panel_df.method == "LLM-DT"]
            for model in LLM_ORDER:
                _draw_curve(
                    ax, llm_panel[llm_panel.feature_set == model],
                    color=COLORS.get(model, "#444"),
                    marker="o",
                    label=f"LLM-PP: {model}",
                    linestyle="-", lw=2.7, ms=10, alpha=0.95, zorder=4,
                )

            ax.set_xticks(xs_pct_all)
            ax.set_xticklabels([f"{p}%" for p in xs_pct_all])
            ax.set_xlim(5, 95)
            if row == len(metrics) - 1:
                ax.set_xlabel(
                    "Training fraction\n(profile length per participant)",
                    fontsize=13, fontweight="bold",
                )
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=14, fontweight="bold")
            ax.set_title(f"{DOMAIN_TITLE[domain]} | {metric_label}",
                         fontsize=14, fontweight="bold", pad=8)
            ax.set_ylim(*_ylim_for(metric_key))
            ax.grid(axis="y", alpha=0.18, linestyle="--", color="#4D4D4D")

            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(xs_pct_all)
            if row == 0:
                ax2.set_xticklabels(
                    [str(SPLIT_TRAIN_N[s]) for s in SPLITS_ALL], fontsize=10
                )
                ax2.set_xlabel("Train n (messages)", fontsize=11,
                               fontweight="bold", labelpad=6)
            else:
                ax2.set_xticks([])
                ax2.set_xticklabels([])
                ax2.set_xlabel("")

            if row == 0 and col == 0:
                ax.annotate(
                    "low-data regime\n(LLM advantage)",
                    xy=(20, _ylim_for(metric_key)[1] - 0.05),
                    fontsize=10, fontweight="bold",
                    color="#8A5A00", ha="center",
                )

    rf_handles = [
        Line2D([0], [0], color=RF_FEATURE_COLORS[fs],
               marker=RF_FEATURE_MARKERS[fs], markersize=9, linewidth=2.0,
               linestyle="--", label=f"RF: {fs}")
        for fs in RF_FEATURE_ORDER
    ]
    llm_handles = [
        Line2D([0], [0], color=COLORS.get(m, "#444"), marker="o",
               markersize=10, linewidth=2.7,
               label=f"LLM-PP: {m}")
        for m in LLM_ORDER
    ]
    legend1 = fig.legend(
        rf_handles, [h.get_label() for h in rf_handles],
        loc="lower center", ncol=4, fontsize=11,
        bbox_to_anchor=(0.5, -0.03), frameon=True, title="Random Forest baselines",
        title_fontsize=11,
    )
    legend1.get_title().set_fontweight("bold")
    fig.add_artist(legend1)
    legend2 = fig.legend(
        llm_handles, [h.get_label() for h in llm_handles],
        loc="lower center", ncol=5, fontsize=11,
        bbox_to_anchor=(0.5, -0.10), frameon=True, title="LLM-PP",
        title_fontsize=11,
    )
    legend2.get_title().set_fontweight("bold")

    fig.suptitle(
        "Apples-to-apples learning curve: Random Forest vs LLM-PP"
        "  (within each split, RF and LLM share the identical test set)",
        fontsize=15, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=[0, 0.10, 1, 0.96])

    save_figure(fig, figures_path("lc_rf_vs_llm_main"))


def plot_spearman(df: pd.DataFrame) -> None:
    """Within-participant Spearman is only well-defined at 1090 and 3070
    (each participant has multiple test messages); 7030 and 9010 collapse
    to ~1 test message per participant and the metric becomes degenerate."""
    splits = ["1090", "3070"]
    sub = df[df["split"].isin(splits)].copy()
    sub["metric_value"] = sub["Spearman_Rho"]
    xs_pct = [SPLIT_TRAIN_PCT[s] for s in splits]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), sharey=True)
    fig.patch.set_facecolor("white")

    for col, domain in enumerate(DOMAIN_ORDER):
        ax = axes[col]
        ax.set_facecolor("white")

        rf_panel = sub[(sub.method == "RF") & (sub.domain == domain)]
        for fs in RF_FEATURE_ORDER:
            _draw_curve(
                ax, rf_panel[rf_panel.feature_set == fs],
                color=RF_FEATURE_COLORS[fs],
                marker=RF_FEATURE_MARKERS[fs],
                label=f"RF: {fs}",
                linestyle="--", lw=2.0, ms=9, alpha=0.95,
            )

        llm_panel = sub[(sub.method == "LLM-DT") & (sub.domain == domain)]
        for model in LLM_ORDER:
            _draw_curve(
                ax, llm_panel[llm_panel.feature_set == model],
                color=COLORS.get(model, "#444"),
                marker="o",
                label=f"LLM-PP: {model}",
                linestyle="-", lw=2.7, ms=10, alpha=0.95,
            )

        ax.axhline(0.0, color="#4D4D4D", linestyle=":", linewidth=1.4, alpha=0.8)
        ax.set_xticks(xs_pct)
        ax.set_xticklabels([f"{p}%" for p in xs_pct])
        ax.set_xlabel("Training fraction", fontsize=13, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Within-participant Spearman ρ",
                          fontsize=13, fontweight="bold")
        ax.set_title(DOMAIN_TITLE[domain], fontsize=14, fontweight="bold", pad=8)
        ax.set_ylim(-0.25, 0.55)
        ax.grid(axis="y", alpha=0.18, linestyle="--", color="#4D4D4D")

    handles, labels = [], []
    for h, lab in zip(*axes[0].get_legend_handles_labels()):
        if lab not in labels:
            handles.append(h)
            labels.append(lab)
    fig.legend(handles, labels,
               loc="lower center", ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, -0.18), frameon=True)

    fig.suptitle(
        "Within-participant rank correlation (well-defined only at low train "
        "fractions where each participant has ≥2 test messages)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout(rect=[0, 0.07, 1, 0.96])

    save_figure(fig, figures_path("lc_rf_vs_llm_spearman"))


def main():
    df = pd.read_csv(CSV_PATH)
    df["split"] = df["split"].astype(str)
    plot_main(df)
    plot_spearman(df)


if __name__ == "__main__":
    main()
