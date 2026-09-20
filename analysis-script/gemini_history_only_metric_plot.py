"""
Plot Gemini full digital-twin vs history-only digital-twin metrics.

Outputs:
  - revision/figures/gemini_history_only_metric_comparison.csv
  - revision/figures/gemini_history_only_metric_comparison.png/pdf

Usage:
  uv run python analysis-script/gemini_history_only_metric_plot.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from revision_utils import (
    apply_repo_plot_style,
    compute_all_metrics_df,
    figures_path,
    load_results_file,
    save_figure,
)


FULL_LABEL = "Full PP"
HISTORY_LABEL = "History Only"
FULL_COLOR = "#0173B2"
HISTORY_COLOR = "#7F7F7F"
MODEL_DIR = Path("results_manuscript_gemini-2.5-pro")

RESULT_FILES = {
    FULL_LABEL: MODEL_DIR / "digital_twin_4_cbtact_7030.json",
    HISTORY_LABEL: MODEL_DIR / "digital_twin_history_only_7030.json",
}

METRIC_ORDER = [
    ("Accuracy", "Accuracy"),
    ("F1", "F1"),
    ("Directional Accuracy", "Directional Accuracy"),
    ("Directional Macro-F1", "Directional Macro-F1"),
]
DOMAINS = ["content", "coping", "quitting"]
DOMAIN_LABELS = ["Content", "Coping", "Quitting"]


def build_metric_df() -> pd.DataFrame:
    rows = []
    for method_label, filepath in RESULT_FILES.items():
        df = load_results_file(str(filepath))
        if df is None:
            raise FileNotFoundError(f"Missing result file: {filepath}")
        metrics = compute_all_metrics_df(df)
        for domain in DOMAINS:
            domain_metrics = metrics[domain]
            for metric_name, _ in METRIC_ORDER:
                rows.append({
                    "Method": method_label,
                    "Domain": domain.capitalize(),
                    "Metric": metric_name,
                    "Value": float(domain_metrics[metric_name]),
                })
    return pd.DataFrame(rows)


def plot_metric_df(metric_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8), sharex=True)
    axes = axes.flatten()
    apply_repo_plot_style(fig, axes)

    x = list(range(len(DOMAIN_LABELS)))
    width = 0.34

    for ax, (metric_name, metric_label) in zip(axes, METRIC_ORDER):
        panel = metric_df[metric_df["Metric"] == metric_name]
        full_vals = [
            float(panel[(panel["Method"] == FULL_LABEL) & (panel["Domain"] == label)]["Value"].iloc[0])
            for label in DOMAIN_LABELS
        ]
        hist_vals = [
            float(panel[(panel["Method"] == HISTORY_LABEL) & (panel["Domain"] == label)]["Value"].iloc[0])
            for label in DOMAIN_LABELS
        ]

        ax.bar([i - width / 2 for i in x], full_vals, width=width, color=FULL_COLOR, label=FULL_LABEL)
        ax.bar([i + width / 2 for i in x], hist_vals, width=width, color=HISTORY_COLOR, label=HISTORY_LABEL)
        ax.set_title(metric_label, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(DOMAIN_LABELS, fontsize=12, fontweight="bold")
        ax.set_ylim(0, min(1.0, max(full_vals + hist_vals) + 0.12))
        ax.tick_params(axis="y", labelsize=12)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        for xpos, val in zip([i - width / 2 for i in x], full_vals):
            ax.text(xpos, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        for xpos, val in zip([i + width / 2 for i in x], hist_vals):
            ax.text(xpos, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    axes[0].set_ylabel("Score", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("Score", fontsize=13, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, prop={"weight": "bold", "size": 12}, frameon=False)
    fig.suptitle(
        "Gemini-2.5-Pro: Full PP vs History-Only Ablation",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(top=0.88, bottom=0.11, left=0.08, right=0.98, hspace=0.26, wspace=0.18)
    save_figure(fig, figures_path("gemini_history_only_metric_comparison"))


def main() -> None:
    metric_df = build_metric_df()
    metric_df.to_csv(Path(figures_path("gemini_history_only_metric_comparison")).with_suffix(".csv"), index=False)
    plot_metric_df(metric_df)


if __name__ == "__main__":
    main()
