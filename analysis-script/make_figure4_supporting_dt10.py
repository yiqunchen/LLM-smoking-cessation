#!/usr/bin/env python3
"""Regenerate Figure 4 supporting message-selection plots.

This script uses the current two-method dt10 message-selection table:
Supervised RF versus LLM-PP. Anchor hybrids are intentionally excluded.

Inputs, in priority order:
  1. revision/figures/progress_summary/message_selection_methods_k7.csv
  2. revision/figures/message_selection_methods_k7.csv

Outputs:
  - figures/figure4/supporting_message_selection_methods_k7.csv
  - figures/figure4/supporting_message_selection_quality_dt10.{png,pdf}
  - figures/figure4/supporting_message_selection_gain_dt10.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_CANDIDATES = [
    ROOT / "revision" / "figures" / "progress_summary" / "message_selection_methods_k7.csv",
    ROOT / "revision" / "figures" / "message_selection_methods_k7.csv",
]
REVISION_OUT = ROOT / "revision" / "figures" / "message_selection_methods_k7.csv"
OUT = ROOT / "figures" / "figure4"
OUT.mkdir(parents=True, exist_ok=True)

DOMAINS = ["Content", "Coping", "Quitting"]
METHODS = ["Supervised RF", "LLM-PP"]
COLORS = {"Supervised RF": "#7F7F7F", "LLM-PP": "#CC78BC"}
MARKERS = {"Supervised RF": "s", "LLM-PP": "o"}
LINESTYLES = {"Supervised RF": "--", "LLM-PP": "-"}
REFERENCE_COLOR = "#4D4D4D"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.linestyle": "--",
    "grid.alpha": 0.12,
    "grid.color": REFERENCE_COLOR,
    "grid.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 400,
})


def _source_path() -> Path:
    for path in SOURCE_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Missing message-selection source CSV. Expected one of: "
        + ", ".join(str(p) for p in SOURCE_CANDIDATES)
    )


def _load_current_methods() -> pd.DataFrame:
    src = _source_path()
    df = pd.read_csv(src)
    if "method_display" not in df.columns:
        raise ValueError(f"{src} does not contain method_display")

    current = df[df["method_display"].isin(METHODS)].copy()
    current = current[current["domain"].isin(DOMAINS)].copy()
    current["method_display"] = pd.Categorical(
        current["method_display"], METHODS, ordered=True
    )
    current["domain"] = pd.Categorical(current["domain"], DOMAINS, ordered=True)
    current = current.sort_values(["domain", "method_display", "K"]).reset_index(drop=True)
    if "method" in current.columns:
        current.insert(
            current.columns.get_loc("method") + 1,
            "method_source",
            current["method"].astype(str),
        )
        current["method"] = current["method_display"].astype(str)

    expected_rows = len(DOMAINS) * len(METHODS) * current["K"].nunique()
    if len(current) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} two-method rows but found {len(current)} from {src}"
        )

    out_source = OUT / "supporting_message_selection_methods_k7.csv"
    current.to_csv(out_source, index=False)
    current.to_csv(REVISION_OUT, index=False)
    print(f"source: {src.relative_to(ROOT)}")
    print(f"wrote {out_source.relative_to(ROOT)}")
    print(f"wrote {REVISION_OUT.relative_to(ROOT)}")
    return current


def _style_axis(ax):
    ax.tick_params(width=1.5, length=5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)


def _plot_method_lines(
    df: pd.DataFrame,
    y_col: str,
    lower_col: str | None,
    upper_col: str | None,
    ylabel: str,
    title: str,
    out_stem: str,
    include_random_oracle: bool,
    extra_stems: list[Path] | None = None,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), sharey=True, constrained_layout=True)
    fig.patch.set_facecolor("white")

    for ax, domain in zip(axes, DOMAINS):
        sub = df[df["domain"] == domain].sort_values("K")
        k_vals = sorted(sub["K"].astype(int).unique())

        if include_random_oracle:
            reference = sub.groupby("K", as_index=False).first().sort_values("K")
            x_ref = reference["K"].to_numpy(dtype=float)
            ax.plot(
                x_ref,
                reference["random_mean"].to_numpy(dtype=float),
                color=REFERENCE_COLOR,
                linestyle=":",
                linewidth=2.1,
                label="Random",
                zorder=2,
            )
            ax.plot(
                x_ref,
                reference["human_oracle_rating"].to_numpy(dtype=float),
                color="#222222",
                linestyle="-.",
                linewidth=2.1,
                label="Human oracle",
                zorder=2,
            )

        for method in METHODS:
            mdf = sub[sub["method_display"] == method].sort_values("K")
            if mdf.empty:
                continue
            x_vals = mdf["K"].to_numpy(dtype=float)
            y_vals = mdf[y_col].to_numpy(dtype=float)
            if lower_col and upper_col and {lower_col, upper_col}.issubset(mdf.columns):
                ax.fill_between(
                    x_vals,
                    mdf[lower_col].to_numpy(dtype=float),
                    mdf[upper_col].to_numpy(dtype=float),
                    color=COLORS[method],
                    alpha=0.12,
                    linewidth=0,
                    zorder=1,
                )
            ax.plot(
                x_vals,
                y_vals,
                marker=MARKERS[method],
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                linewidth=2.4,
                markersize=7.5,
                label=method,
                zorder=3,
            )

        if y_col == "gain_over_random":
            ax.axhline(0, color=REFERENCE_COLOR, linewidth=1.5, linestyle="--", alpha=0.75)
        ax.set_title(domain, fontsize=13, fontweight="bold")
        ax.set_xlabel("K selected messages", fontsize=12, fontweight="bold")
        ax.set_xticks(k_vals)
        _style_axis(ax)

    axes[0].set_ylabel(ylabel, fontsize=12, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    keep = []
    seen = set()
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        keep.append((handle, label))
    fig.legend(
        [h for h, _ in keep],
        [l for _, l in keep],
        loc="lower center",
        ncol=len(keep),
        bbox_to_anchor=(0.5, -0.10),
        frameon=False,
        prop={"weight": "bold", "size": 10.5},
    )
    feature = str(df["feature_set"].iloc[0])
    fig.suptitle(
        f"{title} (dt10, k_train = 7; fixed {feature})",
        fontsize=14.5,
        fontweight="bold",
    )
    stems = [OUT / out_stem]
    if extra_stems:
        stems.extend(extra_stems)
    for stem in stems:
        stem.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT.joinpath(out_stem).relative_to(ROOT)}.png / .pdf")


def main() -> None:
    df = _load_current_methods()
    _plot_method_lines(
        df=df,
        y_col="selected_human_rating",
        lower_col="selected_human_rating_ci_lower",
        upper_col="selected_human_rating_ci_upper",
        ylabel="Mean human rating of selected messages",
        title="Figure 4 supporting. Message-selection quality",
        out_stem="supporting_message_selection_quality_dt10",
        include_random_oracle=True,
        extra_stems=[ROOT / "revision" / "figures" / "message_selection_quality_dt10"],
    )
    _plot_method_lines(
        df=df,
        y_col="gain_over_random",
        lower_col="gain_over_random_ci_lower",
        upper_col="gain_over_random_ci_upper",
        ylabel="Gain in mean human rating over random",
        title="Figure 4 supporting. Message-selection gain over random",
        out_stem="supporting_message_selection_gain_dt10",
        include_random_oracle=False,
        extra_stems=[ROOT / "revision" / "figures" / "message_selection_gain"],
    )


if __name__ == "__main__":
    main()
