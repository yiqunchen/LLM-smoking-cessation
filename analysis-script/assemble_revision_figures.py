#!/usr/bin/env python3
"""Assemble current per-panel manuscript exports into Word-ready figures."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image, JpegImagePlugin  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
WHITE = (255, 255, 255, 255)
DPI = (300, 300)
FIGURE2_PANEL_CONTENT_BOTTOM = 1838
FIGURE2_LEGEND_GAP = 110

FIGURE2_MODEL_COLORS = [
    ("GPT-4o-mini", "#0173B2"),
    ("GPT-5", "#DE8F05"),
    ("DeepSeek-R1", "#029E73"),
    ("Grok-4-Fast", "#CC78BC"),
    ("Gemini-2.5-Pro", "#CA9161"),
]
FIGURE2_BASELINE_LINES = [
    ("LR - demographics", "#E02020", "--"),
    ("RF - demographics", "#7F7F7F", "--"),
    ("LR - demo + history + embedding", "#E02020", "-."),
    ("RF - demo + history + embedding", "#7F7F7F", "-."),
]
FIGURE2_BASELINE_SPECS = [
    ("Demographics", "LR", "LR - demographics", "#E02020", "--"),
    ("Demographics", "RF", "RF - demographics", "#7F7F7F", "--"),
    ("Demographics + History + Message Embedding", "LR", "LR - demo + history + embedding", "#E02020", "-."),
    ("Demographics + History + Message Embedding", "RF", "RF - demo + history + embedding", "#7F7F7F", "-."),
]
FIGURE2_MODEL_ORDER = [label for label, _ in FIGURE2_MODEL_COLORS]
FIGURE2_MODEL_COLOR_MAP = dict(FIGURE2_MODEL_COLORS)
FIGURE2_DOMAIN_ORDER = ["Content", "Coping", "Quitting"]
FIGURE2_METHOD_ORDER = [
    "Zero-shot (all)",
    "Zero-shot (select)",
    "Few-shot (all)",
    "Few-shot (select)",
    "Zero-shot (w/ prob)",
    "PP",
    "Hybrid RF+PP",
]
FIGURE2_METHOD_LABELS = {
    "Zero-shot (all)": "Zero-shot\n(all)",
    "Zero-shot (select)": "Zero-shot\n(select)",
    "Few-shot (all)": "Few-shot\n(all)",
    "Few-shot (select)": "Few-shot\n(select)",
    "Zero-shot (w/ prob)": "Zero-shot\n(w/ prob)",
    "PP": "PP",
    "Hybrid RF+PP": "Hybrid\nRF+PP",
}
FIGURE2_METRIC_LABELS = {
    "Accuracy": ("Actual Accuracy", "Actual accuracy"),
    "F1": ("Actual Macro-F1", "Actual Macro-F1"),
    "QWK": ("Weighted Kappa", "Weighted kappa"),
    "Kappa": ("Cohen's Kappa", "Kappa"),
    "Directional Accuracy": ("Directional Accuracy", "Directional Accuracy"),
    "Directional Macro-F1": ("Directional Macro-F1", "Directional Macro-F1"),
    "Kendall_Tau": ("Kendall's tau", "Kendall's tau"),
}


def _open_rgba(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def _crop_bottom(img: Image.Image, crop_bottom: int | None) -> Image.Image:
    if crop_bottom is None:
        return img
    return img.crop((0, 0, img.width, min(crop_bottom, img.height)))


def _make_figure2_legend(width: int, ncol: int) -> Image.Image:
    height = 760 if ncol <= 3 else 330
    fontsize = 29 if ncol <= 3 else 25
    handles = [
        Patch(facecolor=color, edgecolor="black", linewidth=1.4, label=label)
        for label, color in FIGURE2_MODEL_COLORS
    ]
    handles.extend(
        Line2D([0], [0], color=color, linestyle=linestyle, linewidth=3.4, label=label)
        for label, color, linestyle in FIGURE2_BASELINE_LINES
    )

    fig = plt.figure(figsize=(width / DPI[0], height / DPI[1]), dpi=DPI[0])
    fig.patch.set_facecolor("white")
    legend = fig.legend(
        handles=handles,
        loc="center",
        ncol=ncol,
        frameon=False,
        prop={"weight": "bold", "size": fontsize},
        handlelength=2.2,
        handletextpad=0.6,
        columnspacing=1.2,
        labelspacing=1.15,
        borderaxespad=0.0,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=DPI[0], facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def _append_bottom_label(img: Image.Image, label: str) -> Image.Image:
    label_h = 170
    fig = plt.figure(figsize=(img.width / DPI[0], label_h / DPI[1]), dpi=DPI[0])
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.56, label, ha="center", va="center", fontsize=34, fontweight="bold")
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=DPI[0], facecolor="white")
    plt.close(fig)
    buf.seek(0)
    label_img = Image.open(buf).convert("RGBA")

    combined = Image.new("RGBA", (img.width, img.height + label_h), WHITE)
    combined.alpha_composite(img, (0, 0))
    combined.alpha_composite(label_img, (0, img.height))
    return combined


def assemble_grid(
    panels: list[Path],
    out_prefix: Path,
    ncols: int,
    pad: int = 80,
    gutter: int = 70,
    crop_bottom: int | None = None,
    figure2_shared_legend: bool = False,
    figure2_legend_ncol: int = 3,
    save_pdf: bool = True,
    panel_labels: list[str] | None = None,
) -> None:
    imgs = [_crop_bottom(_open_rgba(p), crop_bottom) for p in panels]
    if panel_labels is not None:
        if len(panel_labels) != len(imgs):
            raise ValueError("panel_labels must match the number of panels")
        imgs = [_append_bottom_label(img, label) for img, label in zip(imgs, panel_labels)]
    cell_w = max(img.width for img in imgs)
    cell_h = max(img.height for img in imgs)
    nrows = (len(imgs) + ncols - 1) // ncols

    canvas_w = 2 * pad + ncols * cell_w + (ncols - 1) * gutter
    canvas_h = 2 * pad + nrows * cell_h + (nrows - 1) * gutter
    legend_img = None
    if figure2_shared_legend:
        legend_img = _make_figure2_legend(canvas_w - 2 * pad, ncol=figure2_legend_ncol)
        canvas_h += FIGURE2_LEGEND_GAP + legend_img.height

    canvas = Image.new("RGBA", (canvas_w, canvas_h), WHITE)

    for idx, img in enumerate(imgs):
        row, col = divmod(idx, ncols)
        x = pad + col * (cell_w + gutter) + (cell_w - img.width) // 2
        y = pad + row * (cell_h + gutter) + (cell_h - img.height) // 2
        canvas.alpha_composite(img, (x, y))

    if legend_img is not None:
        x = (canvas_w - legend_img.width) // 2
        y = pad + nrows * cell_h + (nrows - 1) * gutter + FIGURE2_LEGEND_GAP
        canvas.alpha_composite(legend_img, (x, y))

    rgb = canvas.convert("RGB")
    rgb.save(out_prefix.with_suffix(".png"), dpi=DPI)
    if save_pdf:
        rgb.save(out_prefix.with_suffix(".pdf"), resolution=300.0)


def _figure2_handles() -> list:
    handles = [
        Patch(facecolor=color, edgecolor="black", linewidth=1.2, label=label)
        for label, color in FIGURE2_MODEL_COLORS
    ]
    handles.extend(
        Line2D([0], [0], color=color, linestyle=linestyle, linewidth=2.8, label=label)
        for _, _, label, color, linestyle in FIGURE2_BASELINE_SPECS
    )
    return handles


def _figure2_metric_label(metric: str) -> tuple[str, str]:
    return FIGURE2_METRIC_LABELS.get(metric, (metric, metric))


def _plot_figure2_metric_row(
    axes: np.ndarray,
    df: pd.DataFrame,
    metric: str,
    row_label_size: int,
) -> None:
    title_label, ylabel = _figure2_metric_label(metric)

    for ax, domain in zip(axes, FIGURE2_DOMAIN_ORDER):
        domain_data = df[df["Domain"] == domain].copy()
        llm_data = domain_data[domain_data["Model"].isin(FIGURE2_MODEL_ORDER)].copy()
        pivot = llm_data.pivot_table(
            index="Method",
            columns="Model",
            values=metric,
            aggfunc="first",
        )
        pivot = pivot.reindex([m for m in FIGURE2_METHOD_ORDER if m in pivot.index])
        pivot = pivot[[m for m in FIGURE2_MODEL_ORDER if m in pivot.columns]]
        if pivot.empty:
            if metric == "Kendall_Tau":
                ax.set_title(domain, fontsize=17, fontweight="bold", pad=12)
                ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")
                ax.set_ylim(-1.05, 1.05)
                ax.axhline(
                    0,
                    color="#4D4D4D",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.75,
                )
                ax.set_xticks([])
                ax.text(
                    0.5,
                    0.5,
                    "Undefined\n(tied ranks)",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=15,
                    fontweight="bold",
                    color="#4D4D4D",
                )
                ax.grid(axis="y", alpha=0.18, linestyle="--", color="#4D4D4D", linewidth=0.7)
                ax.tick_params(axis="y", labelsize=14, width=1.8, length=5)
                for tick in ax.get_yticklabels():
                    tick.set_fontweight("bold")
                for spine in ax.spines.values():
                    spine.set_linewidth(1.4)
            else:
                ax.set_visible(False)
            continue
        x = np.arange(len(pivot.index))
        n_models = max(1, len(pivot.columns))
        width = 0.8 / n_models

        for model_idx, model in enumerate(pivot.columns):
            offset = (model_idx - (n_models - 1) / 2) * width
            ax.bar(
                x + offset,
                pivot[model].astype(float).to_numpy(),
                width=width,
                color=FIGURE2_MODEL_COLOR_MAP.get(model, "#999999"),
                edgecolor="black",
                linewidth=1.1,
            )

        baseline_values = []
        for feature_set, classifier, _, color, linestyle in FIGURE2_BASELINE_SPECS:
            subset = domain_data[
                (domain_data["Category"] == "Supervised Baseline")
                & (domain_data["Feature_Set"] == feature_set)
                & (domain_data["Classifier"] == classifier)
            ]
            if subset.empty:
                continue
            value = float(subset[metric].iloc[0])
            if np.isnan(value):
                continue
            baseline_values.append(value)
            ax.axhline(
                value,
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
                alpha=0.9,
            )

        if metric == "Kendall_Tau":
            ax.axhline(
                0,
                color="#4D4D4D",
                linestyle="--",
                linewidth=1.0,
                alpha=0.75,
                zorder=0,
            )
            valid_values = pd.to_numeric(pivot.stack(), errors="coerce").dropna().tolist()
            valid_values.extend(baseline_values)
            if valid_values:
                y_min = min(valid_values)
                y_max = max(valid_values)
                lower = max(-1.02, min(0.0, y_min * 1.18))
                upper = min(1.02, max(0.45, y_max * 1.18))
                ax.set_ylim(lower, upper)
            else:
                ax.set_ylim(-0.05, 0.45)
        else:
            y_max_candidates = [float(np.nanmax(pivot.values))] if not pivot.empty else [0.0]
            if baseline_values:
                y_max_candidates.append(max(baseline_values))
            y_max = max(y_max_candidates)
            ax.set_ylim(0, min(1.02, max(0.45, y_max * 1.18)))

        ax.set_title(
            domain,
            fontsize=17,
            fontweight="bold",
            pad=12,
        )
        ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [FIGURE2_METHOD_LABELS.get(method, method) for method in pivot.index],
            fontsize=13,
            fontweight="bold",
        )
        ax.tick_params(axis="y", labelsize=14, width=1.8, length=5)
        ax.tick_params(axis="x", width=1.8, length=5)
        for tick in ax.get_yticklabels():
            tick.set_fontweight("bold")
        ax.grid(axis="y", alpha=0.18, linestyle="--", color="#4D4D4D", linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.4)

    axes[1].text(
        0.5,
        -0.36,
        title_label,
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=row_label_size,
        fontweight="bold",
        clip_on=False,
    )


def save_figure2_vector_pdf(
    source_table: Path,
    out_prefix: Path,
    metrics: list[str],
) -> None:
    """Save an editable vector PDF for Figure 2 instead of embedding PNG panels."""
    df = pd.read_csv(source_table)
    fig_height = max(20, 6.65 * len(metrics))
    fig, axes = plt.subplots(len(metrics), 3, figsize=(25.5, fig_height), squeeze=False)
    axis_rows = [axes[idx, :] for idx in range(len(metrics))]
    row_label_size = 20
    legend_ncol = 3
    legend_size = 22
    bottom = 0.19
    hspace = 0.52

    fig.patch.set_facecolor("white")
    for idx, metric in enumerate(metrics):
        _plot_figure2_metric_row(axis_rows[idx], df, metric, row_label_size=row_label_size)

    legend = fig.legend(
        handles=_figure2_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=legend_ncol,
        frameon=False,
        prop={"weight": "bold", "size": legend_size},
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=0.9,
        labelspacing=0.9,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    fig.subplots_adjust(left=0.055, right=0.985, top=0.94, bottom=bottom, wspace=0.24, hspace=hspace)
    fig.savefig(out_prefix.with_suffix(".pdf"), format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    figure2 = ROOT / "figures" / "figure2"
    figure2_main = [
        figure2 / "bars_all_methods_accuracy.png",
        figure2 / "bars_all_methods_f1.png",
        figure2 / "bars_all_methods_qwk.png",
    ]
    figure2_appendix = [
        figure2 / "bars_all_methods_kappa.png",
        figure2 / "bars_all_methods_directional_accuracy.png",
        figure2 / "bars_all_methods_directional_macro_f1.png",
        figure2 / "bars_all_methods_kendall_tau.png",
    ]
    figure2_main_labels = ["Overall Accuracy", "Macro-F1", "Weighted Kappa"]
    figure2_appendix_labels = ["Cohen's Kappa", "Directional Accuracy", "Directional Macro-F1", "Kendall's tau"]

    assemble_grid(
        figure2_main,
        figure2 / "figure2_main_top3_vertical_300dpi",
        ncols=1,
        pad=90,
        gutter=60,
        crop_bottom=FIGURE2_PANEL_CONTENT_BOTTOM,
        figure2_shared_legend=True,
        figure2_legend_ncol=3,
        save_pdf=False,
        panel_labels=figure2_main_labels,
    )
    assemble_grid(
        figure2_appendix,
        figure2 / "figure2_appendix_other3_vertical_300dpi",
        ncols=1,
        pad=90,
        gutter=60,
        crop_bottom=FIGURE2_PANEL_CONTENT_BOTTOM,
        figure2_shared_legend=True,
        figure2_legend_ncol=3,
        save_pdf=False,
        panel_labels=figure2_appendix_labels,
    )
    source_table = ROOT / "figures" / "bars_all_methods_source_table.csv"
    save_figure2_vector_pdf(
        source_table,
        figure2 / "figure2_main_top3_vertical_300dpi",
        metrics=["Accuracy", "F1", "QWK"],
    )
    save_figure2_vector_pdf(
        source_table,
        figure2 / "figure2_appendix_other3_vertical_300dpi",
        metrics=["Kappa", "Directional Accuracy", "Directional Macro-F1", "Kendall_Tau"],
    )

    figure3 = ROOT / "figures" / "figure3"
    figure3_panels = [
        figure3 / "figure3_score_distributions_content.png",
        figure3 / "figure3_score_distributions_coping.png",
        figure3 / "figure3_score_distributions_quitting.png",
    ]
    assemble_grid(
        figure3_panels,
        figure3 / "figure3_assembled_300dpi",
        ncols=3,
    )
    assemble_grid(
        figure3_panels,
        figure3 / "figure3_assembled_vertical_300dpi",
        ncols=1,
        pad=90,
        gutter=60,
    )


if __name__ == "__main__":
    main()
