#!/usr/bin/env python3
"""Schematic for the apples-to-apples RF-vs-LLM learning-curve experiment.

Three-panel figure explaining the experimental design:

  Panel A: Dataset (301 participants × ~3 rated messages).
  Panel B: Four canonical digital-twin splits (1090, 3070, 7030, 9010)
           sweep the profile fraction (= LLM in-context history length =
           RF training data).
  Panel C: Within each split, RF and LLM consume the SAME profile rows
           and predict the SAME held-out test rows. Apples-to-apples.

Output:
    revision/figures/lc_rf_vs_llm_schematic.{png,pdf}

Usage:
    uv run python analysis-script/lc_rf_vs_llm_schematic.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_utils import figures_path  # noqa: E402

C_PROFILE = "#4C72B0"
C_TEST    = "#C44E52"
C_RF      = "#7F7F7F"
C_LLM     = "#DE8F05"
C_OK      = "#2E7D32"
C_TEXT    = "#222222"
C_BORDER  = "#1F1F1F"

SPLITS = [
    # (name, n_profile_cells, n_test_cells, train_n_total, test_n_total)
    ("1090",  1, 9,    0, 916),
    ("3070",  3, 7,   22, 894),
    ("7030",  7, 3,  593, 323),
    ("9010",  9, 1,  615, 301),
]


def _arrow(ax, x0, y0, x1, y1, color=C_BORDER, lw=1.6):
    ax.annotate(
        "",
        xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                        shrinkA=2, shrinkB=2),
        annotation_clip=False,
    )


def draw_panel_a(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.96, "A.  Dataset",
            ha="center", va="top", fontsize=15, fontweight="bold",
            color=C_TEXT)

    # Left card: 301 participants
    ax.add_patch(FancyBboxPatch(
        (0.04, 0.18), 0.42, 0.66,
        boxstyle="round,pad=0.010,rounding_size=0.012",
        facecolor="white", edgecolor=C_BORDER, linewidth=1.7,
    ))
    ax.text(0.25, 0.74, "301 participants",
            ha="center", fontsize=13, fontweight="bold", color=C_TEXT)
    ax.text(0.25, 0.66,
            "adults who smoke; ratings of\n"
            "CBT/ACT-coded cessation messages",
            ha="center", fontsize=9.5, color="#555", style="italic")
    n_glyph = 14
    for i in range(n_glyph):
        x = 0.075 + i * (0.345 / n_glyph)
        ax.add_patch(Rectangle((x, 0.42), 0.012, 0.06,
                               facecolor=C_BORDER, edgecolor="none"))
        ax.add_patch(Rectangle((x - 0.005, 0.395), 0.022, 0.025,
                               facecolor=C_BORDER, edgecolor="none"))
    ax.text(0.435, 0.43, "…", fontsize=14, color=C_BORDER, va="center")
    ax.text(0.25, 0.31, "(916 message-rating pairs total)",
            ha="center", fontsize=9.5, color="#666", style="italic")

    # Right card: ~3 messages each
    ax.add_patch(FancyBboxPatch(
        (0.54, 0.18), 0.42, 0.66,
        boxstyle="round,pad=0.010,rounding_size=0.012",
        facecolor="white", edgecolor=C_BORDER, linewidth=1.7,
    ))
    ax.text(0.75, 0.74, "≈ 3 rated messages / participant",
            ha="center", fontsize=12.5, fontweight="bold", color=C_TEXT)
    ax.text(0.75, 0.66, "ratings on a 1–5 scale per domain",
            ha="center", fontsize=9.5, color="#555", style="italic")
    cell_w = 0.06
    cy = 0.42
    for i in range(3):
        cx = 0.62 + i * (cell_w + 0.012)
        ax.add_patch(Rectangle((cx, cy), cell_w, 0.10,
                               facecolor="#F2F2F2",
                               edgecolor=C_BORDER, linewidth=1.4))
        ax.text(cx + cell_w / 2, cy + 0.05, f"M{i+1}",
                ha="center", va="center", fontsize=9.5,
                fontweight="bold", color=C_TEXT)
    ax.text(0.75, 0.30,
            "domains: Content / Coping / Quitting",
            ha="center", fontsize=9.5, color="#666")


def _draw_strip(ax, x0, y0, n_pro, n_te, cell_w=0.018, cell_h=0.07,
                gap=0.0):
    for k in range(n_pro):
        ax.add_patch(Rectangle(
            (x0 + k * (cell_w + gap), y0), cell_w, cell_h,
            facecolor=C_PROFILE, edgecolor=C_BORDER, linewidth=0.6,
        ))
    for k in range(n_te):
        ax.add_patch(Rectangle(
            (x0 + (n_pro + k) * (cell_w + gap), y0), cell_w, cell_h,
            facecolor=C_TEST, edgecolor=C_BORDER, linewidth=0.6,
        ))


def draw_panel_b(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.97,
            "B.  Four canonical splits sweep the profile fraction",
            ha="center", va="top", fontsize=15, fontweight="bold",
            color=C_TEXT)
    ax.text(0.5, 0.89,
            "Each participant's messages are partitioned into "
            "PROFILE (training / in-context history) + TEST (held out). "
            "Same seed across splits.",
            ha="center", va="top", fontsize=10, color="#555", style="italic")

    panel_w = 0.21
    n_panels = 4
    gap = (1.0 - n_panels * panel_w) / (n_panels + 1)
    py = 0.30

    for j, (split, n_pro, n_te, n_train, n_test) in enumerate(SPLITS):
        x0 = gap + j * (panel_w + gap)
        ax.add_patch(FancyBboxPatch(
            (x0, py - 0.04), panel_w, 0.42,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor="#FCFCFC", edgecolor=C_BORDER, linewidth=1.3,
        ))
        ax.text(x0 + panel_w / 2, py + 0.33,
                f"Split {split}\n({n_pro * 10}% / {n_te * 10}%)",
                ha="center", fontsize=10.5, fontweight="bold",
                color=C_TEXT, linespacing=1.1)
        # Strip
        cell_w = 0.018
        strip_x = x0 + (panel_w - cell_w * 10) / 2
        strip_y = py + 0.18
        _draw_strip(ax, strip_x, strip_y, n_pro, n_te, cell_w=cell_w,
                    cell_h=0.06)
        # Bracket lines + labels
        ax.plot(
            [strip_x, strip_x + cell_w * n_pro - 0.001],
            [strip_y - 0.012, strip_y - 0.012],
            color=C_PROFILE, linewidth=2.2,
        )
        ax.plot(
            [strip_x + cell_w * n_pro,
             strip_x + cell_w * 10 - 0.001],
            [strip_y - 0.012, strip_y - 0.012],
            color=C_TEST, linewidth=2.2,
        )
        ax.text(x0 + panel_w / 2, py + 0.07,
                f"profile  n = {n_train}", ha="center",
                fontsize=9.5, color=C_PROFILE, fontweight="bold")
        ax.text(x0 + panel_w / 2, py + 0.01,
                f"test  n = {n_test}", ha="center",
                fontsize=9.5, color=C_TEST, fontweight="bold")

    # Big horizontal arrow at bottom
    arrow_y = -0.05
    _arrow(ax, 0.05, arrow_y, 0.95, arrow_y, color=C_BORDER, lw=1.8)
    ax.text(0.5, arrow_y - 0.07,
            "increasing profile fraction  →  more in-context examples "
            "per participant for both methods",
            ha="center", fontsize=10, color=C_TEXT, style="italic")
    ax.text(0.05, arrow_y + 0.04, "low-data\n(LLM advantage)",
            ha="left", fontsize=9.5, color="#8A5A00", fontweight="bold")
    ax.text(0.95, arrow_y + 0.04, "more data",
            ha="right", fontsize=9.5, color="#444", fontweight="bold")


def draw_panel_c(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.97,
            "C.  Within each split, both methods see IDENTICAL test rows",
            ha="center", va="top", fontsize=15, fontweight="bold",
            color=C_TEXT)

    box_w = 0.30
    box_h = 0.55
    y0 = 0.18

    rf_x = 0.04
    llm_x = 0.66

    # Random Forest box
    ax.add_patch(FancyBboxPatch(
        (rf_x, y0), box_w, box_h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        facecolor="white", edgecolor=C_RF, linewidth=2.4,
    ))
    ax.text(rf_x + box_w / 2, y0 + box_h - 0.04,
            "Random Forest", ha="center", va="top", fontsize=12.5,
            fontweight="bold", color=C_RF)
    ax.text(rf_x + 0.014, y0 + box_h - 0.13, "INPUT",
            fontsize=9, fontweight="bold", color=C_TEXT)
    rf_lines = [
        "trained on PROFILE rows only",
        "feature sets:",
        "    • Demographics",
        "    • Avg-History (mean prior rating)",
        "    • Embedding (1024-d OpenAI)",
        "    • Embedding + Demographics",
    ]
    for i, line in enumerate(rf_lines):
        ax.text(rf_x + 0.020, y0 + box_h - 0.18 - i * 0.06, line,
                fontsize=9.5, va="top", color=C_TEXT)

    # LLM box
    ax.add_patch(FancyBboxPatch(
        (llm_x, y0), box_w, box_h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        facecolor="white", edgecolor=C_LLM, linewidth=2.4,
    ))
    ax.text(llm_x + box_w / 2, y0 + box_h - 0.04,
            "LLM Digital Twin", ha="center", va="top", fontsize=12.5,
            fontweight="bold", color=C_LLM)
    ax.text(llm_x + 0.014, y0 + box_h - 0.13, "INPUT",
            fontsize=9, fontweight="bold", color=C_TEXT)
    llm_lines = [
        "PROFILE rows used as in-context",
        "    history (no fine-tuning)",
        "5 LLMs:",
        "    • GPT-5 / GPT-4o-mini",
        "    • Gemini-2.5-Pro / Grok-4-Fast",
        "    • DeepSeek-R1 (7030 only)",
    ]
    for i, line in enumerate(llm_lines):
        ax.text(llm_x + 0.020, y0 + box_h - 0.18 - i * 0.06, line,
                fontsize=9.5, va="top", color=C_TEXT)

    # Centerpiece
    cx_w = 0.18
    cx_h = 0.40
    cx_x = 0.41
    cx_y = y0 + (box_h - cx_h) / 2
    ax.add_patch(FancyBboxPatch(
        (cx_x, cx_y), cx_w, cx_h,
        boxstyle="round,pad=0.012,rounding_size=0.014",
        facecolor="#FFF4E0", edgecolor=C_TEST, linewidth=2.4,
    ))
    ax.text(cx_x + cx_w / 2, cx_y + cx_h - 0.05,
            "Shared\nTEST set",
            ha="center", va="top", fontsize=11.5,
            fontweight="bold", color=C_TEST, linespacing=1.1)
    ax.text(cx_x + cx_w / 2, cx_y + cx_h - 0.18,
            "same response_ids\n& messages\nper split",
            ha="center", va="top", fontsize=8.5, color="#A1453F",
            style="italic")
    ax.text(cx_x + cx_w / 2, cx_y + 0.10,
            "metrics:\nAccuracy · Macro-F1\nWithin-participant ρ",
            ha="center", va="top", fontsize=8.0, color="#8A5A00")

    # Arrows from each box → centerpiece
    _arrow(ax, rf_x + box_w + 0.003, y0 + box_h / 2,
           cx_x - 0.005, cx_y + cx_h / 2, color=C_TEST, lw=1.8)
    _arrow(ax, llm_x - 0.003, y0 + box_h / 2,
           cx_x + cx_w + 0.005, cx_y + cx_h / 2, color=C_TEST, lw=1.8)
    # Place 'predict' labels in the open gap above the arrows
    label_y = y0 + box_h - 0.02
    ax.text((rf_x + box_w + cx_x) / 2, label_y,
            "predict", ha="center", fontsize=9.5, color=C_TEST,
            fontweight="bold")
    ax.text((llm_x + cx_x + cx_w) / 2, label_y,
            "predict", ha="center", fontsize=9.5, color=C_TEST,
            fontweight="bold")

    # Footer
    ax.text(0.5, 0.10,
            "Apples-to-apples: metrics computed on the same response_ids "
            "for RF and LLM at every split.",
            ha="center", fontsize=11, fontweight="bold", color=C_OK)
    ax.text(0.5, 0.05,
            "Sweeping across splits traces the learning curve "
            "(see lc_rf_vs_llm_main).",
            ha="center", fontsize=9.5, color="#555", style="italic")


def main():
    fig = plt.figure(figsize=(15.0, 13.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.85, 1.05, 1.30],
                          hspace=0.35)

    draw_panel_a(fig.add_subplot(gs[0]))
    draw_panel_b(fig.add_subplot(gs[1]))
    draw_panel_c(fig.add_subplot(gs[2]))

    fig.suptitle(
        "Experimental design: apples-to-apples RF vs LLM Digital-Twin "
        "learning curve",
        fontsize=17, fontweight="bold", y=0.995,
    )
    fig.patch.set_facecolor("white")

    out = figures_path("lc_rf_vs_llm_schematic")
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight",
                facecolor="white")
    fig.savefig(out + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}.png + .pdf")


if __name__ == "__main__":
    main()
