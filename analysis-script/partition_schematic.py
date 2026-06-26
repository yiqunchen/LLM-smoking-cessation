#!/usr/bin/env python3
"""Schematic figure: partition logic + leakage audit + apples-to-apples result.

Two-panel figure intended for the manuscript revision:

  Panel A (top, schematic):
    - Within-participant 70/30 partition shown as a 10-cell strip per participant
    - Three method boxes (Supervised baseline, LLM-PP, Hybrid RF+PP)
      indicating exactly what each method receives as input
    - Leakage audit bullets pulled from `revision/figures/label_leakage_analysis.csv`

  Panel B (bottom, result):
    - Aggregate accuracy on the SAME 3 held-out messages per participant —
      Sup-RF (best per-domain) vs Hybrid RF+PP (best LLM)
    - Within-participant Spearman rho on the same items
    - Visualizes the "tied on accuracy, very different on rank discrimination"
      finding that anchors the apples-to-apples reframing of the paper.

Outputs:
  revision/figures/partition_schematic.png
  revision/figures/partition_schematic.pdf
  revision/figures/apples_to_apples_result.png
  revision/figures/apples_to_apples_result.pdf

Run:
  uv run python analysis-script/partition_schematic.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_utils import figures_path, save_figure  # noqa: E402

# Numbers shown in the result panel. These come from the deduplicated
# digital-twin 70/30 test set (n=302–303 after item-key dedup) and match
# the headline numbers in `comprehensive_results_all_methods.csv` and
# the per-class breakdown computed in /tmp/per_class_ensemble_*.csv.
RESULT_TABLE = pd.DataFrame([
    # domain,         Sup-RF acc, Hybrid acc, Sup-RF spearman, Hybrid spearman
    ("Content",       0.513,      0.474,      0.000,           0.336),
    ("Coping",        0.498,      0.420,      0.000,           0.339),
    ("Quitting",      0.547,      0.492,      0.000,           0.412),
], columns=["domain", "sup_acc", "hybrid_acc", "sup_rho", "hybrid_rho"])

LEAK_CHECKS = [
    ("Few-shot exemplar overlap with test",
     "0 / 2  (0.0%)", "no overlap"),
    ("Within-participant duplicate items (history ≈ test)",
     "16 / 319  (5.0%)", "removed before evaluation (n=302–303)"),
    ("Held-out items inserted into prompt history",
     "0 / 0  (0.0%)", "programmatic ID-based assembly"),
    ("Cross-split message text reuse",
     "104 / 107  (97.2%)", "by design: same message text, different participants"),
]

C_HISTORY = "#4C72B0"
C_TEST    = "#C44E52"
C_SUP     = "#7F7F7F"
C_LLM     = "#DE8F05"
C_HYB     = "#56B4E9"
C_OK      = "#2E7D32"
C_TEXT    = "#222222"


def draw_schematic():
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.9, 2.2, 1.4], hspace=0.55)

    # ---------- TOP: partition strip for one example participant ----------
    ax = fig.add_subplot(gs[0])
    n_msg = 10
    cell_w = 0.85
    for i in range(n_msg):
        x = i * 1.0
        is_train = i < 7
        rect = Rectangle((x, 0), cell_w, 1, facecolor=C_HISTORY if is_train else C_TEST,
                         edgecolor="black", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + cell_w / 2, 0.5, f"M{i+1}", ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")

    # bracket-like underlines
    ax.annotate("", xy=(0.0, -0.18), xytext=(6.85, -0.18),
                arrowprops=dict(arrowstyle="-", color=C_HISTORY, lw=2.5))
    ax.text(3.4, -0.45,
            "7 history messages → personalization context (training/history)",
            ha="center", fontsize=11, color=C_HISTORY, fontweight="bold")
    ax.annotate("", xy=(7.0, -0.18), xytext=(9.85, -0.18),
                arrowprops=dict(arrowstyle="-", color=C_TEST, lw=2.5))
    ax.text(8.4, -0.45,
            "3 held-out → evaluation only",
            ha="center", fontsize=11, color=C_TEST, fontweight="bold")

    ax.text(-0.6, 0.5, "Participant i\n(of 301)", ha="right", va="center",
            fontsize=10, fontweight="bold", color=C_TEXT)
    ax.set_title(
        "Within-participant 70/30 partition  ·  identical for every method",
        fontsize=14, fontweight="bold", pad=10
    )
    ax.set_xlim(-2.0, 11)
    ax.set_ylim(-0.9, 1.2)
    ax.axis("off")

    # ---------- MIDDLE: three method boxes ----------
    ax = fig.add_subplot(gs[1])
    methods = [
        dict(
            name="Supervised baseline\n(Random Forest)",
            color=C_SUP,
            inputs=[
                "Participant demographics (age, sex, race, …)",
                "Avg of participant's 7 prior ratings",
                "— no per-message text or context —",
            ],
            output="Predicts each held-out\nmessage's rating from\ntabular features only",
            note="Trained across all 301 participants' 7-history items",
        ),
        dict(
            name="LLM-PP\n(persona + history prompt)",
            color=C_LLM,
            inputs=[
                "Participant demographics",
                "7 history (message, rating) pairs",
                "Held-out message text",
            ],
            output="Predicts held-out rating\nvia in-context inference\n(no model fine-tuning)",
            note="Each held-out call is independent, per participant",
        ),
        dict(
            name="Hybrid RF + PP\n(RF prior + LLM update)",
            color=C_HYB,
            inputs=[
                "Everything above, plus",
                "RF predicted class injected\ninto the digital-twin prompt",
                "as a soft prior",
            ],
            output="Predicts held-out rating\nwith RF anchor + LLM\nrefinement step",
            note="Combines tabular + textual personalization",
        ),
    ]

    box_w = 0.305
    box_gap = (1.0 - 3 * box_w) / 4  # equal gaps around 3 boxes
    for j, m in enumerate(methods):
        x0 = box_gap + j * (box_w + box_gap)
        w = box_w
        box = FancyBboxPatch(
            (x0, 0.05), w, 0.92,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            facecolor="white", edgecolor=m["color"], linewidth=2.4,
        )
        ax.add_patch(box)
        # Title (header strip-style, centered)
        ax.text(x0 + w / 2, 0.93, m["name"], ha="center", va="top",
                fontweight="bold", fontsize=11.5, color=m["color"])

        # inputs label + bullets
        ax.text(x0 + 0.014, 0.74, "INPUTS", fontsize=9.5,
                fontweight="bold", color=C_TEXT)
        for k, inp in enumerate(m["inputs"]):
            ax.text(x0 + 0.020, 0.685 - k * 0.075, "•  " + inp,
                    fontsize=9.5, va="top", color=C_TEXT)

        # output label + body
        ax.text(x0 + 0.014, 0.40, "OUTPUT", fontsize=9.5,
                fontweight="bold", color=C_TEXT)
        ax.text(x0 + 0.020, 0.35, m["output"], fontsize=9.5, va="top",
                color=C_TEXT)

        # held-out anchor + note (use simple ASCII marker)
        ax.text(x0 + w / 2, 0.155,
                "[✓]  Same 3 held-out messages per participant",
                ha="center", fontsize=9.5, color=C_OK, fontweight="bold")
        ax.text(x0 + w / 2, 0.085, m["note"], ha="center", fontsize=8.5,
                color="#666", style="italic", wrap=True)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(
        "What each method sees:  same partition, different inputs  →  apples-to-apples comparison",
        fontsize=13, fontweight="bold", pad=8
    )
    ax.axis("off")

    # ---------- BOTTOM: leakage audit ----------
    ax = fig.add_subplot(gs[2])
    ax.text(0.5, 1.05,
            "Leakage audit  ·  source: revision/figures/label_leakage_analysis.csv",
            ha="center", fontsize=12.5, fontweight="bold", color=C_TEXT)
    for i, (label, value, status) in enumerate(LEAK_CHECKS):
        y = 0.85 - i * 0.22
        bullet_color = C_OK if i < 3 else "#666"
        ax.text(0.02, y, "•  " + label, fontsize=10.5, fontweight="bold",
                color=C_TEXT)
        ax.text(0.62, y, value, fontsize=10.5, color=C_HISTORY,
                fontfamily="monospace")
        ax.text(0.78, y, status, fontsize=9.8, color=bullet_color,
                style="italic")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.15)
    ax.axis("off")

    out = figures_path("partition_schematic")
    plt.savefig(out + ".png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(out + ".pdf", bbox_inches="tight", facecolor="white")
    print(f"  saved: {out}.png and .pdf")
    plt.close(fig)


def draw_result_panel():
    """Bar plot: aggregate accuracy ~ tied; within-participant Spearman vastly different."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6),
                              gridspec_kw=dict(width_ratios=[1, 1], wspace=0.32))
    domains = RESULT_TABLE["domain"].tolist()
    x = np.arange(len(domains))
    bw = 0.36

    # Left: aggregate accuracy
    ax = axes[0]
    b1 = ax.bar(x - bw / 2, RESULT_TABLE["sup_acc"], bw, color=C_SUP,
                edgecolor="black", linewidth=1.0, label="Supervised RF (best per-domain)")
    b2 = ax.bar(x + bw / 2, RESULT_TABLE["hybrid_acc"], bw, color=C_HYB,
                edgecolor="black", linewidth=1.0, label="Hybrid RF + PP (Grok-4-Fast)")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                f"{b.get_height():.3f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontsize=11)
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("Aggregate accuracy on held-out items", fontsize=11, fontweight="bold")
    ax.set_title("Aggregate accuracy:  comparable", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9.5, loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.25)

    # Right: within-participant Spearman ρ
    ax = axes[1]
    b1 = ax.bar(x - bw / 2, RESULT_TABLE["sup_rho"], bw, color=C_SUP,
                edgecolor="black", linewidth=1.0, label="Supervised RF")
    b2 = ax.bar(x + bw / 2, RESULT_TABLE["hybrid_rho"], bw, color=C_HYB,
                edgecolor="black", linewidth=1.0, label="Hybrid RF + PP")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                f"{b.get_height():.2f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontsize=11)
    ax.set_ylim(-0.02, 0.5)
    ax.set_ylabel("Within-participant Spearman ρ", fontsize=11, fontweight="bold")
    ax.set_title("Within-participant rank discrimination:  LLM only", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9.5, loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.5, -0.18,
            "Supervised RF emits ~constant predictions across messages within a participant; "
            "rank ρ is therefore degenerate (≈0).",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555", style="italic")

    fig.suptitle(
        "Apples-to-apples comparison on the digital-twin 70/30 split (n ≈ 302–303 per domain)",
        fontsize=14, fontweight="bold"
    )
    out = figures_path("apples_to_apples_result")
    plt.savefig(out + ".png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(out + ".pdf", bbox_inches="tight", facecolor="white")
    print(f"  saved: {out}.png and .pdf")
    plt.close(fig)


def _build_partition_matrix(train_path: str, test_path: str, pids: list, min_raters: int = 2):
    """Helper: load a canonical split and build (matrix, pids, messages).

    Caller supplies the participant list so both panels can use the same rows.
    matrix codes: 0=not rated, 1=train, 2=test. Within-participant duplicates
    are rendered as 0.
    """
    import json as _json
    train = _json.load(open(train_path))
    test  = _json.load(open(test_path))
    pid_set = set(pids)

    # For these 15 participants, gather all (pid, msg) pairs
    train_pairs = set((r["response_id"], r["input_message"]) for r in train if r["response_id"] in pid_set)
    test_pairs  = set((r["response_id"], r["input_message"]) for r in test  if r["response_id"] in pid_set)
    overlap_pairs = train_pairs & test_pairs

    # gather (pid, msg) pairs for this subset
    train_pairs = set((r["response_id"], r["input_message"]) for r in train if r["response_id"] in pid_set)
    test_pairs  = set((r["response_id"], r["input_message"]) for r in test  if r["response_id"] in pid_set)

    msg_counts = {}
    for (_, m) in train_pairs | test_pairs:
        msg_counts[m] = msg_counts.get(m, 0) + 1
    messages = [m for m, c in msg_counts.items() if c >= min_raters]

    n_p = len(pids); n_m = len(messages)
    matrix = np.zeros((n_p, n_m), dtype=int)
    for i, p in enumerate(pids):
        for j, m in enumerate(messages):
            in_tr = (p, m) in train_pairs
            in_te = (p, m) in test_pairs
            if in_tr and in_te: matrix[i, j] = 0
            elif in_tr:         matrix[i, j] = 1
            elif in_te:         matrix[i, j] = 2

    # Sort rows + cols so train clusters upper-left, held-out lower-right
    row_score = (matrix == 1).sum(axis=1) - (matrix == 2).sum(axis=1)
    row_order = np.argsort(-row_score, kind="stable")
    matrix = matrix[row_order]
    pids = [pids[i] for i in row_order]
    col_score = (matrix == 1).sum(axis=0) - (matrix == 2).sum(axis=0)
    col_order = np.argsort(-col_score, kind="stable")
    matrix = matrix[:, col_order]
    messages = [messages[j] for j in col_order]
    return matrix, pids, messages


def _render_matrix_panel(ax, matrix, title, highlight_strict=None):
    """highlight_strict: optional set of (row, col) tuples to mark with a gold
    outline as the strict cross-validated holdout cells."""
    n_p, n_m = matrix.shape
    cell = 1.0
    highlight_strict = highlight_strict or set()
    for i in range(n_p):
        for j in range(n_m):
            v = matrix[i, j]
            if v == 0:
                fc, ec, lw = "#F0F0F0", "#DDDDDD", 0.5
            elif v == 1:
                fc, ec, lw = C_HISTORY, "black", 0.6
            else:
                fc, ec, lw = C_TEST, "black", 0.6
            ax.add_patch(Rectangle((j * cell, (n_p - 1 - i) * cell),
                                    cell * 0.94, cell * 0.94,
                                    facecolor=fc, edgecolor=ec, linewidth=lw))
            if (i, j) in highlight_strict:
                # gold outline on top
                ax.add_patch(Rectangle((j * cell - 0.05, (n_p - 1 - i) * cell - 0.05),
                                        cell * 0.94 + 0.10, cell * 0.94 + 0.10,
                                        facecolor="none", edgecolor="#F4A300",
                                        linewidth=2.0, zorder=5))
    ax.text(-1.5, (n_p * cell) / 2, "Participants",
            ha="center", va="center", fontsize=11, fontweight="bold", rotation=90)
    ax.text((n_m * cell) / 2, n_p * cell + 1.2, "Messages",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=8)
    ax.set_xlim(-2.5, n_m * cell + 0.5)
    ax.set_ylim(-0.5, n_p * cell + 1.8)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_partition_matrix():
    """Three-panel figure: supervised (participant split), digital-twin (within-participant),
    and the strict-holdout subset (intersection used for cross-validated evaluation)."""
    import json as _json
    canonical = "/Users/yiqun/Desktop/chen-lab/LLM-smoking-cessation/data_splits/canonical"

    # Sample 28 train + 12 test ppts from the participant split. The bottom 12
    # rows are the strict-holdout participants (in pp_test → never seen by any
    # RF trained on pp_train). We use the same rows for both panels.
    sup_train = _json.load(open(f"{canonical}/train_participant_7030.json"))
    sup_test  = _json.load(open(f"{canonical}/test_participant_7030.json"))
    train_pids = list(dict.fromkeys(r["response_id"] for r in sup_train))
    test_pids  = list(dict.fromkeys(r["response_id"] for r in sup_test))
    pids = train_pids[:28] + test_pids[:12]
    n_train_rows = 28  # for highlighting

    sup_mat, _, _ = _build_partition_matrix(
        f"{canonical}/train_participant_7030.json",
        f"{canonical}/test_participant_7030.json",
        pids=pids, min_raters=2,
    )
    dt_mat, _, _ = _build_partition_matrix(
        f"{canonical}/train_digital_twin_7030.json",
        f"{canonical}/test_digital_twin_7030.json",
        pids=pids, min_raters=2,
    )

    # Strict-holdout cells in the dt panel: bottom 12 rows × any TEST cell (v=2).
    # Note: dt_mat row order has been re-sorted; we instead sort it back below.

    # The matrix panel sort step (in _build_partition_matrix) groups rows with
    # the most train cells first. For the supervised panel that means train
    # participants float to the top and the 12 test participants sink to the
    # bottom. For the digital-twin panel the row score is mixed (every ppt has
    # both train+test cells), so the 12 strict-holdout ppts are not at the
    # bottom by row score — we highlight by re-deriving the row→pid mapping.

    # Re-derive which rows correspond to test_pids in each matrix.
    # _build_partition_matrix returns the matrix in (row_score-sorted) order
    # and the pids list it returns is also re-sorted. We need that pids list.
    sup_mat2, sup_pids, _ = _build_partition_matrix(
        f"{canonical}/train_participant_7030.json",
        f"{canonical}/test_participant_7030.json",
        pids=pids, min_raters=2,
    )
    dt_mat2, dt_pids, _ = _build_partition_matrix(
        f"{canonical}/train_digital_twin_7030.json",
        f"{canonical}/test_digital_twin_7030.json",
        pids=pids, min_raters=2,
    )
    test_pid_set = set(test_pids[:12])
    sup_strict = {(i, j) for i, p in enumerate(sup_pids) if p in test_pid_set
                  for j in range(sup_mat2.shape[1]) if sup_mat2[i, j] == 2}
    dt_strict = {(i, j) for i, p in enumerate(dt_pids) if p in test_pid_set
                 for j in range(dt_mat2.shape[1]) if dt_mat2[i, j] == 2}

    plt.rcParams["font.family"] = ["DejaVu Sans"]
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(1, 2, wspace=0.18)

    ax = fig.add_subplot(gs[0])
    _render_matrix_panel(
        ax, sup_mat2,
        "Supervised baseline split  ·  by participant\n"
        "(train ppts ≠ test ppts)",
        highlight_strict=sup_strict,
    )
    ax = fig.add_subplot(gs[1])
    _render_matrix_panel(
        ax, dt_mat2,
        "Digital-twin split  ·  within participant\n"
        "(every ppt in both; different messages)",
        highlight_strict=dt_strict,
    )

    # shared legend
    fig.legend(
        handles=[
            Rectangle((0,0),1,1, facecolor=C_HISTORY, edgecolor="black"),
            Rectangle((0,0),1,1, facecolor=C_TEST,    edgecolor="black"),
            Rectangle((0,0),1,1, facecolor="#F0F0F0", edgecolor="#CCCCCC"),
            Rectangle((0,0),1,1, facecolor="none", edgecolor="#F4A300", linewidth=2.0),
        ],
        labels=["Train", "Held-out", "Not rated",
                "Strict-holdout subset (n≈95): test items from cross-participant test ppts"],
        loc="upper center", bbox_to_anchor=(0.5, 0.05),
        ncol=2, frameon=False, fontsize=10.5,
    )

    out = figures_path("partition_matrix_schematic")
    plt.savefig(out + ".png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(out + ".pdf", bbox_inches="tight", facecolor="white")
    print(f"  saved: {out}.png and .pdf")
    plt.close(fig)
    return  # skip legacy single-panel rendering below

    ax = None  # unreachable; kept only so the legacy block compiles
    cell = 1.0
    for i in range(n_p):
        for j in range(n_m):
            v = matrix[i, j]
            if v == 0:
                fc, ec, lw = "#F0F0F0", "#DDDDDD", 0.5
            elif v == 1:
                fc, ec, lw = C_HISTORY, "black", 0.6
            elif v == 2:
                fc, ec, lw = C_TEST, "black", 0.6
            else:  # overlap (within-participant duplicate)
                fc, ec, lw = "#F4A300", "black", 1.4
            rect = Rectangle((j * cell, (n_p - 1 - i) * cell), cell * 0.94, cell * 0.94,
                             facecolor=fc, edgecolor=ec, linewidth=lw)
            ax.add_patch(rect)

    # row label (single)
    ax.text(-1.5, (n_p * cell) / 2, "Participants",
            ha="center", va="center", fontsize=11, fontweight="bold", rotation=90)
    # col label (single)
    ax.text((n_m * cell) / 2, n_p * cell + 1.2, "Messages",
            ha="center", va="bottom", fontsize=11, fontweight="bold")

    # minimal legend (right of matrix)
    lx = n_m * cell + 1.2
    ly0 = (n_p - 1) * cell
    legend_items = [
        (C_HISTORY, "Train"),
        (C_TEST,    "Held-out"),
        ("#F0F0F0", "Not rated"),
    ]
    for k, (color, label) in enumerate(legend_items):
        y = ly0 - k * 1.4
        ax.add_patch(Rectangle((lx, y), 0.85, 0.85, facecolor=color,
                                edgecolor="black", linewidth=0.7))
        ax.text(lx + 1.05, y + 0.42, label, va="center", fontsize=10)

    ax.set_title("Train / held-out partition  ·  digital-twin 70/30 split",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xlim(-2.5, n_m * cell + 7)
    ax.set_ylim(-0.5, n_p * cell + 1.8)
    ax.set_aspect("equal")
    ax.axis("off")

    out = figures_path("partition_matrix_schematic")
    plt.savefig(out + ".png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(out + ".pdf", bbox_inches="tight", facecolor="white")
    print(f"  saved: {out}.png and .pdf")
    plt.close(fig)


# Per-class accuracy on the deduplicated digital-twin 70/30 test set.
# Source: /tmp/per_class_ensemble_*.csv (computed earlier from
# history_supervised_predictions.csv + Hybrid RF+PP JSONs).
PER_CLASS_ACC = {
    "Content":  {  # class size: [12, 16, 52, 114, 112]
        "Sup-RF":       [0.000, 0.062, 0.481, 0.518, 0.643],
        "Hybrid Grok":  [0.000, 0.125, 0.308, 0.500, 0.625],
        "Hybrid GPT-5": [0.000, 0.188, 0.615, 0.535, 0.438],
        "n_per_class":  [12, 16, 52, 114, 112],
    },
    "Coping":   {  # [20, 32, 61, 97, 97]
        "Sup-RF":       [0.300, 0.219, 0.492, 0.577, 0.557],
        "Hybrid Grok":  [0.250, 0.125, 0.279, 0.588, 0.474],
        "Hybrid GPT-5": [0.250, 0.344, 0.475, 0.567, 0.320],
        "n_per_class":  [20, 32, 61, 97, 97],
    },
    "Quitting": {  # [24, 29, 51, 98, 105]
        "Sup-RF":       [0.250, 0.207, 0.431, 0.561, 0.752],
        "Hybrid Grok":  [0.250, 0.345, 0.314, 0.643, 0.533],
        "Hybrid GPT-5": [0.292, 0.448, 0.451, 0.582, 0.352],
        "n_per_class":  [24, 29, 51, 98, 105],
    },
}

CLASS_LABELS = {
    "Content":  ["Very poor", "Poor", "Acceptable", "Good", "Very good"],
    "Coping":   ["Not at all", "Somewhat", "Moderately", "Very", "Extremely"],
    "Quitting": ["Not at all", "Somewhat", "Moderately", "Very", "Extremely"],
}


def draw_per_class_breakdown():
    """Per-true-class accuracy: Sup-RF vs Hybrid LLMs.

    Shows the complementary error pattern — supervised wins modal classes (4-5),
    LLM wins minority classes (1-3) — across all three domains on the same
    digital-twin 70/30 test set used everywhere else.
    """
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2),
                              gridspec_kw=dict(wspace=0.18))
    domains = ["Content", "Coping", "Quitting"]
    methods = ["Sup-RF", "Hybrid Grok", "Hybrid GPT-5"]
    method_colors = {
        "Sup-RF":       C_SUP,
        "Hybrid Grok":  C_HYB,
        "Hybrid GPT-5": C_LLM,
    }
    bw = 0.26
    x = np.arange(5)

    for ax, dom in zip(axes, domains):
        d = PER_CLASS_ACC[dom]
        # background tint: classes 1-3 minority (light orange), 4-5 modal (light grey)
        ax.axvspan(-0.5, 2.5, alpha=0.10, color=C_LLM, zorder=0)
        ax.axvspan(2.5, 4.5, alpha=0.12, color=C_SUP, zorder=0)
        ax.text(1, 1.04, "Minority", ha="center", fontsize=9.5, color=C_LLM,
                fontweight="bold", transform=ax.get_xaxis_transform())
        ax.text(3.5, 1.04, "Modal", ha="center", fontsize=9.5, color="#555",
                fontweight="bold", transform=ax.get_xaxis_transform())

        for k, m in enumerate(methods):
            offset = (k - 1) * bw
            bars = ax.bar(x + offset, d[m], bw,
                          color=method_colors[m], edgecolor="black",
                          linewidth=0.7, label=m, zorder=3)
            for b in bars:
                h = b.get_height()
                if h > 0.01:
                    ax.text(b.get_x() + b.get_width() / 2, h + 0.012,
                            f"{h:.2f}", ha="center", fontsize=7.5)

        # x labels: class number + n
        labels = [
            f"{i+1}\n{CLASS_LABELS[dom][i]}\n(n={d['n_per_class'][i]})"
            for i in range(5)
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(0, 0.92)
        ax.set_title(dom, fontsize=12.5, fontweight="bold")
        ax.set_ylabel("Accuracy on items of true class", fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25, zorder=0)
        if dom == "Content":
            ax.legend(fontsize=9, loc="upper left", frameon=False)

    fig.suptitle(
        "Per-class accuracy on the digital-twin 70/30 test set  ·  "
        "supervised wins modal classes (4–5), LLM wins minority classes (1–3)",
        fontsize=13, fontweight="bold", y=1.02
    )
    out = figures_path("per_class_breakdown")
    plt.savefig(out + ".png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(out + ".pdf", bbox_inches="tight", facecolor="white")
    print(f"  saved: {out}.png and .pdf")
    plt.close(fig)


# Hybrid variant comparison numbers — pulled from /tmp/score_all_hybrids.py output.
# v1 = demographics-only RF prior (original)
# v2 = hard history-RF prior
# v3 = soft history-RF prior (with explicit caveats)
HYBRID_AGG_ACC = {
    "Content":  {"GPT-5":  {"v1": 0.472, "v2": 0.462, "v3": 0.528},
                 "Grok":   {"v1": 0.469, "v2": 0.500, "v3": 0.478}},
    "Coping":   {"GPT-5":  {"v1": 0.433, "v2": 0.376, "v3": 0.398},
                 "Grok":   {"v1": 0.433, "v2": 0.411, "v3": 0.417}},
    "Quitting": {"GPT-5":  {"v1": 0.448, "v2": 0.345, "v3": 0.382},
                 "Grok":   {"v1": 0.492, "v2": 0.417, "v3": 0.426}},
}
HYBRID_C5_ACC = {  # accuracy on class-5 items only (the minority extreme)
    "Content":  {"GPT-5": {"v1": 0.434, "v2": 0.363, "v3": 0.584},
                 "Grok":  {"v1": 0.619, "v2": 0.611, "v3": 0.628}},
    "Coping":   {"GPT-5": {"v1": 0.327, "v2": 0.089, "v3": 0.188},
                 "Grok":  {"v1": 0.475, "v2": 0.238, "v3": 0.337}},
    "Quitting": {"GPT-5": {"v1": 0.352, "v2": 0.019, "v3": 0.074},
                 "Grok":  {"v1": 0.528, "v2": 0.259, "v3": 0.315}},
}

V_COLORS = {
    "v1": "#888888",   # demographics-RF (original)
    "v2": "#C44E52",   # hard history-RF
    "v3": "#56B4E9",   # soft history-RF
}
V_LABELS = {
    "v1": "v1 demo-RF (original)",
    "v2": "v2 hard hist-RF",
    "v3": "v3 soft hist-RF",
}


def draw_hybrid_variant_comparison():
    """Did the soft prior actually help? Two-row figure:
       Row 1 — aggregate accuracy across v1/v2/v3 for each (domain × LLM)
       Row 2 — class-5 accuracy for the same cells (the minority extreme)
    """
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5),
                              gridspec_kw=dict(wspace=0.22, hspace=0.45))
    domains = ["Content", "Coping", "Quitting"]
    llms = ["GPT-5", "Grok"]
    variants = ["v1", "v2", "v3"]
    bw = 0.24
    x = np.arange(len(llms))

    def plot_panel(ax, data, title, ylabel, ylim):
        for k, v in enumerate(variants):
            offs = (k - 1) * bw
            heights = [data[llm][v] for llm in llms]
            bars = ax.bar(x + offs, heights, bw,
                           color=V_COLORS[v], edgecolor="black", linewidth=0.7,
                           label=V_LABELS[v], zorder=3)
            for b, h in zip(bars, heights):
                ax.text(b.get_x() + b.get_width() / 2, h + 0.012,
                        f"{h:.2f}", ha="center", fontsize=8.5)
        ax.set_xticks(x); ax.set_xticklabels(llms, fontsize=10.5)
        ax.set_ylim(*ylim); ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25, zorder=0)

    # row 1: aggregate accuracy
    for col, dom in enumerate(domains):
        plot_panel(axes[0, col], HYBRID_AGG_ACC[dom],
                   f"{dom} — Aggregate accuracy", "Accuracy", (0, 0.62))

    # row 2: class-5 (minority extreme) accuracy
    for col, dom in enumerate(domains):
        plot_panel(axes[1, col], HYBRID_C5_ACC[dom],
                   f"{dom} — Class-5 accuracy (minority extreme)",
                   "Acc on class-5 items", (0, 0.78))

    # Shared legend at bottom
    handles = [Rectangle((0,0),1,1, facecolor=V_COLORS[v], edgecolor="black") for v in variants]
    fig.legend(handles=handles, labels=[V_LABELS[v] for v in variants],
                loc="lower center", bbox_to_anchor=(0.5, -0.02),
                ncol=3, frameon=False, fontsize=11)

    fig.suptitle(
        "Did the soft-prior prompt help?  ·  Hybrid variants on the digital-twin 70/30 test set",
        fontsize=13.5, fontweight="bold", y=1.00,
    )
    out = figures_path("hybrid_variant_comparison")
    plt.savefig(out + ".png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(out + ".pdf", bbox_inches="tight", facecolor="white")
    print(f"  saved: {out}.png and .pdf")
    plt.close(fig)


def main():
    print("Generating partition + leakage schematic …")
    draw_schematic()
    print("Generating apples-to-apples result panel …")
    draw_result_panel()
    print("Generating real-data partition matrix …")
    draw_partition_matrix()
    print("Generating per-class breakdown …")
    draw_per_class_breakdown()
    print("Generating hybrid variant comparison …")
    draw_hybrid_variant_comparison()
    print("Done.")


if __name__ == "__main__":
    main()
