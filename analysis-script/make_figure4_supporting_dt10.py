#!/usr/bin/env python3
"""Regenerate the Figure 4 supporting message-selection plot WITHOUT anchor methods.

Shows supervised RF vs LLM-PP gain in mean human rating over random selection,
by domain across K, on the fixed Demographics + History + Message Embedding block.
Anchor hybrids (RF-anchor/LLM-anchor) are dropped per the revised manuscript.

Source : revision/figures/message_selection_methods_k7.csv
Outputs: figures/figure4/supporting_message_selection_gain_dt10.{png,pdf}
"""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "revision" / "figures" / "message_selection_methods_k7.csv"
OUT = ROOT / "figures" / "figure4"
OUT.mkdir(parents=True, exist_ok=True)

DOMAINS = ["Content", "Coping", "Quitting"]
METHODS = ["Supervised RF", "LLM-PP"]
COLORS = {"Supervised RF": "#0072B2", "LLM-PP": "#009E73"}
MARKERS = {"Supervised RF": "s", "LLM-PP": "o"}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.linestyle": "--",
    "grid.alpha": 0.12, "grid.color": "#4D4D4D", "grid.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 400,
})

rows = list(csv.DictReader(open(SRC)))
Ks = sorted({int(r["K"]) for r in rows})

def gain(method, dom, K):
    for r in rows:
        if r["method_display"] == method and r["domain"] == dom and r["K"] == str(K):
            return float(r["gain_over_random"])
    return np.nan

fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
for ax, dom in zip(axes, DOMAINS):
    for m in METHODS:
        ys = [gain(m, dom, K) for K in Ks]
        ax.plot(Ks, ys, marker=MARKERS[m], color=COLORS[m], lw=2.2, ms=7,
                label=m, zorder=3)
    ax.axhline(0, color="#4D4D4D", lw=1.0, ls=":")
    ax.set_title(dom, fontsize=13)
    ax.set_xlabel("K (messages selected)")
    ax.set_xticks(Ks)
    ax.tick_params(length=0)
axes[0].set_ylabel("Gain in mean human rating\nover random selection")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 1.0), fontsize=12)
fig.suptitle("Figure 4 (supporting). Message-selection gain over random — dt10 split",
             y=1.08, fontsize=14)
fig.text(0.5, 0.99, "Feature block = Demographics + History + Message Embedding  |  "
         "RF best for Content; LLM-PP best for Coping and Quitting",
         ha="center", va="top", fontsize=10, color="#4D4D4D")
fig.tight_layout(rect=[0, 0, 1, 0.9])
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"supporting_message_selection_gain_dt10.{ext}", bbox_inches="tight")
plt.close(fig)
print("wrote supporting_message_selection_gain_dt10.png / .pdf")
