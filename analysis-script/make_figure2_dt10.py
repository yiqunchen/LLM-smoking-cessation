#!/usr/bin/env python3
"""Regenerate Figure 2 on the unified dt10 split (single split for the whole paper).

Main panel  : Supervised RF vs LLM-PP, strict shared-row (dt10, k_train=7,
              feature_set=Demographics, N=898 — same held-out rows for both
              methods), across Content/Coping/Quitting on Accuracy, Macro-F1, QWK.
Context panel: Generic zero-/few-shot LLM prompting (mean of 5 LLMs, k_train=7),
              shown SEPARATELY because it covers a different/smaller subset
              (N approx 87 per cell), never merged into the 898-row comparison.

Anchor hybrids (RF-anchor/LLM-anchor) are intentionally NOT plotted (dropped
from the revised manuscript). LLM-PP here is the Grok-4-Fast persona-conditioned
run, matching the ensemble source file.

Sources:
    revision/figures/lc_dt10_ensemble_k7.csv   (RF only / LLM only, per feature_set)
    revision/figures/lc_dt10_generic_llm.csv   (zero/few-shot per model)

Outputs:
    figures/figure2/figure2_main_dt10_rf_vs_llmpp.{png,pdf}
    figures/figure2/figure2_context_generic_llm.{png,pdf}
    figures/figure2/figure2_dt10_source_table.csv
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REV = PROJECT_ROOT / "revision" / "figures"
OUT = PROJECT_ROOT / "figures" / "figure2"
OUT.mkdir(parents=True, exist_ok=True)

DOMAINS = ["content", "coping", "quitting"]
DOMAIN_LABEL = {"content": "Content", "coping": "Coping", "quitting": "Quitting"}
METRICS = [("Accuracy", "Accuracy"), ("F1", "Macro-F1"), ("QWK", "QWK")]

# Okabe-Ito colorblind-safe palette (matches repo convention: RF=blue, LLM-PP=green)
COLOR_RF = "#0072B2"
COLOR_LLMPP = "#009E73"
COLOR_ZS = "#D55E00"
COLOR_FS = "#E69F00"

FONT_CHAIN = ["Arial", "Helvetica", "Helvetica Neue", "Avenir", "DejaVu Sans"]
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": FONT_CHAIN,
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
    "grid.color": "#4D4D4D",
    "grid.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 150,
    "savefig.dpi": 400,
})


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def ensemble_val(rows, feature_set, ensemble, domain, metric):
    for d in rows:
        if (d["feature_set"] == feature_set and d["ensemble"] == ensemble
                and d["domain"] == domain):
            v = d[metric].strip()
            return float(v) if v else float("nan")
    return float("nan")


def generic_mean(rows, group, domain, metric, k="7"):
    vals = [float(d[metric]) for d in rows
            if d["feature_set"] == group and d["domain"] == domain
            and d["k_train"] == k and d[metric].strip()]
    return statistics.mean(vals) if vals else float("nan")


def grouped_bars(ax, data, methods, colors, title, ylim_top):
    """data[method][domain] -> value."""
    x = np.arange(len(DOMAINS))
    n = len(methods)
    w = 0.8 / n
    for i, m in enumerate(methods):
        vals = [data[m][d] for d in DOMAINS]
        bars = ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=m,
                      color=colors[m], edgecolor="white", linewidth=0.8, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + ylim_top * 0.015,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold", color="#222222")
    ax.set_xticks(x)
    ax.set_xticklabels([DOMAIN_LABEL[d] for d in DOMAINS])
    ax.set_ylim(0, ylim_top)
    ax.set_title(title, fontsize=13, pad=8)
    ax.tick_params(length=0)


def make_panel(rows_for, methods, colors, suptitle, subtitle, outstem):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    tops = {"Accuracy": 0.65, "F1": 0.65, "QWK": 0.72}
    for ax, (mcol, mlabel) in zip(axes, METRICS):
        data = {m: {d: rows_for(m, d, mcol) for d in DOMAINS} for m in methods}
        grouped_bars(ax, data, methods, colors, mlabel, tops[mcol])
    axes[0].set_ylabel("Score", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods),
               frameon=False, bbox_to_anchor=(0.5, 0.99), fontsize=12)
    fig.suptitle(suptitle, y=1.06, fontsize=15)
    fig.text(0.5, 0.985, subtitle, ha="center", va="top", fontsize=10.5,
             fontweight="normal", color="#4D4D4D")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{outstem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outstem}.png / .pdf")


def main():
    ens = load(REV / "lc_dt10_ensemble_k7.csv")
    gl = load(REV / "lc_dt10_generic_llm.csv")

    # ---- MAIN: RF vs LLM-PP, Demographics block, N=898 ----
    make_panel(
        rows_for=lambda m, d, mc: ensemble_val(
            ens, "Demographics", "RF only" if m.startswith("Supervised") else "LLM only", d, mc),
        methods=["Supervised RF", "LLM-PP (Grok-4-Fast)"],
        colors={"Supervised RF": COLOR_RF, "LLM-PP (Grok-4-Fast)": COLOR_LLMPP},
        suptitle="Figure 2. Strict shared-row PME prediction (dt10 split)",
        subtitle="k_train = 7  |  feature block = Demographics  |  N = 898 held-out ratings "
                 "(same rows for both methods)  |  301 participants, within-participant",
        outstem="figure2_main_dt10_rf_vs_llmpp",
    )

    # ---- CONTEXT: zero/few-shot, separate coverage ----
    make_panel(
        rows_for=lambda m, d, mc: generic_mean(gl, m, d, mc),
        methods=["Zero-shot", "Few-shot"],
        colors={"Zero-shot": COLOR_ZS, "Few-shot": COLOR_FS},
        suptitle="Figure 2 (context). Generic LLM prompting on dt10",
        subtitle="k_train = 7  |  mean of 5 LLMs  |  N ≈ 87 per cell (different/smaller "
                 "coverage — NOT comparable to the 898-row panel above)",
        outstem="figure2_context_generic_llm",
    )

    # ---- source table for provenance ----
    with open(OUT / "figure2_dt10_source_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panel", "method", "domain", "metric", "value", "N", "split",
                    "feature_set", "source_file"])
        for d in DOMAINS:
            for mc, _ in METRICS:
                w.writerow(["main", "Supervised RF", d, mc,
                            f"{ensemble_val(ens,'Demographics','RF only',d,mc):.4f}", 898,
                            "dt10_k7", "Demographics", "lc_dt10_ensemble_k7.csv"])
                w.writerow(["main", "LLM-PP (Grok-4-Fast)", d, mc,
                            f"{ensemble_val(ens,'Demographics','LLM only',d,mc):.4f}", 898,
                            "dt10_k7", "Demographics", "lc_dt10_ensemble_k7.csv"])
            for grp in ("Zero-shot", "Few-shot"):
                for mc, _ in METRICS:
                    w.writerow(["context", grp, d, mc,
                                f"{generic_mean(gl,grp,d,mc):.4f}", 87, "dt10_k7",
                                "n/a", "lc_dt10_generic_llm.csv"])
    print("wrote figure2_dt10_source_table.csv")


if __name__ == "__main__":
    main()
