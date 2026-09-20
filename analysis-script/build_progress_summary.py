#!/usr/bin/env python3
"""Build a clean apples-to-apples progress summary for collaborator review.

IMPORTANT: A previous version of this script merged rows across files at
the same k_train, but those files use DIFFERENT test sets at the same
k_train (Generic LLM ≈ 300, LLM-PP ≈ 800, RF/Ensemble ≈ 2700 at k=1).
So cross-file comparisons at the same k_train are NOT apples-to-apples.

This version sources every cross-method comparison from the ensemble files
(`lc_dt10_ensemble_k{1,3,7}.csv`), where RF, LLM-PP, and hybrid rows are
evaluated on the SAME test set within each (k_train, feature_set).

Outputs:
    revision/figures/progress_summary/table_accuracy.{md,csv}
    revision/figures/progress_summary/table_macro_f1.{md,csv}
    revision/figures/progress_summary/table_qwk.{md,csv}
    revision/figures/progress_summary/table_spearman_rho.{md,csv}
    revision/figures/progress_summary/sweep_by_k_train.md
    revision/figures/progress_summary/ai_vs_ml_strict_fixed_feature.csv
    revision/figures/progress_summary/message_selection_best_by_k.csv
    revision/figures/progress_summary/message_selection_methods_k7.csv
    revision/figures/progress_summary/confusion_predictions_k7.csv
    revision/figures/progress_summary/confusion_matrices_k7.csv
    revision/figures/progress_summary/confusion_metrics_k7.csv
    revision/figures/progress_summary/clean_results.md
    revision/figures/progress_summary/ai_vs_ml_strict_snapshot.{png,pdf}
    revision/figures/progress_summary/ai_vs_ml_strict_scaling.{png,pdf}
    revision/figures/progress_summary/ordinal_qwk_spearman_snapshot.{png,pdf}
    revision/figures/progress_summary/confusion_matrices_k7.{png,pdf}
    revision/figures/progress_summary/message_selection_gain.{png,pdf}
    revision/figures/progress_summary/learning_curves.{png,pdf}
    revision/figures/progress_summary/progress_summary_all.md
    revision/figures/progress_summary/test_set_audit.md

The Generic LLM (zero/few-shot) is reported separately with an explicit
test-set caveat, since it was evaluated on a different (smaller) subset.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from revision_utils import compute_all_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REV_DIR = PROJECT_ROOT / "revision" / "figures"
OUT_DIR = REV_DIR / "progress_summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS = ["content", "coping", "quitting"]
DOMAIN_LABEL = {"content": "Content", "coping": "Coping", "quitting": "Quitting"}
METRICS = [
    ("Accuracy", "Accuracy"),
    ("F1", "Macro-F1"),
    ("QWK", "QWK"),
    ("Spearman_Rho", "Within-participant Spearman rho"),
]
METRIC_SLUG = {
    "Accuracy": "accuracy",
    "F1": "macro_f1",
    "QWK": "qwk",
    "Spearman_Rho": "spearman_rho",
}

FONT_CHAIN = ["Arial", "Helvetica", "Helvetica Neue", "Avenir",
              "Avenir Next", "DejaVu Sans"]
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": FONT_CHAIN,
    "font.size": 13,
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
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 150,
    "savefig.dpi": 400,
})

COLOR_RF = "#7F7F7F"
COLOR_GROK = "#CC78BC"
COLOR_HYBRID = "#029E73"
COLOR_REFERENCE = "#4D4D4D"
MODEL_COLORS = {
    "GPT-4o-mini": "#0173B2",
    "GPT-5": "#DE8F05",
    "DeepSeek-R1": "#029E73",
    "Grok-4-Fast": "#CC78BC",
    "Gemini-2.5-Pro": "#CA9161",
}
PLOT_FAMILIES = [
    "Supervised RF",
    "LLM-PP (Grok-4-Fast)",
]
MESSAGE_SELECTION_FEATURE_SET = "Demographics + History + Message Embedding"
METHOD_DISPLAY = {
    "Supervised RF": "Supervised RF",
    "LLM-PP (Grok-4-Fast)": "LLM-PP",
    "Hybrid — RF-anchor (70/30)": "RF-anchor",
    "Hybrid — LLM-anchor (30/70)": "LLM-anchor",
}
METHOD_COLORS = {
    "Supervised RF": "#7F7F7F",
    "LLM-PP (Grok-4-Fast)": "#CC78BC",
    "Hybrid — RF-anchor (70/30)": "#029E73",
    "Hybrid — LLM-anchor (30/70)": "#DE8F05",
}
METHOD_MARKERS = {
    "Supervised RF": "s",
    "LLM-PP (Grok-4-Fast)": "o",
    "Hybrid — RF-anchor (70/30)": "D",
    "Hybrid — LLM-anchor (30/70)": "v",
}
NORMAL_CI_Z = 1.96
RATING_MAPS_LOCAL = {
    "content": {"Very poor": 1, "Poor": 2, "Acceptable": 3, "Good": 4, "Very good": 5},
    "coping": {
        "Not at all helpful": 1, "Somewhat helpful": 2,
        "Moderately helpful": 3, "Very helpful": 4,
        "Extremely helpful": 5, "Not Helpful": 1,
    },
    "quitting": {
        "Not at all helpful": 1, "Somewhat helpful": 2,
        "Moderately helpful": 3, "Very helpful": 4,
        "Extremely helpful": 5, "Not Helpful": 1,
    },
}
ORDINAL_LABELS = [1, 2, 3, 4, 5]


# ----------------------- audit: how big is each test set? --------------

def audit_test_sets() -> pd.DataFrame:
    rows = []
    files = {
        "Generic LLM (lc_dt10_generic_llm.csv)": ("lc_dt10_generic_llm.csv", "covered_n"),
        "LLM-PP     (lc_dt10_llmdt.csv)":         ("lc_dt10_llmdt.csv",       "covered_n"),
        "RF         (lc_dt10_rf.csv)":            ("lc_dt10_rf.csv",          "test_n"),
        "Ensemble k=1 (lc_dt10_ensemble_k1.csv)": ("lc_dt10_ensemble_k1.csv", "N"),
        "Ensemble k=3 (lc_dt10_ensemble_k3.csv)": ("lc_dt10_ensemble_k3.csv", "N"),
        "Ensemble k=7 (lc_dt10_ensemble_k7.csv)": ("lc_dt10_ensemble_k7.csv", "N"),
    }
    for label, (fname, n_col) in files.items():
        df = pd.read_csv(REV_DIR / fname)
        if "k_train" in df.columns:
            grp = df.groupby("k_train")[n_col].agg(["min", "max", "mean"]).round(0).astype(int)
        else:
            grp = pd.DataFrame({"min":[df[n_col].min()],
                                 "max":[df[n_col].max()],
                                 "mean":[int(df[n_col].mean())]},
                                index=pd.Index([3], name="k_train"))
        for kt, row in grp.iterrows():
            rows.append({"source": label, "k_train": int(kt),
                          "min_N": int(row["min"]), "max_N": int(row["max"])})
    out = pd.DataFrame(rows).pivot_table(
        index="source", columns="k_train",
        values="max_N", aggfunc="first")
    out.columns = [f"k={int(c)}" for c in out.columns]
    return out


def write_audit():
    audit = audit_test_sets()
    md = ["# Test-set size audit (max N per source × k_train)\n",
          "These columns show the test-set size each method was evaluated on. ",
          "When the numbers in a row of the comparison tables differ across ",
          "*sources*, they are NOT on the same test set and should not be ",
          "directly compared.\n"]
    md.append("| source | " + " | ".join(audit.columns) + " |")
    md.append("|" + "|".join(["---"] * (1 + len(audit.columns))) + "|")
    for src in audit.index:
        vals = ["—" if pd.isna(audit.loc[src, c]) else str(int(audit.loc[src, c]))
                for c in audit.columns]
        md.append(f"| {src} | " + " | ".join(vals) + " |")
    md.append("")
    md.append("**Implication.** The cross-method comparison tables in this folder ")
    md.append("are sourced ONLY from the ensemble files, where all method rows ")
    md.append("share a single test set within a fixed `(k_train, feature_set)` ")
    md.append("block. Generic LLM rows live in their own table because they are ")
    md.append("evaluated on a different (smaller) population.")
    (OUT_DIR / "test_set_audit.md").write_text("\n".join(md))


# ----------------------- apples-to-apples (ensemble files) -------------

ENS_FAMILY_FROM_ROW = {
    "RF only":              "Supervised RF",
    "LLM only":             "LLM-PP (Grok-4-Fast)",
}

FAMILY_ORDER_APPLES = [
    "Supervised RF",
    "LLM-PP (Grok-4-Fast)",
]


def load_ensemble_long() -> pd.DataFrame:
    parts = []
    for k, path in [(1, REV_DIR / "lc_dt10_ensemble_k1.csv"),
                     (3, REV_DIR / "lc_dt10_ensemble_k3.csv"),
                     (7, REV_DIR / "lc_dt10_ensemble_k7.csv")]:
        if path.exists():
            df = pd.read_csv(path)
            df["k_train"] = k
            parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df["family"] = df["ensemble"].map(ENS_FAMILY_FROM_ROW)
    df = df.dropna(subset=["family"])
    df["domain"] = df["domain"].astype(str).str.lower()
    return df


def best_within_family_apples(df: pd.DataFrame, k_train: int, metric: str) -> pd.DataFrame:
    """For each (family, domain), pick the feature_set that maximizes the metric."""
    sub = df[df["k_train"] == k_train].dropna(subset=[metric]).copy()
    idx = sub.groupby(["family", "domain"])[metric].idxmax()
    return sub.loc[idx].reset_index(drop=True)


def build_apples_table(df: pd.DataFrame, metric_col: str, k_train: int) -> pd.DataFrame:
    best = best_within_family_apples(df, k_train, metric_col)
    pivot = (best.pivot_table(index="family", columns="domain",
                               values=metric_col, aggfunc="first")
                  .reindex(index=FAMILY_ORDER_APPLES, columns=DOMAINS))
    pivot["Mean"] = pivot.mean(axis=1)
    pivot.columns = [DOMAIN_LABEL.get(c, c) for c in pivot.columns]
    pivot = pivot.round(3)

    cfg = (best.set_index(["family", "domain"])["feature_set"]
                .unstack("domain")
                .reindex(index=FAMILY_ORDER_APPLES, columns=DOMAINS))
    cfg.columns = [f"best feature set ({DOMAIN_LABEL[c]})" for c in cfg.columns]

    n_per_dom = (best.set_index(["family", "domain"])["N"]
                      .unstack("domain")
                      .reindex(index=FAMILY_ORDER_APPLES, columns=DOMAINS))
    n_per_dom.columns = [f"N ({DOMAIN_LABEL[c]})" for c in n_per_dom.columns]

    out = pd.concat([pivot, cfg, n_per_dom], axis=1)
    out.index.name = "Method"
    return out


# ----------------------- generic-LLM context table ---------------------

def build_generic_llm_table(metric_col: str) -> pd.DataFrame:
    """Generic-LLM zero/few-shot best-per-domain at each k_train, on its own
    (smaller) test set. Reported separately because the test set differs from
    the ensemble files."""
    df = pd.read_csv(REV_DIR / "lc_dt10_generic_llm.csv")
    df["domain"] = df["domain"].astype(str).str.lower()
    rows = []
    for kt in sorted(df["k_train"].unique()):
        sub = df[df["k_train"] == kt].dropna(subset=[metric_col])
        if sub.empty:
            continue
        for variant in ["Zero-shot", "Few-shot"]:
            ssub = sub[sub["feature_set"] == variant]
            if ssub.empty:
                continue
            row = {"k_train": int(kt), "variant": variant,
                    "N (any domain)": int(ssub["covered_n"].iloc[0])}
            for dom in DOMAINS:
                ddom = ssub[ssub["domain"] == dom]
                if ddom.empty:
                    row[DOMAIN_LABEL[dom]] = np.nan
                else:
                    row[DOMAIN_LABEL[dom]] = round(ddom[metric_col].max(), 3)
            row["Mean"] = round(np.mean([row[DOMAIN_LABEL[d]] for d in DOMAINS]), 3)
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------- markdown writers ------------------------------

def write_apples_table(per_k_tables: dict[int, pd.DataFrame], metric_label: str,
                        slug: str, generic_llm: pd.DataFrame):
    """Write one .md / .csv pair per metric, with one panel per k_train."""
    csv_path = OUT_DIR / f"table_{slug}.csv"
    md_path = OUT_DIR / f"table_{slug}.md"

    # combined long-form CSV
    parts = []
    for kt, t in per_k_tables.items():
        tt = t.copy()
        tt.insert(0, "k_train", kt)
        parts.append(tt.reset_index())
    pd.concat(parts, ignore_index=True).to_csv(csv_path, index=False)

    metric_cols = [DOMAIN_LABEL[d] for d in DOMAINS] + ["Mean"]

    lines = [
        f"# {metric_label} — apples-to-apples by k_train",
        "",
        "All values are sourced from the ensemble files. Within a fixed ",
        "`(k_train, feature_set)` block, all method rows use the same test ",
        "rows. The table below picks the best feature set separately for ",
        "each method × domain × metric, so the exact feature set and N are ",
        "reported in the details block. Use `clean_results.md` for the ",
        "strict fixed-feature apples-to-apples read.",
        "",
    ]
    for kt in sorted(per_k_tables.keys()):
        out = per_k_tables[kt]
        n_cols = [c for c in out.columns if c.startswith("N (")]
        cfg_cols = [c for c in out.columns if c.startswith("best ")]
        n_vals = pd.to_numeric(out[n_cols].stack(), errors="coerce").dropna()
        if n_vals.empty:
            n_label = "N unavailable"
        else:
            n_min, n_max = int(n_vals.min()), int(n_vals.max())
            n_label = f"test N = {n_min}" if n_min == n_max else f"test N range = {n_min}-{n_max}"
        lines.append(f"## k_train = {kt}  ({n_label})")
        lines.append("")
        lines.append("| Method | " + " | ".join(metric_cols) + " |")
        lines.append("|" + "|".join(["---"] * (1 + len(metric_cols))) + "|")
        for fam in out.index:
            vals = [f"{out.loc[fam, c]:.3f}" if pd.notna(out.loc[fam, c]) else "—"
                    for c in metric_cols]
            lines.append(f"| {fam} | " + " | ".join(vals) + " |")
        lines.append("")
        lines.append("<details><summary>Best feature set used per cell</summary>")
        lines.append("")
        lines.append("| Method | " + " | ".join(cfg_cols) + " | " + " | ".join(n_cols) + " |")
        lines.append("|" + "|".join(["---"] * (1 + len(cfg_cols) + len(n_cols))) + "|")
        for fam in out.index:
            cfg_vals = [str(out.loc[fam, c]) if pd.notna(out.loc[fam, c]) else "—"
                        for c in cfg_cols]
            n_vals = [str(int(out.loc[fam, c])) if pd.notna(out.loc[fam, c]) else "—"
                      for c in n_cols]
            lines.append(f"| {fam} | " + " | ".join(cfg_vals + n_vals) + " |")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.append("---")
    lines.append("## Generic LLM (separate test set — DO NOT directly compare)")
    lines.append("")
    lines.append("Generic LLM zero-/few-shot was evaluated on the dt10 message "
                 "subset (much smaller test set than the ensemble grid above). ")
    lines.append("Reported here for context only.")
    lines.append("")
    if slug == "spearman_rho":
        lines.append("Generic LLM Spearman rows are omitted here because the aggregate "
                     "dt10 CSV does not carry the valid-participant counts needed "
                     "to interpret tied/degenerate rank correlations. Use the "
                     "dedicated `spearman_rank_summary.csv` artifact for that "
                     "separate analysis.")
        lines.append("")
    elif not generic_llm.empty:
        cols = ["k_train", "variant", "Content", "Coping", "Quitting", "Mean", "N (any domain)"]
        cols = [c for c in cols if c in generic_llm.columns]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, row in generic_llm.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if pd.isna(v):
                    vals.append("—")
                elif isinstance(v, float):
                    vals.append(f"{v:.3f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")
    md_path.write_text("\n".join(lines))
    return csv_path, md_path


# ----------------------- k_train sweep (apples-to-apples) --------------

def build_kt_sweep(df_apples: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    rows = []
    for fam in FAMILY_ORDER_APPLES:
        for kt in sorted(df_apples["k_train"].unique()):
            sub = df_apples[(df_apples["family"] == fam)
                            & (df_apples["k_train"] == kt)].dropna(subset=[metric_col])
            if sub.empty:
                rows.append({"family": fam, "k_train": int(kt), metric_col: np.nan,
                             "N": np.nan})
                continue
            per_dom = sub.groupby("domain")[metric_col].max()
            n_typical = int(sub["N"].iloc[0])
            rows.append({"family": fam, "k_train": int(kt),
                          metric_col: float(per_dom.mean()),
                          "N": n_typical})
    out = (pd.DataFrame(rows)
              .pivot(index="family", columns="k_train", values=metric_col)
              .reindex(FAMILY_ORDER_APPLES)
              .round(3))
    return out


def write_sweep(df_apples: pd.DataFrame):
    md = ["# Mean-across-domains by k_train (apples-to-apples)\n",
          "Source: ensemble files. Each cell = mean over {Content, Coping, ",
          "Quitting} of the per-domain best feature set, with N tracked in ",
          "the metric-specific tables. Each k_train column corresponds to a ",
          "different test set (different intersection of users with both RF ",
          "and LLM-PP predictions at that history depth), so absolute values ",
          "across columns are not strictly comparable — only the ranking of ",
          "rows within a column is. For the strict feature-matched read, see ",
          "`clean_results.md`.\n"]
    for metric_col, metric_label in METRICS:
        sweep = build_kt_sweep(df_apples, metric_col)
        sweep.to_csv(OUT_DIR / f"sweep_{METRIC_SLUG[metric_col]}.csv")
        md.append(f"## {metric_label}\n")
        md.append("| Method | " + " | ".join(f"k={c}" for c in sweep.columns) + " |")
        md.append("|" + "|".join(["---"] * (1 + len(sweep.columns))) + "|")
        for fam in sweep.index:
            vals = [f"{sweep.loc[fam, c]:.3f}" if pd.notna(sweep.loc[fam, c]) else "—"
                    for c in sweep.columns]
            md.append(f"| {fam} | " + " | ".join(vals) + " |")
        md.append("")
    (OUT_DIR / "sweep_by_k_train.md").write_text("\n".join(md))


# ----------------------- learning-curve plot ---------------------------

# Aesthetic: keep the lc_rf_vs_llm_plot.py palette family.
#  - Supervised RF: dark gray, dashed (the "baseline" feel from that figure).
#  - LLM-PP (Grok-4-Fast): pink/magenta, solid (matches COLORS['Grok-4-Fast']).
#  - Hybrid blends: greens of varying weight to read as a family.
FAMILY_STYLE_APPLES = {
    "Supervised RF":                  dict(color=METHOD_COLORS["Supervised RF"], marker="s", ls="--",
                                            lw=2.0, ms=9, alpha=0.95, zorder=3),
    "LLM-PP (Grok-4-Fast)":           dict(color=METHOD_COLORS["LLM-PP (Grok-4-Fast)"], marker="o", ls="-",
                                            lw=2.7, ms=10, alpha=0.95, zorder=4),
    "Hybrid — Mean (50/50)":          dict(color="#7DA87B", marker="D", ls="-",
                                            lw=2.0, ms=8, alpha=0.85, zorder=2),
    "Hybrid — RF-anchor (70/30)":     dict(color=METHOD_COLORS["Hybrid — RF-anchor (70/30)"], marker="D", ls="-",
                                            lw=2.4, ms=9, alpha=0.95, zorder=5),
    "Hybrid — LLM-anchor (30/70)":    dict(color=METHOD_COLORS["Hybrid — LLM-anchor (30/70)"], marker="v", ls="-",
                                            lw=2.2, ms=9, alpha=0.95, zorder=4),
    # The new soft stacked LR — give it the strongest visual weight so the
    # collaborator's eye lands here.
    "Hybrid — Stacked LR (soft)":     dict(color="#E02020", marker="P", ls="-",
                                            lw=3.0, ms=12, alpha=0.95, zorder=6),
    "Hybrid — Conf-Gated (entropy)":  dict(color="#DE8F05", marker="X", ls="-",
                                            lw=2.6, ms=11, alpha=0.95, zorder=7),
    "Hybrid — Conf-Gated (max-p)":    dict(color="#FFC658", marker="x", ls="--",
                                            lw=2.0, ms=10, alpha=0.85, zorder=2),
}

# Approximate train-message totals (300 users × k_train messages each, less the
# small dropout from missing demographics/embeddings).  Used for the secondary
# (top) x-axis on the first row.
TRAIN_N_BY_K = {1: 301, 3: 903, 7: 2107}


def best_curve_apples(df: pd.DataFrame, family: str, domain: str, metric: str) -> pd.DataFrame:
    sub = df[(df["family"] == family) & (df["domain"] == domain)].dropna(subset=[metric])
    if sub.empty:
        return sub
    return (sub.groupby("k_train")[metric].max()
                .reset_index().sort_values("k_train"))


def plot_learning_curves(df_apples: pd.DataFrame, out_stem: Path):
    ks = sorted(df_apples["k_train"].dropna().unique().astype(int).tolist())
    fig, axes = plt.subplots(
        len(METRICS), len(DOMAINS),
        figsize=(15.5, 3.25 * len(METRICS) + 1.2),
        sharex=True, sharey="row",
    )
    fig.patch.set_facecolor("white")

    # Pre-compute per-row y-limits with a small margin.
    row_ylims = {}
    for ri, (metric_col, _) in enumerate(METRICS):
        all_vals = df_apples[metric_col].dropna().values
        if len(all_vals) == 0:
            row_ylims[ri] = (0, 1)
            continue
        lo = float(np.nanmin(all_vals))
        hi = float(np.nanmax(all_vals))
        pad = max(0.01, 0.08 * (hi - lo))
        row_ylims[ri] = (lo - pad, hi + pad)

    for ri, (metric_col, metric_label) in enumerate(METRICS):
        for ci, dom in enumerate(DOMAINS):
            ax = axes[ri, ci]
            ax.set_facecolor("white")

            for fam in PLOT_FAMILIES:
                cur = best_curve_apples(df_apples, fam, dom, metric_col)
                if cur.empty:
                    continue
                ax.plot(cur["k_train"], cur[metric_col],
                        label=METHOD_DISPLAY.get(fam, fam),
                        **FAMILY_STYLE_APPLES.get(fam, {}))

            ax.set_xticks(ks)
            ax.set_xlim(min(ks) - 0.4, max(ks) + 0.4)
            ax.set_ylim(*row_ylims[ri])
            ax.grid(axis="y", alpha=0.18, linestyle="--", color="#4D4D4D")

            if ri == 0:
                ax.set_title(DOMAIN_LABEL[dom],
                             fontsize=14, fontweight="bold", pad=8)
            if ci == 0:
                ax.set_ylabel(metric_label, fontsize=14, fontweight="bold")
            if ri == len(METRICS) - 1:
                ax.set_xlabel("k_train  (prior messages per user)",
                              fontsize=13, fontweight="bold")

            # Twin x-axis (top) on first row only — show train-set message count.
            if ri == 0:
                ax2 = ax.twiny()
                ax2.set_xlim(ax.get_xlim())
                ax2.set_xticks(ks)
                ax2.set_xticklabels([str(TRAIN_N_BY_K.get(k, "?")) for k in ks],
                                     fontsize=10)
                ax2.set_xlabel("Train n  (messages, ≈ 301 users × k)",
                                fontsize=11, fontweight="bold", labelpad=6)

    handles = [
        Line2D([0], [0],
                color=FAMILY_STYLE_APPLES[f]["color"],
                marker=FAMILY_STYLE_APPLES[f]["marker"],
                markersize=FAMILY_STYLE_APPLES[f].get("ms", 9),
                linewidth=FAMILY_STYLE_APPLES[f]["lw"],
                linestyle=FAMILY_STYLE_APPLES[f]["ls"],
                label=METHOD_DISPLAY.get(f, f))
        for f in PLOT_FAMILIES
    ]
    legend = fig.legend(handles, [h.get_label() for h in handles],
                         loc="lower center", ncol=2, fontsize=11,
                         bbox_to_anchor=(0.5, -0.04),
                         frameon=True, title="Method",
                         title_fontsize=11)
    legend.get_title().set_fontweight("bold")

    fig.suptitle(
        f"Apples-to-apples scaling: supervised RF versus LLM-PP "
        f"(k_train in {{{', '.join(str(k) for k in ks)}}})",
        fontsize=15, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(f"{out_stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------------- clean results ---------------------------------

def fixed_feature_means(df_apples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric_col, metric_label in METRICS:
        for (kt, feature_set, family), sub in df_apples.groupby(
            ["k_train", "feature_set", "family"]
        ):
            vals = sub.dropna(subset=[metric_col])
            if vals.empty:
                value = np.nan
                n_min = np.nan
                n_max = np.nan
            else:
                value = float(vals.groupby("domain")[metric_col].max().mean())
                n_min = int(vals["N"].min())
                n_max = int(vals["N"].max())
            rows.append({
                "k_train": int(kt),
                "feature_set": feature_set,
                "method": family,
                "metric": metric_label,
                "metric_col": metric_col,
                "mean_across_domains": value,
                "N_min": n_min,
                "N_max": n_max,
            })
    out = pd.DataFrame(rows)
    out["method"] = pd.Categorical(out["method"], FAMILY_ORDER_APPLES, ordered=True)
    return out.sort_values(["metric_col", "k_train", "feature_set", "method"])


def strict_feature_summary(fixed: pd.DataFrame) -> pd.DataFrame:
    """Pick the feature-set block with the best observed method for each
    metric × k_train, then report supervised RF and LLM-PP values from that
    same feature-set block.
    """
    rows = []
    for metric_col, metric_label in METRICS:
        sub_metric = fixed[fixed["metric_col"] == metric_col].dropna(
            subset=["mean_across_domains"]
        )
        for kt, sub_k in sub_metric.groupby("k_train"):
            if sub_k.empty:
                continue
            block_scores = sub_k.groupby("feature_set")["mean_across_domains"].max()
            feature_set = str(block_scores.idxmax())
            block = sub_k[sub_k["feature_set"] == feature_set]

            def val(method: str) -> float:
                s = block[block["method"] == method]["mean_across_domains"]
                return float(s.iloc[0]) if len(s) else np.nan

            best_overall = block.loc[block["mean_across_domains"].idxmax()]
            n_min = int(block["N_min"].dropna().min()) if block["N_min"].notna().any() else np.nan
            n_max = int(block["N_max"].dropna().max()) if block["N_max"].notna().any() else np.nan
            rows.append({
                "k_train": int(kt),
                "metric": metric_label,
                "feature_set": feature_set,
                "N_min": n_min,
                "N_max": n_max,
                "best_overall_method": str(best_overall["method"]),
                "best_overall": float(best_overall["mean_across_domains"]),
                "supervised_rf": val("Supervised RF"),
                "llm_pp_grok": val("LLM-PP (Grok-4-Fast)"),
            })
    return pd.DataFrame(rows)


def message_selection_best() -> pd.DataFrame:
    path = REV_DIR / "llm_selection_quality.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    idx = df.groupby(["Domain", "K"])["LLM Selection (Human Rating)"].idxmax()
    out = df.loc[idx, [
        "Domain",
        "K",
        "Model",
        "LLM Selection (Human Rating)",
        "LLM SE",
        "LLM CI Lower",
        "LLM CI Upper",
        "Random Mean",
        "Random SE",
        "Random CI Lower",
        "Random CI Upper",
        "Human Oracle (Human Rating)",
        "Human Oracle SE",
        "Human Oracle CI Lower",
        "Human Oracle CI Upper",
        "Bootstrap_N",
        "N Messages",
    ]].copy()
    out["LLM_minus_Random"] = out["LLM Selection (Human Rating)"] - out["Random Mean"]
    out["Oracle_minus_LLM"] = (
        out["Human Oracle (Human Rating)"] - out["LLM Selection (Human Rating)"]
    )
    return out.sort_values(["Domain", "K"]).reset_index(drop=True)


def _grok_dt10_path(k_train: int) -> Path:
    return PROJECT_ROOT / "results_manuscript_x-ai_grok-4-fast" / f"digital_twin_dt10_k{k_train}.json"


def _ensemble_prediction(rf_pred: int, llm_pred: int, method: str) -> int:
    if method == "Supervised RF":
        return int(rf_pred)
    if method == "LLM-PP (Grok-4-Fast)":
        return int(llm_pred)
    if method == "Hybrid — RF-anchor (70/30)":
        return int(max(1, min(5, round(0.7 * rf_pred + 0.3 * llm_pred))))
    if method == "Hybrid — LLM-anchor (30/70)":
        return int(max(1, min(5, round(0.3 * rf_pred + 0.7 * llm_pred))))
    raise ValueError(method)


def aligned_rf_llm_predictions(k_train: int = 7) -> pd.DataFrame:
    """Row-level RF + Grok LLM predictions on the same held-out examples."""
    rf_path = REV_DIR / "lc_dt10_rf_predictions.csv"
    llm_path = _grok_dt10_path(k_train)
    if not rf_path.exists() or not llm_path.exists():
        return pd.DataFrame()

    rf = pd.read_csv(rf_path)
    rf = rf[rf["k_train"] == k_train].copy()
    if rf.empty:
        return pd.DataFrame()
    rf = rf.rename(columns={"predicted_num": "rf_pred"})

    with open(llm_path) as f:
        data = json.load(f)
    llm_rows = []
    for rec in data.values():
        if not isinstance(rec, dict):
            continue
        rid = rec.get("response_id")
        msg = rec.get("input_message")
        if rid is None or msg is None:
            continue
        for dom in DOMAINS:
            gt = RATING_MAPS_LOCAL[dom].get(rec.get(f"ground_truth_{dom}"))
            pred = RATING_MAPS_LOCAL[dom].get(rec.get(f"predicted_{dom}"))
            if gt is None or pred is None:
                continue
            llm_rows.append({
                "response_id": rid,
                "input_message": msg,
                "domain": dom,
                "llm_gt": int(gt),
                "llm_pred": int(pred),
            })
    llm = pd.DataFrame(llm_rows)
    if llm.empty:
        return pd.DataFrame()

    merged = rf.merge(llm, on=["response_id", "input_message", "domain"],
                      how="inner")
    if merged.empty:
        return pd.DataFrame()

    merged["ground_truth_num"] = merged["ground_truth_num"].astype(int)
    merged = merged[merged["ground_truth_num"] == merged["llm_gt"]].copy()
    merged["rf_pred"] = merged["rf_pred"].astype(int)
    merged["llm_pred"] = merged["llm_pred"].astype(int)
    return merged


def method_prediction_rows(k_train: int = 7, feature_set: str | None = None) -> pd.DataFrame:
    """Long row-level predictions for supervised RF and LLM-PP."""
    base = aligned_rf_llm_predictions(k_train)
    if base.empty:
        return pd.DataFrame()
    if feature_set is not None:
        base = base[base["feature_set"] == feature_set].copy()
    if base.empty:
        return pd.DataFrame()

    rows = []
    for method in PLOT_FAMILIES:
        out = base[[
            "k_train", "feature_set", "domain", "response_id",
            "input_message", "ground_truth_num", "rf_pred", "llm_pred",
        ]].copy()
        out["method"] = method
        out["method_display"] = METHOD_DISPLAY[method]
        out["predicted_num"] = [
            _ensemble_prediction(rf_pred, llm_pred, method)
            for rf_pred, llm_pred in zip(out["rf_pred"], out["llm_pred"])
        ]
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def _mean_se(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _normal_ci(mean: float, se: float) -> tuple[float, float]:
    return float(mean - NORMAL_CI_Z * se), float(mean + NORMAL_CI_Z * se)


def _message_selection_se_intervals(
    gt_vals: np.ndarray,
    pred_vals: np.ndarray,
    k_values: list[int],
) -> dict[int, dict[str, float]]:
    """Normal-approximation 95% CIs from message-level SE = SD / sqrt(n)."""
    gt_vals = np.asarray(gt_vals, dtype=float)
    pred_vals = np.asarray(pred_vals, dtype=float)
    n_messages = len(gt_vals)
    valid_k = [k for k in k_values if k <= n_messages]
    random_mean = float(np.mean(gt_vals))
    random_se = _mean_se(gt_vals)
    random_ci_lower, random_ci_upper = _normal_ci(random_mean, random_se)
    pred_order = np.argsort(-pred_vals, kind="mergesort")
    oracle_order = np.argsort(-gt_vals, kind="mergesort")

    stats = {}
    for k in valid_k:
        selected_vals = gt_vals[pred_order[:k]]
        oracle_vals = gt_vals[oracle_order[:k]]
        selected = float(np.mean(selected_vals))
        oracle = float(np.mean(oracle_vals))
        selected_se = _mean_se(selected_vals)
        oracle_se = _mean_se(oracle_vals)
        gain = selected - random_mean
        oracle_gain = oracle - random_mean
        gain_se = float(np.sqrt(selected_se ** 2 + random_se ** 2))
        oracle_gain_se = float(np.sqrt(oracle_se ** 2 + random_se ** 2))
        selected_ci_lower, selected_ci_upper = _normal_ci(selected, selected_se)
        oracle_ci_lower, oracle_ci_upper = _normal_ci(oracle, oracle_se)
        gain_ci_lower, gain_ci_upper = _normal_ci(gain, gain_se)
        oracle_gain_ci_lower, oracle_gain_ci_upper = _normal_ci(
            oracle_gain, oracle_gain_se
        )
        stats[k] = {
            "selected_human_rating_se": selected_se,
            "selected_human_rating_ci_lower": selected_ci_lower,
            "selected_human_rating_ci_upper": selected_ci_upper,
            "random_mean_se": random_se,
            "random_mean_ci_lower": random_ci_lower,
            "random_mean_ci_upper": random_ci_upper,
            "gain_over_random_se": gain_se,
            "gain_over_random_ci_lower": gain_ci_lower,
            "gain_over_random_ci_upper": gain_ci_upper,
            "human_oracle_rating_se": oracle_se,
            "human_oracle_rating_ci_lower": oracle_ci_lower,
            "human_oracle_rating_ci_upper": oracle_ci_upper,
            "oracle_gain_over_random_se": oracle_gain_se,
            "oracle_gain_over_random_ci_lower": oracle_gain_ci_lower,
            "oracle_gain_over_random_ci_upper": oracle_gain_ci_upper,
            "ci_method": "normal_approx_message_se",
            "ci_z": NORMAL_CI_Z,
        }
    return stats


def message_selection_methods(k_train: int = 7) -> pd.DataFrame:
    """Method-level message-selection curves on the same k_train ensemble rows.

    Use a single fixed RF feature block across domains, then reuse it for
    LLM-PP. This keeps every method and domain on the same held-out rows and
    avoids domain-wise feature cherry-picking.
    """
    rf_path = REV_DIR / "lc_dt10_rf_predictions.csv"
    if not rf_path.exists():
        return pd.DataFrame()

    merged = aligned_rf_llm_predictions(k_train)
    if merged.empty:
        return pd.DataFrame()

    k_values = [5, 10, 15, 20, 25]
    all_rows = []
    for feature_set, fs_df in merged.groupby("feature_set"):
        for dom, dom_df in fs_df.groupby("domain"):
            for method in PLOT_FAMILIES:
                pred = [
                    _ensemble_prediction(rf_pred, llm_pred, method)
                    for rf_pred, llm_pred in zip(dom_df["rf_pred"], dom_df["llm_pred"])
                ]
                msg_df = (
                    dom_df.assign(selection_pred=pred)
                          .groupby("input_message", as_index=False)
                          .agg(gt=("ground_truth_num", "mean"),
                               pred=("selection_pred", "mean"))
                )
                random_mean = float(msg_df["gt"].mean())
                n_messages = int(len(msg_df))
                gt_vals = msg_df["gt"].to_numpy(dtype=float)
                pred_vals = msg_df["pred"].to_numpy(dtype=float)
                ci_stats = _message_selection_se_intervals(gt_vals, pred_vals, k_values)
                for k in k_values:
                    if k > n_messages:
                        continue
                    selected = float(
                        msg_df.sort_values("pred", ascending=False,
                                           kind="mergesort")
                              .head(k)["gt"].mean()
                    )
                    oracle = float(
                        msg_df.sort_values("gt", ascending=False,
                                           kind="mergesort")
                              .head(k)["gt"].mean()
                    )
                    row = {
                        "k_train": k_train,
                        "feature_set": feature_set,
                        "domain": DOMAIN_LABEL[dom],
                        "method": method,
                        "method_display": METHOD_DISPLAY[method],
                        "K": k,
                        "selected_human_rating": selected,
                        "random_mean": random_mean,
                        "human_oracle_rating": oracle,
                        "gain_over_random": selected - random_mean,
                        "oracle_gain_over_random": oracle - random_mean,
                        "n_messages": n_messages,
                    }
                    row.update(ci_stats[k])
                    all_rows.append(row)
    all_df = pd.DataFrame(all_rows)
    if all_df.empty:
        return all_df

    expected_domains = set(DOMAIN_LABEL.values())
    preferred = MESSAGE_SELECTION_FEATURE_SET
    preferred_df = all_df[all_df["feature_set"] == preferred].copy()
    if set(preferred_df["domain"].unique()) == expected_domains:
        filtered = preferred_df
        rule = f"fixed feature set across all domains: {preferred}"
    else:
        candidates: list[tuple[str, float]] = []
        for feature_set, fs_df in all_df.groupby("feature_set"):
            if set(fs_df["domain"].unique()) != expected_domains:
                continue
            rf_df = fs_df[fs_df["method"] == "Supervised RF"]
            candidates.append((str(feature_set), float(rf_df["gain_over_random"].mean())))
        if not candidates:
            return pd.DataFrame()
        fallback = sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]
        filtered = all_df[all_df["feature_set"] == fallback].copy()
        rule = (
            f"fixed feature set requested: {preferred}; unavailable for all "
            f"domains, used {fallback}"
        )
    filtered["feature_selection_rule"] = rule
    return filtered.sort_values(["domain", "method", "K"]).reset_index(drop=True)


def _fmt(x: float) -> str:
    return "—" if pd.isna(x) else f"{float(x):.3f}"


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                vals.append(_fmt(val))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def _style_axis(ax):
    ax.set_facecolor("white")
    ax.grid(axis="y", alpha=0.12, linestyle="--",
            color=COLOR_REFERENCE, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", width=1.5, length=5, direction="out")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")


def _save_plot(fig, out_stem: Path):
    fig.savefig(f"{out_stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def prediction_metric_summary(pred_rows: pd.DataFrame) -> pd.DataFrame:
    if pred_rows.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["k_train", "feature_set", "domain", "method", "method_display"]
    for keys, sub in pred_rows.groupby(group_cols):
        kt, feature_set, domain, method, display = keys
        metrics = compute_all_metrics(
            sub["ground_truth_num"].to_numpy(dtype=int),
            sub["predicted_num"].to_numpy(dtype=int),
            sub["response_id"].to_numpy(),
        )
        rows.append({
            "k_train": int(kt),
            "feature_set": feature_set,
            "domain": DOMAIN_LABEL.get(domain, domain),
            "method": method,
            "method_display": display,
            "N": int(metrics.get("N", len(sub))),
            "Accuracy": metrics.get("Accuracy", np.nan),
            "F1": metrics.get("F1", np.nan),
            "Kappa": metrics.get("Kappa", np.nan),
            "QWK": metrics.get("QWK", np.nan),
            "Spearman_Rho": metrics.get("Spearman_Rho", np.nan),
        })
    out = pd.DataFrame(rows)
    out["method"] = pd.Categorical(out["method"], PLOT_FAMILIES, ordered=True)
    out["domain"] = pd.Categorical(out["domain"],
                                   [DOMAIN_LABEL[d] for d in DOMAINS],
                                   ordered=True)
    return out.sort_values(["domain", "method"]).reset_index(drop=True)


def confusion_matrix_long(pred_rows: pd.DataFrame) -> pd.DataFrame:
    if pred_rows.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["k_train", "feature_set", "domain", "method", "method_display"]
    for keys, sub in pred_rows.groupby(group_cols):
        kt, feature_set, domain, method, display = keys
        cm = np.zeros((len(ORDINAL_LABELS), len(ORDINAL_LABELS)), dtype=int)
        for truth, pred in zip(sub["ground_truth_num"], sub["predicted_num"]):
            if truth in ORDINAL_LABELS and pred in ORDINAL_LABELS:
                cm[ORDINAL_LABELS.index(int(truth)), ORDINAL_LABELS.index(int(pred))] += 1
        row_totals = cm.sum(axis=1)
        for i, truth in enumerate(ORDINAL_LABELS):
            for j, pred in enumerate(ORDINAL_LABELS):
                n = int(cm[i, j])
                rows.append({
                    "k_train": int(kt),
                    "feature_set": feature_set,
                    "domain": DOMAIN_LABEL.get(domain, domain),
                    "method": method,
                    "method_display": display,
                    "true_rating": truth,
                    "predicted_rating": pred,
                    "n": n,
                    "row_total": int(row_totals[i]),
                    "row_percent": n / row_totals[i] if row_totals[i] else np.nan,
                })
    out = pd.DataFrame(rows)
    out["method"] = pd.Categorical(out["method"], PLOT_FAMILIES, ordered=True)
    out["domain"] = pd.Categorical(out["domain"],
                                   [DOMAIN_LABEL[d] for d in DOMAINS],
                                   ordered=True)
    return out.sort_values(["domain", "method", "true_rating", "predicted_rating"])


def _confusion_array(sub: pd.DataFrame) -> np.ndarray:
    cm = np.zeros((len(ORDINAL_LABELS), len(ORDINAL_LABELS)), dtype=int)
    for truth, pred in zip(sub["ground_truth_num"], sub["predicted_num"]):
        if truth in ORDINAL_LABELS and pred in ORDINAL_LABELS:
            cm[ORDINAL_LABELS.index(int(truth)), ORDINAL_LABELS.index(int(pred))] += 1
    return cm


def plot_confusion_matrices(pred_rows: pd.DataFrame, metric_summary: pd.DataFrame,
                            out_stem: Path):
    if pred_rows.empty or metric_summary.empty:
        return
    domains = [DOMAIN_LABEL[d] for d in DOMAINS]
    fig, axes = plt.subplots(
        len(domains), len(PLOT_FAMILIES),
        figsize=(14.8, 9.6),
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "ordinal_confusion", ["#FFFFFF", "#DDEAF3", "#0173B2"]
    )
    im = None
    for ri, domain_label in enumerate(domains):
        for ci, method in enumerate(PLOT_FAMILIES):
            ax = axes[ri, ci]
            sub = pred_rows[
                (pred_rows["domain"].map(DOMAIN_LABEL) == domain_label)
                & (pred_rows["method"] == method)
            ]
            cm = _confusion_array(sub)
            denom = cm.sum(axis=1, keepdims=True)
            pct = np.divide(
                cm, denom,
                out=np.zeros(cm.shape, dtype=float),
                where=denom > 0,
            ) * 100
            im = ax.imshow(pct, vmin=0, vmax=100, cmap=cmap)

            for i in range(len(ORDINAL_LABELS)):
                for j in range(len(ORDINAL_LABELS)):
                    if cm[i, j] == 0:
                        continue
                    color = "white" if pct[i, j] >= 55 else "#222222"
                    ax.text(
                        j, i, f"{pct[i, j]:.0f}\n({cm[i, j]})",
                        ha="center", va="center",
                        fontsize=6.8, fontweight="bold", color=color,
                    )

            met = metric_summary[
                (metric_summary["domain"] == domain_label)
                & (metric_summary["method"] == method)
            ]
            if met.empty:
                metric_line = ""
            else:
                r = met.iloc[0]
                metric_line = f"QWK={_fmt(r['QWK'])}  Acc={_fmt(r['Accuracy'])}"
            title = metric_line
            if ri == 0:
                title = f"{METHOD_DISPLAY[method]}\n{metric_line}"
            ax.set_title(title, fontsize=10.5, fontweight="bold")

            ax.set_xticks(range(len(ORDINAL_LABELS)))
            ax.set_yticks(range(len(ORDINAL_LABELS)))
            ax.set_xticklabels([str(x) for x in ORDINAL_LABELS])
            ax.set_yticklabels([str(x) for x in ORDINAL_LABELS])
            if ri == len(domains) - 1:
                ax.set_xlabel("Predicted rating", fontsize=10.5, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{domain_label}\nTrue rating",
                              fontsize=10.5, fontweight="bold")
            else:
                ax.set_yticklabels([])
            ax.grid(False)
            ax.tick_params(axis="both", width=1.2, length=3, direction="out")
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontweight("bold")
                lbl.set_fontsize(9)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.72, pad=0.012)
        cbar.set_label("Row percent", fontsize=11, fontweight="bold")
        for lbl in cbar.ax.get_yticklabels():
            lbl.set_fontweight("bold")

    feature = str(pred_rows["feature_set"].iloc[0])
    k_train = int(pred_rows["k_train"].iloc[0])
    fig.suptitle(
        f"Ordinal confusion matrices: strict QWK feature block "
        f"({feature}, k_train = {k_train})",
        fontsize=15, fontweight="bold",
    )
    _save_plot(fig, out_stem)


def rank_signal_best_available(df_apples: pd.DataFrame, k_train: int) -> pd.DataFrame:
    """Best available per-domain Spearman for the selected plotting families.

    This is intentionally less strict than the fixed-feature aggregate table:
    supervised RF is often tied/constant within participants for demographic
    and history-only feature sets, so the strict fixed-feature Spearman can be
    undefined. For the rank-signal panel we show the best non-degenerate
    supervised RF signal alongside the selected LLM/hybrid methods.
    """
    rows = []
    for fam in PLOT_FAMILIES:
        sub = df_apples[
            (df_apples["k_train"] == k_train)
            & (df_apples["family"] == fam)
        ].dropna(subset=["Spearman_Rho"])
        vals = []
        features = []
        for dom in DOMAINS:
            d = sub[sub["domain"] == dom]
            if d.empty:
                continue
            best = d.loc[d["Spearman_Rho"].idxmax()]
            vals.append(float(best["Spearman_Rho"]))
            features.append(f"{DOMAIN_LABEL[dom]}={best['feature_set']}")
        rows.append({
            "method": fam,
            "display": METHOD_DISPLAY.get(fam, fam),
            "spearman_rho": float(np.mean(vals)) if vals else np.nan,
            "feature_sets": "; ".join(features) if features else "",
        })
    return pd.DataFrame(rows)


def plot_ordinal_qwk_spearman_snapshot(strict: pd.DataFrame,
                                       df_apples: pd.DataFrame,
                                       out_stem: Path):
    latest_k = int(strict["k_train"].max())
    latest = strict[strict["k_train"] == latest_k].set_index("metric")
    if "QWK" not in latest.index:
        return

    method_cols = [
        ("Supervised RF", "supervised_rf", METHOD_COLORS["Supervised RF"]),
        ("LLM-PP", "llm_pp_grok", METHOD_COLORS["LLM-PP (Grok-4-Fast)"]),
    ]
    qwk_vals = [latest.loc["QWK", col] for _, col, _ in method_cols]
    qwk_feature = str(latest.loc["QWK", "feature_set"])

    rank = rank_signal_best_available(df_apples, latest_k)
    rank_vals = [
        float(rank[rank["display"] == label]["spearman_rho"].iloc[0])
        if len(rank[rank["display"] == label]) else np.nan
        for label, _, _ in method_cols
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6),
                             constrained_layout=True)
    fig.patch.set_facecolor("white")
    labels = [label for label, _, _ in method_cols]
    colors = [color for _, _, color in method_cols]

    panels = [
        (axes[0], qwk_vals, f"QWK, strict fixed-feature ({qwk_feature})",
         "Mean QWK across domains", False),
        (axes[1], rank_vals, "Spearman, best available non-degenerate",
         "Mean Spearman rho", True),
    ]
    for ax, vals, title, ylabel, zero_line in panels:
        x = np.arange(len(labels))
        bars = ax.bar(x, vals, width=0.58, color=colors, alpha=0.78,
                      edgecolor=colors, linewidth=1.7)
        ax.bar_label(bars, labels=[_fmt(v) for v in vals], padding=3,
                     fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold")
        if zero_line:
            ax.axhline(0, color=COLOR_REFERENCE, linewidth=1.6,
                       linestyle="--", alpha=0.75)
            ax.set_ylim(min(-0.04, float(np.nanmin(vals)) - 0.03),
                        max(0.20, float(np.nanmax(vals)) + 0.04))
        else:
            ax.set_ylim(0, max(0.64, float(np.nanmax(vals)) + 0.08))
        _style_axis(ax)

    fig.suptitle(f"QWK versus within-participant Spearman (k_train = {latest_k})",
                 fontsize=15, fontweight="bold")
    _save_plot(fig, out_stem)


def plot_ai_vs_ml_strict_snapshot(strict: pd.DataFrame, df_apples: pd.DataFrame,
                                  out_stem: Path):
    latest_k = int(strict["k_train"].max())
    latest = strict[strict["k_train"] == latest_k].set_index("metric")

    class_metrics = ["Accuracy", "Macro-F1", "QWK"]
    method_cols = [
        ("Supervised RF", "supervised_rf", METHOD_COLORS["Supervised RF"]),
        ("LLM-PP", "llm_pp_grok", METHOD_COLORS["LLM-PP (Grok-4-Fast)"]),
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(14.0, 5.2),
        gridspec_kw={"width_ratios": [2.4, 1.0]},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")

    ax = axes[0]
    x = np.arange(len(class_metrics))
    width = 0.34
    for i, (label, col, color) in enumerate(method_cols):
        vals = [latest.loc[m, col] for m in class_metrics]
        bars = ax.bar(
            x + (i - 0.5) * width, vals, width=width,
            color=color, alpha=0.78, edgecolor=color, linewidth=1.7,
            label=label,
        )
        ax.bar_label(bars, labels=[_fmt(v) for v in vals], padding=3,
                     fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(class_metrics)
    ax.set_ylim(0, max(0.64, float(latest.loc[class_metrics, "best_overall"].max()) + 0.10))
    ax.set_ylabel("Mean across domains", fontsize=13, fontweight="bold")
    ax.set_title(f"Strict fixed-feature comparison (k_train = {latest_k})",
                 fontsize=14, fontweight="bold")
    _style_axis(ax)

    ax = axes[1]
    rank = rank_signal_best_available(df_apples, latest_k)
    rank_vals = rank["spearman_rho"].to_numpy(dtype=float)
    rank_labels = rank["display"].tolist()
    rank_colors = [METHOD_COLORS[m] for m in rank["method"]]
    bars = ax.bar(rank_labels, rank_vals, width=0.55, color=rank_colors,
                  alpha=0.78, edgecolor=rank_colors, linewidth=1.7)
    ax.bar_label(bars, labels=[_fmt(v) for v in rank_vals], padding=3,
                 fontsize=9, fontweight="bold")
    ax.axhline(0, color=COLOR_REFERENCE, linewidth=1.6, linestyle="--", alpha=0.75)
    ax.set_ylim(min(-0.04, float(np.nanmin(rank_vals)) - 0.03),
                max(0.20, float(np.nanmax(rank_vals)) + 0.04))
    ax.set_ylabel("Mean Spearman rho", fontsize=13, fontweight="bold")
    ax.set_title("Best available rank signal", fontsize=14, fontweight="bold")
    _style_axis(ax)

    handles = [
        Patch(facecolor=color, edgecolor=color, alpha=0.78, label=label)
        for label, _, color in method_cols
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.08), frameon=False,
               prop={"weight": "bold", "size": 11})
    _save_plot(fig, out_stem)


def plot_ai_vs_ml_strict_scaling(strict: pd.DataFrame, df_apples: pd.DataFrame,
                                 out_stem: Path):
    metrics = ["Accuracy", "Macro-F1", "QWK", "Within-participant Spearman rho"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    method_cols = [
        ("Supervised RF", "Supervised RF", "supervised_rf", METHOD_COLORS["Supervised RF"], "s", "--"),
        ("LLM-PP", "LLM-PP (Grok-4-Fast)", "llm_pp_grok", METHOD_COLORS["LLM-PP (Grok-4-Fast)"], "o", "-"),
    ]
    k_values = sorted(strict["k_train"].unique())
    spearman_sweep = build_kt_sweep(df_apples, "Spearman_Rho")

    for ax, metric in zip(axes.ravel(), metrics):
        sub = strict[strict["metric"] == metric].sort_values("k_train")
        for label, family, col, color, marker, ls in method_cols:
            if metric == "Within-participant Spearman rho":
                vals = [spearman_sweep.loc[family, k] for k in k_values]
                x_vals = k_values
            else:
                vals = sub[col].astype(float).to_numpy()
                x_vals = sub["k_train"]
            ax.plot(x_vals, vals, marker=marker, linestyle=ls,
                    color=color, linewidth=2.4, markersize=7.5,
                    alpha=0.95, label=label)
        ax.set_xticks(k_values)
        ax.set_xlabel("k_train", fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean across domains", fontsize=12, fontweight="bold")
        ax.set_title(metric, fontsize=13, fontweight="bold")
        if metric != "Within-participant Spearman rho":
            vals = sub[["supervised_rf", "llm_pp_grok"]].to_numpy(dtype=float)
            ax.set_ylim(0, max(0.64, np.nanmax(vals) + 0.08))
        else:
            ax.axhline(0, color=COLOR_REFERENCE, linewidth=1.5,
                       linestyle="--", alpha=0.75)
            vals = spearman_sweep.loc[[m[1] for m in method_cols], k_values].to_numpy(dtype=float)
            ax.set_ylim(min(-0.04, np.nanmin(vals) - 0.03),
                        max(0.20, np.nanmax(vals) + 0.04))
        _style_axis(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.05), frameon=False,
               prop={"weight": "bold", "size": 11})
    fig.suptitle("Strict fixed-feature AI-vs-ML scaling summary",
                 fontsize=15, fontweight="bold")
    _save_plot(fig, out_stem)


def plot_message_selection_gain(message_methods: pd.DataFrame, out_stem: Path):
    if message_methods.empty:
        return

    msg = message_methods.copy()
    domains = ["Content", "Coping", "Quitting"]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True,
                             constrained_layout=True)
    fig.patch.set_facecolor("white")

    for ax, domain in zip(axes, domains):
        sub = msg[msg["domain"] == domain].sort_values("K")
        k_vals = sorted(sub["K"].unique())
        for method in PLOT_FAMILIES:
            mdf = sub[sub["method"] == method].sort_values("K")
            if mdf.empty:
                continue
            x_vals = mdf["K"].to_numpy(dtype=float)
            y_vals = mdf["gain_over_random"].to_numpy(dtype=float)
            if {
                "gain_over_random_ci_lower",
                "gain_over_random_ci_upper",
            }.issubset(mdf.columns):
                ax.fill_between(
                    x_vals,
                    mdf["gain_over_random_ci_lower"].to_numpy(dtype=float),
                    mdf["gain_over_random_ci_upper"].to_numpy(dtype=float),
                    color=METHOD_COLORS[method],
                    alpha=0.12,
                    linewidth=0,
                    zorder=1,
                )
            ax.plot(
                x_vals, y_vals,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linewidth=2.4,
                markersize=7.5,
                linestyle="--" if method == "Supervised RF" else "-",
                alpha=0.95,
                zorder=3,
                label=METHOD_DISPLAY[method],
            )
        ax.axhline(0, color=COLOR_REFERENCE, linewidth=1.6,
                   linestyle="--", alpha=0.75)
        ax.set_xticks(k_vals)
        ax.set_xticklabels([str(int(k)) for k in k_vals])
        ax.set_xlabel("K selected messages", fontsize=12, fontweight="bold")
        feature = sub["feature_set"].iloc[0] if not sub.empty else ""
        ax.set_title(f"{domain} — fixed {feature}", fontsize=13, fontweight="bold")
        _style_axis(ax)

    axes[0].set_ylabel("Gain in mean human rating over random",
                       fontsize=12, fontweight="bold")
    if {
        "gain_over_random_ci_lower",
        "gain_over_random_ci_upper",
    }.issubset(msg.columns):
        y_max = float(msg["gain_over_random_ci_upper"].max()) + 0.08
        y_min = min(-0.08, float(msg["gain_over_random_ci_lower"].min()) - 0.04)
    else:
        y_max = float(msg["gain_over_random"].max()) + 0.08
        y_min = min(-0.08, float(msg["gain_over_random"].min()) - 0.04)
    for ax in axes:
        ax.set_ylim(y_min, y_max)

    method_handles = [
        Line2D([0], [0],
               color=METHOD_COLORS[m],
               marker=METHOD_MARKERS[m],
               linewidth=2.4,
               linestyle="--" if m == "Supervised RF" else "-",
               label=METHOD_DISPLAY[m])
        for m in PLOT_FAMILIES
    ]
    fig.legend(
        method_handles,
        [h.get_label() for h in method_handles],
        loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12),
        frameon=False, prop={"weight": "bold", "size": 10.5},
    )
    fig.suptitle("Message-selection gain over random with demographics + history + message embedding (95% CI bands, k_train = 7)",
                 fontsize=15, fontweight="bold")
    _save_plot(fig, out_stem)


def write_clean_results(df_apples: pd.DataFrame):
    fixed = fixed_feature_means(df_apples)
    fixed_path = OUT_DIR / "ai_vs_ml_fixed_feature_mean_by_k.csv"
    fixed.to_csv(fixed_path, index=False)

    strict = strict_feature_summary(fixed)
    strict_path = OUT_DIR / "ai_vs_ml_strict_fixed_feature.csv"
    strict.to_csv(strict_path, index=False)

    message = message_selection_best()
    msg_path = OUT_DIR / "message_selection_best_by_k.csv"
    if not message.empty:
        message.to_csv(msg_path, index=False)

    message_methods = message_selection_methods(k_train=7)
    msg_methods_path = OUT_DIR / "message_selection_methods_k7.csv"
    if not message_methods.empty:
        message_methods.to_csv(msg_methods_path, index=False)

    latest_k = int(df_apples["k_train"].max())
    qwk_feature_set = ""
    qwk_row = strict[(strict["k_train"] == latest_k) & (strict["metric"] == "QWK")]
    if not qwk_row.empty:
        qwk_feature_set = str(qwk_row["feature_set"].iloc[0])

    confusion_preds = method_prediction_rows(
        k_train=latest_k,
        feature_set=qwk_feature_set if qwk_feature_set else None,
    )
    confusion_metrics = prediction_metric_summary(confusion_preds)
    confusion_long = confusion_matrix_long(confusion_preds)
    if not confusion_preds.empty:
        confusion_preds.to_csv(OUT_DIR / f"confusion_predictions_k{latest_k}.csv",
                               index=False)
    if not confusion_metrics.empty:
        confusion_metrics.to_csv(OUT_DIR / f"confusion_metrics_k{latest_k}.csv",
                                 index=False)
    if not confusion_long.empty:
        confusion_long.to_csv(OUT_DIR / f"confusion_matrices_k{latest_k}.csv",
                              index=False)

    plot_ai_vs_ml_strict_snapshot(strict, df_apples, OUT_DIR / "ai_vs_ml_strict_snapshot")
    plot_ai_vs_ml_strict_scaling(strict, df_apples, OUT_DIR / "ai_vs_ml_strict_scaling")
    plot_ordinal_qwk_spearman_snapshot(strict, df_apples,
                                       OUT_DIR / "ordinal_qwk_spearman_snapshot")
    plot_confusion_matrices(confusion_preds, confusion_metrics,
                            OUT_DIR / f"confusion_matrices_k{latest_k}")
    plot_message_selection_gain(message_methods, OUT_DIR / "message_selection_gain")

    latest = strict[strict["k_train"] == latest_k].copy()
    latest["N"] = latest.apply(
        lambda r: str(int(r["N_min"])) if r["N_min"] == r["N_max"]
        else f"{int(r['N_min'])}-{int(r['N_max'])}",
        axis=1,
    )
    latest_display = latest[[
        "metric", "feature_set", "N", "best_overall_method", "best_overall",
        "supervised_rf", "llm_pp_grok",
    ]].copy()

    msg_display = pd.DataFrame()
    if not message.empty:
        msg_display = message[message["K"].isin([5, 10, 25])].copy()
        msg_display = msg_display[[
            "Domain", "K", "Model", "LLM Selection (Human Rating)",
            "Random Mean", "Human Oracle (Human Rating)",
            "LLM_minus_Random", "Oracle_minus_LLM", "N Messages",
        ]]

    msg_method_display = pd.DataFrame()
    if not message_methods.empty:
        msg_method_display = message_methods[message_methods["K"].isin([5, 10, 25])].copy()
        if {
            "gain_over_random_ci_lower",
            "gain_over_random_ci_upper",
        }.issubset(msg_method_display.columns):
            msg_method_display["gain_over_random_95ci"] = msg_method_display.apply(
                lambda r: (
                    f"[{r['gain_over_random_ci_lower']:.3f}, "
                    f"{r['gain_over_random_ci_upper']:.3f}]"
                ),
                axis=1,
            )
        display_cols = [
            "domain", "feature_set", "method_display", "K",
            "selected_human_rating", "random_mean", "gain_over_random",
        ]
        if "gain_over_random_95ci" in msg_method_display.columns:
            display_cols.append("gain_over_random_95ci")
        display_cols.extend(["human_oracle_rating", "n_messages"])
        msg_method_display = msg_method_display[display_cols]

    confusion_metric_display = pd.DataFrame()
    if not confusion_metrics.empty:
        confusion_metric_display = confusion_metrics[[
            "domain", "feature_set", "method_display", "N",
            "Accuracy", "QWK", "Spearman_Rho",
        ]].copy()

    lines = [
        "# Clean Updated Results",
        "",
        "## Recommended Messaging",
        "",
        "- The whole comparison is strictly two methods: supervised RF versus LLM-PP (Grok-4-Fast) on the identical held-out rows. No blended or ensemble methods are reported.",
        "- Frame supervised RF as the strongest aggregate classifier, but as a within-participant participant-calibrated baseline (it predicts each person's typical rating from a demographic fingerprint), not as demographic generalization.",
        "- Frame LLM-PP as competitive and complementary: it is strongest relative to RF at low history / cold start, and it is the better choice for coping and quitting message selection.",
        "- RF's within-participant Spearman is undefined because it is message-blind (one value per participant); LLM-PP can actually rank a participant's messages.",
        "- Generic zero-/few-shot LLM results are context only, because they were scored on a smaller dt10 subset and should not be mixed into the AI-vs-ML table.",
        "",
        "## Reviewer-Facing Sequence",
        "",
        "1. Start with source discipline: the AI-vs-ML comparison uses only shared-row ensemble files, while generic LLM results stay in a separate context table.",
        "2. Answer the aggregate-performance question with the strict fixed-feature snapshot and scaling curves.",
        "3. Address ordinal-rating concerns with QWK, the QWK-vs-Spearman diagnostic, and the true-vs-predicted confusion matrices.",
        "4. Address the Spearman surprise explicitly: Spearman is within-participant and tie-sensitive, whereas QWK is an all-row ordinal-agreement metric.",
        "5. Answer the message-selection question with gain over random for supervised RF and LLM-PP, highlighting LLM-PP's edge in the coping and quitting domains.",
        "6. Use the rating-distribution histograms as context for class imbalance and demographic subgroup checks, not as model-performance claims.",
        "",
        "## Strict AI-vs-ML Snapshot",
        "",
        "Each row below uses one fixed feature-set block at the latest available ",
        f"`k_train={latest_k}`. Within that block, supervised RF and LLM-PP share ",
        "the same held-out rows. Values are means across Content, ",
        "Coping, and Quitting.",
        "",
    ]
    lines.extend(_markdown_table(latest_display, list(latest_display.columns)))
    lines.extend([
        "",
        "For Spearman rho, `—` means that the fixed-feature RF predictions ",
        "were degenerate for within-participant ranking in that block ",
        "(for example, tied predictions), not that a result was fabricated ",
        "or silently unavailable.",
        "",
        "Full strict fixed-feature summaries are saved in ",
        "`ai_vs_ml_strict_fixed_feature.csv`; full fixed-feature method means ",
        "are saved in `ai_vs_ml_fixed_feature_mean_by_k.csv`.",
        "",
        "## Plots",
        "",
        "- `ai_vs_ml_strict_snapshot.png` / `.pdf` — latest strict fixed-feature snapshot.",
        "- `ai_vs_ml_strict_scaling.png` / `.pdf` — strict fixed-feature scaling across k.",
        "- `ordinal_qwk_spearman_snapshot.png` / `.pdf` — strict QWK beside best-available within-participant Spearman.",
        f"- `confusion_matrices_k{latest_k}.png` / `.pdf` — row-normalized true-vs-predicted rating confusion matrices.",
        "- `message_selection_gain.png` / `.pdf` — supervised RF versus LLM-PP message-selection gain over random.",
        "- In `ai_vs_ml_strict_snapshot`, the rank-signal panel uses the best available non-degenerate Spearman signal for each selected method; the aggregate-metric panel remains strict fixed-feature.",
        "- Companion revision histograms outside this folder: `class_distribution_by_domain`, `all_rating_subgroup_histograms`, `profile_test_similarity_distribution`, and `figure3_score_distributions_<domain>`.",
        "",
        "## Ordinal Diagnostics",
        "",
        "QWK and Spearman answer different questions. QWK is an absolute ",
        "ordinal-agreement metric over all held-out message ratings; ",
        "within-participant Spearman is computed inside each participant and ",
        "then averaged, so it is sensitive to tied predictions and to the ",
        "small number of held-out messages per participant.",
        "",
    ])
    if confusion_metric_display.empty:
        lines.append("Confusion-matrix diagnostics were not regenerated.")
        lines.append("")
    else:
        lines.extend([
            f"The confusion matrices use the strict QWK feature block ",
            f"`{qwk_feature_set}` at `k_train={latest_k}`.",
            "",
        ])
        lines.extend(_markdown_table(confusion_metric_display,
                                     list(confusion_metric_display.columns)))
        lines.extend([
            "",
            f"Full row-level confusion counts are saved in ",
            f"`confusion_matrices_k{latest_k}.csv`; per-row predictions are ",
            f"saved in `confusion_predictions_k{latest_k}.csv`.",
            "",
        ])

    lines.extend([
        "## Method-Level Message Selection",
        "",
    ])
    if msg_method_display.empty:
        lines.append("`message_selection_methods_k7.csv` is missing, so method-level message selection was not regenerated.")
    else:
        selection_feature_sets = sorted(
            str(x) for x in message_methods["feature_set"].dropna().unique()
        )
        selection_feature_text = (
            selection_feature_sets[0]
            if len(selection_feature_sets) == 1
            else ", ".join(selection_feature_sets)
        )
        lines.extend([
            f"All domains use the fixed `{selection_feature_text}` RF feature ",
            "block, then the same held-out rows are reused for LLM-PP. This ",
            "keeps supervised and LLM message selection on the same rows and ",
            "avoids domain-wise feature-set cherry-picking.",
            "",
            "Supervised RF message selection treats the fitted RF model as a ",
            "scoring rule: each held-out participant-message row receives a ",
            "predicted numeric PME rating, predictions are averaged at the ",
            "message level, messages are ranked by that predicted score, and ",
            "the top K messages are evaluated by their observed human ratings. ",
            "The table reports the selected-message human rating and gain over ",
            "random selection, with the human-oracle ranking as an upper bound. ",
            "The gain plot and `gain_over_random_95ci` column use 95% normal-",
            "approximation confidence intervals from message-level standard ",
            "errors (SD divided by sqrt(n)); the figure renders these intervals ",
            "as shaded bands.",
            "",
        ])
        lines.extend(_markdown_table(msg_method_display, list(msg_method_display.columns)))
        lines.extend([
            "",
            "Full method-level grid is saved in `message_selection_methods_k7.csv`.",
            "",
        ])

    lines.extend([
        "## LLM-Only Message-Selection Context",
        "",
    ])
    if msg_display.empty:
        lines.append("`llm_selection_quality.csv` is missing, so message-selection results were not regenerated.")
    else:
        lines.extend([
            "Best LLM by domain and K, evaluated as the mean human rating of ",
            "the messages selected by LLM score. Random and human-oracle rows ",
            "come from the same 108-message domain pools with 2,000 bootstrap ",
            "resamples.",
            "",
        ])
        lines.extend(_markdown_table(msg_display, list(msg_display.columns)))
        lines.extend([
            "",
            "Full K grid is saved in `message_selection_best_by_k.csv`.",
        ])

    lines.extend([
        "",
        "## Source Discipline",
        "",
        "- AI-vs-ML aggregate comparisons use only `lc_dt10_ensemble_k1.csv`, `lc_dt10_ensemble_k3.csv`, and `lc_dt10_ensemble_k7.csv`.",
        "- The best-feature-set tables remain available as descriptive summaries, with feature set and N listed per cell.",
        "- No placeholder or fabricated revision values are generated here.",
    ])

    (OUT_DIR / "clean_results.md").write_text("\n".join(lines))


# ----------------------- summary doc -----------------------------------

def write_summary_doc():
    md = []
    md.append("# Cross-method progress summary\n")
    md.append("**Important caveat.** Earlier versions of this summary mixed ")
    md.append("rows from different files at the same k_train. Those files use ")
    md.append("DIFFERENT test sets at the same k_train (Generic LLM ≈ 300, ")
    md.append("LLM-PP ≈ 800, RF/Ensemble ≈ 2700 at k=1), so the apples-to-")
    md.append("apples comparison was broken. See `test_set_audit.md`.\n")
    md.append("This version sources the comparison only from the ensemble ")
    md.append("files (`lc_dt10_ensemble_k1.csv`, `lc_dt10_ensemble_k3.csv`, ")
    md.append("and `lc_dt10_ensemble_k7.csv`), where all method rows are ")
    md.append("evaluated on the same test set inside each fixed `(k_train, ")
    md.append("feature_set)` block. Generic LLM is reported separately with ")
    md.append("its own (smaller) test set.\n")
    md.append("## Clean updated read")
    md.append("- [`clean_results.md`](clean_results.md) — concise messaging, ")
    md.append("strict fixed-feature AI-vs-ML summary, and message-selection ")
    md.append("headline table.\n")
    md.append("## Tables (apples-to-apples, one panel per k_train)")
    for slug, label in zip(["accuracy", "macro_f1", "qwk", "spearman_rho"],
                            ["Accuracy", "Macro-F1", "QWK", "Within-participant Spearman rho"]):
        md.append(f"- {label}: [`table_{slug}.md`](table_{slug}.md) "
                  f"(CSV: `table_{slug}.csv`)")
    md.append("")
    md.append("## Mean across domains, by k_train")
    md.append("- [`sweep_by_k_train.md`](sweep_by_k_train.md)\n")
    md.append("## Learning curves")
    md.append("- `learning_curves.png` / `learning_curves.pdf` — apples-to-")
    md.append("apples scaling at k_train in {1, 3, 7}.\n")
    md.append("## Clean plots")
    md.append("- `ai_vs_ml_strict_snapshot.png` / `.pdf` — latest strict ")
    md.append("fixed-feature AI-vs-ML snapshot.")
    md.append("- `ai_vs_ml_strict_scaling.png` / `.pdf` — strict fixed-feature ")
    md.append("scaling across k.")
    md.append("- `message_selection_gain.png` / `.pdf` — supervised RF and ")
    md.append("LLM-PP message-selection gain over random.\n")
    md.append("## Test-set audit")
    md.append("- [`test_set_audit.md`](test_set_audit.md) — N per source × k_train.\n")
    md.append("## Notes")
    md.append("- Each k_train evaluates on a different shared test set (the ")
    md.append("intersection of users with both an RF and an LLM-PP prediction ")
    md.append("at that history depth: N≈2700 at k=1, N≈2100 at k=3, N≈900 at ")
    md.append("k=7). Within a fixed feature-set block, all method rows share ")
    md.append("that test set; absolute values across k columns are not strictly ")
    md.append("comparable, so read the curves as the relative ranking of ")
    md.append("methods at each k.")
    md.append("- If you want Generic LLM in the apples-to-apples grid, score ")
    md.append("the Generic LLM on the ensemble file's test set rather than ")
    md.append("the smaller dt10 set.")
    (OUT_DIR / "progress_summary_all.md").write_text("\n".join(md))


# ----------------------- main ------------------------------------------

def main():
    write_audit()

    df_apples = load_ensemble_long()
    print(f"apples-to-apples rows: {len(df_apples)} "
          f"(k_train values: {sorted(df_apples['k_train'].unique())})")

    ks = sorted(int(k) for k in df_apples["k_train"].unique())
    for metric_col, metric_label in METRICS:
        slug = METRIC_SLUG[metric_col]
        per_k_tables = {kt: build_apples_table(df_apples, metric_col, kt) for kt in ks}
        gen = build_generic_llm_table(metric_col)
        csv_p, md_p = write_apples_table(per_k_tables, metric_label, slug=slug,
                                          generic_llm=gen)
        print(f"  wrote {md_p.relative_to(PROJECT_ROOT)}")

    write_sweep(df_apples)
    print(f"  wrote {(OUT_DIR / 'sweep_by_k_train.md').relative_to(PROJECT_ROOT)}")

    fig_stem = OUT_DIR / "learning_curves"
    plot_learning_curves(df_apples, fig_stem)
    print(f"  wrote {fig_stem.with_suffix('.png').relative_to(PROJECT_ROOT)}")

    write_clean_results(df_apples)
    print(f"  wrote {(OUT_DIR / 'clean_results.md').relative_to(PROJECT_ROOT)}")

    write_summary_doc()
    print(f"  wrote {(OUT_DIR / 'progress_summary_all.md').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
