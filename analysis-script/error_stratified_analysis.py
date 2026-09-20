#!/usr/bin/env python3
"""Where does best-LLM-PP win vs best-RF, and where does it lose?

For k=3 on the dt10 split (N=2102 test messages, 301 users in shared
train/test):

  - RF    = the per-domain best feature set on this same test set
            (Demographics or Demo+History — picked from lc_dt10_rf.csv).
  - LLM-PP = Grok-4-Fast Digital Twin (best LLM-PP overall).

Per row, we log: (gt, rf_pred, llm_pred, |rf_err|, |llm_err|, winner).
Then we stratify by:
  - ground-truth rating (1..5)
  - quit_intention
  - smoking_status
  - age band (<30 / 30-49 / 50+)

For each stratum:
  - N
  - RF accuracy, LLM-PP accuracy
  - Oracle accuracy = % of rows where AT LEAST one of the two is correct
                       (the best a perfect router could do)
  - Oracle uplift   = oracle - max(rf, llm)  →  headroom for routing
  - Per-row win rates (LLM beats RF strictly / tie / RF beats LLM)

Outputs:
  revision/figures/error_stratified/per_row_k3.csv
  revision/figures/error_stratified/by_gt_rating.{md,csv}
  revision/figures/error_stratified/by_quit_intention.{md,csv}
  revision/figures/error_stratified/by_smoking_status.{md,csv}
  revision/figures/error_stratified/by_age_band.{md,csv}
  revision/figures/error_stratified/win_loss_bars.{png,pdf}
  revision/figures/error_stratified/headline.md  (top strata where LLM-PP wins)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REV_DIR = PROJECT_ROOT / "revision" / "figures"

K_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 3
OUT_DIR = REV_DIR / "error_stratified" / f"k{K_TRAIN}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DOMAINS = ["content", "coping", "quitting"]
DOMAIN_LABEL = {"content": "Content", "coping": "Coping", "quitting": "Quitting"}

RATING_MAPS = {
    "content": {"Very poor": 1, "Poor": 2, "Acceptable": 3, "Good": 4, "Very good": 5},
    "coping":  {"Not at all helpful": 1, "Somewhat helpful": 2,
                 "Moderately helpful": 3, "Very helpful": 4, "Extremely helpful": 5},
    "quitting": {"Not at all helpful": 1, "Somewhat helpful": 2,
                 "Moderately helpful": 3, "Very helpful": 4, "Extremely helpful": 5},
}


def best_rf_feature_set_per_domain() -> dict[str, str]:
    """Pick the RF feature set with highest accuracy at k=3 on the same test
    set we're scoring (lc_dt10_rf.csv)."""
    df = pd.read_csv(REV_DIR / "lc_dt10_rf.csv")
    df = df[df["k_train"] == K_TRAIN]
    out = {}
    for dom in DOMAINS:
        sub = df[df["domain"] == dom].dropna(subset=["Accuracy"])
        if sub.empty:
            continue
        best = sub.loc[sub["Accuracy"].idxmax()]
        out[dom] = best["feature_set"]
    return out


def load_rf_predictions() -> dict[tuple[str, str, str], int]:
    """(rid, msg, dom) -> RF predicted integer for the chosen best feature set."""
    df = pd.read_csv(REV_DIR / "lc_dt10_rf_predictions.csv")
    df = df[df["k_train"] == K_TRAIN]
    best = best_rf_feature_set_per_domain()
    out = {}
    for dom in DOMAINS:
        fs = best.get(dom)
        if fs is None:
            continue
        sub = df[(df["feature_set"] == fs) & (df["domain"] == dom)]
        for _, r in sub.iterrows():
            out[(r["response_id"], r["input_message"], dom)] = int(r["predicted_num"])
    return out, best


def load_llm_predictions_and_meta() -> tuple[dict, dict]:
    """Returns (llm_pred dict, metadata dict). LLM = Grok-4-Fast."""
    path = (PROJECT_ROOT / "results_manuscript_x-ai_grok-4-fast"
            / f"digital_twin_dt10_k{K_TRAIN}.json")
    with open(path) as f:
        data = json.load(f)
    pred = {}
    meta = {}
    for v in data.values():
        if not isinstance(v, dict):
            continue
        rid = v.get("response_id")
        msg = v.get("input_message")
        if rid is None or msg is None:
            continue
        for dom in DOMAINS:
            p = RATING_MAPS[dom].get(v.get(f"predicted_{dom}"))
            g = RATING_MAPS[dom].get(v.get(f"ground_truth_{dom}"))
            if p is None or g is None:
                continue
            pred[(rid, msg, dom)] = (int(g), int(p))
        meta[(rid, msg)] = v.get("metadata", {})
    return pred, meta


# ---------------- per-row table ----------------------------------------

def build_per_row(rf_pred, llm_pred_with_gt, meta) -> pd.DataFrame:
    rows = []
    for key, (gt, p_llm) in llm_pred_with_gt.items():
        p_rf = rf_pred.get(key)
        if p_rf is None:
            continue
        rid, msg, dom = key
        m = meta.get((rid, msg), {})
        rf_err = abs(p_rf - gt)
        llm_err = abs(p_llm - gt)
        rf_correct = int(p_rf == gt)
        llm_correct = int(p_llm == gt)
        if rf_correct and not llm_correct:
            winner = "RF"
        elif llm_correct and not rf_correct:
            winner = "LLM"
        elif rf_correct and llm_correct:
            winner = "tie_both_correct"
        else:
            winner = "tie_both_wrong"
        rows.append({
            "response_id": rid,
            "domain": dom,
            "gt": gt,
            "rf_pred": p_rf,
            "llm_pred": p_llm,
            "rf_correct": rf_correct,
            "llm_correct": llm_correct,
            "rf_err": rf_err,
            "llm_err": llm_err,
            "winner": winner,
            "quit_intention": m.get("quit_intention", "Unknown"),
            "smoking_status": m.get("smoking_status", "Unknown"),
            "age_years": m.get("age_years"),
        })
    df = pd.DataFrame(rows)
    df["age_band"] = pd.cut(
        df["age_years"].astype(float),
        bins=[0, 29.999, 49.999, 200],
        labels=["<30", "30-49", "50+"],
    ).astype(object)
    return df


# ---------------- stratum summary --------------------------------------

def stratum_summary(df: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for stratum, g in df.groupby(by, dropna=False):
        if len(g) == 0:
            continue
        rf_acc = g["rf_correct"].mean()
        llm_acc = g["llm_correct"].mean()
        oracle_acc = ((g["rf_correct"] | g["llm_correct"]).mean())
        # Within-row deltas
        n_llm_beats = ((g["llm_err"] < g["rf_err"]).sum())
        n_rf_beats = ((g["rf_err"] < g["llm_err"]).sum())
        n_tie = ((g["rf_err"] == g["llm_err"]).sum())
        # Mean absolute error (ordinal)
        rf_mae = g["rf_err"].mean()
        llm_mae = g["llm_err"].mean()
        rows.append({
            by: stratum,
            "N": len(g),
            "RF_acc": round(rf_acc, 3),
            "LLM_acc": round(llm_acc, 3),
            "Oracle_acc": round(oracle_acc, 3),
            "Oracle_uplift": round(oracle_acc - max(rf_acc, llm_acc), 3),
            "LLM_beats_RF_%": round(100 * n_llm_beats / len(g), 1),
            "RF_beats_LLM_%": round(100 * n_rf_beats / len(g), 1),
            "tie_%": round(100 * n_tie / len(g), 1),
            "RF_MAE": round(rf_mae, 3),
            "LLM_MAE": round(llm_mae, 3),
        })
    return pd.DataFrame(rows)


def write_stratum(df_strat: pd.DataFrame, by: str, by_label: str):
    csv_p = OUT_DIR / f"by_{by}.csv"
    md_p = OUT_DIR / f"by_{by}.md"
    df_strat.to_csv(csv_p, index=False)
    md = [f"# Error stratification by {by_label}  (k_train = {K_TRAIN}, "
          f"all 3 domains pooled)\n",
          "Per-row comparison of best-RF vs best-LLM-PP (Grok-4-Fast). "
          "`Oracle_acc` = % of rows where at least one of the two is correct; "
          "`Oracle_uplift` = oracle minus the better of RF / LLM (headroom for "
          "a perfect row-level router). `LLM_beats_RF_%` = % of rows where the "
          "LLM is strictly closer to ground truth than RF (by absolute "
          "ordinal error).\n"]
    cols = list(df_strat.columns)
    md.append("| " + " | ".join(cols) + " |")
    md.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in df_strat.iterrows():
        vals = [str(r[c]) for c in cols]
        md.append("| " + " | ".join(vals) + " |")
    md.append("")
    md_p.write_text("\n".join(md))
    return md_p


# ---------------- bar chart --------------------------------------------

def plot_winloss_bars(df: pd.DataFrame, out_stem: Path):
    """Stacked bars: % LLM-beats-RF / tie / RF-beats-LLM, per stratum, per
    stratification axis. One panel per axis."""
    axes_specs = [
        ("gt", "Ground-truth rating", [1, 2, 3, 4, 5]),
        ("quit_intention", "Quit intention", None),
        ("smoking_status", "Smoking status", None),
        ("age_band", "Age band", ["<30", "30-49", "50+"]),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.patch.set_facecolor("white")
    axs = axs.flatten()
    for ax, (col, label, order) in zip(axs, axes_specs):
        if order is None:
            order = (df.groupby(col).size()
                       .sort_values(ascending=False)
                       .index.tolist())
        rows = []
        for stratum in order:
            sub = df[df[col] == stratum]
            if len(sub) == 0:
                continue
            n = len(sub)
            rows.append({
                "stratum": str(stratum),
                "N": n,
                "LLM_beats": (sub["llm_err"] < sub["rf_err"]).sum() / n,
                "tie":       (sub["llm_err"] == sub["rf_err"]).sum() / n,
                "RF_beats":  (sub["rf_err"] < sub["llm_err"]).sum() / n,
            })
        sdf = pd.DataFrame(rows)
        if sdf.empty:
            ax.set_visible(False)
            continue
        x = np.arange(len(sdf))
        ax.bar(x, sdf["LLM_beats"], color="#CC78BC", label="LLM-PP closer")
        ax.bar(x, sdf["tie"], bottom=sdf["LLM_beats"],
                color="#BBBBBB", label="tie")
        ax.bar(x, sdf["RF_beats"],
                bottom=sdf["LLM_beats"] + sdf["tie"],
                color="#262626", label="RF closer")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}\n(N={n})" for s, n in
                              zip(sdf["stratum"], sdf["N"])],
                             fontsize=10, rotation=0)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction of rows", fontsize=11)
        ax.axhline(0.5, color="#666", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.grid(axis="y", alpha=0.18, linestyle="--", color="#4D4D4D")
    handles, labels_ = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=3,
                fontsize=11, bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.suptitle(f"Per-row win/tie/loss: best-LLM-PP (Grok-4-Fast) vs best-RF, "
                 f"k_train={K_TRAIN}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(f"{out_stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------- headline find-er ------------------------------------

def write_headline(df: pd.DataFrame):
    """Identify strata where LLM-PP genuinely beats RF (acc + lower MAE)."""
    md = ["# Headline: where best-LLM-PP genuinely beats best-RF\n",
          f"k_train = {K_TRAIN}; LLM-PP = Grok-4-Fast; RF = best feature set "
          "per domain on this same test set.\n",
          "We flag a stratum as a *real* LLM-PP win when **all three** of "
          "the following hold (any single signal is noise on small N):\n",
          "1. LLM accuracy ≥ RF accuracy (within rounding).",
          "2. LLM mean absolute error ≤ RF MAE.",
          "3. % of rows where LLM is strictly closer to GT than RF "
          "is ≥ 50% (more than half of disagreements go LLM's way).\n"]
    found_any = False
    for axis, label in [("gt", "ground-truth rating"),
                         ("quit_intention", "quit intention"),
                         ("smoking_status", "smoking status"),
                         ("age_band", "age band"),
                         ("domain", "rating domain")]:
        s = stratum_summary(df, axis)
        winners = s[(s["LLM_acc"] >= s["RF_acc"] - 0.005)
                     & (s["LLM_MAE"] <= s["RF_MAE"] + 0.005)
                     & (s["LLM_beats_RF_%"] + s["tie_%"]
                          >= s["RF_beats_LLM_%"] + s["tie_%"])
                     & (s["LLM_beats_RF_%"] >= s["RF_beats_LLM_%"])]
        if not winners.empty:
            found_any = True
            md.append(f"## By {label}\n")
            cols = ["N", "RF_acc", "LLM_acc", "Oracle_acc", "Oracle_uplift",
                     "LLM_beats_RF_%", "RF_beats_LLM_%", "tie_%",
                     "RF_MAE", "LLM_MAE"]
            md.append("| " + label + " | " + " | ".join(cols) + " |")
            md.append("|" + "|".join(["---"] * (1 + len(cols))) + "|")
            for _, r in winners.iterrows():
                stratum = str(r[axis])
                vals = [str(r[c]) for c in cols]
                md.append("| " + stratum + " | " + " | ".join(vals) + " |")
            md.append("")
    if not found_any:
        md.append("**No stratum currently meets all three criteria.** That's "
                  "consistent with the global pattern: at this k_train and "
                  "with shared train/test users, RF wins almost everywhere.\n")
    md.append("\n## Oracle ceiling (headroom for a perfect row-level router)")
    s_overall = stratum_summary(df, "domain")
    md.append("\n| domain | RF_acc | LLM_acc | Oracle_acc | Oracle_uplift |")
    md.append("|---|---|---|---|---|")
    for _, r in s_overall.iterrows():
        md.append(f"| {r['domain']} | {r['RF_acc']} | {r['LLM_acc']} | "
                   f"{r['Oracle_acc']} | {r['Oracle_uplift']} |")
    md.append("")
    md.append("If Oracle_uplift is large (≥ 0.05), there *is* a real "
              "complementary signal that a smart router could harvest. "
              "If small, the two predictors are largely redundant and no "
              "ensemble can save us.\n")
    (OUT_DIR / "headline.md").write_text("\n".join(md))


def main():
    rf_pred, best_rf = load_rf_predictions()
    llm_pred, meta = load_llm_predictions_and_meta()
    df = build_per_row(rf_pred, llm_pred, meta)
    df.to_csv(OUT_DIR / f"per_row_k{K_TRAIN}.csv", index=False)
    print(f"per-row rows: {len(df)} (best RF feature set: {best_rf})")

    # Stratifications
    for col, label in [("gt", "ground-truth rating"),
                        ("quit_intention", "quit intention"),
                        ("smoking_status", "smoking status"),
                        ("age_band", "age band"),
                        ("domain", "rating domain")]:
        s = stratum_summary(df, col)
        write_stratum(s, col, label)
        print(f"  wrote {OUT_DIR / f'by_{col}.md'}")

    plot_winloss_bars(df, OUT_DIR / "win_loss_bars")
    print(f"  wrote {OUT_DIR / 'win_loss_bars.png'}")

    write_headline(df)
    print(f"  wrote {OUT_DIR / 'headline.md'}")


if __name__ == "__main__":
    main()
