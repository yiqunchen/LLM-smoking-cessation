#!/usr/bin/env python3
"""
Spearman rank analysis for the revision.

This script extracts the already-saved predictions and summarizes
participant-level Spearman rho by domain for:
  1. Baselines
  2. Generic LLM methods
  3. Personalized LLM methods

Rationale:
  Spearman rho is invariant to within-participant affine rescaling,
  so it provides complementary robustness evidence for concerns about
  participant-specific rating scale shifts (for example z-scoring).
  It does not replace a full z-score retraining analysis.

Outputs:
  - revision/figures/spearman_rank_results.csv
  - revision/figures/spearman_rank_summary.csv
  - revision/figures/spearman_rank_comparison.png/pdf
  - revision/figures/spearman_rank_notes.md
"""

from __future__ import annotations

import os
import sys
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import text_baselines as tb
from revision_utils import (
    COLORS,
    DOMAINS,
    METHOD_CONFIGS,
    MODEL_CONFIGS,
    apply_repo_plot_style,
    compute_all_metrics,
    figures_path,
    load_canonical_data,
    load_results_aligned,
    save_figure,
)


GROUP_COLORS = {
    "Best Baseline": COLORS["Random Forest"],
    "Best Generic LLM": COLORS["GPT-4o-mini"],
    "Best Personalized LLM": COLORS["GPT-5"],
}


def compute_spearman_support(gt: np.ndarray,
                             pred: np.ndarray,
                             response_ids: np.ndarray) -> tuple[float, int, int]:
    """Return mean per-participant Spearman, valid participant count, total count."""
    total_participants = len(np.unique(response_ids))
    rhos = []

    for pid in np.unique(response_ids):
        mask = response_ids == pid
        g = gt[mask]
        p = pred[mask]
        if len(g) <= 1:
            continue
        if len(np.unique(g)) <= 1 or len(np.unique(p)) <= 1:
            continue
        rho, _ = spearmanr(g, p)
        if not np.isnan(rho):
            rhos.append(float(rho))

    mean_rho = float(np.mean(rhos)) if rhos else np.nan
    return mean_rho, len(rhos), total_participants


def collect_llm_rows() -> pd.DataFrame:
    """Recompute Spearman-focused rows from saved LLM prediction JSON files."""
    rows = []

    for model_id, model_cfg in MODEL_CONFIGS.items():
        for method_file, method_cfg in METHOD_CONFIGS.items():
            df = load_results_aligned(model_id, method_file)
            if df is None:
                continue

            for domain in DOMAINS:
                gt_col = f"gt_{domain}_num"
                pred_col = f"pred_{domain}_num"
                valid = df[gt_col].notna() & df[pred_col].notna()
                gt = df.loc[valid, gt_col].astype(int).to_numpy()
                pred = df.loc[valid, pred_col].astype(int).to_numpy()
                response_ids = df.loc[valid, "response_id"].to_numpy()

                metrics = compute_all_metrics(gt, pred, response_ids)
                spearman_rho, valid_p, total_p = compute_spearman_support(
                    gt, pred, response_ids
                )

                rows.append({
                    "System_Type": "LLM",
                    "Comparison_Group": (
                        "Generic LLM"
                        if method_cfg["category"] == "Generic LLM"
                        else "Personalized LLM"
                    ),
                    "Split_Type": (
                        "Participant 70/30"
                        if method_cfg["category"] == "Generic LLM"
                        else "PP 70/30"
                    ),
                    "Category": method_cfg["category"],
                    "Domain": domain.capitalize(),
                    "System_Label": f"{model_cfg['display']} / {method_cfg['display']}",
                    "Model": model_cfg["display"],
                    "Method": method_cfg["display"],
                    "Baseline": np.nan,
                    "Accuracy": metrics.get("Accuracy", np.nan),
                    "QWK": metrics.get("QWK", np.nan),
                    "Spearman_Rho": spearman_rho,
                    "Spearman_Valid_Participants": valid_p,
                    "Spearman_Total_Participants": total_p,
                    "Spearman_Coverage_Pct": (
                        100.0 * valid_p / total_p if total_p else np.nan
                    ),
                })

    return pd.DataFrame(rows)


def collect_baseline_rows() -> pd.DataFrame:
    """Recompute Spearman-focused rows from baseline models."""
    train_data, test_data = load_canonical_data("7030", "participant")
    emb_matrix, emb_lookup = tb._load_embeddings()

    baselines: list[tuple[str, Callable[[str], tuple[np.ndarray, np.ndarray, np.ndarray]]]] = [
        ("Participant Mean", lambda d: tb.baseline_participant_mean(train_data, test_data, d)),
        ("Message Mean", lambda d: tb.baseline_message_mean(train_data, test_data, d)),
        ("TF-IDF + LR", lambda d: tb.baseline_tfidf_model(train_data, test_data, d, "lr")),
        ("TF-IDF + RF", lambda d: tb.baseline_tfidf_model(train_data, test_data, d, "rf")),
        ("Demographics LR", lambda d: tb.baseline_demographics_model(train_data, test_data, d, "lr")),
        ("Demographics RF", lambda d: tb.baseline_demographics_model(train_data, test_data, d, "rf")),
        (
            "Embedding + LR",
            lambda d: tb.baseline_embedding_model(train_data, test_data, d, emb_matrix, emb_lookup, "lr"),
        ),
        (
            "Embedding + RF",
            lambda d: tb.baseline_embedding_model(train_data, test_data, d, emb_matrix, emb_lookup, "rf"),
        ),
        (
            "TF-IDF + Demo + LR",
            lambda d: tb.baseline_tfidf_demographics_model(train_data, test_data, d, "lr"),
        ),
        (
            "TF-IDF + Demo + RF",
            lambda d: tb.baseline_tfidf_demographics_model(train_data, test_data, d, "rf"),
        ),
        (
            "Embedding + Demo + LR",
            lambda d: tb.baseline_embedding_demographics_model(
                train_data, test_data, d, emb_matrix, emb_lookup, "lr"
            ),
        ),
        (
            "Embedding + Demo + RF",
            lambda d: tb.baseline_embedding_demographics_model(
                train_data, test_data, d, emb_matrix, emb_lookup, "rf"
            ),
        ),
    ]

    rows = []
    for baseline_name, runner in baselines:
        for domain in DOMAINS:
            gt, pred, response_ids = runner(domain)
            metrics = compute_all_metrics(gt, pred, response_ids)
            spearman_rho, valid_p, total_p = compute_spearman_support(
                gt, pred, response_ids
            )

            rows.append({
                "System_Type": "Baseline",
                "Comparison_Group": "Baseline",
                "Split_Type": "Participant 70/30",
                "Category": "Baseline",
                "Domain": domain.capitalize(),
                "System_Label": baseline_name,
                "Model": np.nan,
                "Method": np.nan,
                "Baseline": baseline_name,
                "Accuracy": metrics.get("Accuracy", np.nan),
                "QWK": metrics.get("QWK", np.nan),
                "Spearman_Rho": spearman_rho,
                "Spearman_Valid_Participants": valid_p,
                "Spearman_Total_Participants": total_p,
                "Spearman_Coverage_Pct": (
                    100.0 * valid_p / total_p if total_p else np.nan
                ),
            })

    return pd.DataFrame(rows)


def build_summary(all_results: pd.DataFrame) -> pd.DataFrame:
    """Create a best-of summary for baseline/generic/personalized by domain."""
    summary_rows = []

    selectors = [
        ("Best Baseline", all_results["Comparison_Group"] == "Baseline"),
        ("Best Generic LLM", all_results["Comparison_Group"] == "Generic LLM"),
        ("Best Personalized LLM", all_results["Comparison_Group"] == "Personalized LLM"),
    ]

    for domain in ["Content", "Coping", "Quitting"]:
        for group_label, mask in selectors:
            sub = all_results[mask & (all_results["Domain"] == domain)].copy()
            if sub.empty:
                continue
            sub = sub.sort_values(
                ["Spearman_Rho", "Spearman_Valid_Participants", "Accuracy"],
                ascending=[False, False, False],
            )
            best = sub.iloc[0]
            summary_rows.append({
                "Domain": domain,
                "Group": group_label,
                "System_Label": best["System_Label"],
                "Category": best["Category"],
                "Split_Type": best["Split_Type"],
                "Model": best["Model"],
                "Method": best["Method"],
                "Baseline": best["Baseline"],
                "Accuracy": best["Accuracy"],
                "QWK": best["QWK"],
                "Spearman_Rho": best["Spearman_Rho"],
                "Spearman_Valid_Participants": int(best["Spearman_Valid_Participants"]),
                "Spearman_Total_Participants": int(best["Spearman_Total_Participants"]),
                "Spearman_Coverage_Pct": best["Spearman_Coverage_Pct"],
            })

    return pd.DataFrame(summary_rows)


def plot_summary(summary_df: pd.DataFrame):
    """Plot best baseline/generic/personalized Spearman by domain."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5), sharey=True)
    domain_order = ["Content", "Coping", "Quitting"]
    group_order = ["Best Baseline", "Best Generic LLM", "Best Personalized LLM"]
    y_min = min(-0.25, float(summary_df["Spearman_Rho"].min()) - 0.1)
    y_max = max(1.05, float(summary_df["Spearman_Rho"].max()) + 0.1)

    for ax, domain in zip(axes, domain_order):
        dom = summary_df[summary_df["Domain"] == domain].copy()
        dom["Group"] = pd.Categorical(dom["Group"], categories=group_order, ordered=True)
        dom = dom.sort_values("Group")
        x = np.arange(len(dom))
        heights = dom["Spearman_Rho"].to_numpy()
        colors = [GROUP_COLORS[g] for g in dom["Group"]]

        bars = ax.bar(
            x,
            heights,
            color=colors,
            edgecolor="black",
            linewidth=1.4,
            width=0.68,
        )

        ax.axhline(0, color="#4D4D4D", linewidth=1.2, linestyle="--")
        ax.set_title(domain, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(
            ["Baseline\n(participant)", "Generic\n(participant)", "Personalized\n(digital twin)"],
            rotation=0,
            fontsize=11,
            fontweight="bold",
        )
        ax.set_ylim(y_min, y_max)

        for bar, (_, row) in zip(bars, dom.iterrows()):
            value = row["Spearman_Rho"]
            valid_n = int(row["Spearman_Valid_Participants"])
            total_n = int(row["Spearman_Total_Participants"])
            label = f"{value:.3f}\n(n={valid_n}/{total_n})"
            offset = 0.03 if value >= 0 else -0.03
            va = "bottom" if value >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                label,
                ha="center",
                va=va,
                fontsize=9.5,
                fontweight="bold",
            )

    axes[0].set_ylabel("Mean Participant Spearman Rho", fontsize=12, fontweight="bold")
    fig.suptitle(
        "Rank-Order Agreement by Domain\nParticipant-Split Baselines/Generic Methods and PP Methods",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    apply_repo_plot_style(fig, axes)
    fig.tight_layout()
    save_figure(fig, figures_path("spearman_rank_comparison"))


def write_notes(summary_df: pd.DataFrame):
    """Write a short markdown note for the response letter."""
    path = figures_path("spearman_rank_notes") + ".md"
    domain_lines = []
    for domain in ["Content", "Coping", "Quitting"]:
        dom = summary_df[summary_df["Domain"] == domain].set_index("Group")
        if dom.empty:
            continue
        baseline = dom.loc["Best Baseline"]
        personalized = dom.loc["Best Personalized LLM"]
        domain_lines.append(
            f"- {domain}: best baseline rho `{baseline['Spearman_Rho']:.3f}` "
            f"({int(baseline['Spearman_Valid_Participants'])}/"
            f"{int(baseline['Spearman_Total_Participants'])} participants), "
            f"best generic rho `{dom.loc['Best Generic LLM', 'Spearman_Rho']:.3f}` "
            f"({int(dom.loc['Best Generic LLM', 'Spearman_Valid_Participants'])}/"
            f"{int(dom.loc['Best Generic LLM', 'Spearman_Total_Participants'])} participants), "
            f"best personalized rho `{personalized['Spearman_Rho']:.3f}` "
            f"({int(personalized['Spearman_Valid_Participants'])}/"
            f"{int(personalized['Spearman_Total_Participants'])} participants)."
        )

    lines = [
        "# Spearman Rank Analysis",
        "",
        "Spearman rho is computed per participant and then averaged across",
        "participants with at least two distinct observed ratings and two",
        "distinct predicted ratings in the relevant held-out canonical split.",
        "",
        "Why this matters:",
        "",
        "- Spearman rho is invariant to within-participant affine rescaling, so it is complementary evidence for concerns about participant-specific rating-scale shifts such as z-scoring.",
        "- It does not replace a full z-score retraining analysis.",
        "- The saved artifacts live on different canonical splits: baselines and generic LLMs use the participant 70/30 split, while personalized digital-twin methods use the digital-twin 70/30 split.",
        "- Because each participant has only three held-out messages, Spearman values are discrete and the number of valid participants can be small for methods that predict tied ratings often.",
        "",
        "Best rows by domain:",
        "",
        *domain_lines,
        "",
        "Artifacts:",
        "",
        "- [spearman_rank_results.csv](spearman_rank_results.csv)",
        "- [spearman_rank_summary.csv](spearman_rank_summary.csv)",
        "- [spearman_rank_comparison.png](spearman_rank_comparison.png)",
        "- [spearman_rank_comparison.pdf](spearman_rank_comparison.pdf)",
    ]

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")
    print(f"  Saved: {path}")


def main():
    print("=" * 70)
    print("Spearman Rank Analysis")
    print("=" * 70)

    print("\n[1/4] Collecting baseline rows ...")
    baseline_df = collect_baseline_rows()

    print("[2/4] Collecting LLM rows ...")
    llm_df = collect_llm_rows()

    all_results = pd.concat([baseline_df, llm_df], ignore_index=True)
    all_results = all_results.sort_values(
        ["Domain", "Comparison_Group", "Spearman_Rho", "Accuracy"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)

    results_path = figures_path("spearman_rank_results") + ".csv"
    all_results.to_csv(results_path, index=False)
    print(f"[3/4] Saved CSV: {results_path}")
    print(f"       Rows: {len(all_results)}")

    summary_df = build_summary(all_results)
    summary_path = figures_path("spearman_rank_summary") + ".csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"       Saved summary: {summary_path}")
    print(summary_df[[
        "Domain", "Group", "Split_Type", "System_Label", "Spearman_Rho",
        "Spearman_Valid_Participants", "Spearman_Total_Participants"
    ]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n[4/4] Creating figure and notes ...")
    plot_summary(summary_df)
    write_notes(summary_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
