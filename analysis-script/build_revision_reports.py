#!/usr/bin/env python3
"""
Build a markdown summary for the revision outputs.

The report is written to:
  revision/figures/README.md
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from revision_utils import FIGURES_DIR


def csv_path(name: str) -> str:
    return os.path.join(FIGURES_DIR, f"{name}.csv")


def exists(name: str, suffix: str) -> bool:
    return os.path.exists(os.path.join(FIGURES_DIR, f"{name}.{suffix}"))


def read_csv(name: str) -> pd.DataFrame | None:
    path = csv_path(name)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def table_md(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a GitHub-flavored markdown table."""
    if df is None or df.empty:
        return "_No rows_"

    rendered = df.fillna("NA").astype(str)
    columns = list(rendered.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(row[col] for col in columns) + " |"
        for _, row in rendered.iterrows()
    ]
    return "\n".join([header, divider, *rows])


def bullet_files(entries: list[tuple[str, list[str]]]) -> list[str]:
    lines = []
    for label, files in entries:
        links = []
        for file_name in files:
            if os.path.exists(os.path.join(FIGURES_DIR, file_name)):
                links.append(f"[{file_name}]({file_name})")
        if links:
            lines.append(f"- {label}: " + ", ".join(links))
    return lines


def summarize_class_imbalance() -> list[str]:
    dist = read_csv("class_distribution")
    qwk = read_csv("qwk_results")
    lines = []
    if dist is None or qwk is None:
        return ["- Missing class imbalance outputs."]

    test_summary = (
        dist.assign(Is_Positive=dist["Rating"].isin([4, 5]))
        .groupby("Domain", as_index=False)
        .agg(
            Test_N=("Test_N", "sum"),
            Test_Positive_N=("Is_Positive", lambda s: 0),
        )
    )
    pos = (
        dist[dist["Rating"].isin([4, 5])]
        .groupby("Domain", as_index=False)["Test_N"]
        .sum()
        .rename(columns={"Test_N": "Test_Positive_N"})
    )
    test_summary = test_summary.drop(columns=["Test_Positive_N"]).merge(pos, on="Domain", how="left")
    test_summary["Test_Positive_Pct"] = test_summary["Test_Positive_N"] / test_summary["Test_N"] * 100
    mode_lookup = (
        dist.sort_values(["Domain", "Test_N", "Rating"], ascending=[True, False, False])
        .drop_duplicates("Domain")
        .set_index("Domain")["Rating"]
        .to_dict()
    )
    test_summary["Mode_Test_Rating"] = test_summary["Domain"].map(mode_lookup)

    best_qwk = (
        qwk.sort_values(["Domain", "QWK"], ascending=[True, False])
        .drop_duplicates("Domain")[["Domain", "Model", "Method", "QWK", "Accuracy"]]
        .copy()
    )
    best_qwk["QWK"] = best_qwk["QWK"].map(lambda x: fmt(x, 3))
    best_qwk["Accuracy"] = best_qwk["Accuracy"].map(lambda x: fmt(x, 3))

    split_note = ""
    if "Split" in qwk.columns and qwk["Split"].nunique() == 1:
        split_note = f" All QWK rows use `{qwk['Split'].iloc[0]}` and the same source policy as `../../figures/bars_all_methods_source_table.csv`."

    lines.append("- Cleaned PP 70/30 test-set positive skew remains clear: ratings 4-5 account for "
                 + ", ".join(
                     f"{row.Domain} {row.Test_Positive_Pct:.1f}%"
                     for row in test_summary.itertuples()
                 )
                 + "." + split_note)
    lines.append("- Best quadratic weighted kappa by domain:")
    lines.append(table_md(best_qwk))
    if exists("all_rating_subgroup_histograms", "png"):
        lines.append("- Raw subgroup rating histograms are descriptive full PP train+test context, not a held-out model-comparison metric.")
    lines.extend(bullet_files([
        ("Artifacts", [
            "class_distribution.csv",
            "class_distribution_by_domain.png",
            "class_distribution_by_domain.pdf",
            "all_rating_subgroup_histograms.png",
            "all_rating_subgroup_histograms.pdf",
            "qwk_results.csv",
        ]),
    ]))
    return lines


def summarize_cost() -> list[str]:
    df = read_csv("cost_latency_analysis")
    if df is None or df.empty:
        return ["- Missing cost analysis outputs."]

    cheapest = df.loc[df["Cost_Per_Participant"].idxmin()]
    priciest = df.loc[df["Cost_Per_Participant"].idxmax()]
    top_total = df.sort_values("Total_Cost", ascending=False).head(5).copy()
    top_total["Cost_Per_Participant"] = top_total["Cost_Per_Participant"].map(lambda x: fmt(x, 4))
    top_total["Total_Cost"] = top_total["Total_Cost"].map(lambda x: fmt(x, 3))

    return [
        f"- Cheapest per-participant estimate: {cheapest['Model']} / {cheapest['Method']} "
        f"at ${fmt(cheapest['Cost_Per_Participant'], 4)}.",
        f"- Most expensive per-participant estimate: {priciest['Model']} / {priciest['Method']} "
        f"at ${fmt(priciest['Cost_Per_Participant'], 4)}.",
        "- Highest total-study cost combinations:",
        table_md(top_total[["Model", "Method", "Cost_Per_Participant", "Total_Cost"]]),
        *bullet_files([
            ("Artifacts", [
                "cost_latency_analysis.csv",
                "cost_per_participant.png",
                "cost_per_participant.pdf",
            ]),
        ]),
    ]


def summarize_hybrid() -> list[str]:
    df = read_csv("hybrid_lr_vs_rf_comparison")
    if df is None or df.empty:
        return ["- Missing hybrid LR vs RF outputs."]

    test_df = df[df["Split"] == "Test"].copy()
    winners = []
    for domain in sorted(test_df["Domain"].unique()):
        dom = test_df[test_df["Domain"] == domain].sort_values("Accuracy", ascending=False)
        winner = dom.iloc[0]
        winners.append({
            "Domain": domain,
            "Winner": winner["Model"],
            "Accuracy": fmt(winner["Accuracy"]),
            "F1": fmt(winner["F1"]),
            "QWK": fmt(winner["QWK"]),
        })

    return [
        "- Held-out test comparison between demographic-only Logistic Regression and Random Forest:",
        table_md(pd.DataFrame(winners)),
        *bullet_files([
            ("Artifacts", [
                "hybrid_lr_vs_rf_comparison.csv",
                "hybrid_lr_vs_rf.png",
                "hybrid_lr_vs_rf.pdf",
            ]),
        ]),
    ]


def summarize_leakage() -> list[str]:
    df = read_csv("label_leakage_analysis")
    if df is None or df.empty:
        return ["- Missing label leakage outputs."]

    leak = df[["Check_Type", "N_Overlapping", "Total_Checked", "Pct_Overlap", "Risk_Level"]].copy()
    leak.loc[leak["Risk_Level"].isna() & (leak["Pct_Overlap"] == 0), "Risk_Level"] = "None"
    leak["Pct_Overlap"] = leak["Pct_Overlap"].map(lambda x: fmt(x, 2))
    return [
        "- Leakage checks:",
        table_md(leak),
        *bullet_files([
            ("Artifacts", [
                "label_leakage_analysis.csv",
                "profile_test_similarity_distribution.png",
                "profile_test_similarity_distribution.pdf",
            ]),
        ]),
    ]


def summarize_pairwise() -> list[str]:
    df = read_csv("pairwise_significance_tests")
    if df is None or df.empty:
        return ["- Pairwise significance outputs not yet generated."]

    sig = df[df["Significant"]].copy()
    lines = [
        f"- Pairwise tests completed: {len(df)} total comparisons, {len(sig)} significant after Bonferroni correction."
    ]
    if not sig.empty:
        top = sig.sort_values("p_corrected").head(10).copy()
        top["Diff"] = top["Diff"].map(lambda x: fmt(x, 3))
        top["p_corrected"] = top["p_corrected"].map(lambda x: fmt(x, 4))
        lines.append("- Significant comparisons with the smallest corrected p-values:")
        lines.append(table_md(top[["Domain", "Metric", "Model_A", "Model_B", "Diff", "p_corrected"]]))

    lines.extend(bullet_files([
        ("Artifacts", [
            "pairwise_significance_tests.csv",
            "pairwise_significance_heatmap.png",
            "pairwise_significance_heatmap.pdf",
        ]),
    ]))
    return lines


def summarize_accuracy_ci() -> list[str]:
    df = read_csv("accuracy_confidence_intervals")
    if df is None or df.empty:
        return ["- Accuracy confidence-interval artifact not yet generated."]

    preview = (
        df.sort_values(["Domain", "Method", "Accuracy"], ascending=[True, True, False])
        .groupby(["Domain", "Method"], as_index=False)
        .first()[[
            "Domain", "Method", "Model", "Accuracy",
            "Accuracy_CI_Lower", "Accuracy_CI_Upper"
        ]]
        .copy()
    )
    for col in ["Accuracy", "Accuracy_CI_Lower", "Accuracy_CI_Upper"]:
        preview[col] = preview[col].map(lambda x: fmt(x))

    boot_n = int(df["Bootstrap_N"].iloc[0]) if "Bootstrap_N" in df.columns else None
    split_note = ""
    if "Split" in df.columns and df["Split"].nunique() == 1:
        split_note = f" on `{df['Split'].iloc[0]}`"

    return [
        f"- Revision-folder accuracy CI artifact generated with {boot_n} bootstrap resamples per row{split_note}.",
        "- Highest-accuracy row within each domain/method bucket:",
        table_md(preview),
        *bullet_files([
            ("Artifacts", [
                "accuracy_confidence_intervals.csv",
                "accuracy_confidence_intervals.png",
                "accuracy_confidence_intervals.pdf",
                "accuracy_confidence_intervals.md",
            ]),
        ]),
    ]


def summarize_cbt_act() -> list[str]:
    df = read_csv("cbt_act_comparison")
    if df is None or df.empty:
        return ["- CBT vs ACT outputs not yet generated."]

    acc = df[df["Metric"] == "Accuracy"].copy()
    summary = (
        acc.groupby(["Domain", "Category"], as_index=False)["Value"]
        .mean()
        .pivot(index="Domain", columns="Category", values="Value")
        .reset_index()
    )
    if "ACT" in summary.columns and "CBT" in summary.columns:
        summary["ACT_minus_CBT"] = summary["ACT"] - summary["CBT"]
    for col in [c for c in summary.columns if c != "Domain"]:
        summary[col] = summary[col].map(lambda x: fmt(x))

    return [
        "- Mean accuracy across selected model/method combinations by therapy family:",
        table_md(summary),
        *bullet_files([
            ("Artifacts", [
                "cbt_act_comparison.csv",
                "cbt_vs_act_performance.png",
                "cbt_vs_act_performance.pdf",
            ]),
        ]),
    ]


def summarize_selection_quality() -> list[str]:
    df = read_csv("llm_selection_quality")
    if df is None or df.empty:
        lines = [
            "- Original Figure 4 selection-quality artifacts are kept in the manuscript figure directory:",
            "  [../../figures/llm_selection_quality.csv](../../figures/llm_selection_quality.csv), "
            "  [../../figures/llm_selection_quality.png](../../figures/llm_selection_quality.png), and "
            "  [../../figures/llm_selection_quality.pdf](../../figures/llm_selection_quality.pdf).",
        ]
        if exists("message_selection_methods_k7", "csv"):
            lines.extend([
                "- The strict supervised/LLM gain-over-random benchmark uses the fixed `Demo+History` feature block across all domains:",
                "  [message_selection_gain.png](message_selection_gain.png), "
                "  [message_selection_gain.pdf](message_selection_gain.pdf), and "
                "  [message_selection_methods_k7.csv](message_selection_methods_k7.csv).",
            ])
        return lines

    preview = (
        df.sort_values(["Domain", "K", "Model"])
        .groupby(["Domain", "K"], as_index=False)
        .first()[[
            "Domain",
            "K",
            "Human Oracle (Human Rating)",
            "Human Oracle SE",
            "Random Mean",
            "Random SE",
        ]]
        .head(9)
        .copy()
    )
    for col in [c for c in preview.columns if c not in ["Domain", "K"]]:
        preview[col] = preview[col].map(lambda x: fmt(x, 3))

    boot_n = int(df["Bootstrap_N"].iloc[0]) if "Bootstrap_N" in df.columns else None
    n_messages = int(df["N Messages"].iloc[0]) if "N Messages" in df.columns else None

    lines = [
        f"- Selection-quality Figure 4 artifact refreshed with uncertainty for random, human oracle, and LLM curves ({boot_n} bootstrap resamples; {n_messages} messages per domain).",
        "- Preview of oracle vs random summary rows:",
        table_md(preview),
        *bullet_files([
            ("Artifacts", [
                "llm_selection_quality.csv",
                "llm_selection_quality.png",
                "llm_selection_quality.pdf",
            ]),
        ]),
    ]
    if exists("message_selection_methods_k7", "csv"):
        lines.extend([
            "- Strict supporting method-level selection benchmark uses the fixed `Demo+History` feature block across all domains:",
            "  [message_selection_gain.png](message_selection_gain.png), "
            "  [message_selection_gain.pdf](message_selection_gain.pdf), and "
            "  [message_selection_methods_k7.csv](message_selection_methods_k7.csv).",
        ])
    return lines


def summarize_demographics() -> list[str]:
    df = read_csv("demographic_subgroup_results")
    if df is None or df.empty:
        return ["- Missing demographic subgroup outputs."]

    gap_rows = []
    for subgroup_type in ["Race", "Gender", "Age"]:
        sub = df[df["Subgroup_Type"] == subgroup_type].copy()
        if sub.empty:
            continue
        grouped = sub.groupby(["Domain", "Subgroup_Type"])["Accuracy"].agg(["min", "max"]).reset_index()
        grouped["Gap"] = grouped["max"] - grouped["min"]
        for row in grouped.itertuples():
            gap_rows.append({
                "Domain": row.Domain,
                "Subgroup_Type": row.Subgroup_Type,
                "Gap": fmt(row.Gap),
            })

    flagged = int(df["Small_N_Flag"].sum()) if "Small_N_Flag" in df.columns else 0
    return [
        f"- Demographic subgroup analysis generated {len(df)} rows; {flagged} entries are flagged as n < 20.",
        "- Accuracy gaps across subgroup families:",
        table_md(pd.DataFrame(gap_rows)) if gap_rows else "- No subgroup gap summary available.",
        *bullet_files([
            ("Artifacts", [
                "demographic_subgroup_results.csv",
                "demographic_subgroup_race.png",
                "demographic_subgroup_race.pdf",
                "demographic_subgroup_gender.png",
                "demographic_subgroup_gender.pdf",
            ]),
        ]),
    ]


def summarize_text_baselines() -> list[str]:
    df = read_csv("text_baseline_results")
    qwk = read_csv("qwk_results")
    if df is None or df.empty:
        return ["- Missing text baseline outputs."]

    best_baselines = (
        df.sort_values(["Domain", "Accuracy"], ascending=[True, False])
        .drop_duplicates("Domain")[["Domain", "Baseline", "Accuracy", "F1", "QWK"]]
        .copy()
    )
    best_baselines["Accuracy"] = best_baselines["Accuracy"].map(lambda x: fmt(x))
    best_baselines["F1"] = best_baselines["F1"].map(lambda x: fmt(x))
    best_baselines["QWK"] = best_baselines["QWK"].map(lambda x: fmt(x))

    lines = [
        "- Best text-aware baseline per domain:",
        table_md(best_baselines),
    ]

    if qwk is not None and not qwk.empty:
        best_llm = (
            qwk[qwk["Method"].isin(["PP", "Hybrid RF+PP"])]
            .sort_values(["Domain", "Accuracy"], ascending=[True, False])
            .drop_duplicates("Domain")[["Domain", "Model", "Method", "Accuracy", "QWK"]]
            .copy()
        )
        best_llm["Accuracy"] = best_llm["Accuracy"].map(lambda x: fmt(x))
        best_llm["QWK"] = best_llm["QWK"].map(lambda x: fmt(x))
        lines.append("- Best personalized LLM result per domain for context:")
        lines.append(table_md(best_llm))

    lines.extend(bullet_files([
        ("Artifacts", [
            "text_baseline_results.csv",
            "all_baselines_comparison.png",
            "all_baselines_comparison.pdf",
            "baseline_feature_ablation.png",
            "baseline_feature_ablation.pdf",
        ]),
    ]))
    return lines


def summarize_spearman() -> list[str]:
    summary = read_csv("spearman_rank_summary")
    if summary is None or summary.empty:
        return [
            "- The old standalone Spearman rank artifacts were removed because they mixed participant-split and PP-split rows.",
            "- Use the current strict diagnostics instead: [progress_summary/ordinal_qwk_spearman_snapshot.png](progress_summary/ordinal_qwk_spearman_snapshot.png), [progress_summary/table_spearman_rho.csv](progress_summary/table_spearman_rho.csv), and [progress_summary/table_spearman_rho.md](progress_summary/table_spearman_rho.md).",
        ]

    display = summary[[
        "Domain",
        "Group",
        "Split_Type",
        "System_Label",
        "Spearman_Rho",
        "Spearman_Valid_Participants",
        "Spearman_Total_Participants",
    ]].copy()
    display["Spearman_Rho"] = display["Spearman_Rho"].map(lambda x: fmt(x))

    return [
        "- Per-participant Spearman rho provides a rank-based complement to the absolute-rating metrics.",
        "- Baseline/generic rows come from the participant 70/30 split; personalized rows come from the PP 70/30 split.",
        "- Best row in each comparison bucket by domain:",
        table_md(display),
        *bullet_files([
            ("Artifacts", [
                "spearman_rank_results.csv",
                "spearman_rank_summary.csv",
                "spearman_rank_comparison.png",
                "spearman_rank_comparison.pdf",
                "spearman_rank_notes.md",
            ]),
        ]),
    ]


def summarize_figure_redesign() -> list[str]:
    df = read_csv("figure_redesign_results")
    if df is None or df.empty:
        return [
            "- The old generic/personalized redesign grids were removed because they conflicted with the original Figure 1-4 manuscript architecture.",
            "- Current Figure 2 bars are kept in [../../figures](../../figures), with source audit [../../figures/bars_all_methods_source_audit.csv](../../figures/bars_all_methods_source_audit.csv).",
            "- Current Figure 3 score-distribution panels are kept in the manuscript figure directory: [../../figures/figure3_score_distributions_content.png](../../figures/figure3_score_distributions_content.png), [../../figures/figure3_score_distributions_coping.png](../../figures/figure3_score_distributions_coping.png), and [../../figures/figure3_score_distributions_quitting.png](../../figures/figure3_score_distributions_quitting.png).",
        ]

    generic_pngs = sorted(
        file_name for file_name in os.listdir(FIGURES_DIR)
        if file_name.startswith("bars_generic_methods_") and file_name.endswith(".png")
    )
    personalized_pngs = sorted(
        file_name for file_name in os.listdir(FIGURES_DIR)
        if file_name.startswith("bars_personalized_methods_") and file_name.endswith(".png")
    )
    figure3_pngs = sorted(
        file_name for file_name in os.listdir(FIGURES_DIR)
        if file_name.startswith("figure3_score_distributions_") and file_name.endswith(".png")
    )

    lines = [
        f"- Redesign source table saved with {len(df)} rows.",
        f"- Generated {len(generic_pngs)} generic-method figures, {len(personalized_pngs)} personalized-method figures, and {len(figure3_pngs)} enlarged Figure 3 domain panels.",
    ]
    lines.extend(bullet_files([
        ("Artifacts", ["figure_redesign_results.csv"]),
    ]))
    for bucket, files in [
        ("Generic method figures", generic_pngs),
        ("Personalized method figures", personalized_pngs),
        ("Large Figure 3 panels", figure3_pngs),
    ]:
        if files:
            lines.append(f"- {bucket}: " + ", ".join(f"[{file_name}]({file_name})" for file_name in files))
    return lines


def summarize_ablation() -> list[str]:
    df = read_csv("ablation_results")
    if df is None or df.empty:
        if exists("ablation_status", "md"):
            return [
                "- No real ablation comparison outputs are present under the no-mock-data policy.",
                "- Status note: [ablation_status.md](ablation_status.md)",
            ]
        return ["- Ablation comparison outputs are not present. The prompts are prepared, but no observed message-only/profile-only/history-only runs exist in the repo yet."]

    source_counts = (
        df.groupby("Data_Source", as_index=False)
        .size()
        .rename(columns={"size": "Rows"})
        if "Data_Source" in df.columns
        else pd.DataFrame([{"Data_Source": "unknown", "Rows": len(df)}])
    )

    lines = [
        "- Ablation status:",
        table_md(source_counts),
    ]

    observed_conditions = set(df["Condition"].unique()) if "Condition" in df.columns else set()
    comparison_conditions = observed_conditions - {"full-PP"}
    if not comparison_conditions:
        lines.append("- No observed `message-only`, `profile-only`, or `history-only` outputs exist in this repo, so no no-mock ablation comparison figure is produced.")
        lines.append("- The observed artifact here is only the full PP reference rows saved in `ablation_results.csv`.")
        if exists("ablation_status", "md"):
            lines.append("- Status note: [ablation_status.md](ablation_status.md)")

    lines.extend(bullet_files([
        ("Artifacts", [
            "ablation_results.csv",
            "ablation_study.png",
            "ablation_study.pdf",
        ]),
    ]))
    return lines


def build_report() -> str:
    sections = [
        ("Class Imbalance and QWK", summarize_class_imbalance()),
        ("Cost Analysis", summarize_cost()),
        ("Hybrid LR vs RF", summarize_hybrid()),
        ("Label Leakage", summarize_leakage()),
        ("Pairwise Significance", summarize_pairwise()),
        ("Accuracy Confidence Intervals", summarize_accuracy_ci()),
        ("CBT vs ACT", summarize_cbt_act()),
        ("Selection Quality", summarize_selection_quality()),
        ("Demographic Subgroups", summarize_demographics()),
        ("Text Baselines", summarize_text_baselines()),
        ("Spearman Rank Summary", summarize_spearman()),
        ("Figure Redesign", summarize_figure_redesign()),
        ("Ablation Status", summarize_ablation()),
    ]

    lines = [
        "# Revision Results",
        "",
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.",
        "",
        "All figures in this directory are saved as 400 dpi PNG plus matching PDF.",
        "Current model-comparison artifacts use the cleaned canonical PP 70/30 source unless a section explicitly says it is descriptive full-data context or a separate sensitivity analysis.",
        "Missing split-specific methods are omitted rather than backfilled from another split.",
        "",
    ]

    for title, section_lines in sections:
        lines.append(f"## {title}")
        lines.append("")
        lines.extend(section_lines)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main():
    report_path = os.path.join(FIGURES_DIR, "README.md")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(build_report())
    print(f"Saved markdown summary: {report_path}")


if __name__ == "__main__":
    main()
