"""
Create combined all-method LLM comparison figures on the exact digital-twin overlap.

Methods shown:
  - Zero-shot (all)
  - Zero-shot (select)
  - Few-shot (all)
  - Few-shot (select)
  - PP
  - Hybrid RF+PP

Every bar is computed on the exact same participant-message subset across all
5 models x 6 methods on the canonical 70/30 PP split, and the
supervised LR/RF reference lines are recomputed on that same shared subset.

Outputs:
  - revision/figures/bars_all_methods_<metric>.csv
  - revision/figures/bars_all_methods_<metric>.png/.pdf
  - revision/figures/all_methods_global_common_subset_results.csv
  - revision/figures/all_methods_global_common_subset_coverage.csv
  - revision/figures/all_methods_global_baseline_subset.csv
  - revision/figures/all_methods_global_baseline_best.csv

Usage:
  uv run python analysis-script/qwk_method_comparison.py
  uv run python analysis-script/qwk_method_comparison.py --plot-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from revision_utils import (
    MODEL_CONFIGS,
    COLORS,
    apply_repo_plot_style,
    compute_all_metrics,
    figures_path,
    load_results_aligned,
    save_figure,
)
from filter_duplicates import get_duplicate_signatures


METHOD_ORDER = [
    "Zero-shot (all)",
    "Zero-shot (select)",
    "Few-shot (all)",
    "Few-shot (select)",
    "PP",
    "Hybrid RF+PP",
]
METHOD_FILES = {
    "Zero-shot (all)": "generic_llm_1_zero_shot_dt7030.json",
    "Zero-shot (select)": "generic_llm_2_zero_shot_select_dt7030.json",
    "Few-shot (all)": "generic_llm_3_few_shot_dt7030.json",
    "Few-shot (select)": "generic_llm_4_few_shot_select_dt7030.json",
    "PP": "digital_twin_4_cbtact_7030.json",
    "Hybrid RF+PP": "hybrid",
}
METHOD_LABELS = {
    "Zero-shot (all)": "Zero-shot\n(all)",
    "Zero-shot (select)": "Zero-shot\n(select)",
    "Few-shot (all)": "Few-shot\n(all)",
    "Few-shot (select)": "Few-shot\n(select)",
    "PP": "PP",
    "Hybrid RF+PP": "Hybrid\nRF+PP",
}
MODEL_ORDER = [cfg["display"] for cfg in MODEL_CONFIGS.values()]
DOMAIN_ORDER = ["Content", "Coping", "Quitting"]
METRICS = [
    ("Accuracy", "Accuracy", "accuracy"),
    ("F1", "Macro F1", "f1"),
    ("Kappa", "Cohen's Kappa", "kappa"),
    ("QWK", "Quadratic Weighted Kappa", "qwk"),
    ("Directional Accuracy", "Directional Accuracy", "directional_accuracy"),
    ("Directional Macro-F1", "Directional Macro-F1", "directional_macro_f1"),
    ("Spearman_Rho", "Spearman Rho", "spearman_rho"),
]
BASELINE_COLORS = {
    "LR": COLORS["Logistic Regression"],
    "RF": COLORS["Random Forest"],
}


def _normalize_message(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _build_item_key(row: pd.Series) -> str:
    return "||".join([
        str(row.get("response_id", "")),
        _normalize_message(row.get("input_message", "")),
        str(row.get("ground_truth_content", "NA")),
        str(row.get("ground_truth_design", "NA")),
        str(row.get("ground_truth_coping", "NA")),
        str(row.get("ground_truth_quitting", "NA")),
    ])


def _attach_item_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Item_Key"] = out.apply(_build_item_key, axis=1)
    return out.drop_duplicates(subset=["Item_Key"]).reset_index(drop=True)


def _duplicate_item_keys() -> set[str]:
    keys = set()
    for response_id, message, ratings_tuple in get_duplicate_signatures():
        ratings = dict(ratings_tuple)
        keys.add("||".join([
            str(response_id),
            _normalize_message(message),
            str(ratings.get("content", "NA")),
            str(ratings.get("design", "NA")),
            str(ratings.get("coping", "NA")),
            str(ratings.get("quitting", "NA")),
        ]))
    return keys


def collect_global_common_subset_results() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    dup_keys = _duplicate_item_keys()
    raw_results = {}
    for model_id, cfg in MODEL_CONFIGS.items():
        model_name = cfg["display"]
        for method_name in METHOD_ORDER:
            df = load_results_aligned(model_id, METHOD_FILES[method_name])
            if df is None:
                continue
            keyed = _attach_item_keys(df)
            raw_results[(model_name, method_name)] = keyed[~keyed["Item_Key"].isin(dup_keys)].reset_index(drop=True)

    metric_rows = []
    coverage_rows = []
    common_keys = {}

    for domain_label in DOMAIN_ORDER:
        domain = domain_label.lower()
        series_sets = {}
        for (model_name, method_name), df in raw_results.items():
            gt_col = f"gt_{domain}_num"
            pred_col = f"pred_{domain}_num"
            valid = df[gt_col].notna() & df[pred_col].notna()
            series_sets[(model_name, method_name)] = set(df.loc[valid, "Item_Key"])

        shared = set.intersection(*series_sets.values()) if series_sets else set()
        common_keys[domain_label] = shared

        for (model_name, method_name), df in raw_results.items():
            gt_col = f"gt_{domain}_num"
            pred_col = f"pred_{domain}_num"
            valid = df[gt_col].notna() & df[pred_col].notna()
            aligned = df.loc[valid & df["Item_Key"].isin(shared)].copy()
            metrics = compute_all_metrics(
                aligned[gt_col].astype(int).to_numpy(),
                aligned[pred_col].astype(int).to_numpy(),
                aligned["response_id"].to_numpy(),
            )
            metric_rows.append({
                "Model": model_name,
                "Method": method_name,
                "Domain": domain_label,
                **metrics,
            })
            coverage_rows.append({
                "Model": model_name,
                "Method": method_name,
                "Domain": domain_label,
                "N_Available": int(valid.sum()),
                "N_Common": len(shared),
            })

    return pd.DataFrame(metric_rows), common_keys, pd.DataFrame(coverage_rows)


def load_global_baseline_subset(common_keys: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(figures_path("history_supervised_predictions")).with_suffix(".csv")
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline predictions: {path}")

    pred_df = pd.read_csv(path)
    metric_rows = []
    best_rows = []

    for domain in DOMAIN_ORDER:
        panel = pred_df[
            (pred_df["Domain"] == domain)
            & (pred_df["Item_Key"].isin(common_keys.get(domain, set())))
        ].copy()
        domain_metric_rows = []
        for (feature_set, classifier), sub in panel.groupby(["Feature_Set", "Classifier"]):
            sub = sub.drop_duplicates(subset=["Item_Key"]).reset_index(drop=True)
            metrics = compute_all_metrics(
                sub["Ground_Truth_Num"].astype(int).to_numpy(),
                sub["Predicted_Num"].astype(int).to_numpy(),
                sub["response_id"].to_numpy(),
            )
            row = {
                "Feature_Set": feature_set,
                "Classifier": classifier,
                "Domain": domain,
                **metrics,
            }
            domain_metric_rows.append(row)
            metric_rows.append(row)
        domain_metric_df = pd.DataFrame(domain_metric_rows)
        best_lr_row = {"Domain": domain, "Series": "Best baseline LR"}
        best_rf_row = {"Domain": domain, "Series": "Best baseline RF"}
        for metric_name, _, _ in METRICS:
            for clf in ["LR", "RF"]:
                clf_df = domain_metric_df[domain_metric_df["Classifier"] == clf]
                best = clf_df.sort_values(metric_name, ascending=False).iloc[0]
                target = best_lr_row if clf == "LR" else best_rf_row
                target["Baseline"] = f"{best['Feature_Set']} ({clf})"
                target[metric_name] = float(best[metric_name])
        best_rows.extend([best_lr_row, best_rf_row])

    return pd.DataFrame(metric_rows), pd.DataFrame(best_rows)


def baseline_best_from_metric_df(baseline_metric_df: pd.DataFrame) -> pd.DataFrame:
    best_rows = []
    if baseline_metric_df.empty:
        return pd.DataFrame(best_rows)

    for domain in DOMAIN_ORDER:
        domain_metric_df = baseline_metric_df[baseline_metric_df["Domain"] == domain].copy()
        if domain_metric_df.empty:
            continue
        best_lr_row = {"Domain": domain, "Series": "Best baseline LR"}
        best_rf_row = {"Domain": domain, "Series": "Best baseline RF"}
        for metric_name, _, _ in METRICS:
            if metric_name not in domain_metric_df.columns:
                continue
            for clf in ["LR", "RF"]:
                clf_df = domain_metric_df[domain_metric_df["Classifier"] == clf].copy()
                clf_df = clf_df[clf_df[metric_name].notna()]
                if clf_df.empty:
                    continue
                best = clf_df.sort_values(metric_name, ascending=False).iloc[0]
                target = best_lr_row if clf == "LR" else best_rf_row
                target["Baseline"] = f"{best['Feature_Set']} ({clf})"
                target[metric_name] = float(best[metric_name])
        best_rows.extend([best_lr_row, best_rf_row])

    return pd.DataFrame(best_rows)


def common_counts_from_coverage(coverage_df: pd.DataFrame) -> dict:
    if coverage_df.empty or "Domain" not in coverage_df.columns or "N_Common" not in coverage_df.columns:
        return {}
    return (
        coverage_df.groupby("Domain")["N_Common"]
        .max()
        .astype(int)
        .to_dict()
    )


def metric_limits(metric_name: str, values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return (0.0, 1.0)
    if metric_name == "Spearman_Rho":
        return (
            min(-0.15, float(finite.min()) - 0.05),
            max(0.15, float(finite.max()) + 0.05),
        )
    if metric_name in {"Kappa", "QWK"}:
        return (
            min(-0.05, float(finite.min()) - 0.03),
            min(1.0, float(finite.max()) + 0.08),
        )
    return (0.0, min(1.0, float(finite.max()) + 0.08))


def save_plot_data(
    llm_df: pd.DataFrame,
    baseline_best_df: pd.DataFrame,
    metric_name: str,
    metric_slug: str,
) -> None:
    out = llm_df[["Model", "Method", "Domain", metric_name]].copy()
    out["Series"] = "LLM"
    base = baseline_best_df.rename(columns={"Baseline": "Model"})[["Model", "Series", "Domain", metric_name]].copy()
    base["Method"] = "Baseline reference"
    combined = pd.concat([out, base], ignore_index=True)
    combined.to_csv(
        Path(figures_path(f"bars_all_methods_{metric_slug}")).with_suffix(".csv"),
        index=False,
        float_format="%.4f",
    )


def create_plot(
    llm_df: pd.DataFrame,
    baseline_best_df: pd.DataFrame,
    common_counts: dict,
    metric_name: str,
    metric_label: str,
    metric_slug: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), sharey=(metric_name != "Spearman_Rho"))
    apply_repo_plot_style(fig, axes)

    x = np.arange(len(METHOD_ORDER))
    width = 0.15
    offsets = np.linspace(-2 * width, 2 * width, len(MODEL_ORDER))

    all_values = np.concatenate([
        llm_df[metric_name].to_numpy(dtype=float),
        baseline_best_df[metric_name].to_numpy(dtype=float),
    ])
    y_min, y_max = metric_limits(metric_name, all_values)

    for ax_idx, domain in enumerate(DOMAIN_ORDER):
        ax = axes[ax_idx]
        domain_df = llm_df[llm_df["Domain"] == domain].copy()

        for model_idx, model_name in enumerate(MODEL_ORDER):
            model_df = domain_df[domain_df["Model"] == model_name]
            vals = []
            for method in METHOD_ORDER:
                row = model_df[model_df["Method"] == method]
                vals.append(float(row[metric_name].iloc[0]) if len(row) == 1 else np.nan)
            vals_arr = np.array(vals, dtype=float)
            ax.bar(
                x + offsets[model_idx],
                vals_arr,
                width=width * 0.95,
                color=COLORS[model_name],
                edgecolor="black",
                linewidth=0.9,
                label=model_name if ax_idx == 0 else None,
                zorder=3,
            )

        base_sub = baseline_best_df[baseline_best_df["Domain"] == domain]
        lr_row = base_sub[(base_sub["Series"] == "Best baseline LR") & base_sub[metric_name].notna()]
        rf_row = base_sub[(base_sub["Series"] == "Best baseline RF") & base_sub[metric_name].notna()]
        if not lr_row.empty:
            ax.axhline(
                float(lr_row[metric_name].iloc[0]),
                color=BASELINE_COLORS["LR"],
                linestyle="--",
                linewidth=2.8,
                alpha=1.0,
                label="Best baseline LR" if ax_idx == 0 else None,
                zorder=5,
            )
        if not rf_row.empty:
            ax.axhline(
                float(rf_row[metric_name].iloc[0]),
                color=BASELINE_COLORS["RF"],
                linestyle="-.",
                linewidth=2.8,
                alpha=1.0,
                label="Best baseline RF" if ax_idx == 0 else None,
                zorder=5,
            )

        if metric_name in {"Kappa", "QWK", "Spearman_Rho"}:
            ax.axhline(0, color="#4D4D4D", linestyle="--", linewidth=1.5, alpha=0.7, zorder=2)
        ax.set_title(f"{domain} (common N={common_counts.get(domain, 0)})", fontsize=15, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], fontsize=12, fontweight="bold")
        ax.set_xlabel("Method", fontsize=13, fontweight="bold")
        ax.set_ylim(y_min, y_max)
        ax.tick_params(axis="both", labelsize=12)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        if ax_idx == 0:
            ax.set_ylabel(metric_label, fontsize=13, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        prop={"weight": "bold", "size": 14},
        frameon=False,
    )
    fig.suptitle(
        f"{metric_label} Across Main LLM Prompting Methods on the Canonical 70/30 PP Common Subset",
        fontsize=17,
        fontweight="bold",
        y=1.03,
    )
    fig.subplots_adjust(top=0.82, bottom=0.16, left=0.07, right=0.99, wspace=0.08)
    save_figure(fig, figures_path(f"bars_all_methods_{metric_slug}"))


def parse_args():
    parser = argparse.ArgumentParser(description="Build all-method common-subset diagnostic bars.")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Reload saved revision/figures common-subset CSVs and regenerate bars only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coverage_path = Path(figures_path("all_methods_global_common_subset_coverage")).with_suffix(".csv")
    llm_path = Path(figures_path("all_methods_global_common_subset_results")).with_suffix(".csv")
    baseline_metric_path = Path(figures_path("all_methods_global_baseline_subset")).with_suffix(".csv")
    baseline_best_path = Path(figures_path("all_methods_global_baseline_best")).with_suffix(".csv")

    if args.plot_only:
        required = [coverage_path, llm_path]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing cached source CSV(s): " + ", ".join(missing))
        print("Reloading cached all-method common-subset tables...")
        coverage_df = pd.read_csv(coverage_path)
        llm_df = pd.read_csv(llm_path)
        if baseline_best_path.exists():
            baseline_best_df = pd.read_csv(baseline_best_path)
        elif baseline_metric_path.exists():
            baseline_metric_df = pd.read_csv(baseline_metric_path)
            baseline_best_df = baseline_best_from_metric_df(baseline_metric_df)
            baseline_best_df.to_csv(baseline_best_path, index=False, float_format="%.4f")
        else:
            raise FileNotFoundError(
                f"Missing cached baseline source CSV: {baseline_best_path} "
                f"or {baseline_metric_path}"
            )
    else:
        llm_df, common_keys, coverage_df = collect_global_common_subset_results()
        coverage_df.to_csv(coverage_path, index=False)
        llm_df.to_csv(llm_path, index=False, float_format="%.4f")

        baseline_metric_df, baseline_best_df = load_global_baseline_subset(common_keys)
        baseline_metric_df.to_csv(baseline_metric_path, index=False, float_format="%.4f")
        baseline_best_df.to_csv(baseline_best_path, index=False, float_format="%.4f")

    common_counts = common_counts_from_coverage(coverage_df)
    for metric_name, metric_label, metric_slug in METRICS:
        save_plot_data(llm_df, baseline_best_df, metric_name, metric_slug)
        create_plot(llm_df, baseline_best_df, common_counts, metric_name, metric_label, metric_slug)


if __name__ == "__main__":
    main()
