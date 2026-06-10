#!/usr/bin/env python3
"""
Figure redesign outputs for the revision.

Addresses reviewer concerns that the original multi-panel figures were too dense
to read. This script splits the old "all methods" bar charts into:

1. Generic LLM methods only
2. Personalized methods only (PP + Hybrid, with best supervised
   baseline reference)

It also creates larger score-distribution panels for the personalized and
supervised results, split by domain.

Outputs:
  - revision/figures/figure_redesign_results.csv
  - revision/figures/bars_generic_methods_<metric>.png/.pdf
  - revision/figures/bars_personalized_methods_<metric>.png/.pdf
  - revision/figures/figure3_score_distributions_<domain>.png/.pdf
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from revision_utils import (
    collect_all_results,
    load_results_aligned,
    MODEL_CONFIGS,
    DOMAINS,
    COLORS,
    apply_repo_plot_style,
    compute_all_metrics,
    figures_path,
    save_figure,
)
from filter_duplicates import get_duplicate_signatures


GENERIC_METHODS = [
    "Zero-shot (all)",
    "Zero-shot (select)",
    "Few-shot (all)",
    "Few-shot (select)",
    "Zero-shot (w/ prob)",
]
GENERIC_METHOD_FILES = {
    "Zero-shot (all)": "generic_llm_1_zero_shot.json",
    "Zero-shot (select)": "generic_llm_2_zero_shot_select.json",
    "Few-shot (all)": "generic_llm_3_few_shot.json",
    "Few-shot (select)": "generic_llm_4_few_shot_select.json",
    "Zero-shot (w/ prob)": "generic_llm_5_continuous.json",
}

PERSONALIZED_METHODS = [
    "PP",
    "Hybrid RF+PP",
]
PERSONALIZED_METHOD_FILES = {
    "PP": "digital_twin_4_cbtact_7030.json",
    "Hybrid RF+PP": "hybrid",
}

METHOD_LABELS = {
    "Zero-shot (all)": "Zero-shot\n(all)",
    "Zero-shot (select)": "Zero-shot\n(select)",
    "Few-shot (all)": "Few-shot\n(all)",
    "Few-shot (select)": "Few-shot\n(select)",
    "Zero-shot (w/ prob)": "Prob.\nratings",
    "PP": "PP",
    "Hybrid RF+PP": "Hybrid\nRF+PP",
}

METRICS = [
    ("Accuracy", "Accuracy", "accuracy"),
    ("F1", "Macro F1", "f1"),
    ("Kappa", "Cohen's kappa", "kappa"),
    ("QWK", "Quadratic Weighted Kappa", "qwk"),
    ("Directional Accuracy", "Directional Accuracy", "directional_accuracy"),
    ("Directional Macro-F1", "Directional Macro-F1", "directional_macro_f1"),
    ("Spearman_Rho", "Spearman rho", "spearman_rho"),
]
RATING_VALUES = [1, 2, 3, 4, 5]

PERSONALIZED_COLORS = {
    "PP": "#0173B2",
    "Hybrid RF+PP": "#DE8F05",
}

BASELINE_COLORS = {
    "LR": COLORS["Logistic Regression"],
    "RF": COLORS["Random Forest"],
}

REFERENCE_GRAY = "#4D4D4D"
PREFERRED_SUPERVISED_FEATURE = "Demographics + History + Message Embedding"

FIGURE3_DISTRIBUTION_ROWS = [
    {
        "label": "Zero-shot (select)",
        "method": "Zero-shot (select)",
        "kind": "llm",
        "method_file": "generic_llm_2_zero_shot_select_dt7030.json",
    },
    {
        "label": "Supervised",
        "method": "Supervised",
        "kind": "supervised_group",
        "predictors": [
            {
                "classifier": "RF",
                "feature_set": "Demographics",
                "series_label": "RF (demo)",
                "color": COLORS["Random Forest"],
            },
            {
                "classifier": "LR",
                "feature_set": PREFERRED_SUPERVISED_FEATURE,
                "series_label": "LR (full)",
                "color": COLORS["Logistic Regression"],
            },
        ],
    },
    {
        "label": "Few-shot (select)",
        "method": "Few-shot (select)",
        "kind": "llm",
        "method_file": "generic_llm_4_few_shot_select_dt7030.json",
    },
    {
        "label": "PP",
        "method": "PP",
        "kind": "llm",
        "method_file": "digital_twin_4_cbtact_7030.json",
    },
]


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


def collect_personalized_common_subset_results() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Compute personalized metrics on the exact common item subset across all model-method pairs."""
    dup_keys = _duplicate_item_keys()
    raw_results = {}
    for model_id, cfg in MODEL_CONFIGS.items():
        model_name = cfg["display"]
        for method_name, method_file in PERSONALIZED_METHOD_FILES.items():
            df = load_results_aligned(model_id, method_file)
            if df is None:
                continue
            keyed = _attach_item_keys(df)
            raw_results[(model_name, method_name)] = keyed[~keyed["Item_Key"].isin(dup_keys)].reset_index(drop=True)

    metric_rows = []
    coverage_rows = []
    common_keys = {}

    for domain_label in ["Content", "Coping", "Quitting"]:
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


def load_baseline_reference(common_keys: dict) -> tuple[dict, pd.DataFrame]:
    """Load best history-aware supervised LR and RF baselines on the same common item subset."""
    path = figures_path("history_supervised_predictions") + ".csv"
    if not os.path.exists(path):
        return {}, pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        return {}, pd.DataFrame()

    refs = {}
    metric_rows = []
    for domain in ["Content", "Coping", "Quitting"]:
        sub = df[
            (df["Domain"] == domain)
            & (df["Item_Key"].isin(common_keys.get(domain, set())))
        ].copy()
        if sub.empty:
            continue
        refs[domain] = {}
        domain_rows = []
        for (feature_set, classifier), panel in sub.groupby(["Feature_Set", "Classifier"]):
            metrics = compute_all_metrics(
                panel["Ground_Truth_Num"].astype(int).to_numpy(),
                panel["Predicted_Num"].astype(int).to_numpy(),
                panel["response_id"].to_numpy(),
            )
            row = {
                "Feature_Set": feature_set,
                "Classifier": classifier,
                "Domain": domain,
                **metrics,
            }
            domain_rows.append(row)
            metric_rows.append(row)
        domain_metrics = pd.DataFrame(domain_rows)
        for metric_name, _, _ in METRICS:
            best_lr = domain_metrics[domain_metrics["Classifier"] == "LR"].sort_values(metric_name, ascending=False).iloc[0]
            best_rf = domain_metrics[domain_metrics["Classifier"] == "RF"].sort_values(metric_name, ascending=False).iloc[0]
            refs[domain][f"{metric_name}_lr_value"] = float(best_lr[metric_name])
            refs[domain][f"{metric_name}_rf_value"] = float(best_rf[metric_name])
            refs[domain][f"{metric_name}_lr_label"] = f"{best_lr['Feature_Set']} (LR)"
            refs[domain][f"{metric_name}_rf_label"] = f"{best_rf['Feature_Set']} (RF)"
    return refs, pd.DataFrame(metric_rows)


def metric_limits(metric_name: str, values: np.ndarray) -> tuple[float, float]:
    """Consistent axis limits for each metric."""
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (0.0, 1.0)

    if metric_name == "Spearman_Rho":
        low = min(-0.4, float(values.min()) - 0.05)
        high = max(0.5, float(values.max()) + 0.05)
        return (low, high)

    high = max(0.45, float(values.max()) * 1.15)
    return (0.0, min(high, 1.0))


def collect_generic_common_subset_results() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Compute generic-method metrics on the exact common item subset across all model-method pairs."""
    raw_results = {}
    for model_id, cfg in MODEL_CONFIGS.items():
        model_name = cfg["display"]
        for method_name, method_file in GENERIC_METHOD_FILES.items():
            df = load_results_aligned(model_id, method_file)
            if df is None:
                continue
            raw_results[(model_name, method_name)] = _attach_item_keys(df)

    metric_rows = []
    coverage_rows = []
    common_keys = {}

    for domain_label in ["Content", "Coping", "Quitting"]:
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


def create_generic_method_bars(results_df: pd.DataFrame, common_counts: dict) -> list[str]:
    """Create split-out replacement for the overloaded Figure 2 generic methods."""
    generated = []
    model_order = [cfg["display"] for cfg in MODEL_CONFIGS.values()]

    for metric_name, metric_label, metric_slug in METRICS:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), sharey=(metric_name != "Spearman_Rho"))
        all_values = []

        for ax_idx, domain in enumerate(["Content", "Coping", "Quitting"]):
            ax = axes[ax_idx]
            subset = results_df[
                (results_df["Domain"] == domain)
                & (results_df["Method"].isin(GENERIC_METHODS))
            ].copy()

            method_order = [m for m in GENERIC_METHODS if m in subset["Method"].unique()]
            if subset.empty or not method_order:
                ax.set_visible(False)
                continue

            x = np.arange(len(method_order))
            width = 0.15

            for model_idx, model_name in enumerate(model_order):
                model_subset = subset[subset["Model"] == model_name]
                vals = []
                for method_name in method_order:
                    row = model_subset[model_subset["Method"] == method_name]
                    val = row[metric_name].iloc[0] if len(row) == 1 else np.nan
                    vals.append(val)
                vals_arr = np.array(vals, dtype=float)
                all_values.append(vals_arr)

                offset = (model_idx - (len(model_order) - 1) / 2) * width
                ax.bar(
                    x + offset,
                    vals_arr,
                    width=width * 0.95,
                    color=COLORS.get(model_name, "#999999"),
                    edgecolor="black",
                    linewidth=0.9,
                    label=model_name if ax_idx == 0 else None,
                )

            ax.set_title(
                f"{domain} (common N={common_counts.get(domain, 0)})",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xticks(x)
            ax.set_xticklabels([METHOD_LABELS[m] for m in method_order], fontsize=9, fontweight="bold")
            if ax_idx == 0:
                ax.set_ylabel(metric_label, fontsize=12, fontweight="bold")

        if all_values:
            y_min, y_max = metric_limits(metric_name, np.concatenate(all_values))
            for ax in axes:
                if ax.get_visible():
                    ax.set_ylim(y_min, y_max)
                    ax.set_xlabel("Method", fontsize=12, fontweight="bold")

        apply_repo_plot_style(fig, axes)
        for ax in axes:
            if ax is None or not ax.get_visible():
                continue
            ax.tick_params(axis="both", labelsize=13, width=1.8,
                           length=6, direction="out")
            ax.xaxis.label.set_fontsize(15)
            ax.yaxis.label.set_fontsize(15)
            ax.title.set_fontsize(16)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight("bold")
        legend = fig.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=5,
            fontsize=10,
            frameon=True,
        )
        for text in legend.get_texts():
            text.set_fontweight("bold")

        fig.suptitle(
            f"Generic LLM Methods on the Common Held-Out Participant Subset: {metric_label}",
            fontsize=15,
            fontweight="bold",
            y=1.10,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        output_stem = figures_path(f"bars_generic_methods_{metric_slug}")
        save_figure(fig, output_stem)
        generated.append(os.path.basename(output_stem))

    return generated


def create_personalized_method_bars(
    results_df: pd.DataFrame,
    baseline_refs: dict,
    common_counts: dict,
) -> list[str]:
    """Create cleaner personalized-method figures with best supervised reference lines."""
    generated = []
    model_order = [cfg["display"] for cfg in MODEL_CONFIGS.values()]

    for metric_name, metric_label, metric_slug in METRICS:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), sharey=(metric_name != "Spearman_Rho"))
        plotted_values = []

        for ax_idx, domain in enumerate(["Content", "Coping", "Quitting"]):
            ax = axes[ax_idx]
            subset = results_df[
                (results_df["Domain"] == domain)
                & (results_df["Method"].isin(PERSONALIZED_METHODS))
            ].copy()

            x = np.arange(len(model_order))
            width = 0.32

            for method_idx, method_name in enumerate(PERSONALIZED_METHODS):
                vals = []
                for model_name in model_order:
                    row = subset[
                        (subset["Model"] == model_name)
                        & (subset["Method"] == method_name)
                    ]
                    val = row[metric_name].iloc[0] if len(row) == 1 else np.nan
                    vals.append(val)
                vals_arr = np.array(vals, dtype=float)
                plotted_values.append(vals_arr)

                offset = (method_idx - 0.5) * width
                bars = ax.bar(
                    x + offset,
                    vals_arr,
                    width=width * 0.95,
                    color=PERSONALIZED_COLORS[method_name],
                    edgecolor="black",
                    linewidth=0.9,
                    label=method_name if ax_idx == 0 else None,
                )

                for bar, value in zip(bars, vals_arr):
                    if np.isfinite(value):
                        va = "bottom" if metric_name != "Spearman_Rho" or value >= 0 else "top"
                        delta = 0.01 if metric_name != "Spearman_Rho" or value >= 0 else -0.01
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            value + delta,
                            f"{value:.2f}",
                            ha="center",
                            va=va,
                            fontsize=7,
                        )

            ref = baseline_refs.get(domain, {})
            lr_value = ref.get(f"{metric_name}_lr_value", np.nan)
            rf_value = ref.get(f"{metric_name}_rf_value", np.nan)
            if np.isfinite(lr_value):
                ax.axhline(
                    lr_value,
                    color=BASELINE_COLORS["LR"],
                    linestyle="--",
                    linewidth=2.8,
                    alpha=1.0,
                    zorder=5,
                )
                plotted_values.append(np.array([lr_value], dtype=float))
            if np.isfinite(rf_value):
                ax.axhline(
                    rf_value,
                    color=BASELINE_COLORS["RF"],
                    linestyle="-.",
                    linewidth=2.8,
                    alpha=1.0,
                    zorder=5,
                )
                plotted_values.append(np.array([rf_value], dtype=float))

            ax.set_title(
                f"{domain} (common N={common_counts.get(domain, 0)})",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(model_order, rotation=25, ha="right", fontsize=9, fontweight="bold")
            if ax_idx == 0:
                ax.set_ylabel(metric_label, fontsize=12, fontweight="bold")

        if plotted_values:
            y_min, y_max = metric_limits(metric_name, np.concatenate(plotted_values))
            for ax in axes:
                ax.set_ylim(y_min, y_max)
                ax.set_xlabel("Model", fontsize=12, fontweight="bold")

        apply_repo_plot_style(fig, axes)
        legend_handles = [
            Patch(facecolor=PERSONALIZED_COLORS["PP"], label="PP"),
            Patch(facecolor=PERSONALIZED_COLORS["Hybrid RF+PP"], label="Hybrid RF+PP"),
            Line2D([0], [0], color=BASELINE_COLORS["LR"], linestyle="--",
                   linewidth=2.8, label="Best supervised LR"),
            Line2D([0], [0], color=BASELINE_COLORS["RF"], linestyle="-.",
                   linewidth=2.8, label="Best supervised RF"),
        ]
        legend = fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=5,
            fontsize=10,
            frameon=True,
        )
        for text in legend.get_texts():
            text.set_fontweight("bold")

        fig.suptitle(
            f"PP, Hybrid, and ML Baselines on the Common Held-Out 70/30 Subset: {metric_label}",
            fontsize=15,
            fontweight="bold",
            y=1.10,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        output_stem = figures_path(f"bars_personalized_methods_{metric_slug}")
        save_figure(fig, output_stem)
        generated.append(os.path.basename(output_stem))

    return generated


def _rating_percentages(values: np.ndarray) -> np.ndarray:
    """Return percent mass on the 1-5 rating scale."""
    arr = pd.Series(values).dropna().astype(int)
    arr = arr[(arr >= 1) & (arr <= 5)]
    if arr.empty:
        return np.zeros(5)
    counts = np.bincount(arr.to_numpy(), minlength=6)[1:6].astype(float)
    return counts / counts.sum() * 100


def _append_score_distribution_records(
    records: list[dict],
    domain: str,
    display_order: int,
    row_label: str,
    method: str,
    series_type: str,
    series_label: str,
    values: np.ndarray,
    n: int,
    color: str,
) -> None:
    percentages = _rating_percentages(values)
    for rating, pct in zip(RATING_VALUES, percentages):
        records.append({
            "Domain": domain.capitalize(),
            "Display_Order": display_order,
            "Row_Label": row_label,
            "Label": row_label,
            "Method": method,
            "Series_Type": series_type,
            "Series_Label": series_label,
            "Series": series_label,
            "Rating": rating,
            "Percent": float(pct),
            "N": int(n),
            "Color": color,
        })


def build_score_distribution_source_table() -> pd.DataFrame:
    """Compute and saveable source table for Figure 3 distribution panels."""
    tables = []
    for domain in DOMAINS:
        records = _method_distribution_rows(domain)
        if records:
            tables.append(pd.DataFrame(records))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _distribution_plot_rows_from_table(domain_df: pd.DataFrame) -> list[dict]:
    """Rehydrate Figure 3 plotting rows from saved distribution percentages."""
    rows = []
    for row_order in sorted(domain_df["Display_Order"].dropna().astype(int).unique()):
        panel = domain_df[domain_df["Display_Order"].astype(int) == row_order].copy()
        if panel.empty:
            continue
        first = panel.iloc[0]
        if "Series_Type" not in panel.columns:
            panel["Series_Type"] = panel["Series"]
            panel["Series_Label"] = panel["Series"]

        human = (
            panel[panel["Series_Type"] == "Human reference"]
            .set_index("Rating")
            .reindex(RATING_VALUES)["Percent"]
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        pred_series = []
        pred_panel = panel[panel["Series_Type"] == "Prediction"].copy()
        for series_label in pred_panel["Series_Label"].dropna().drop_duplicates():
            series_df = pred_panel[pred_panel["Series_Label"] == series_label].copy()
            pred = (
                series_df
                .set_index("Rating")
                .reindex(RATING_VALUES)["Percent"]
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            pred_series.append({
                "label": str(series_label),
                "color": str(series_df["Color"].iloc[0]),
                "pred_pct": pred,
                "n": int(series_df["N"].iloc[0]),
            })
        rows.append({
            "label": first.get("Row_Label", first["Label"]),
            "method": first["Method"],
            "human_pct": human,
            "human_n": int(
                panel[panel["Series_Type"] == "Human reference"]["N"].iloc[0]
            ) if not panel[panel["Series_Type"] == "Human reference"].empty else 0,
            "pred_series": pred_series,
        })
    return rows


def _method_distribution_rows(domain: str) -> list[dict]:
    """Build real-data distribution records for Figure 3."""
    dup_keys = _duplicate_item_keys()
    records = []

    # Use one complete cleaned DT run for the top human-rating reference row.
    reference_df = None
    for model_id in MODEL_CONFIGS:
        candidate = load_results_aligned(model_id, "digital_twin_4_cbtact_7030.json")
        if candidate is not None:
            keyed = _attach_item_keys(candidate)
            reference_df = keyed[~keyed["Item_Key"].isin(dup_keys)].reset_index(drop=True)
            break

    gt_col = f"gt_{domain}_num"
    pred_col = f"pred_{domain}_num"
    if reference_df is not None and gt_col in reference_df.columns:
        valid = reference_df[gt_col].notna()
        reference_gt = reference_df.loc[valid, gt_col].astype(int).to_numpy()
    else:
        reference_gt = np.asarray([], dtype=int)

    supervised_path = figures_path("history_supervised_predictions") + ".csv"
    supervised = pd.read_csv(supervised_path) if os.path.exists(supervised_path) else pd.DataFrame()
    if not supervised.empty:
        supervised = supervised[~supervised["Item_Key"].isin(dup_keys)].copy()

    for display_order, row_spec in enumerate(FIGURE3_DISTRIBUTION_ROWS):
        series_records_start = len(records)
        row_human = reference_gt

        if row_spec["kind"] == "llm":
            for model_id, cfg in MODEL_CONFIGS.items():
                df = load_results_aligned(model_id, row_spec["method_file"])
                if df is None:
                    continue
                keyed = _attach_item_keys(df)
                keyed = keyed[~keyed["Item_Key"].isin(dup_keys)].reset_index(drop=True)
                if gt_col not in keyed.columns or pred_col not in keyed.columns:
                    continue
                valid = keyed[gt_col].notna() & keyed[pred_col].notna()
                if not valid.any():
                    continue
                _append_score_distribution_records(
                    records,
                    domain=domain,
                    display_order=display_order,
                    row_label=row_spec["label"],
                    method=row_spec["method"],
                    series_type="Prediction",
                    series_label=cfg["display"],
                    values=keyed.loc[valid, pred_col].astype(int).to_numpy(),
                    n=int(valid.sum()),
                    color=COLORS.get(cfg["display"], "#888888"),
                )

        elif row_spec["kind"] == "supervised" and not supervised.empty:
            sub = supervised[
                (supervised["Domain"] == domain.capitalize())
                & (supervised["Classifier"] == row_spec["classifier"])
                & (supervised["Feature_Set"] == row_spec["feature_set"])
            ].copy()
            valid = sub["Ground_Truth_Num"].notna() & sub["Predicted_Num"].notna() if not sub.empty else pd.Series(dtype=bool)
            if not sub.empty and valid.any():
                row_human = sub.loc[valid, "Ground_Truth_Num"].astype(int).to_numpy()
                _append_score_distribution_records(
                    records,
                    domain=domain,
                    display_order=display_order,
                    row_label=row_spec["label"],
                    method=row_spec["method"],
                    series_type="Prediction",
                    series_label=row_spec["series_label"],
                    values=sub.loc[valid, "Predicted_Num"].astype(int).to_numpy(),
                    n=int(valid.sum()),
                    color=row_spec["color"],
                )

        elif row_spec["kind"] == "supervised_group" and not supervised.empty:
            row_human = np.asarray([], dtype=int)
            for predictor in row_spec["predictors"]:
                sub = supervised[
                    (supervised["Domain"] == domain.capitalize())
                    & (supervised["Classifier"] == predictor["classifier"])
                    & (supervised["Feature_Set"] == predictor["feature_set"])
                ].copy()
                valid = (
                    sub["Ground_Truth_Num"].notna() & sub["Predicted_Num"].notna()
                    if not sub.empty
                    else pd.Series(dtype=bool)
                )
                if sub.empty or not valid.any():
                    continue
                if len(row_human) == 0:
                    row_human = sub.loc[valid, "Ground_Truth_Num"].astype(int).to_numpy()
                _append_score_distribution_records(
                    records,
                    domain=domain,
                    display_order=display_order,
                    row_label=row_spec["label"],
                    method=row_spec["method"],
                    series_type="Prediction",
                    series_label=predictor["series_label"],
                    values=sub.loc[valid, "Predicted_Num"].astype(int).to_numpy(),
                    n=int(valid.sum()),
                    color=predictor["color"],
                )

        if len(records) > series_records_start and len(row_human):
            human_records = []
            _append_score_distribution_records(
                human_records,
                domain=domain,
                display_order=display_order,
                row_label=row_spec["label"],
                method=row_spec["method"],
                series_type="Human reference",
                series_label="Human ratings",
                values=row_human,
                n=len(row_human),
                color=REFERENCE_GRAY,
            )
            records[series_records_start:series_records_start] = human_records

    return records


def create_score_distribution_redesign(distribution_df: pd.DataFrame | None = None) -> list[str]:
    """Create vertical domain-specific score-distribution panels."""
    generated = []
    if distribution_df is None:
        distribution_df = build_score_distribution_source_table()
    if distribution_df.empty or "Domain" not in distribution_df.columns:
        return generated

    for domain in DOMAINS:
        domain_label = domain.capitalize()
        domain_df = distribution_df[distribution_df["Domain"] == domain_label].copy()
        rows = _distribution_plot_rows_from_table(domain_df)
        if not rows:
            continue

        nrows = len(rows)
        fig_height = max(8.4, 1.16 * nrows + 2.2)
        fig, axes = plt.subplots(nrows, 1, figsize=(12.8, fig_height), sharex=True)
        if nrows == 1:
            axes = np.array([axes])
        fig.patch.set_facecolor("white")

        x = np.arange(1, 6)
        width = 0.36
        max_pct = 0.0
        for row in rows:
            max_pct = max(max_pct, float(row["human_pct"].max()))
            for series in row["pred_series"]:
                max_pct = max(max_pct, float(series["pred_pct"].max()))
        y_max = min(100, max(35, np.ceil((max_pct + 6) / 5) * 5))

        for idx, (ax, row) in enumerate(zip(axes, rows)):
            human_pct = row["human_pct"]
            pred_series = row["pred_series"]
            ax.set_facecolor("white")
            ax.grid(axis="y", alpha=0.12, color=REFERENCE_GRAY,
                    linestyle="--", linewidth=0.6)
            ax.set_axisbelow(True)

            ax.bar(
                x,
                human_pct,
                width=0.82,
                color=REFERENCE_GRAY,
                alpha=0.16,
                edgecolor=REFERENCE_GRAY,
                linewidth=0.9,
                label="Human ratings" if idx == 0 else None,
                zorder=1,
            )

            if len(pred_series) == 1:
                offsets = np.array([0.0])
                bar_width = 0.34
            else:
                total_width = 0.78
                bar_width = min(0.13, total_width / max(1, len(pred_series)) * 0.86)
                offsets = np.linspace(
                    -total_width / 2 + bar_width / 2,
                    total_width / 2 - bar_width / 2,
                    len(pred_series),
                )

            for offset, series in zip(offsets, pred_series):
                ax.bar(
                    x + offset,
                    series["pred_pct"],
                    width=bar_width,
                    color=series["color"],
                    alpha=0.88,
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=2,
                )

            ax.set_ylim(0, y_max)
            ax.set_yticks([0, y_max / 2, y_max])
            ax.set_ylabel("%", fontsize=11, fontweight="bold")
            ax.text(
                -0.10,
                0.50,
                row["label"],
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=11,
                fontweight="bold",
            )
            ax.tick_params(axis="both", labelsize=10, width=1.5,
                           length=5, direction="out")
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight("bold")
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
            if idx < nrows - 1:
                ax.tick_params(labelbottom=False)

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels([str(i) for i in x], fontsize=12, fontweight="bold")
        axes[-1].set_xlabel("Rating", fontsize=13, fontweight="bold")

        handles = [
            Patch(facecolor=REFERENCE_GRAY, alpha=0.16, edgecolor=REFERENCE_GRAY,
                  label="Human ratings"),
        ]
        seen_labels = set()
        for row in rows:
            for series in row["pred_series"]:
                if series["label"] in seen_labels:
                    continue
                handles.append(Patch(
                    facecolor=series["color"],
                    alpha=0.88,
                    edgecolor="black",
                    label=series["label"],
                ))
                seen_labels.add(series["label"])
        legend = fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),
            ncol=4,
            frameon=False,
            prop={"weight": "bold", "size": 10},
        )
        for text in legend.get_texts():
            text.set_fontweight("bold")

        fig.suptitle(
            f"Rating Distributions: {domain.capitalize()} Domain",
            fontsize=17,
            fontweight="bold",
            y=0.988,
        )
        fig.subplots_adjust(left=0.18, right=0.97, top=0.935,
                            bottom=0.15, hspace=0.42)

        output_stem = figures_path(f"figure3_score_distributions_{domain}")
        save_figure(fig, output_stem)
        generated.append(os.path.basename(output_stem))

    return generated


def _coverage_counts(coverage_df: pd.DataFrame) -> dict:
    if coverage_df.empty or "Domain" not in coverage_df.columns or "N_Common" not in coverage_df.columns:
        return {}
    return (
        coverage_df.groupby("Domain")["N_Common"]
        .max()
        .astype(int)
        .to_dict()
    )


def _baseline_refs_from_metrics(metric_df: pd.DataFrame) -> dict:
    refs = {}
    if metric_df.empty:
        return refs
    for domain, domain_metrics in metric_df.groupby("Domain"):
        refs[domain] = {}
        for metric_name, _, _ in METRICS:
            if metric_name not in domain_metrics.columns:
                continue
            for clf, prefix in [("LR", "lr"), ("RF", "rf")]:
                clf_df = domain_metrics[domain_metrics["Classifier"] == clf].copy()
                clf_df = clf_df[clf_df[metric_name].notna()]
                if clf_df.empty:
                    continue
                best = clf_df.sort_values(metric_name, ascending=False).iloc[0]
                refs[domain][f"{metric_name}_{prefix}_value"] = float(best[metric_name])
                refs[domain][f"{metric_name}_{prefix}_label"] = f"{best['Feature_Set']} ({clf})"
    return refs


def parse_args():
    parser = argparse.ArgumentParser(description="Build revision redesign figures.")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Reload saved revision/figures source CSVs and regenerate PNG/PDF outputs only.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("Revision Figure Redesign")
    print("=" * 70)

    csv_path = figures_path("figure_redesign_results") + ".csv"
    generic_results_path = figures_path("generic_common_subset_results") + ".csv"
    generic_coverage_path = figures_path("generic_common_subset_coverage") + ".csv"
    personalized_results_path = figures_path("personalized_common_subset_results") + ".csv"
    common_coverage_path = figures_path("personalized_common_subset_coverage") + ".csv"
    baseline_common_path = figures_path("history_supervised_common_subset") + ".csv"
    distribution_path = figures_path("figure3_score_distribution_source") + ".csv"

    if args.plot_only:
        required = [
            generic_results_path,
            generic_coverage_path,
            personalized_results_path,
            common_coverage_path,
            baseline_common_path,
            distribution_path,
        ]
        missing = [path for path in required if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(
                "Missing cached source CSV(s): " + ", ".join(missing)
            )
        print("Reloading cached source CSVs for plot-only rebuild...")
        generic_results_df = pd.read_csv(generic_results_path)
        generic_coverage_df = pd.read_csv(generic_coverage_path)
        personalized_results_df = pd.read_csv(personalized_results_path)
        common_coverage_df = pd.read_csv(common_coverage_path)
        baseline_common_df = pd.read_csv(baseline_common_path)
        distribution_df = pd.read_csv(distribution_path)
        baseline_refs = _baseline_refs_from_metrics(baseline_common_df)
    else:
        results_df = collect_all_results()
        if results_df.empty:
            raise RuntimeError("No results were collected; cannot build redesign figures.")

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        results_df.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"Saved source CSV: {csv_path}")

        generic_results_df, generic_common_keys, generic_coverage_df = collect_generic_common_subset_results()
        generic_results_df.to_csv(generic_results_path, index=False, float_format="%.4f")
        print(f"Saved generic common-subset metrics: {generic_results_path}")

        generic_coverage_df.to_csv(generic_coverage_path, index=False)
        print(f"Saved generic common-subset coverage: {generic_coverage_path}")

        personalized_results_df, common_keys, common_coverage_df = collect_personalized_common_subset_results()
        personalized_results_df.to_csv(personalized_results_path, index=False, float_format="%.4f")
        print(f"Saved personalized common-subset metrics: {personalized_results_path}")

        common_coverage_df.to_csv(common_coverage_path, index=False)
        print(f"Saved personalized common-subset coverage: {common_coverage_path}")

        baseline_refs, baseline_common_df = load_baseline_reference(common_keys)
        baseline_common_df.to_csv(baseline_common_path, index=False, float_format="%.4f")
        print(f"Saved baseline common-subset metrics: {baseline_common_path}")

        distribution_df = build_score_distribution_source_table()
        distribution_df.to_csv(distribution_path, index=False, float_format="%.4f")
        print(f"Saved Figure 3 distribution source: {distribution_path}")

    generic_common_counts = _coverage_counts(generic_coverage_df)
    common_counts = _coverage_counts(common_coverage_df)
    print("\nCreating figures from saved source tables...")
    generic_files = create_generic_method_bars(generic_results_df, generic_common_counts)
    personalized_files = create_personalized_method_bars(
        personalized_results_df,
        baseline_refs,
        common_counts,
    )
    distribution_files = create_score_distribution_redesign(distribution_df)

    print("\nGenerated redesign outputs:")
    for stem in generic_files + personalized_files + distribution_files:
        print(f"  - {stem}.png/.pdf")

    print("\nDone.")


if __name__ == "__main__":
    main()
