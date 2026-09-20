"""
Demographic Subgroup Analysis for LLM Message Rating

Analyzes model performance across demographic subgroups to address
reviewer R2's fairness concern. Uses the digital twin method
(digital_twin_4_cbtact_7030.json) which has the largest test set.

For each model x domain x subgroup, computes metrics with bootstrap CIs
and flags subgroups with n < 20.

Outputs:
    revision/figures/demographic_subgroup_results.csv
    revision/figures/demographic_subgroup_race.png/pdf
    revision/figures/demographic_subgroup_gender.png/pdf

Usage:
    uv run python analysis-script/demographic_subgroup_analysis.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "analysis-script")
from revision_utils import (
    load_results_aligned,
    compute_all_metrics,
    MODEL_CONFIGS,
    DOMAINS,
    COLORS,
    save_figure,
    figures_path,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHOD_FILE = "digital_twin_4_cbtact_7030.json"
N_BOOTSTRAP = 1000
SMALL_N_THRESHOLD = 20
SEED = 42

# ---------------------------------------------------------------------------
# Demographic grouping helpers
# ---------------------------------------------------------------------------

def classify_race(race_val):
    """Map race_ethnicity string into White / Black / African American / Other."""
    if not isinstance(race_val, str) or race_val.strip() == "":
        return "Other"
    race = race_val.strip()
    if race == "White":
        return "White"
    if race == "Black or African American":
        return "Black / African American"
    return "Other"


def classify_gender(gender_val):
    """Map gender_identity string into Male / Female / Other."""
    if not isinstance(gender_val, str) or gender_val.strip() == "":
        return "Other"
    gender = gender_val.strip()
    if gender == "Male":
        return "Male"
    if gender == "Female":
        return "Female"
    return "Other"


def classify_age(age_val):
    """Map age_years into 18-24 / 25-30 bins."""
    try:
        age = int(age_val)
    except (ValueError, TypeError):
        return None
    if 18 <= age <= 24:
        return "18-24"
    if 25 <= age <= 30:
        return "25-30"
    return None


def extract_subgroups(df):
    """Add subgroup columns to DataFrame based on metadata."""
    races, genders, ages = [], [], []
    for _, row in df.iterrows():
        meta = row.get("metadata", {})
        if not isinstance(meta, dict):
            meta = {}
        races.append(classify_race(meta.get("race_ethnicity")))
        genders.append(classify_gender(meta.get("gender_identity")))
        ages.append(classify_age(meta.get("age_years")))
    df = df.copy()
    df["subgroup_race"] = races
    df["subgroup_gender"] = genders
    df["subgroup_age"] = ages
    return df


# ---------------------------------------------------------------------------
# Bootstrap CI computation
# ---------------------------------------------------------------------------

def bootstrap_accuracy(gt, pred, n_boot=N_BOOTSTRAP, seed=SEED):
    """Compute bootstrap 95% CI for accuracy.

    Returns (lower, upper) or (np.nan, np.nan) if not computable.
    """
    from sklearn.metrics import accuracy_score
    rng = np.random.RandomState(seed)
    n = len(gt)
    if n < 2:
        return np.nan, np.nan
    boot_vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            val = accuracy_score(gt[idx], pred[idx])
            if not np.isnan(val):
                boot_vals.append(val)
        except Exception:
            continue
    if len(boot_vals) < 10:
        return np.nan, np.nan
    return float(np.percentile(boot_vals, 2.5)), float(np.percentile(boot_vals, 97.5))


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis():
    """Run demographic subgroup analysis across all models and domains."""
    print("=" * 70)
    print("Demographic Subgroup Analysis")
    print("=" * 70)

    all_rows = []

    for model_id, model_cfg in MODEL_CONFIGS.items():
        model_name = model_cfg["display"]
        print("\n--- " + model_name + " ---")

        df = load_results_aligned(model_id, METHOD_FILE)
        if df is None:
            print("  WARNING: No results for " + model_name + ", skipping.")
            continue

        df = extract_subgroups(df)
        print("  Loaded " + str(len(df)) + " records")

        # Define subgroup definitions: (subgroup_type, column, ordered_groups)
        subgroup_defs = [
            ("Race", "subgroup_race", ["White", "Black / African American", "Other"]),
            ("Gender", "subgroup_gender", ["Male", "Female", "Other"]),
            ("Age", "subgroup_age", ["18-24", "25-30"]),
        ]

        for domain in DOMAINS:
            gt_col = f"gt_{domain}_num"
            pred_col = f"pred_{domain}_num"
            if gt_col not in df.columns or pred_col not in df.columns:
                continue

            valid_mask = df[gt_col].notna() & df[pred_col].notna()
            df_valid = df[valid_mask]

            for sg_type, sg_col, sg_groups in subgroup_defs:
                for sg_name in sg_groups:
                    mask = df_valid[sg_col] == sg_name
                    sub = df_valid[mask]
                    n = len(sub)

                    if n == 0:
                        continue

                    gt = sub[gt_col].values.astype(int)
                    pred = sub[pred_col].values.astype(int)
                    rids = sub["response_id"].values if "response_id" in sub.columns else None

                    metrics = compute_all_metrics(gt, pred, rids)

                    # Bootstrap CI for accuracy
                    ci_lower, ci_upper = bootstrap_accuracy(gt, pred)

                    row = {
                        "Model": model_name,
                        "Domain": domain.capitalize(),
                        "Subgroup_Type": sg_type,
                        "Subgroup": sg_name,
                        "N": n,
                        "Accuracy": metrics.get("Accuracy", np.nan),
                        "F1": metrics.get("F1", np.nan),
                        "Kappa": metrics.get("Kappa", np.nan),
                        "QWK": metrics.get("QWK", np.nan),
                        "Accuracy_CI_Lower": ci_lower,
                        "Accuracy_CI_Upper": ci_upper,
                        "Small_N_Flag": n < SMALL_N_THRESHOLD,
                    }
                    all_rows.append(row)

    results_df = pd.DataFrame(all_rows)

    # Save CSV
    out_csv = figures_path("demographic_subgroup_results") + ".csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    results_df.to_csv(out_csv, index=False, float_format="%.4f")
    print("\nSaved CSV: " + out_csv)

    return results_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_subgroup_bars(results_df, subgroup_type, subgroups_order, filename):
    """Create 1x3 subplot (by domain) grouped bar chart with hue = subgroup.

    Each cluster of bars is a model; within-cluster bars are subgroups.
    Error bars from bootstrap CIs. Hatching for small-n subgroups.
    """
    domains = [d.capitalize() for d in DOMAINS]
    models_order = [cfg["display"] for cfg in MODEL_CONFIGS.values()]
    # Keep only models actually present
    available_models = results_df["Model"].unique()
    models_order = [m for m in models_order if m in available_models]

    n_models = len(models_order)
    n_subgroups = len(subgroups_order)

    if subgroup_type == "Gender":
        sg_colors = {
            "Male": "#0173B2",
            "Female": "#DE8F05",
            "Other": "#7F7F7F",
        }
    elif subgroup_type == "Race":
        sg_colors = {
            "White": "#7F7F7F",
            "Black / African American": "#029E73",
            "Other": "#CA9161",
        }
    else:
        sg_palette = plt.cm.Set2(np.linspace(0, 0.6, n_subgroups))
        sg_colors = {sg: sg_palette[i] for i, sg in enumerate(subgroups_order)}

    hatch_pattern = "//"

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
    fig.suptitle(
        "Accuracy by " + subgroup_type + " Subgroup (PP LLM)",
        fontsize=18, fontweight="bold", y=1.02,
    )

    bar_width = 0.8 / n_subgroups

    for ax_idx, domain in enumerate(domains):
        ax = axes[ax_idx]
        subset = results_df[
            (results_df["Domain"] == domain)
            & (results_df["Subgroup_Type"] == subgroup_type)
        ]

        x_positions = np.arange(n_models)

        for sg_idx, sg_name in enumerate(subgroups_order):
            sg_data = subset[subset["Subgroup"] == sg_name]

            values = []
            ci_lowers = []
            ci_uppers = []
            hatches = []
            for model in models_order:
                row = sg_data[sg_data["Model"] == model]
                if len(row) == 1:
                    r = row.iloc[0]
                    acc = r["Accuracy"]
                    values.append(acc)
                    ci_lo = (
                        acc - r["Accuracy_CI_Lower"]
                        if not np.isnan(r["Accuracy_CI_Lower"])
                        else 0
                    )
                    ci_hi = (
                        r["Accuracy_CI_Upper"] - acc
                        if not np.isnan(r["Accuracy_CI_Upper"])
                        else 0
                    )
                    ci_lowers.append(max(ci_lo, 0))
                    ci_uppers.append(max(ci_hi, 0))
                    hatches.append(r["Small_N_Flag"])
                else:
                    values.append(0)
                    ci_lowers.append(0)
                    ci_uppers.append(0)
                    hatches.append(False)

            offset = (sg_idx - (n_subgroups - 1) / 2) * bar_width
            positions = x_positions + offset

            bars = ax.bar(
                positions,
                values,
                width=bar_width * 0.9,
                color=sg_colors[sg_name],
                edgecolor="black",
                linewidth=0.5,
                label=sg_name if ax_idx == 0 else None,
                yerr=[ci_lowers, ci_uppers],
                capsize=2,
                error_kw={"linewidth": 0.8},
            )

            # Apply hatching for small-n bars
            for bar, is_small in zip(bars, hatches):
                if is_small:
                    bar.set_hatch(hatch_pattern)
                    bar.set_edgecolor("gray")

        ax.set_title(domain, fontsize=16, fontweight="bold")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(models_order, rotation=30, ha="right", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 0.75)
        ax.set_ylabel("Accuracy" if ax_idx == 0 else "", fontsize=14, fontweight="bold")
        ax.tick_params(axis="y", labelsize=14, width=1.8, length=6)
        ax.grid(axis="y", alpha=0.3)

    plotted_subset = results_df[
        (results_df["Subgroup_Type"] == subgroup_type)
        & (results_df["Subgroup"].isin(subgroups_order))
    ].copy()

    # Build legend handles
    legend_handles = [
        mpatches.Patch(
            facecolor=sg_colors[sg], edgecolor="black", linewidth=0.5, label=sg
        )
        for sg in subgroups_order
    ]
    if plotted_subset["Small_N_Flag"].any():
        hatch_patch = mpatches.Patch(
            facecolor="white",
            edgecolor="gray",
            hatch=hatch_pattern,
            label="n < " + str(SMALL_N_THRESHOLD) + " (interpret with caution)",
        )
        legend_handles.append(hatch_patch)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(legend_handles),
        fontsize=12,
        bbox_to_anchor=(0.5, 0.99),
        frameon=True,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, figures_path(filename))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    results_df = run_analysis()

    if results_df.empty:
        print("No results to plot.")
        return

    # Print summary
    print("\n" + "=" * 70)
    print("Summary of subgroup sizes:")
    for sg_type in ["Race", "Gender", "Age"]:
        sub = results_df[results_df["Subgroup_Type"] == sg_type]
        if sub.empty:
            continue
        sample = sub[sub["Domain"] == "Content"]
        if sample.empty:
            sample = sub
        print("\n  " + sg_type + ":")
        for _, r in sample.drop_duplicates(subset=["Model", "Subgroup"]).iterrows():
            flag = " *" if r["Small_N_Flag"] else ""
            print("    {:20s} {:10s} n={:4d}{}".format(
                r["Model"], r["Subgroup"], int(r["N"]), flag))

    # Small-n warnings
    small_n = results_df[results_df["Small_N_Flag"]]
    if not small_n.empty:
        print("\n  WARNING: {} subgroup entries have n < {}.".format(
            len(small_n), SMALL_N_THRESHOLD))
        print("  These are marked with hatching in plots and flagged in CSV.")

    # --- Plots ---
    print("\n" + "=" * 70)
    print("Creating subgroup plots...")

    plot_subgroup_bars(
        results_df,
        subgroup_type="Race",
        subgroups_order=["White", "Black / African American", "Other"],
        filename="demographic_subgroup_race",
    )

    plot_subgroup_bars(
        results_df,
        subgroup_type="Gender",
        subgroups_order=["Male", "Female"],
        filename="demographic_subgroup_gender",
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
