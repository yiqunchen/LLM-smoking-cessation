#!/usr/bin/env python3
"""Plot Reviewer 3 prompt ablations for model families with complete results.

This interim analysis intentionally includes only the four model families with
all four 898-row canonical dt10-k7 condition files.  It refuses incomplete
inputs, so DeepSeek-R1 cannot enter the source tables or figures until its
four condition runs are complete.  A five-model companion script is retained
for the final rerun.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data_splits" / "canonical" / "test_dt10_k7.json"
OUTDIR = ROOT / "revision" / "figures" / "reviewer_ablations_dt10"
DOMAINS = ("content", "design", "coping", "quitting")
RATING_SCALES = {
    "content": {"Very poor": 1, "Poor": 2, "Acceptable": 3, "Good": 4, "Very good": 5},
    "design": {"Very poor": 1, "Poor": 2, "Acceptable": 3, "Good": 4, "Very good": 5},
    "coping": {
        "Not at all helpful": 1, "Not Helpful": 1, "Somewhat helpful": 2,
        "Moderately helpful": 3, "Very helpful": 4, "Extremely helpful": 5,
    },
    "quitting": {
        "Not at all helpful": 1, "Not Helpful": 1, "Somewhat helpful": 2,
        "Moderately helpful": 3, "Very helpful": 4, "Extremely helpful": 5,
    },
}
CONDITIONS = (
    ("pp_cbtact", "PP + history\n+CBT/ACT"),
    ("full_pp_no_cbtact", "PP + history\n(no CBT/ACT)"),
    ("history_ratings_only", "History +\nratings only"),
    ("history_text_only", "History text\nonly"),
)
MODELS = (
    ("GPT-4o-mini", "#0173B2", "results_reviewer_ablations_openai_gpt-4o-mini"),
    ("GPT-5", "#DE8F05", "results_reviewer_ablations_openai_gpt-5"),
    ("Grok-4.3", "#CC78BC", "results_reviewer_ablations_x-ai_grok-4.3"),
    ("Gemini-2.5-Pro", "#CA9161", "results_reviewer_ablations_google_gemini-2.5-pro"),
)
FONT_CHAIN = ["Helvetica", "Arial", "DejaVu Sans"]

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": FONT_CHAIN, "font.size": 12,
    "font.weight": "bold", "axes.labelsize": 13, "axes.titlesize": 13,
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.grid.axis": "y", "grid.linestyle": "--", "grid.alpha": 0.12,
    "grid.color": "#4D4D4D", "grid.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150, "savefig.dpi": 400,
})


def load_and_validate(path: Path, test: list[dict], condition: str) -> dict[str, dict]:
    """Load an exact, canonical 898-row condition result without substitutions."""
    if not path.exists():
        raise SystemExit(f"Missing required complete result: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    expected = {str(index) for index in range(len(test))}
    if not isinstance(rows, dict) or set(rows) != expected:
        raise SystemExit(f"Incomplete result for {condition}: {path}")
    for index, test_row in enumerate(test):
        row = rows[str(index)]
        if (row.get("response_id") != test_row.get("response_id")
                or row.get("input_message") != test_row.get("input_message")):
            raise SystemExit(f"Misaligned canonical test row {index}: {path}")
        ratings = test_row.get("ratings", {}) or {}
        for domain in DOMAINS:
            if row.get(f"ground_truth_{domain}") != ratings.get(domain):
                raise SystemExit(f"Ground-truth mismatch at row {index}, {domain}: {path}")
    return rows


def measure(rows: dict[str, dict], n_test: int) -> list[dict]:
    values: list[dict] = []
    for domain in DOMAINS:
        truth = [rows[str(index)][f"ground_truth_{domain}"] for index in range(n_test)]
        prediction = [rows[str(index)][f"predicted_{domain}"] for index in range(n_test)]
        ordinal_truth = [RATING_SCALES[domain][value] for value in truth]
        ordinal_prediction = [RATING_SCALES[domain][value] for value in prediction]
        values.append({
            "domain": domain.capitalize(),
            "accuracy": float(np.mean(np.asarray(truth) == np.asarray(prediction))),
            "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
            "qwk": float(cohen_kappa_score(ordinal_truth, ordinal_prediction, weights="quadratic")),
        })
    return values


def style_axis(ax: plt.Axes) -> None:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Arial")
        label.set_fontweight("bold")
    ax.tick_params(length=0)


def panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(0.015, 0.98, letter, transform=ax.transAxes, fontweight="bold", fontsize=14,
            ha="left", va="top", color="#222222")


def plot_overall(summary: pd.DataFrame) -> None:
    """Figure 1: matched model trajectories for the three Figure-2 metrics."""
    metric_specs = (
        ("accuracy", "Exact accuracy", (0.0, 0.80)),
        ("macro_f1", "Macro-F1", (0.0, 0.75)),
        ("qwk", "Quadratic-weighted κ", (-0.05, 0.70)),
    )
    figure = plt.figure(figsize=(14.2, 4.9), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=(0.20, 1.0))
    axes = [figure.add_subplot(grid[1, column]) for column in range(3)]
    xs = np.arange(len(CONDITIONS))
    for index, (axis, (metric, title, limits)) in enumerate(zip(axes, metric_specs)):
        for model, color, _directory in MODELS:
            subset = summary[summary["model"] == model].set_index("condition_key")
            axis.plot(xs, [subset.loc[key, metric] for key, _label in CONDITIONS], marker="o",
                      markersize=6.5, linewidth=2.3, color=color, label=model, zorder=3)
        axis.set_title(title, pad=8)
        axis.set_ylim(*limits)
        axis.set_xticks(xs, [label for _key, label in CONDITIONS])
        style_axis(axis)
        panel_label(axis, chr(ord("a") + index))
    axes[0].set_ylabel("Mean score across four rating dimensions")
    handles, labels = axes[0].get_legend_handles_labels()
    legend = figure.legend(handles, labels, loc="upper center", ncol=4,
                           bbox_to_anchor=(0.5, 0.91), frameon=False,
                           prop={"family": "Arial", "weight": "bold", "size": 10.5})
    figure.suptitle("Reviewer 3 prompt-component sensitivity analysis", y=0.99,
                     fontweight="bold", fontsize=15)
    figure.text(0.5, 0.945,
                "Canonical dt10-k7 held-out test set: N = 898 messages; four complete model families",
                ha="center", va="top", fontsize=10.5, fontweight="normal", color="#4D4D4D")
    for extension in ("png", "pdf"):
        figure.savefig(OUTDIR / f"reviewer_ablation_completed_models_overall_dt10.{extension}",
                       bbox_inches="tight")
    plt.close(figure)


def plot_accuracy_by_domain(records: pd.DataFrame) -> None:
    """Figure 2: domain-level check that is not hidden by the four-domain mean."""
    figure = plt.figure(figsize=(12.2, 8.7), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=(0.18, 1.0, 1.0))
    axes = np.asarray([
        [figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])],
        [figure.add_subplot(grid[2, 0]), figure.add_subplot(grid[2, 1])],
    ])
    xs = np.arange(len(CONDITIONS))
    for index, (axis, domain) in enumerate(zip(axes.flat, ("Content", "Design", "Coping", "Quitting"))):
        domain_rows = records[records["domain"] == domain]
        for model, color, _directory in MODELS:
            subset = domain_rows[domain_rows["model"] == model].set_index("condition_key")
            axis.plot(xs, [subset.loc[key, "accuracy"] for key, _label in CONDITIONS], marker="o",
                      markersize=6, linewidth=2.2, color=color, label=model, zorder=3)
        axis.set_title(domain, pad=7)
        axis.set_ylim(0.0, 0.82)
        axis.set_xticks(xs, [label for _key, label in CONDITIONS])
        style_axis(axis)
        panel_label(axis, chr(ord("a") + index))
    axes[0, 0].set_ylabel("Exact accuracy")
    axes[1, 0].set_ylabel("Exact accuracy")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = figure.legend(handles, labels, loc="lower center", ncol=4,
                           bbox_to_anchor=(0.5, -0.045), frameon=False,
                           prop={"family": "Arial", "weight": "bold", "size": 10.5})
    figure.suptitle("Reviewer 3 prompt-component sensitivity analysis by rating dimension", y=0.99,
                     fontweight="bold", fontsize=15)
    figure.text(0.5, 0.945,
                "Canonical dt10-k7 held-out test set: N = 898 messages; four complete model families",
                ha="center", va="top", fontsize=10.5, fontweight="normal", color="#4D4D4D")
    for extension in ("png", "pdf"):
        figure.savefig(OUTDIR / f"reviewer_ablation_completed_models_accuracy_by_domain_dt10.{extension}",
                       bbox_inches="tight")
    plt.close(figure)


def plot_score_histograms(rows_by_model_condition: dict[tuple[str, str], dict[str, dict]],
                          n_test: int) -> pd.DataFrame:
    """Plot observed-versus-predicted discrete rating frequencies without pooling domains.

    Each domain has a separate 4-by-4 grid.  A row is a model family and a
    column is a reviewer-requested prompt configuration.  Keeping the true
    score bars in every panel makes distributional shifts visible alongside the
    accuracy summaries, while preserving the matched 898-message comparison.
    """
    count_records: list[dict] = []
    for domain in DOMAINS:
        figure = plt.figure(figsize=(14.6, 14.2), constrained_layout=True)
        # Reserve a dedicated title/legend band so the first-row condition
        # headers cannot collide with the figure-level annotation.
        grid = figure.add_gridspec(5, 4, height_ratios=(0.60, 1.0, 1.0, 1.0, 1.0))
        axes = np.asarray([
            [figure.add_subplot(grid[row + 1, column]) for column in range(4)]
            for row in range(4)
        ])
        panel_counts: list[int] = []
        for model, color, _directory in MODELS:
            for condition_key, _condition_label in CONDITIONS:
                result_rows = rows_by_model_condition[(model, condition_key)]
                scale = RATING_SCALES[domain]
                for kind, field in (("Observed", f"ground_truth_{domain}"),
                                    ("Predicted", f"predicted_{domain}")):
                    scores = [scale.get(result_rows[str(index)][field]) for index in range(n_test)]
                    if any(score is None for score in scores):
                        raise SystemExit(f"Unmapped {domain} {kind.lower()} label for {model}: {condition_key}")
                    for rating in range(1, 6):
                        count = int(sum(score == rating for score in scores))
                        panel_counts.append(count)
                        count_records.append({
                            "split": "dt10_k7", "n_test": n_test, "domain": domain.capitalize(),
                            "model": model, "model_color": color, "condition_key": condition_key,
                            "condition": dict(CONDITIONS)[condition_key].replace("\n", " "),
                            "series": kind, "ordinal_rating": rating, "count": count,
                            "proportion": count / n_test,
                        })
        ymax = max(panel_counts) * 1.12
        x = np.arange(1, 6, dtype=float)
        for row, (model, color, _directory) in enumerate(MODELS):
            for column, (condition_key, condition_label) in enumerate(CONDITIONS):
                axis = axes[row, column]
                result_rows = rows_by_model_condition[(model, condition_key)]
                scale = RATING_SCALES[domain]
                truth = [scale[result_rows[str(index)][f"ground_truth_{domain}"]] for index in range(n_test)]
                prediction = [scale[result_rows[str(index)][f"predicted_{domain}"]] for index in range(n_test)]
                truth_counts = [truth.count(rating) for rating in range(1, 6)]
                pred_counts = [prediction.count(rating) for rating in range(1, 6)]
                axis.bar(x - 0.19, truth_counts, width=0.36, color="#7F7F7F", alpha=0.48,
                         edgecolor="#4D4D4D", linewidth=0.8, label="Observed" if row == column == 0 else None)
                axis.bar(x + 0.19, pred_counts, width=0.36, color=color, alpha=0.88,
                         edgecolor=color, linewidth=0.8, label="Predicted" if row == column == 0 else None)
                axis.set_ylim(0, ymax)
                axis.set_xticks(x)
                if row == 0:
                    axis.set_title(condition_label, pad=10, fontsize=11.5)
                if column == 0:
                    axis.set_ylabel(f"{model}\nFrequency", color=color, labelpad=14, fontsize=11.5)
                if row == len(MODELS) - 1:
                    axis.set_xlabel("Ordinal rating")
                style_axis(axis)
        figure.suptitle(f"Reviewer 3 prompt-component score distributions — {domain.capitalize()}",
                         y=0.99, fontsize=15, fontweight="bold")
        figure.text(0.5, 0.955,
                    "Observed versus predicted rating frequencies; canonical dt10-k7 test set, N = 898",
                    ha="center", va="top", fontsize=10.5, fontweight="normal", color="#4D4D4D")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=2,
                      frameon=False, prop={"family": "Arial", "weight": "bold", "size": 10.5})
        for extension in ("png", "pdf"):
            figure.savefig(OUTDIR / f"reviewer_ablation_completed_models_score_histogram_{domain}_dt10.{extension}",
                           bbox_inches="tight")
        plt.close(figure)
    return pd.DataFrame(count_records)


def main() -> None:
    test = json.loads(TEST_PATH.read_text(encoding="utf-8"))
    if len(test) != 898:
        raise SystemExit("Expected the 898-row canonical dt10-k7 test file")
    records: list[dict] = []
    rows_by_model_condition: dict[tuple[str, str], dict[str, dict]] = {}
    for model, color, directory in MODELS:
        for condition_key, condition_label in CONDITIONS:
            rows = load_and_validate(ROOT / directory / f"{condition_key}_dt10_k7.json", test,
                                     f"{model}: {condition_key}")
            rows_by_model_condition[(model, condition_key)] = rows
            for result in measure(rows, len(test)):
                records.append({
                    "split": "dt10_k7", "n_test": len(test), "model": model, "model_color": color,
                    "condition_key": condition_key, "condition": condition_label.replace("\n", " "),
                    **result,
                })
    metrics = pd.DataFrame(records)
    summary = metrics.groupby(["model", "model_color", "condition_key", "condition"], as_index=False)[
        ["accuracy", "macro_f1", "qwk"]
    ].mean()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTDIR / "reviewer_ablation_completed_models_metrics_dt10.csv", index=False)
    summary.to_csv(OUTDIR / "reviewer_ablation_completed_models_summary_dt10.csv", index=False)
    plot_overall(summary)
    plot_accuracy_by_domain(metrics)
    histogram_counts = plot_score_histograms(rows_by_model_condition, len(test))
    histogram_counts.to_csv(OUTDIR / "reviewer_ablation_completed_models_histogram_counts_dt10.csv", index=False)
    print(f"Wrote validated four-model reviewer figures to {OUTDIR}")


if __name__ == "__main__":
    main()
