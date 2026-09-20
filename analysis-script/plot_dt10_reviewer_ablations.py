#!/usr/bin/env python3
"""Create dt10-k7-only figures for the Reviewer 3 PP prompt ablations.

No figure is produced until every required condition contains all 898
canonical test rows.  This prevents partial API checkpoints from appearing as
reviewer results.  The two figures compare PP with CBT/ACT and the three
requested ablations for the same Grok-4.3 model and exact test rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data_splits" / "canonical" / "test_dt10_k7.json"
REVIEWER_RESULTS = ROOT / "results_reviewer_ablations_x-ai_grok-4.3"
OUTDIR = ROOT / "revision" / "figures" / "reviewer_ablations_dt10"
DOMAINS = ("content", "design", "coping", "quitting")
RATING_SCALES = {
    "content": {"Very poor": 1, "Poor": 2, "Acceptable": 3, "Good": 4, "Very good": 5},
    "design": {"Very poor": 1, "Poor": 2, "Acceptable": 3, "Good": 4, "Very good": 5},
    "coping": {"Not at all helpful": 1, "Not Helpful": 1, "Somewhat helpful": 2,
               "Moderately helpful": 3, "Very helpful": 4, "Extremely helpful": 5},
    "quitting": {"Not at all helpful": 1, "Not Helpful": 1, "Somewhat helpful": 2,
                 "Moderately helpful": 3, "Very helpful": 4, "Extremely helpful": 5},
}
SPECS = (
    ("pp_cbtact", "PP + CBT/ACT", REVIEWER_RESULTS / "pp_cbtact_dt10_k7.json", "#CC78BC"),
    ("full_pp_no_cbtact", "PP, no CBT/ACT", REVIEWER_RESULTS / "full_pp_no_cbtact_dt10_k7.json", "#0173B2"),
    ("history_ratings_only", "History + ratings", REVIEWER_RESULTS / "history_ratings_only_dt10_k7.json", "#029E73"),
    ("history_text_only", "History text only", REVIEWER_RESULTS / "history_text_only_dt10_k7.json", "#7F7F7F"),
)
FONT_CHAIN = ["Avenir", "Avenir Next", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": FONT_CHAIN, "font.size": 13,
    "font.weight": "bold", "axes.labelsize": 15, "axes.titlesize": 15,
    "axes.labelweight": "bold", "axes.titleweight": "bold", "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.linewidth": 1.4, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.linestyle": "-", "grid.alpha": 0.12, "grid.color": "#4D4D4D",
    "grid.linewidth": 0.6, "xtick.major.width": 1.3, "ytick.major.width": 1.3,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Required result is absent: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a keyed row dictionary: {path}")
    return payload


def validate_rows(rows: dict[str, dict], test: list[dict], label: str) -> None:
    expected = {str(i) for i in range(len(test))}
    if set(rows) != expected:
        raise ValueError(f"{label} is incomplete or has mismatched row keys: {len(rows)}/{len(test)}")
    for index, item in enumerate(test):
        row = rows[str(index)]
        if row.get("response_id") != item.get("response_id"):
            raise ValueError(f"{label} row {index} has a different participant")
        if row.get("input_message") != item.get("input_message"):
            raise ValueError(f"{label} row {index} has a different test message")
        for domain in DOMAINS:
            if row.get(f"ground_truth_{domain}") != (item.get("ratings") or {}).get(domain):
                raise ValueError(f"{label} row {index} has mismatched ground truth for {domain}")
            if not row.get(f"predicted_{domain}"):
                raise ValueError(f"{label} row {index} lacks predicted_{domain}")


def domain_metrics(rows: dict[str, dict], indices: np.ndarray) -> dict[str, tuple[float, float]]:
    output = {}
    for domain in DOMAINS:
        truth = np.array([rows[str(i)][f"ground_truth_{domain}"] for i in indices])
        prediction = np.array([rows[str(i)][f"predicted_{domain}"] for i in indices])
        accuracy = float(np.mean(truth == prediction))
        ordinal_truth = np.array([RATING_SCALES[domain][value] for value in truth])
        ordinal_prediction = np.array([RATING_SCALES[domain][value] for value in prediction])
        qwk = float(cohen_kappa_score(ordinal_truth, ordinal_prediction, weights="quadratic"))
        output[domain] = (accuracy, qwk)
    return output


def bootstrap(rows: dict[str, dict], test: list[dict], n_bootstrap: int, seed: int) -> dict[str, tuple[float, float]]:
    participant_rows: dict[str, list[int]] = {}
    for index, item in enumerate(test):
        participant_rows.setdefault(str(item["response_id"]), []).append(index)
    participants = np.array(list(participant_rows))
    rng = np.random.default_rng(seed)
    metrics = {domain: {"accuracy": [], "qwk": []} for domain in DOMAINS}
    for _ in range(n_bootstrap):
        sampled = rng.choice(participants, size=len(participants), replace=True)
        indices = np.array([i for pid in sampled for i in participant_rows[pid]])
        sample = domain_metrics(rows, indices)
        for domain, (accuracy, qwk) in sample.items():
            metrics[domain]["accuracy"].append(accuracy)
            metrics[domain]["qwk"].append(qwk)
    return {
        domain: (
            (float(np.quantile(metrics[domain]["accuracy"], .025)), float(np.quantile(metrics[domain]["accuracy"], .975))),
            (float(np.quantile(metrics[domain]["qwk"], .025)), float(np.quantile(metrics[domain]["qwk"], .975))),
        )
        for domain in DOMAINS
    }


def make_summary(rows_by_condition: dict[str, dict[str, dict]], test: list[dict], n_bootstrap: int) -> pd.DataFrame:
    records = []
    full_indices = np.arange(len(test))
    for index, (key, label, _path, color) in enumerate(SPECS):
        observed = domain_metrics(rows_by_condition[key], full_indices)
        intervals = bootstrap(rows_by_condition[key], test, n_bootstrap, seed=20260918 + index)
        for domain, (accuracy, qwk) in observed.items():
            acc_ci, qwk_ci = intervals[domain]
            records.append({
                "split": "dt10_k7", "n_test": len(test), "model": "Grok-4.3",
                "condition_key": key, "condition": label, "color": color, "domain": domain.capitalize(),
                "accuracy": accuracy, "accuracy_ci_low": acc_ci[0], "accuracy_ci_high": acc_ci[1],
                "qwk": qwk, "qwk_ci_low": qwk_ci[0], "qwk_ci_high": qwk_ci[1],
            })
    return pd.DataFrame(records)


def style_ticks(ax):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontfamily("Avenir")


def plot_overall(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.1), constrained_layout=True)
    condition_order = [spec[0] for spec in SPECS]
    labels = [spec[1] for spec in SPECS]
    colors = [spec[3] for spec in SPECS]
    for ax, metric, title in zip(axes, ("accuracy", "qwk"), ("Mean accuracy", "Mean quadratic-weighted κ")):
        values, lowers, uppers = [], [], []
        for key in condition_order:
            subset = summary[summary.condition_key == key]
            values.append(subset[metric].mean())
            lowers.append(subset[f"{metric}_ci_low"].mean())
            uppers.append(subset[f"{metric}_ci_high"].mean())
        values = np.asarray(values)
        errors = np.vstack([values - np.asarray(lowers), np.asarray(uppers) - values])
        ax.bar(np.arange(len(values)), values, color=colors, edgecolor=colors, alpha=.72, linewidth=1.7,
               yerr=errors, capsize=4, error_kw={"elinewidth": 1.5, "ecolor": "#4D4D4D"})
        ax.set_xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_ylim((0, 0.75) if metric == "accuracy" else (0, 0.65))
        style_ticks(ax)
    fig.suptitle("Reviewer 3 prompt-component comparison — canonical dt10-k7 test set (N = 898)", fontweight="bold", fontsize=16)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTDIR / f"reviewer_ablation_overall_dt10.{suffix}", bbox_inches="tight", dpi=400)
    plt.close(fig)


def plot_domains(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), sharey=True, constrained_layout=True)
    condition_order = [spec[0] for spec in SPECS]
    labels = ["PP +\nCBT/ACT", "PP, no\nCBT/ACT", "History +\nratings", "History text\nonly"]
    colors = [spec[3] for spec in SPECS]
    for ax, domain in zip(axes, ("Content", "Design", "Coping", "Quitting")):
        subset = summary[summary.domain == domain].set_index("condition_key").loc[condition_order]
        values = subset.accuracy.to_numpy()
        errors = np.vstack([values - subset.accuracy_ci_low.to_numpy(), subset.accuracy_ci_high.to_numpy() - values])
        ax.bar(np.arange(len(values)), values, color=colors, edgecolor=colors, alpha=.72, linewidth=1.6,
               yerr=errors, capsize=3, error_kw={"elinewidth": 1.4, "ecolor": "#4D4D4D"})
        ax.set_xticks(np.arange(len(labels)), labels)
        ax.set_title(domain)
        ax.set_ylim(0, .82)
        style_ticks(ax)
    axes[0].set_ylabel("Accuracy")
    fig.suptitle("Prediction accuracy by rating dimension — canonical dt10-k7 (N = 898)", fontweight="bold", fontsize=16)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTDIR / f"reviewer_ablation_accuracy_by_domain_dt10.{suffix}", bbox_inches="tight", dpi=400)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.n_bootstrap < 100:
        parser.error("Use at least 100 participant-level bootstrap replicates")
    test = json.loads(TEST_PATH.read_text(encoding="utf-8"))
    if len(test) != 898:
        raise SystemExit("Expected the canonical 898-row dt10-k7 test file")
    rows_by_condition = {}
    for key, label, path, _color in SPECS:
        rows = load_rows(path)
        validate_rows(rows, test, label)
        rows_by_condition[key] = rows
    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary = make_summary(rows_by_condition, test, args.n_bootstrap)
    summary.to_csv(OUTDIR / "reviewer_ablation_metrics_dt10.csv", index=False)
    plot_overall(summary)
    plot_domains(summary)
    print(f"Wrote observed dt10-k7 figures and metrics to {OUTDIR}")


if __name__ == "__main__":
    main()
