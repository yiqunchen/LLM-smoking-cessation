#!/usr/bin/env python3
"""Create validated five-model figures for the Reviewer 3 prompt ablations.

The script refuses to pool or plot a model until all four conditions contain
the exact 898 canonical dt10-k7 rows.  It emits observed metrics only—never
placeholder values for unfinished model runs.
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
TEST_PATH = ROOT / "data/splits" / "canonical" / "test_dt10_k7.json"
OUTDIR = ROOT / "figures" / "prompt_ablations"
DOMAINS = ("content", "coping", "quitting")
RATING_SCALES = {
    "content": {"Very poor": 1, "Poor": 2, "Acceptable": 3, "Good": 4, "Very good": 5},
    "coping": {"Not at all helpful": 1, "Not Helpful": 1, "Somewhat helpful": 2,
               "Moderately helpful": 3, "Very helpful": 4, "Extremely helpful": 5},
    "quitting": {"Not at all helpful": 1, "Not Helpful": 1, "Somewhat helpful": 2,
                 "Moderately helpful": 3, "Very helpful": 4, "Extremely helpful": 5},
}
CONDITIONS = (
    ("pp_cbtact", "PP +\nCBT/ACT"),
    ("full_pp_no_cbtact", "PP, no\nCBT/ACT"),
    ("history_ratings_only", "History +\nratings"),
    ("history_text_only", "History text\nonly"),
)
MODELS = (
    ("GPT-4o-mini", "#0173B2", "results/prompt_ablations/gpt-4o-mini"),
    ("GPT-5", "#DE8F05", "results/prompt_ablations/gpt-5"),
    ("DeepSeek-R1", "#029E73", "results/prompt_ablations/deepseek-r1"),
    ("Grok-4.3", "#CC78BC", "results/prompt_ablations/grok-4.3"),
    ("Gemini-2.5-Pro", "#CA9161", "results/prompt_ablations/gemini-2.5-pro"),
)
DIRECTIONAL_BUCKET_MAP = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
METRIC_SPECS = {
    "accuracy": ("Exact accuracy", (0, .75)),
    "macro_f1": ("Macro-F1", (0, .75)),
    "qwk": ("QWK", (-.05, .65)),
    "directional_accuracy": ("Directional accuracy", (0, 1.0)),
    "directional_macro_f1": ("Directional macro-F1", (0, 1.0)),
}
PANEL_YLABELS = {
    "accuracy": "Exact accuracy",
    "macro_f1": "Macro-F1",
    "qwk": "QWK",
    "directional_accuracy": "Directional accuracy",
    "directional_macro_f1": "Directional macro-F1",
}
FONT_CHAIN = ["Helvetica", "Arial", "DejaVu Sans"]

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": FONT_CHAIN, "font.size": 13,
    "font.weight": "bold", "axes.labelsize": 15, "axes.titlesize": 15,
    "axes.labelweight": "bold", "axes.titleweight": "bold", "figure.facecolor": "white",
    "axes.facecolor": "white", "figure.constrained_layout.use": True, "axes.linewidth": 1.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.grid.axis": "y", "grid.linestyle": "-", "grid.alpha": .12,
    "grid.color": "#4D4D4D", "grid.linewidth": .6, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load_and_validate(path: Path, test: list[dict], condition: str) -> dict[str, dict]:
    if not path.exists():
        raise SystemExit(f"Missing required result: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    expected = {str(index) for index in range(len(test))}
    if not isinstance(rows, dict) or set(rows) != expected:
        raise SystemExit(f"Incomplete result for {condition}: {path}")
    for index, item in enumerate(test):
        row = rows[str(index)]
        if row.get("response_id") != item.get("response_id") or row.get("input_message") != item.get("input_message"):
            raise SystemExit(f"Misaligned dt10-k7 row {index}: {path}")
    return rows


def metrics(rows: dict[str, dict], test: list[dict]) -> list[dict]:
    output = []
    for domain in DOMAINS:
        truth = [rows[str(index)][f"ground_truth_{domain}"] for index in range(len(test))]
        prediction = [rows[str(index)][f"predicted_{domain}"] for index in range(len(test))]
        ordinal_truth = [RATING_SCALES[domain][value] for value in truth]
        ordinal_prediction = [RATING_SCALES[domain][value] for value in prediction]
        directional_truth = [DIRECTIONAL_BUCKET_MAP[value] for value in ordinal_truth]
        directional_prediction = [DIRECTIONAL_BUCKET_MAP[value] for value in ordinal_prediction]
        output.append({
            "domain": domain.capitalize(),
            "accuracy": float(np.mean(np.asarray(truth) == np.asarray(prediction))),
            "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
            "qwk": float(cohen_kappa_score(ordinal_truth, ordinal_prediction, weights="quadratic")),
            "directional_accuracy": float(np.mean(np.asarray(directional_truth) == np.asarray(directional_prediction))),
            "directional_macro_f1": float(f1_score(directional_truth, directional_prediction, average="macro", zero_division=0)),
        })
    return output


def style(ax):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Helvetica")
        label.set_fontweight("bold")


def plot_by_domain(records: pd.DataFrame, metric: str, ylabel: str, name: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharex=True, constrained_layout=True)
    xs = np.arange(len(CONDITIONS))
    for axis, domain in zip(axes.flat, ("Content", "Coping", "Quitting")):
        domain_df = records[records.domain == domain]
        for model, color, _directory in MODELS:
            subset = domain_df[domain_df.model == model].set_index("condition_key")
            axis.plot(xs, [subset.loc[key, metric] for key, _label in CONDITIONS],
                      marker="o", color=color, linewidth=2.2, markersize=6, label=model)
        axis.set_title(domain)
        axis.set_ylim(METRIC_SPECS[metric][1])
        axis.set_xticks(xs, [label for _key, label in CONDITIONS])
        style(axis)
    axes[0].set_ylabel(PANEL_YLABELS[metric])
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.03), ncol=5,
                        prop={"family": "Helvetica", "weight": "bold", "size": 10}, frameon=False)
    for extension in ("png", "pdf"):
        fig.savefig(OUTDIR / f"{name}.{extension}",
                    bbox_inches="tight", bbox_extra_artists=(legend,), dpi=400)
    plt.close(fig)


def main() -> None:
    test = json.loads(TEST_PATH.read_text(encoding="utf-8"))
    if len(test) != 898:
        raise SystemExit("Expected the 898-row canonical dt10-k7 test file")
    records = []
    for model, color, directory in MODELS:
        for condition_key, condition_label in CONDITIONS:
            rows = load_and_validate(ROOT / directory / f"{condition_key}_dt10_k7.json", test, f"{model}: {condition_key}")
            for row in metrics(rows, test):
                records.append({
                    "split": "dt10_k7", "n_test": len(test), "model": model, "model_color": color,
                    "condition_key": condition_key, "condition": condition_label.replace("\n", " "), **row,
                })
    results = pd.DataFrame(records)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTDIR / "reviewer_ablation_all_models_metrics_dt10.csv", index=False)
    metric_columns = list(METRIC_SPECS)
    for metric, (ylabel, _ylim) in METRIC_SPECS.items():
        plot_by_domain(results, metric, ylabel, f"reviewer_ablation_all_models_{metric}_by_domain_dt10")
    print(f"Wrote validated five-model reviewer figures to {OUTDIR}")


if __name__ == "__main__":
    main()
