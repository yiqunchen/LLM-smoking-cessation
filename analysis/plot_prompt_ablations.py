#!/usr/bin/env python3
"""Figures for the prompt ablations on the canonical dt10-k7 split, by domain.

Reads the bootstrap tables written by ``bootstrap_prompt_ablations.py`` plus the
raw result JSONs.  Only models whose four conditions are complete (those present
in the bootstrap table) are drawn; pending models are listed in
``prompt_ablation_figure_status_dt10.csv`` and never drawn as placeholders.
Every metric is reported separately for Content, Coping, and Quitting.

Outputs (figures/prompt_ablations/, PNG + PDF unless noted):
  prompt_ablation_metrics_by_domain_dt10             accuracy / macro-F1 / QWK (rows) x domains (columns)
  prompt_ablation_directional_by_domain_dt10         directional accuracy / macro-F1 x domains
  prompt_ablation_rating_distributions_dt10          human vs predicted rating distributions, conditions x domains
  prompt_ablation_pairwise_<metric>_dt10             paired differences for all condition pairs, colour = verdict
  prompt_ablation_rating_distribution_dt10.csv       long table behind the distribution figures
  prompt_ablation_prediction_summary_dt10.csv        mean/SD of predictions, bias, MAE, within-1 rate
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


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
    ("pp_cbtact", "PP +\nCBT/ACT", "PP + CBT/ACT"),
    ("full_pp_no_cbtact", "PP, no\nCBT/ACT", "PP, no CBT/ACT"),
    ("history_ratings_only", "History +\nratings", "History + ratings"),
    ("history_text_only", "History text\nonly", "History text only"),
)
CONDITION_COLORS = {"pp_cbtact": "#0173B2", "full_pp_no_cbtact": "#DE8F05",
                    "history_ratings_only": "#029E73", "history_text_only": "#CC78BC"}
OBSERVED_COLOR = "#4D4D4D"
MODELS = (
    ("GPT-4o-mini", "#0173B2", "results/prompt_ablations/gpt-4o-mini"),
    ("GPT-5", "#DE8F05", "results/prompt_ablations/gpt-5"),
    ("DeepSeek-R1", "#029E73", "results/prompt_ablations/deepseek-r1"),
    ("Grok-4.3", "#CC78BC", "results/prompt_ablations/grok-4.3"),
    ("Gemini-2.5-Pro", "#CA9161", "results/prompt_ablations/gemini-2.5-pro"),
)
METRIC_SPECS = {
    "accuracy": ("Exact accuracy", (0, .75)),
    "macro_f1": ("Macro-F1", (0, .75)),
    "qwk": ("QWK", (-.05, .65)),
    "directional_accuracy": ("Directional accuracy", (0, 1.0)),
    "directional_macro_f1": ("Directional macro-F1", (0, 1.0)),
}
STANDARD_METRICS = ("accuracy", "macro_f1", "qwk")
DIRECTIONAL_METRICS = ("directional_accuracy", "directional_macro_f1")
FONT_CHAIN = ["Helvetica", "Arial", "DejaVu Sans"]
LEGEND_FONT = {"family": "Helvetica", "weight": "bold", "size": 14}

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


def style(ax) -> None:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Helvetica")
        label.set_fontweight("bold")


def save(fig, name: str, legend=None) -> None:
    extra = (legend,) if legend is not None else ()
    for extension in ("png", "pdf"):
        # No timestamps in the PDF metadata, so an unchanged figure is byte-identical across rebuilds.
        fig.savefig(OUTDIR / f"{name}.{extension}", bbox_inches="tight", bbox_extra_artists=extra, dpi=400,
                    metadata={"CreationDate": None, "ModDate": None} if extension == "pdf" else None)
    plt.close(fig)


def slug(model: str) -> str:
    return model.lower().replace("-", "_").replace(".", "_")


def load_rows(path: Path, test: list[dict]) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    expected = {str(index) for index in range(len(test))}
    if not isinstance(rows, dict) or set(rows) != expected:
        raise SystemExit(f"Incomplete result: {path}")
    for index, item in enumerate(test):
        row = rows[str(index)]
        if row.get("response_id") != item.get("response_id") or row.get("input_message") != item.get("input_message"):
            raise SystemExit(f"Misaligned dt10-k7 row {index}: {path}")
    return rows


# --------------------------------------------------------------------------
# Metric figures with 95% CI whiskers
# --------------------------------------------------------------------------

def metric_panel(ax, metrics: pd.DataFrame, models: list[tuple[str, str]], metric: str, domain: str) -> None:
    """Point estimates only; the bootstrap intervals are reported in the tables and the Word report."""
    xs = np.arange(len(CONDITIONS))
    for model, color in models:
        block = metrics[(metrics.model == model) & (metrics.domain == domain) & (metrics.metric == metric)]
        block = block.set_index("condition_key").reindex([key for key, *_ in CONDITIONS])
        ax.plot(xs, block["estimate"].to_numpy(), marker="o", color=color, linewidth=2.2, markersize=6, label=model)
    lower, upper = METRIC_SPECS[metric][1]
    sub = metrics[(metrics.metric == metric) & (metrics.model.isin([m for m, _ in models]))]
    lower = min(lower, float(np.floor((sub["estimate"].min() - .02) * 20) / 20))
    upper = max(upper, float(np.ceil((sub["estimate"].max() + .02) * 20) / 20))
    ax.set_ylim(lower, upper)
    ax.set_xticks(xs, [label for _key, label, _short in CONDITIONS])
    style(ax)


def figure_metric_grid(metrics: pd.DataFrame, models: list[tuple[str, str]], metric_list: tuple[str, ...], name: str) -> None:
    rows = len(metric_list)
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4.4 * rows), sharex=True, constrained_layout=True)
    axes = np.atleast_2d(axes)
    for r, metric in enumerate(metric_list):
        for c, domain in enumerate(("Content", "Coping", "Quitting")):
            ax = axes[r, c]
            metric_panel(ax, metrics, models, metric, domain)
            if r == 0:
                ax.set_title(domain)
            if c == 0:
                ax.set_ylabel(METRIC_SPECS[metric][0])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, -.01), ncol=len(models), prop=LEGEND_FONT)
    save(fig, name, legend)


# --------------------------------------------------------------------------
# Rating distributions
# --------------------------------------------------------------------------

def distributions(predictions: dict[str, dict[str, dict[str, np.ndarray]]], truth: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long tables: rating-level proportions and (predicted - observed) proportions, plus summary statistics."""
    dist_rows, summary_rows = [], []
    levels = np.arange(1, 6)
    errors = np.arange(-4, 5)
    for domain in DOMAINS:
        y = truth[domain]
        counts = np.bincount(y, minlength=6)[1:]
        for level, count in zip(levels, counts):
            dist_rows.append({"model": "Observed", "condition_key": "observed", "condition": "Observed", "domain": domain.capitalize(),
                              "kind": "rating", "value": int(level), "count": int(count), "proportion": count / len(y)})
    for model, by_condition in predictions.items():
        for condition_key, _label, short in CONDITIONS:
            for domain in DOMAINS:
                y, p = truth[domain], by_condition[condition_key][domain]
                counts = np.bincount(p, minlength=6)[1:]
                for level, count in zip(levels, counts):
                    dist_rows.append({"model": model, "condition_key": condition_key, "condition": short, "domain": domain.capitalize(),
                                      "kind": "rating", "value": int(level), "count": int(count), "proportion": count / len(p)})
                error = p - y
                ecounts = np.bincount(error + 4, minlength=9)
                for value, count in zip(errors, ecounts):
                    dist_rows.append({"model": model, "condition_key": condition_key, "condition": short, "domain": domain.capitalize(),
                                      "kind": "signed_error", "value": int(value), "count": int(count), "proportion": count / len(p)})
                summary_rows.append({
                    "model": model, "condition_key": condition_key, "condition": short, "domain": domain.capitalize(),
                    "n": int(len(p)), "observed_mean": float(y.mean()), "observed_sd": float(y.std(ddof=1)),
                    "predicted_mean": float(p.mean()), "predicted_sd": float(p.std(ddof=1)),
                    "bias_predicted_minus_observed": float(error.mean()), "mean_absolute_error": float(np.abs(error).mean()),
                    "exact_rate": float((error == 0).mean()), "within_one_rate": float((np.abs(error) <= 1).mean()),
                    "over_rate": float((error > 0).mean()), "under_rate": float((error < 0).mean()),
                    "predicted_level_proportions": " ".join(f"{v:.3f}" for v in np.bincount(p, minlength=6)[1:] / len(p)),
                })
    return pd.DataFrame(dist_rows), pd.DataFrame(summary_rows)


def figure_rating_grid(dist: pd.DataFrame, models: list[tuple[str, str]]) -> None:
    """Observed vs predicted rating distributions: rows = prompt conditions, columns = domains.

    Same construction as the manuscript's score-distribution figure: the human
    rating distribution is a wide light-gray background bar and each model's
    predicted distribution is a narrow coloured bar drawn on top of it.
    """
    levels = np.arange(1, 6)
    shown = dist[(dist.kind == "rating") & (dist.model.isin(["Observed"] + [m for m, _ in models]))]
    y_max = 100 if shown["proportion"].max() > .9 else 90
    n_rows, n_cols = len(CONDITIONS), 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2.9 * n_rows), sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_2d(axes)
    total_width = .78
    bar_width = min(.13, total_width / max(1, len(models)) * .86)
    offsets = np.linspace(-total_width / 2 + bar_width / 2, total_width / 2 - bar_width / 2, len(models)) if len(models) > 1 else [0.0]
    for r, (key, _label, short) in enumerate(CONDITIONS):
        for c, domain in enumerate(("Content", "Coping", "Quitting")):
            ax = axes[r, c]
            ax.grid(axis="y", alpha=.12, color=OBSERVED_COLOR, linestyle="--", linewidth=.6)
            human = dist[(dist.model == "Observed") & (dist.domain == domain) & (dist.kind == "rating")].set_index("value").reindex(levels)
            ax.bar(levels, 100 * human["proportion"].to_numpy(), width=.82, color=OBSERVED_COLOR, alpha=.16,
                   edgecolor=OBSERVED_COLOR, linewidth=.9, zorder=2)
            for (model, color), offset in zip(models, offsets):
                block = dist[(dist.model == model) & (dist.condition_key == key) & (dist.domain == domain) & (dist.kind == "rating")]
                block = block.set_index("value").reindex(levels)
                ax.bar(levels + offset, 100 * block["proportion"].to_numpy(), width=bar_width, color=color, alpha=.88,
                       edgecolor="black", linewidth=.8, zorder=3)
            ax.set_ylim(0, y_max)
            ax.set_yticks([0, y_max / 2, y_max], ["0%", f"{y_max // 2}%", f"{y_max}%"])
            ax.set_xticks(levels, [str(level) for level in levels])
            if r == 0:
                ax.set_title(domain)
            if c == 0:
                ax.set_ylabel(short.replace(", ", ",\n").replace(" + ", " +\n"))
            if r == n_rows - 1:
                ax.set_xlabel("Rating")
            style(ax)
    handles = [Patch(facecolor=OBSERVED_COLOR, alpha=.16, edgecolor=OBSERVED_COLOR, label="Human ratings")]
    handles += [Patch(facecolor=color, alpha=.88, edgecolor="black", label=model) for model, color in models]
    legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.01), ncol=len(handles), prop=LEGEND_FONT)
    save(fig, "prompt_ablation_rating_distributions_dt10", legend)


# --------------------------------------------------------------------------
# Pairwise comparison heat maps
# --------------------------------------------------------------------------

def figure_pairwise(pairwise: pd.DataFrame, models: list[tuple[str, str]], metric: str) -> None:
    short = {key: s for key, _l, s in CONDITIONS}
    keys = [key for key, *_ in CONDITIONS]
    pairs = [(keys[i], keys[j]) for i in range(len(keys)) for j in range(i + 1, len(keys))]
    row_labels, cells, verdicts = [], [], []
    for model, _color in models:
        for domain in ("Content", "Coping", "Quitting"):
            row_labels.append(f"{model} · {domain}")
            block = pairwise[(pairwise.model == model) & (pairwise.domain == domain) & (pairwise.metric == metric)]
            values, flags = [], []
            for reference, comparison in pairs:
                hit = block[(block.reference_key == reference) & (block.comparison_key == comparison)].iloc[0]
                values.append(hit["difference"])
                flags.append(hit["verdict"])
            cells.append(values)
            verdicts.append(flags)
    colors = {"comparison better": mpl.colors.to_rgba("#0173B2", .45), "reference better": mpl.colors.to_rgba("#DE8F05", .45),
              "comparable": (1, 1, 1, 1)}
    rgba = np.array([[colors[v] for v in row] for row in verdicts])
    fig, ax = plt.subplots(figsize=(13, .42 * len(row_labels) + 1.8), constrained_layout=True)
    ax.imshow(rgba, aspect="auto", interpolation="nearest")
    for r, row in enumerate(cells):
        for c, value in enumerate(row):
            ax.text(c, r, f"{value:+.3f}", ha="center", va="center", fontsize=10, family="Helvetica",
                    weight="bold" if verdicts[r][c] != "comparable" else "normal", color="black")
    ax.set_xticks(range(len(pairs)), [f"{short[b]}\n− {short[a]}" for a, b in pairs], fontsize=10)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=10)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
    for r in range(3, len(row_labels), 3):
        ax.axhline(r - .5, color="#4D4D4D", linewidth=1.2)
    for c in range(1, len(pairs)):
        ax.axvline(c - .5, color="#4D4D4D", linewidth=.4, alpha=.5)
    ax.tick_params(length=0)
    style(ax)
    ax.set_xlabel(f"Δ {METRIC_SPECS[metric][0]} (comparison − reference)")
    handles = [Patch(facecolor=colors["comparison better"], edgecolor="#4D4D4D", label="Comparison better (95% CI > 0)"),
               Patch(facecolor=colors["reference better"], edgecolor="#4D4D4D", label="Reference better (95% CI < 0)"),
               Patch(facecolor="white", edgecolor="#4D4D4D", label="Comparable (CI includes 0)")]
    legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.01), ncol=3, prop=LEGEND_FONT)
    save(fig, f"prompt_ablation_pairwise_{metric}_dt10", legend)


# --------------------------------------------------------------------------

def main() -> None:
    metrics_path = OUTDIR / "prompt_ablation_bootstrap_metrics_dt10.csv"
    pairwise_path = OUTDIR / "prompt_ablation_bootstrap_pairwise_dt10.csv"
    for path in (metrics_path, pairwise_path):
        if not path.exists():
            raise SystemExit(f"Missing {path.name}; run analysis/bootstrap_prompt_ablations.py first")
    metrics = pd.read_csv(metrics_path)
    pairwise = pd.read_csv(pairwise_path)
    test = json.loads(TEST_PATH.read_text(encoding="utf-8"))
    if len(test) != 898:
        raise SystemExit("Expected the 898-row canonical dt10-k7 test file")
    completed = [(model, color) for model, color, _dir in MODELS if model in set(metrics["model"])]
    pending = [model for model, _color, _dir in MODELS if model not in set(metrics["model"])]
    if not completed:
        raise SystemExit("No complete model families in the bootstrap table")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"model": m, "status": "plotted"} for m, _ in completed] + [{"model": m, "status": "pending"} for m in pending]) \
        .to_csv(OUTDIR / "prompt_ablation_figure_status_dt10.csv", index=False)

    truth = {domain: np.asarray([RATING_SCALES[domain][item["ratings"][domain]] for item in test], dtype=int) for domain in DOMAINS}
    predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    directories = {model: directory for model, _color, directory in MODELS}
    for model, _color in completed:
        predictions[model] = {}
        for key, *_ in CONDITIONS:
            rows = load_rows(ROOT / directories[model] / f"{key}_dt10_k7.json", test)
            predictions[model][key] = {
                domain: np.asarray([RATING_SCALES[domain][rows[str(i)][f"predicted_{domain}"]] for i in range(len(test))], dtype=int)
                for domain in DOMAINS
            }
    dist, summary = distributions(predictions, truth)
    dist.to_csv(OUTDIR / "prompt_ablation_rating_distribution_dt10.csv", index=False)
    summary.to_csv(OUTDIR / "prompt_ablation_prediction_summary_dt10.csv", index=False)

    for metric in METRIC_SPECS:
        figure_pairwise(pairwise, completed, metric)
    figure_metric_grid(metrics, completed, STANDARD_METRICS, "prompt_ablation_metrics_by_domain_dt10")
    figure_metric_grid(metrics, completed, DIRECTIONAL_METRICS, "prompt_ablation_directional_by_domain_dt10")
    figure_rating_grid(dist, completed)
    print(f"Wrote prompt-ablation figures for {len(completed)} complete model(s) to {OUTDIR}"
          + (f"; pending: {', '.join(pending)}" if pending else ""))


if __name__ == "__main__":
    main()
