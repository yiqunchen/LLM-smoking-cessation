#!/usr/bin/env python3
"""Cacheable clustered-bootstrap CIs for the Reviewer 3 prompt ablations.

The canonical dt10-k7 test set contains repeated messages within participants.
This script therefore resamples participants (clusters), retaining all of each
sampled participant's held-out messages through sample weights.  Results are
cached separately by model and input SHA-256 hashes.  After an incomplete
model finishes, rerunning this script computes that model only and reassembles
the combined reviewer tables without re-bootstraping completed model families.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data/splits" / "canonical" / "test_dt10_k7.json"
OUTDIR = ROOT / "figures" / "prompt_ablations"
CACHE_DIR = OUTDIR / "bootstrap_cache_dt10"
SCHEMA_VERSION = 3
DOMAINS = ("content", "coping", "quitting")
RATING_SCALES = {
    "content": {"Very poor": 1, "Poor": 2, "Acceptable": 3, "Good": 4, "Very good": 5},
    "coping": {
        "Not at all helpful": 1, "Not Helpful": 1, "Somewhat helpful": 2,
        "Moderately helpful": 3, "Very helpful": 4, "Extremely helpful": 5,
    },
    "quitting": {
        "Not at all helpful": 1, "Not Helpful": 1, "Somewhat helpful": 2,
        "Moderately helpful": 3, "Very helpful": 4, "Extremely helpful": 5,
    },
}
METRICS = ("accuracy", "macro_f1", "qwk", "directional_accuracy", "directional_macro_f1")
BASELINE = "pp_cbtact"
CONDITIONS = (
    ("pp_cbtact", "PP + history + CBT/ACT"),
    ("full_pp_no_cbtact", "PP + history (no CBT/ACT)"),
    ("history_ratings_only", "History + ratings only"),
    ("history_text_only", "History text only"),
)
MODELS = (
    ("GPT-4o-mini", "#0173B2", "results/prompt_ablations/gpt-4o-mini"),
    ("GPT-5", "#DE8F05", "results/prompt_ablations/gpt-5"),
    ("DeepSeek-R1", "#029E73", "results/prompt_ablations/deepseek-r1"),
    ("Grok-4.3", "#CC78BC", "results/prompt_ablations/grok-4.3"),
    ("Gemini-2.5-Pro", "#CA9161", "results/prompt_ablations/gemini-2.5-pro"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json_write(path: Path, payload: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def model_cache_name(model: str) -> str:
    return model.lower().replace("-", "_").replace(".", "_")


def complete_rows(directory: Path, condition: str, test: list[dict]) -> dict[str, dict] | None:
    """Return a validated condition file, or None when it remains incomplete."""
    path = directory / f"{condition}_dt10_k7.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    expected = {str(index) for index in range(len(test))}
    if not isinstance(rows, dict) or set(rows) != expected:
        return None
    for index, item in enumerate(test):
        row = rows[str(index)]
        if (row.get("response_id") != item.get("response_id")
                or row.get("input_message") != item.get("input_message")):
            raise SystemExit(f"Misaligned canonical row {index}: {path}")
        ratings = item.get("ratings", {}) or {}
        for domain in DOMAINS:
            if row.get(f"ground_truth_{domain}") != ratings.get(domain):
                raise SystemExit(f"Ground-truth mismatch at row {index}, {domain}: {path}")
    return rows


def model_inputs(directory: Path, test: list[dict]) -> tuple[dict[str, dict[str, dict]] | None, dict[str, str]]:
    rows_by_condition: dict[str, dict[str, dict]] = {}
    input_hashes = {"canonical_test": sha256(TEST_PATH)}
    for condition, _label in CONDITIONS:
        path = directory / f"{condition}_dt10_k7.json"
        rows = complete_rows(directory, condition, test)
        if rows is None:
            return None, {}
        rows_by_condition[condition] = rows
        input_hashes[condition] = sha256(path)
    return rows_by_condition, input_hashes


def ordinal_metrics(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray | None) -> dict[str, float]:
    """Fast weighted metrics for fixed five-level ordinal scales.

    The fixed 1--5 scale is deliberate: every bootstrap replicate uses the
    same category spacing for quadratic weighted kappa, even if a rare label
    is absent in a particular participant resample.
    """
    if weights is None:
        weights = np.ones(len(y_true), dtype=float)
    else:
        weights = weights.astype(float, copy=False)
    n_levels = 5
    matrix = np.bincount(
        (y_true - 1) * n_levels + (y_pred - 1), weights=weights, minlength=n_levels ** 2
    ).reshape(n_levels, n_levels)
    total = matrix.sum()
    accuracy = float(np.trace(matrix) / total)
    row_sum = matrix.sum(axis=1)
    col_sum = matrix.sum(axis=0)
    denominator = row_sum + col_sum
    f1_values = np.divide(2 * np.diag(matrix), denominator, out=np.zeros(n_levels), where=denominator > 0)
    ordinal_weights = (np.subtract.outer(np.arange(n_levels), np.arange(n_levels)) ** 2) / (n_levels - 1) ** 2
    observed = matrix / total
    expected = np.outer(row_sum, col_sum) / total ** 2
    expected_disagreement = float(np.sum(ordinal_weights * expected))
    qwk = float(1 - np.sum(ordinal_weights * observed) / expected_disagreement) if expected_disagreement else float("nan")
    directional_truth = np.select([y_true <= 2, y_true == 3], [0, 1], default=2)
    directional_prediction = np.select([y_pred <= 2, y_pred == 3], [0, 1], default=2)
    directional_matrix = np.bincount(
        directional_truth * 3 + directional_prediction, weights=weights, minlength=9
    ).reshape(3, 3)
    directional_denominator = directional_matrix.sum(axis=1) + directional_matrix.sum(axis=0)
    directional_f1 = np.divide(
        2 * np.diag(directional_matrix), directional_denominator,
        out=np.zeros(3), where=directional_denominator > 0,
    )
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "qwk": qwk,
        "directional_accuracy": float(np.trace(directional_matrix) / total),
        "directional_macro_f1": float(np.mean(directional_f1)),
    }


def weighted_metrics(truth: dict[str, np.ndarray], prediction: dict[str, np.ndarray],
                     weights: np.ndarray | None) -> dict[str, dict[str, float]]:
    """Compute fixed-scale ordinal metrics separately for each tested domain."""
    result: dict[str, dict[str, float]] = {}
    for domain in DOMAINS:
        y_true = truth[domain]
        y_pred = prediction[domain]
        result[domain.capitalize()] = ordinal_metrics(y_true, y_pred, weights)
    return result


def bootstrap_one_model(model: str, color: str, rows_by_condition: dict[str, dict[str, dict]],
                        test: list[dict], n_bootstrap: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Run a participant-clustered, paired percentile bootstrap for one model."""
    participant_ids = np.asarray([str(row.get("response_id", "")) for row in test], dtype=object)
    clusters, cluster_index = np.unique(participant_ids, return_inverse=True)
    if len(clusters) < 2:
        raise SystemExit("Participant-clustered bootstrap requires at least two participants")
    truth = {
        domain: np.asarray([RATING_SCALES[domain][test[index]["ratings"][domain]] for index in range(len(test))], dtype=int)
        for domain in DOMAINS
    }
    predictions = {
        condition: {
            domain: np.asarray([
                RATING_SCALES[domain][rows[str(index)][f"predicted_{domain}"]]
                for index in range(len(test))
            ], dtype=int)
            for domain in DOMAINS
        }
        for condition, rows in rows_by_condition.items()
    }
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(clusters), size=(n_bootstrap, len(clusters)), endpoint=False)
    scopes = tuple(domain.capitalize() for domain in DOMAINS)
    point = {condition: weighted_metrics(truth, pred, None) for condition, pred in predictions.items()}
    samples = {
        condition: {scope: {metric: np.empty(n_bootstrap, dtype=float) for metric in METRICS} for scope in scopes}
        for condition in predictions
    }
    for replicate, draw in enumerate(draws):
        cluster_counts = np.bincount(draw, minlength=len(clusters))
        row_weights = cluster_counts[cluster_index]
        for condition, pred in predictions.items():
            values = weighted_metrics(truth, pred, row_weights)
            for scope in scopes:
                for metric in METRICS:
                    samples[condition][scope][metric][replicate] = values[scope][metric]
    ci_low, ci_high = 2.5, 97.5
    metric_rows: list[dict] = []
    delta_rows: list[dict] = []
    labels = dict(CONDITIONS)
    for condition, _label in CONDITIONS:
        for scope in scopes:
            for metric in METRICS:
                values = samples[condition][scope][metric]
                metric_rows.append({
                    "model": model, "model_color": color, "condition_key": condition,
                    "condition": labels[condition], "domain": scope, "metric": metric,
                    "estimate": point[condition][scope][metric],
                    "ci_low": float(np.percentile(values, ci_low)),
                    "ci_high": float(np.percentile(values, ci_high)),
                    "bootstrap_replicates": n_bootstrap, "resampling_unit": "participant",
                    "n_participants": len(clusters), "n_test_messages": len(test), "seed": seed,
                })
        if condition == BASELINE:
            continue
        for scope in scopes:
            for metric in METRICS:
                difference = samples[condition][scope][metric] - samples[BASELINE][scope][metric]
                delta_rows.append({
                    "model": model, "model_color": color, "baseline_key": BASELINE,
                    "baseline": labels[BASELINE], "comparison_key": condition,
                    "comparison": labels[condition], "domain": scope, "metric": metric,
                    "difference": point[condition][scope][metric] - point[BASELINE][scope][metric],
                    "ci_low": float(np.percentile(difference, ci_low)),
                    "ci_high": float(np.percentile(difference, ci_high)),
                    "bootstrap_replicates": n_bootstrap, "resampling_unit": "participant",
                    "n_participants": len(clusters), "n_test_messages": len(test), "seed": seed,
                })
    return metric_rows, delta_rows


def cache_is_valid(cache: dict, model: str, input_hashes: dict[str, str], n_bootstrap: int, seed: int) -> bool:
    return (
        cache.get("schema_version") == SCHEMA_VERSION
        and cache.get("model") == model
        and cache.get("input_sha256") == input_hashes
        and cache.get("n_bootstrap") == n_bootstrap
        and cache.get("seed") == seed
        and isinstance(cache.get("metrics"), list)
        and isinstance(cache.get("deltas"), list)
    )


def formatted_ci(row: pd.Series, value: str = "estimate") -> str:
    return f"{row[value]:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional third-party package."""
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        cells = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_markdown_tables(metrics: pd.DataFrame, deltas: pd.DataFrame) -> None:
    by_domain = metrics.copy()
    by_domain["CI"] = by_domain.apply(formatted_ci, axis=1)
    pivot = by_domain.pivot(index=["domain", "model", "condition"], columns="metric", values="CI").reset_index()
    standard_metrics = ["accuracy", "macro_f1", "qwk"]
    directional_metrics = ["directional_accuracy", "directional_macro_f1"]
    pivot_standard = pivot[["domain", "model", "condition", *standard_metrics]].copy()
    pivot_directional = pivot[["domain", "model", "condition", *directional_metrics]].copy()
    pivot_standard = pivot_standard.rename(columns={"domain": "Domain", "model": "Model", "condition": "Configuration", "accuracy": "Accuracy (95% CI)",
                                  "macro_f1": "Macro-F1 (95% CI)", "qwk": "QWK (95% CI)"})
    pivot_directional = pivot_directional.rename(columns={"domain": "Domain", "model": "Model", "condition": "Configuration",
        "directional_accuracy": "Directional accuracy (95% CI)",
        "directional_macro_f1": "Directional macro-F1 (95% CI)"})
    lines = [
        "# Reviewer 3 prompt ablations: domain-specific clustered-bootstrap confidence intervals",
        "",
        "Participant-clustered percentile bootstrap (2,000 replicates by default); each model/configuration uses the shared canonical dt10-k7 test set (898 messages from 301 participants). Metrics are reported separately for Content, Coping, and Quitting; no cross-domain mean is calculated.",
        "",
        markdown_table(pivot_standard),
        "",
    ]
    (OUTDIR / "reviewer_ablation_bootstrap_by_domain_table_dt10.md").write_text("\n".join(lines), encoding="utf-8")
    directional_lines = [
        "# Reviewer 3 prompt ablations: domain-specific directional performance",
        "",
        "Directional ratings use the established three-bin definition: low (1–2), neutral (3), and high (4–5). Intervals use the same participant-clustered percentile bootstrap.",
        "",
        markdown_table(pivot_directional),
        "",
    ]
    (OUTDIR / "reviewer_ablation_bootstrap_directional_by_domain_table_dt10.md").write_text("\n".join(directional_lines), encoding="utf-8")
    delta = deltas.copy()
    delta["CI"] = delta.apply(lambda row: formatted_ci(row, "difference"), axis=1)
    delta = delta[["model", "comparison", "metric", "CI"]].rename(
        columns={"model": "Model", "comparison": "Compared with PP + history + CBT/ACT", "metric": "Metric", "CI": "Difference (95% CI)"}
    )
    delta_lines = [
        "# Paired configuration differences from PP + history + CBT/ACT",
        "",
        "Positive differences favor the comparison configuration. Each interval uses the same participant-bootstrap replicate for both configurations.",
        "",
        markdown_table(delta),
        "",
    ]
    (OUTDIR / "reviewer_ablation_bootstrap_deltas_by_domain_vs_pp_cbtact_dt10.md").write_text("\n".join(delta_lines), encoding="utf-8")
    directional_delta = delta[delta["Metric"].isin(directional_metrics)]
    directional_delta_lines = [
        "# Paired directional-performance differences from PP + history + CBT/ACT",
        "",
        "Positive differences favor the comparison configuration. Directional ratings use low (1–2), neutral (3), and high (4–5).",
        "",
        markdown_table(directional_delta),
        "",
    ]
    (OUTDIR / "reviewer_ablation_bootstrap_directional_deltas_by_domain_vs_pp_cbtact_dt10.md").write_text("\n".join(directional_delta_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--force", action="store_true", help="recompute valid per-model caches")
    args = parser.parse_args()
    if args.n_bootstrap < 200:
        parser.error("Use at least 200 bootstrap replicates")
    test = json.loads(TEST_PATH.read_text(encoding="utf-8"))
    if not test:
        raise SystemExit("Canonical test file is empty")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict] = []
    all_deltas: list[dict] = []
    status_rows: list[dict] = []
    for model_index, (model, color, relative_dir) in enumerate(MODELS):
        rows_by_condition, input_hashes = model_inputs(ROOT / relative_dir, test)
        if rows_by_condition is None:
            status_rows.append({"model": model, "status": "pending", "detail": "one or more condition files are incomplete"})
            print(f"PENDING: {model}; no bootstrap run until all four conditions have {len(test)} rows.", flush=True)
            continue
        cache_path = CACHE_DIR / f"{model_cache_name(model)}.json"
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else None
        if cache is not None and not args.force and cache_is_valid(cache, model, input_hashes, args.n_bootstrap, args.seed):
            metrics, deltas = cache["metrics"], cache["deltas"]
            status = "cached"
            print(f"CACHED: {model}; input hashes and bootstrap settings match.", flush=True)
        else:
            print(f"BOOTSTRAP: {model}; {args.n_bootstrap} participant-clustered replicates.", flush=True)
            metrics, deltas = bootstrap_one_model(model, color, rows_by_condition, test, args.n_bootstrap,
                                                  args.seed + model_index * 1009)
            atomic_json_write(cache_path, {
                "schema_version": SCHEMA_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
                "model": model, "input_sha256": input_hashes, "n_bootstrap": args.n_bootstrap,
                "seed": args.seed, "metrics": metrics, "deltas": deltas,
            })
            status = "computed"
        all_metrics.extend(metrics)
        all_deltas.extend(deltas)
        status_rows.append({"model": model, "status": status, "detail": "four complete conditions"})
    status = pd.DataFrame(status_rows)
    status.to_csv(OUTDIR / "reviewer_ablation_bootstrap_status_dt10.csv", index=False)
    if not all_metrics:
        raise SystemExit("No complete model families are available for bootstrap analysis")
    metrics_df = pd.DataFrame(all_metrics)
    deltas_df = pd.DataFrame(all_deltas)
    metrics_df.to_csv(OUTDIR / "reviewer_ablation_bootstrap_metrics_dt10.csv", index=False)
    deltas_df.to_csv(OUTDIR / "reviewer_ablation_bootstrap_deltas_vs_pp_cbtact_dt10.csv", index=False)
    write_markdown_tables(metrics_df, deltas_df)
    print(f"Wrote bootstrap tables for {metrics_df['model'].nunique()} complete model families; status table includes pending models.")


if __name__ == "__main__":
    main()
