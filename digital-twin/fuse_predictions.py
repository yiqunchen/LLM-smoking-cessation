#!/usr/bin/env python3
"""Fuse multiple prediction sources for the digital-twin message split.

Requirements:
- Train/test splits defined in digital-twin/splits/messages_{train,test}.txt
- Predictions CSVs with columns: participant_id,input_message,dimension,predicted
  (one row per (participant,message,dimension)).
- Truth labels supplied via data/processed_llm_data.json.

The script fits a Ridge regression on the training split using supplied model
predictions + the participant baseline and evaluates on both train and test.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

CD_MAP = {
    "Very poor": 1,
    "Poor": 2,
    "Acceptable": 3,
    "Good": 4,
    "Very good": 5,
}
CQ_MAP = {
    "Not at all helpful": 1,
    "Slightly helpful": 2,
    "Moderately helpful": 3,
    "Very helpful": 4,
    "Extremely helpful": 5,
}
DIMENSIONS = ["content", "design", "coping", "quitting"]


@dataclass
class BaselineStats:
    sum_by_pid_dim: Dict[Tuple[str, str], float]
    count_by_pid_dim: Dict[Tuple[str, str], int]
    global_sum_by_dim: Dict[str, float]
    global_count_by_dim: Dict[str, int]

    def baseline(self, pid: str, dim: str, label: int | None = None, drop_self: bool = False) -> float:
        key = (pid, dim)
        total = self.sum_by_pid_dim.get(key, 0.0)
        count = self.count_by_pid_dim.get(key, 0)
        if count == 0:
            gs = self.global_sum_by_dim.get(dim, 0.0)
            gc = self.global_count_by_dim.get(dim, 0)
            return (gs / gc) if gc else 3.0
        if drop_self and label is not None and count > 1:
            return (total - label) / (count - 1)
        if drop_self and label is not None and count == 1:
            gs = self.global_sum_by_dim.get(dim, 0.0)
            gc = self.global_count_by_dim.get(dim, 0)
            if gc > 1:
                return (gs - label) / (gc - 1)
        return total / count


def load_truth(data_json: str) -> pd.DataFrame:
    with open(data_json, "r") as f:
        data = json.load(f)
    rows: List[Dict[str, str | int]] = []
    for item in data:
        pid = item["response_id"]
        msg = item["input_message"]
        ratings = item.get("ratings", {}) or {}
        for dim in ["content", "design"]:
            lbl = CD_MAP.get(ratings.get(dim))
            if lbl is not None:
                rows.append({
                    "participant_id": pid,
                    "input_message": msg,
                    "dimension": dim,
                    "label": lbl,
                })
        for dim in ["coping", "quitting"]:
            lbl = CQ_MAP.get(ratings.get(dim))
            if lbl is not None:
                rows.append({
                    "participant_id": pid,
                    "input_message": msg,
                    "dimension": dim,
                    "label": lbl,
                })
    return pd.DataFrame(rows)


def load_split(path: str) -> set[str]:
    with open(path, "r") as f:
        return {line.strip() for line in f if line.strip()}


def load_predictions(path: str, name: str) -> Dict[Tuple[str, str, str], int]:
    df = pd.read_csv(path)
    required_cols = {"participant_id", "input_message", "dimension", "predicted"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Prediction file {path} missing columns: {missing}")
    mapping: Dict[Tuple[str, str, str], int] = {}
    for row in df.itertuples(index=False):
        key = (row.participant_id, row.input_message, row.dimension)
        mapping[key] = int(getattr(row, "predicted"))
    return mapping


def compute_baseline_stats(train_df: pd.DataFrame) -> BaselineStats:
    grouped = train_df.groupby(["participant_id", "dimension"])['label'].agg(['sum', 'count']).reset_index()
    sum_by = {(row.participant_id, row.dimension): float(row['sum']) for _, row in grouped.iterrows()}
    count_by = {(row.participant_id, row.dimension): int(row['count']) for _, row in grouped.iterrows()}
    global_group = train_df.groupby('dimension')['label'].agg(['sum', 'count']).reset_index()
    global_sum = {row.dimension: float(row['sum']) for _, row in global_group.iterrows()}
    global_count = {row.dimension: int(row['count']) for _, row in global_group.iterrows()}
    return BaselineStats(sum_by, count_by, global_sum, global_count)


def build_dataset(
    df_truth: pd.DataFrame,
    model_preds: Dict[str, Dict[Tuple[str, str, str], int]],
    baseline_stats: BaselineStats,
    drop_self: bool,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rows = []
    features: List[List[float]] = []
    labels: List[int] = []
    skipped = 0
    for row in df_truth.itertuples(index=False):
        key = (row.participant_id, row.input_message, row.dimension)
        feature_row: List[float] = []
        missing = False
        for name, mapping in model_preds.items():
            value = mapping.get(key)
            if value is None:
                missing = True
                break
            feature_row.append(float(value))
        if missing:
            skipped += 1
            continue
        baseline = baseline_stats.baseline(row.participant_id, row.dimension, row.label if drop_self else None, drop_self=drop_self)
        feature_row.append(baseline)
        features.append(feature_row)
        labels.append(row.label)
        rows.append({
            "participant_id": row.participant_id,
            "input_message": row.input_message,
            "dimension": row.dimension,
            "label": row.label,
        })
    if skipped:
        print(f"Warning: skipped {skipped} rows due to missing predictions")
    if not features:
        raise RuntimeError("No rows available after merging predictions")
    X = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float)
    meta_df = pd.DataFrame(rows)
    return X, y, meta_df


def evaluate_predictions(df: pd.DataFrame, preds: np.ndarray) -> Dict[str, Dict[str, float]]:
    df = df.copy()
    df["predicted"] = preds
    metrics: Dict[str, Dict[str, float]] = {}
    for dim in DIMENSIONS:
        sub = df[df["dimension"] == dim]
        if sub.empty:
            metrics[dim] = {"accuracy": float('nan'), "mae": float('nan'), "mse": float('nan')}
            continue
        y_true = sub["label"].values
        y_pred = sub["predicted"].values
        metrics[dim] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mse": float(mean_squared_error(y_true, y_pred)),
        }
    overall = {
        "accuracy": float(accuracy_score(df["label"], df["predicted"])),
        "mae": float(mean_absolute_error(df["label"], df["predicted"])),
        "mse": float(mean_squared_error(df["label"], df["predicted"])),
    }
    metrics["overall"] = overall
    return metrics


def write_markdown(metrics: Dict[str, Dict[str, float]], title: str, path: str) -> None:
    lines = [f"# {title}", "", "| Dimension | Accuracy | MAE | MSE |", "| --- | --- | --- | --- |"]
    for dim in DIMENSIONS:
        m = metrics.get(dim, {})
        lines.append(
            f"| {dim} | {m.get('accuracy', 'NA')} | {m.get('mae', 'NA')} | {m.get('mse', 'NA')} |"
        )
    overall = metrics.get("overall", {})
    lines.append("")
    lines.append(
        f"**Overall** — Accuracy: {overall.get('accuracy', 'NA')}, MAE: {overall.get('mae', 'NA')}, MSE: {overall.get('mse', 'NA')}"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


def save_predictions(df: pd.DataFrame, preds: np.ndarray, out_path: str) -> pd.DataFrame:
    df = df.copy()
    df["predicted"] = preds
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse model predictions using Ridge regression")
    parser.add_argument("--data-json", default="data/processed_llm_data.json")
    parser.add_argument("--messages-train", default="digital-twin/splits/messages_train.txt")
    parser.add_argument("--messages-test", default="digital-twin/splits/messages_test.txt")
    parser.add_argument(
        "--train-model",
        action="append",
        default=[],
        help="Training predictions as name=path.csv (repeat per model)",
    )
    parser.add_argument(
        "--test-model",
        action="append",
        default=[],
        help="Test predictions as name=path.csv (repeat per model)",
    )
    parser.add_argument("--out-dir", default="digital-twin/fusion")
    args = parser.parse_args()

    if not args.train_model or not args.test_model:
        raise ValueError("Provide at least one --train-model and --test-model entry")

    train_pairs = {}
    for entry in args.train_model:
        if "=" not in entry:
            raise ValueError(f"Invalid --train-model entry: {entry}")
        name, path = entry.split("=", 1)
        train_pairs[name.strip()] = path.strip()
    test_pairs = {}
    for entry in args.test_model:
        if "=" not in entry:
            raise ValueError(f"Invalid --test-model entry: {entry}")
        name, path = entry.split("=", 1)
        test_pairs[name.strip()] = path.strip()

    model_names = sorted(set(train_pairs.keys()) & set(test_pairs.keys()))
    if not model_names:
        raise ValueError("No overlapping model names across train/test predictions")
    missing_train = set(test_pairs.keys()) - set(train_pairs.keys())
    missing_test = set(train_pairs.keys()) - set(test_pairs.keys())
    if missing_train:
        print(f"Warning: ignoring test-only models {missing_train}")
    if missing_test:
        print(f"Warning: ignoring train-only models {missing_test}")

    truth = load_truth(args.data_json)
    train_msgs = load_split(args.messages_train)
    test_msgs = load_split(args.messages_test)
    truth_train = truth[truth["input_message"].isin(train_msgs)].copy()
    truth_test = truth[truth["input_message"].isin(test_msgs)].copy()

    train_pred_maps = {name: load_predictions(train_pairs[name], name) for name in model_names}
    test_pred_maps = {name: load_predictions(test_pairs[name], name) for name in model_names}

    baseline_stats = compute_baseline_stats(truth_train)

    X_train, y_train, meta_train = build_dataset(truth_train, train_pred_maps, baseline_stats, drop_self=True)
    X_test, y_test, meta_test = build_dataset(truth_test, test_pred_maps, baseline_stats, drop_self=False)

    feature_names = [f"model_{name}" for name in model_names] + ["participant_baseline"]

    model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    model.fit(X_train, y_train)

    train_continuous = model.predict(X_train)
    test_continuous = model.predict(X_test)
    train_preds = np.clip(np.round(train_continuous), 1, 5).astype(int)
    test_preds = np.clip(np.round(test_continuous), 1, 5).astype(int)

    train_metrics = evaluate_predictions(meta_train, train_preds)
    test_metrics = evaluate_predictions(meta_test, test_preds)

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "ridge_coefficients.npy"), model.coef_)
    with open(os.path.join(args.out_dir, "ridge_details.json"), "w") as f:
        json.dump(
            {
                "feature_names": feature_names,
                "coefficients": model.coef_.tolist(),
                "intercept": float(model.intercept_),
                "chosen_alpha": float(model.alpha_),
            },
            f,
            indent=2,
        )

    train_pred_df = save_predictions(meta_train, train_preds, os.path.join(args.out_dir, "train_fusion_predictions.csv"))
    test_pred_df = save_predictions(meta_test, test_preds, os.path.join(args.out_dir, "test_fusion_predictions.csv"))

    train_errors = train_pred_df[train_pred_df["label"] != train_pred_df["predicted"]]
    test_errors = test_pred_df[test_pred_df["label"] != test_pred_df["predicted"]]
    train_errors.to_csv(os.path.join(args.out_dir, "train_errors.csv"), index=False)
    test_errors.to_csv(os.path.join(args.out_dir, "test_errors.csv"), index=False)

    write_markdown(train_metrics, "Fusion Metrics (train split)", os.path.join(args.out_dir, "fusion_metrics_train.md"))
    write_markdown(test_metrics, "Fusion Metrics (test split)", os.path.join(args.out_dir, "fusion_metrics_test.md"))

    print("Fusion complete. Key artifacts:")
    print(f"- Train metrics: {os.path.join(args.out_dir, 'fusion_metrics_train.md')}")
    print(f"- Test metrics: {os.path.join(args.out_dir, 'fusion_metrics_test.md')}")
    print(f"- Ridge details: {os.path.join(args.out_dir, 'ridge_details.json')}")


if __name__ == "__main__":
    main()
