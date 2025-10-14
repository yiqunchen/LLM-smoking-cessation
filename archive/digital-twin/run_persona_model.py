#!/usr/bin/env python3
"""Baseline models using persona attributes only (no message text).

Supports message-based and participant-based splits.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
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


def load_dataset(data_json: str) -> pd.DataFrame:
    with open(data_json, "r") as f:
        data = json.load(f)
    rows: List[Dict[str, Any]] = []
    for item in data:
        pid = item["response_id"]
        msg = item["input_message"]
        ratings = item.get("ratings", {}) or {}
        for dim in ["content", "design"]:
            label = CD_MAP.get(ratings.get(dim))
            if label is not None:
                rows.append({
                    "participant_id": pid,
                    "input_message": msg,
                    "dimension": dim,
                    "label": label,
                })
        for dim in ["coping", "quitting"]:
            label = CQ_MAP.get(ratings.get(dim))
            if label is not None:
                rows.append({
                    "participant_id": pid,
                    "input_message": msg,
                    "dimension": dim,
                    "label": label,
                })
    return pd.DataFrame(rows)


def load_personas(personas_path: str) -> Dict[str, Dict[str, Any]]:
    with open(personas_path, "r") as f:
        personas = json.load(f)
    return {p["participant_id"]: p for p in personas}


def flatten_persona(persona: Dict[str, Any]) -> Dict[str, Any]:
    features: Dict[str, Any] = {}
    if not persona:
        return features
    for section in ["demographics", "smoking_behavior", "psychosocial"]:
        section_data = persona.get(section) or {}
        for key, value in section_data.items():
            if isinstance(value, (int, float)):
                features[f"{section}.{key}"] = value
            else:
                val = str(value).strip()
                if val:
                    features[f"{section}.{key}={val}"] = 1
    return features


def load_split(path: str) -> set[str]:
    with open(path, "r") as f:
        return {line.strip() for line in f if line.strip()}


def evaluate_split(
    df_all: pd.DataFrame,
    personas: Dict[str, Dict[str, Any]],
    split_type: str,
    train_ids: set[str],
    test_ids: set[str],
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for dim in DIMENSIONS:
        df_dim = df_all[df_all["dimension"] == dim].copy()
        if split_type == "message":
            train_mask = df_dim["input_message"].isin(train_ids)
            test_mask = df_dim["input_message"].isin(test_ids)
        else:
            train_mask = df_dim["participant_id"].isin(train_ids)
            test_mask = df_dim["participant_id"].isin(test_ids)
        train_df = df_dim[train_mask]
        test_df = df_dim[test_mask]
        if train_df.empty or test_df.empty:
            metrics[dim] = {"accuracy": np.nan, "mae": np.nan, "mse": np.nan}
            continue

        feature_rows: List[Dict[str, Any]] = []
        labels: List[int] = []
        for row in train_df.itertuples(index=False):
            persona = personas.get(row.participant_id)
            if not persona:
                continue
            feat = flatten_persona(persona)
            if not feat:
                continue
            feature_rows.append(feat)
            labels.append(row.label)
        if not feature_rows:
            metrics[dim] = {"accuracy": np.nan, "mae": np.nan, "mse": np.nan}
            continue
        vec = DictVectorizer(sparse=True)
        X_train = vec.fit_transform(feature_rows)
        y_train = np.array(labels)

        clf = LogisticRegression(max_iter=5000, multi_class="multinomial", solver="lbfgs")
        clf.fit(X_train, y_train)

        test_features = []
        y_true = []
        for row in test_df.itertuples(index=False):
            persona = personas.get(row.participant_id)
            if not persona:
                continue
            feat = flatten_persona(persona)
            if not feat:
                continue
            test_features.append(feat)
            y_true.append(row.label)
        if not test_features:
            metrics[dim] = {"accuracy": np.nan, "mae": np.nan, "mse": np.nan}
            continue
        X_test = vec.transform(test_features)
        y_true = np.array(y_true)
        y_pred = clf.predict(X_test)
        metrics[dim] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mse": float(mean_squared_error(y_true, y_pred)),
        }
    return metrics


def summarize_markdown(metrics: Dict[str, Dict[str, float]], title: str, out_path: str) -> None:
    lines = [f"# {title}", "", "| Dimension | Accuracy | MAE | MSE |", "| --- | --- | --- | --- |"]
    for dim in DIMENSIONS:
        m = metrics.get(dim, {})
        acc = m.get("accuracy")
        mae = m.get("mae")
        mse = m.get("mse")
        lines.append(
            f"| {dim} | {acc if acc is not None else 'NA'} | {mae if mae is not None else 'NA'} | {mse if mse is not None else 'NA'} |"
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Persona-only baseline evaluator")
    parser.add_argument("--data-json", default="data/processed_llm_data.json")
    parser.add_argument("--personas-json", default="digital-twin/personas/personas.json")
    parser.add_argument("--split-type", choices=["message", "participant"], default="message")
    parser.add_argument("--messages-train", default="digital-twin/splits/messages_train.txt")
    parser.add_argument("--messages-test", default="digital-twin/splits/messages_test.txt")
    parser.add_argument("--participants-train", default="digital-twin/splits/participants_train.txt")
    parser.add_argument("--participants-test", default="digital-twin/splits/participants_test.txt")
    parser.add_argument("--out-dir", default="digital-twin/persona_model")
    args = parser.parse_args()

    df_all = load_dataset(args.data_json)
    personas = load_personas(args.personas_json)

    if args.split_type == "message":
        train_ids = load_split(args.messages_train)
        test_ids = load_split(args.messages_test)
    else:
        train_ids = load_split(args.participants_train)
        test_ids = load_split(args.participants_test)

    metrics = evaluate_split(df_all, personas, args.split_type, train_ids, test_ids)
    title = "Persona-only Logistic Regression" + (" (message split)" if args.split_type == "message" else " (participant split)")
    out_md = os.path.join(args.out_dir, f"metrics_{args.split_type}.md")
    summarize_markdown(metrics, title, out_md)
    print(f"Wrote metrics to {out_md}")


if __name__ == "__main__":
    main()
