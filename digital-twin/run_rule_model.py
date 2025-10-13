#!/usr/bin/env python3
"""Rule-based message scorer combined with participant baselines.

Workflow:
1. Use export_rule_examples.py to inspect the training split and craft rules.
2. Encode rules in JSON (see --rules-json format) derived from TRAIN only.
3. Run this script to fit a lightweight linear model on the training split,
   report validation metrics, freeze coefficients, and optionally evaluate on
   the held-out test split (without touching its labels during fitting).
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

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
FEATURE_NAMES = [
    "baseline_mean",
    "positive_weight_sum",
    "negative_weight_sum",
    "net_weight_sum",
    "token_length",
    "auto_positive_weight_sum",
    "auto_negative_weight_sum",
    "auto_net_weight_sum",
]


@dataclass
class RuleSet:
    positive: List[Tuple[str, float]]
    negative: List[Tuple[str, float]]


@dataclass
class BaselineStats:
    sum_by_pid: Dict[str, float]
    count_by_pid: Dict[str, int]
    global_sum: float
    global_count: int

    def global_mean(self) -> float:
        if self.global_count == 0:
            return 3.0
        return self.global_sum / self.global_count


@dataclass
class AutoLexicon:
    positive: Dict[str, float]
    negative: Dict[str, float]


def load_dataset(data_json: str) -> pd.DataFrame:
    with open(data_json, "r") as f:
        raw = json.load(f)
    rows = []
    for item in raw:
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


def load_split(path: str) -> set[str]:
    with open(path, "r") as f:
        return {line.strip() for line in f if line.strip()}


def parse_rules(rules_json: str) -> Dict[str, RuleSet]:
    with open(rules_json, "r") as f:
        raw = json.load(f)
    parsed: Dict[str, RuleSet] = {}
    for dim in DIMENSIONS:
        dim_rules = raw.get(dim, {}) or {}
        pos_entries = dim_rules.get("positive", [])
        neg_entries = dim_rules.get("negative", [])
        parsed[dim] = RuleSet(
            positive=_parse_rule_entries(pos_entries),
            negative=_parse_rule_entries(neg_entries),
        )
    return parsed


def _parse_rule_entries(entries: List[Any]) -> List[Tuple[str, float]]:
    parsed: List[Tuple[str, float]] = []
    for item in entries:
        if isinstance(item, str):
            parsed.append((item.lower(), 1.0))
        elif isinstance(item, dict):
            phrase = item.get("phrase")
            if not phrase:
                continue
            weight = float(item.get("weight", 1.0))
            parsed.append((phrase.lower(), weight))
    return parsed


def build_baseline_stats(df: pd.DataFrame) -> BaselineStats:
    grouped = df.groupby("participant_id")["label"].agg(["sum", "count"]).reset_index()
    sum_by_pid = {row["participant_id"]: float(row["sum"]) for _, row in grouped.iterrows()}
    count_by_pid = {row["participant_id"]: int(row["count"]) for _, row in grouped.iterrows()}
    global_sum = float(df["label"].sum())
    global_count = int(len(df))
    return BaselineStats(sum_by_pid, count_by_pid, global_sum, global_count)


def compute_baseline(pid: str, stats: BaselineStats, remove_label: float | None = None) -> float:
    sum_pid = stats.sum_by_pid.get(pid)
    count_pid = stats.count_by_pid.get(pid)
    if sum_pid is None or count_pid is None or count_pid == 0:
        base = stats.global_mean()
    else:
        if remove_label is not None and count_pid > 1:
            base = (sum_pid - remove_label) / (count_pid - 1)
        elif remove_label is not None and count_pid == 1:
            global_count = stats.global_count - 1
            if global_count > 0:
                base = (stats.global_sum - remove_label) / global_count
            else:
                base = stats.global_mean()
        else:
            base = sum_pid / count_pid
    return float(base)


def count_rule_hits(message: str, rules: RuleSet) -> Tuple[float, float, float]:
    text = message.lower()
    pos_sum = 0.0
    neg_sum = 0.0
    for phrase, weight in rules.positive:
        if phrase and phrase in text:
            pos_sum += weight
    for phrase, weight in rules.negative:
        if phrase and phrase in text:
            neg_sum += weight
    net = pos_sum - neg_sum
    return pos_sum, neg_sum, net


def derive_auto_lexicon(
    df_dim: pd.DataFrame,
    top_k: int,
    min_count: int,
    include_bigrams: bool,
) -> AutoLexicon:
    if df_dim.empty or top_k <= 0:
        return AutoLexicon(positive={}, negative={})
    global_mean = df_dim["label"].mean()
    token_stats: Dict[str, Tuple[float, int]] = {}
    for row in df_dim.itertuples(index=False):
        tokens = re.findall(r"[a-z0-9']+", row.input_message.lower())
        token_set = set(tokens)
        if include_bigrams and len(tokens) >= 2:
            token_set.update(f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1))
        for tok in token_set:
            total, count = token_stats.get(tok, (0.0, 0))
            token_stats[tok] = (total + row.label, count + 1)
    scored: List[Tuple[str, float, int]] = []
    for tok, (label_sum, count) in token_stats.items():
        if count < min_count:
            continue
        avg = label_sum / count
        diff = avg - global_mean
        if diff == 0:
            continue
        scored.append((tok, diff, count))
    if not scored:
        return AutoLexicon(positive={}, negative={})
    pos = sorted(
        (s for s in scored if s[1] > 0),
        key=lambda x: (x[1], x[2]),
        reverse=True,
    )[:top_k]
    neg = sorted(
        (s for s in scored if s[1] < 0),
        key=lambda x: (x[1], x[2]),
    )[:top_k]
    return AutoLexicon(
        positive={tok: float(diff) for tok, diff, _ in pos},
        negative={tok: float(-diff) for tok, diff, _ in neg},
    )


def tokenize_for_lexicon(message: str) -> Tuple[set[str], set[str]]:
    tokens = re.findall(r"[a-z0-9']+", message.lower())
    token_set = set(tokens)
    bigrams = set()
    if len(tokens) >= 2:
        bigrams = {f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)}
    return token_set, bigrams


def compute_auto_weights(message: str, lexicon: AutoLexicon) -> Tuple[float, float, float]:
    if not lexicon.positive and not lexicon.negative:
        return 0.0, 0.0, 0.0
    token_set, bigram_set = tokenize_for_lexicon(message)
    pos_sum = 0.0
    neg_sum = 0.0
    for token, weight in lexicon.positive.items():
        if (" " in token and token in bigram_set) or (" " not in token and token in token_set):
            pos_sum += weight
    for token, weight in lexicon.negative.items():
        if (" " in token and token in bigram_set) or (" " not in token and token in token_set):
            neg_sum += weight
    net = pos_sum - neg_sum
    return pos_sum, neg_sum, net


def featurize_dataframe(
    df: pd.DataFrame,
    rules: RuleSet,
    stats: BaselineStats,
    auto_lexicon: AutoLexicon,
    drop_self: bool,
) -> np.ndarray:
    feats = np.zeros((len(df), len(FEATURE_NAMES)), dtype=float)
    for idx, row in enumerate(df.itertuples(index=False)):
        label = getattr(row, "label") if drop_self else None
        baseline = compute_baseline(row.participant_id, stats, remove_label=label if drop_self else None)
        pos_sum, neg_sum, net_sum = count_rule_hits(row.input_message, rules)
        token_length = len(row.input_message.split())
        auto_pos, auto_neg, auto_net = compute_auto_weights(row.input_message, auto_lexicon)
        feats[idx, :] = [
            baseline,
            pos_sum,
            neg_sum,
            net_sum,
            token_length,
            auto_pos,
            auto_neg,
            auto_net,
        ]
    return feats


def make_estimator(model_type: str, random_state: int) -> Any:
    if model_type == "linear":
        return LinearRegression()
    if model_type == "logistic":
        return LogisticRegression(multi_class="multinomial", max_iter=1000, random_state=random_state)
    if model_type == "rf":
        return RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
    if model_type == "gbm":
        return GradientBoostingRegressor(random_state=random_state)
    raise ValueError(f"Unknown model type: {model_type}")


def fit_dimension_model(
    dim: str,
    df_dim: pd.DataFrame,
    rules: RuleSet,
    auto_lexicon: AutoLexicon,
    val_size: float,
    random_state: int,
    model_type: str,
) -> Tuple[Any, Dict[str, Any], BaselineStats]:
    if df_dim.empty:
        raise ValueError(f"No training data for dimension {dim}")
    if df_dim["label"].nunique() > 1 and 0 < val_size < 1.0:
        train_df, val_df = train_test_split(
            df_dim,
            test_size=val_size,
            random_state=random_state,
            stratify=df_dim["label"],
        )
    else:
        train_df, val_df = df_dim, pd.DataFrame(columns=df_dim.columns)

    train_stats = build_baseline_stats(train_df)
    X_train = featurize_dataframe(train_df, rules, train_stats, auto_lexicon, drop_self=True)
    y_train = train_df["label"].values

    model = make_estimator(model_type, random_state)
    model.fit(X_train, y_train)

    metrics: Dict[str, Any] = {"dimension": dim, "val_accuracy": None, "val_mae": None, "val_mse": None}

    if not val_df.empty:
        X_val = featurize_dataframe(val_df, rules, train_stats, auto_lexicon, drop_self=False)
        y_val = val_df["label"].values
        if model_type == "logistic":
            y_pred = model.predict(X_val).astype(int)
        else:
            y_pred = np.clip(np.round(model.predict(X_val)), 1, 5)
        metrics.update(
            {
                "val_accuracy": float(np.mean(y_val == y_pred)),
                "val_mae": float(np.mean(np.abs(y_val - y_pred))),
                "val_mse": float(np.mean((y_val - y_pred) ** 2)),
            }
        )

    # Refit on full dimension df for frozen coefficients and stats
    full_stats = build_baseline_stats(df_dim)
    X_full = featurize_dataframe(df_dim, rules, full_stats, auto_lexicon, drop_self=True)
    y_full = df_dim["label"].values
    full_model = make_estimator(model_type, random_state)
    full_model.fit(X_full, y_full)

    return full_model, metrics, full_stats


def predict_dimension(
    model: Any,
    df_dim: pd.DataFrame,
    rules: RuleSet,
    stats: BaselineStats,
    auto_lexicon: AutoLexicon,
    model_type: str,
    drop_self: bool,
) -> np.ndarray:
    X = featurize_dataframe(df_dim, rules, stats, auto_lexicon, drop_self=drop_self)
    if model_type == "logistic":
        preds = model.predict(X).astype(int)
    else:
        preds = np.clip(np.round(model.predict(X)), 1, 5).astype(int)
    return preds


def evaluate_predictions(preds: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    accuracy = float(np.mean(preds == truth))
    mae = float(np.mean(np.abs(preds - truth)))
    mse = float(np.mean((preds - truth) ** 2))
    return {"accuracy": accuracy, "mae": mae, "mse": mse}


def write_metrics(path: str, header: str, entries: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [f"# {header}", "", "| Dimension | Accuracy | MAE | MSE |", "| --- | --- | --- | --- |"]
    for ent in entries:
        acc = ent.get("accuracy")
        mae = ent.get("mae")
        mse = ent.get("mse")
        lines.append(
            f"| {ent['dimension']} | {acc if acc is not None else 'NA'} | {mae if mae is not None else 'NA'} | {mse if mse is not None else 'NA'} |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate rule-based models")
    parser.add_argument("--data-json", default="data/processed_llm_data.json")
    parser.add_argument("--split-type", choices=["message", "participant"], default="message")
    parser.add_argument("--messages-train", default="digital-twin/splits/messages_train.txt")
    parser.add_argument("--messages-test", default="digital-twin/splits/messages_test.txt")
    parser.add_argument("--participants-train", default="digital-twin/splits/participants_train.txt")
    parser.add_argument("--participants-test", default="digital-twin/splits/participants_test.txt")
    parser.add_argument("--rules-json", required=True, help="Rule definitions derived from training")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out-dir", default="digital-twin/rule_model")
    parser.add_argument("--skip-test", action="store_true", help="Skip evaluation on held-out test split")
    parser.add_argument(
        "--model-type",
        choices=["linear", "logistic", "rf", "gbm"],
        default="linear",
        help="Base estimator for the rule model",
    )
    parser.add_argument("--auto-top-k", type=int, default=25, help="Auto lexicon size per polarity")
    parser.add_argument(
        "--auto-min-count",
        type=int,
        default=10,
        help="Minimum occurrences for auto lexicon tokens",
    )
    parser.add_argument(
        "--auto-bigrams",
        action="store_true",
        help="Include bigrams when deriving auto lexicon",
    )
    parser.add_argument(
        "--train-pred-csv",
        type=str,
        default=None,
        help="Optional path to save combined training predictions",
    )
    parser.add_argument(
        "--test-pred-csv",
        type=str,
        default=None,
        help="Optional path to save combined test predictions",
    )
    args = parser.parse_args()

    df_all = load_dataset(args.data_json)
    if args.split_type == "message":
        train_msgs = load_split(args.messages_train)
        test_msgs = load_split(args.messages_test)
        df_train = df_all[df_all["input_message"].isin(train_msgs)].copy()
        df_test = df_all[df_all["input_message"].isin(test_msgs)].copy()
    else:
        train_p = load_split(args.participants_train)
        test_p = load_split(args.participants_test)
        df_train = df_all[df_all["participant_id"].isin(train_p)].copy()
        df_test = df_all[df_all["participant_id"].isin(test_p)].copy()

    rules_by_dim = parse_rules(args.rules_json)

    os.makedirs(args.out_dir, exist_ok=True)
    val_reports = []
    metadata: Dict[str, Dict[str, Any]] = {}
    full_models: Dict[str, Tuple[Any, BaselineStats, AutoLexicon]] = {}
    train_pred_rows: Optional[List[Dict[str, Any]]] = [] if args.train_pred_csv else None
    test_pred_rows: Optional[List[Dict[str, Any]]] = [] if args.test_pred_csv else None
    auto_lexica: Dict[str, Dict[str, Dict[str, float]]] = {}

    for dim in DIMENSIONS:
        dim_train = df_train[df_train["dimension"] == dim].copy()
        auto_lex = derive_auto_lexicon(
            dim_train,
            top_k=args.auto_top_k,
            min_count=args.auto_min_count,
            include_bigrams=args.auto_bigrams,
        )
        model, val_metrics, stats = fit_dimension_model(
            dim,
            dim_train,
            rules_by_dim[dim],
            auto_lex,
            args.val_size,
            args.random_state,
            args.model_type,
        )
        val_reports.append({
            "dimension": dim,
            "accuracy": val_metrics.get("val_accuracy"),
            "mae": val_metrics.get("val_mae"),
            "mse": val_metrics.get("val_mse"),
        })
        model_path = os.path.join(args.out_dir, f"{dim}_model.joblib")
        joblib.dump(model, model_path)
        metadata[dim] = {
            "model_type": args.model_type,
            "model_path": model_path,
            "train_samples": int(len(dim_train)),
            "feature_names": FEATURE_NAMES,
        }
        auto_lexica[dim] = {
            "positive": auto_lex.positive,
            "negative": auto_lex.negative,
        }
        stats_path = os.path.join(args.out_dir, f"{dim}_baseline_stats.json")
        with open(stats_path, "w") as f:
            json.dump(
                {
                    "sum_by_pid": stats.sum_by_pid,
                    "count_by_pid": stats.count_by_pid,
                    "global_sum": stats.global_sum,
                    "global_count": stats.global_count,
                },
                f,
                indent=2,
            )
        metadata[dim]["baseline_stats_path"] = stats_path
        metadata[dim]["auto_lexicon_positive"] = len(auto_lex.positive)
        metadata[dim]["auto_lexicon_negative"] = len(auto_lex.negative)
        full_models[dim] = (model, stats, auto_lex)

        if train_pred_rows is not None:
            train_preds = predict_dimension(
                model,
                dim_train,
                rules_by_dim[dim],
                stats,
                auto_lex,
                args.model_type,
                drop_self=True,
            )
            for row, pred in zip(dim_train.itertuples(index=False), train_preds):
                train_pred_rows.append({
                    "participant_id": row.participant_id,
                    "input_message": row.input_message,
                    "dimension": row.dimension,
                    "predicted": int(pred),
                })

        if test_pred_rows is not None:
            dim_test = df_test[df_test["dimension"] == dim].copy()
            if not dim_test.empty:
                test_preds = predict_dimension(
                    model,
                    dim_test,
                    rules_by_dim[dim],
                    stats,
                    auto_lex,
                    args.model_type,
                    drop_self=False,
                )
                for row, pred in zip(dim_test.itertuples(index=False), test_preds):
                    test_pred_rows.append({
                        "participant_id": row.participant_id,
                        "input_message": row.input_message,
                        "dimension": row.dimension,
                        "predicted": int(pred),
                    })

    metrics_path = os.path.join(args.out_dir, "validation_metrics.md")
    write_metrics(metrics_path, "Validation Metrics (train split only)", val_reports)
    print(f"Wrote validation metrics to {metrics_path}")

    meta_path = os.path.join(args.out_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to {meta_path}")

    lexicon_path = os.path.join(args.out_dir, "auto_lexicon.json")
    with open(lexicon_path, "w") as f:
        json.dump(auto_lexica, f, indent=2)
    print(f"Saved auto lexicon to {lexicon_path}")

    if train_pred_rows is not None and args.train_pred_csv:
        train_df_out = pd.DataFrame(train_pred_rows)
        train_df_out.sort_values(["participant_id", "input_message", "dimension"], inplace=True)
        os.makedirs(os.path.dirname(args.train_pred_csv), exist_ok=True)
        train_df_out.to_csv(args.train_pred_csv, index=False)
        print(f"Saved train predictions to {args.train_pred_csv}")

    if test_pred_rows is not None and args.test_pred_csv:
        test_df_out = pd.DataFrame(test_pred_rows)
        test_df_out.sort_values(["participant_id", "input_message", "dimension"], inplace=True)
        os.makedirs(os.path.dirname(args.test_pred_csv), exist_ok=True)
        test_df_out.to_csv(args.test_pred_csv, index=False)
        print(f"Saved test predictions to {args.test_pred_csv}")

    if not args.skip_test:
        test_reports = []
        for dim in DIMENSIONS:
            dim_test = df_test[df_test["dimension"] == dim].copy()
            if dim_test.empty:
                test_reports.append({"dimension": dim, "accuracy": None, "mae": None, "mse": None})
                continue
            model, stats, auto_lex = full_models[dim]
            preds = predict_dimension(
                model,
                dim_test,
                rules_by_dim[dim],
                stats,
                auto_lex,
                args.model_type,
                drop_self=False,
            )
            metrics = evaluate_predictions(preds, dim_test["label"].values)
            metrics["dimension"] = dim
            test_reports.append(metrics)
            pred_path = os.path.join(args.out_dir, f"predictions_{dim}.csv")
            out_df = dim_test.copy()
            out_df["predicted"] = preds
            out_df.to_csv(pred_path, index=False)
        test_metrics_path = os.path.join(args.out_dir, "test_metrics.md")
        write_metrics(test_metrics_path, "Held-out Test Metrics", test_reports)
        print(f"Wrote test metrics to {test_metrics_path}")


if __name__ == "__main__":
    main()
