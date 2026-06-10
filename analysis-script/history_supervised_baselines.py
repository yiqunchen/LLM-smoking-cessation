#!/usr/bin/env python3
"""
Digital-twin supervised baselines with coarse history-score features.

Runs dense supervised baselines on the canonical 70/30 digital-twin split.
History conditions use average prior rating scores rather than richer
message-history features. The full results table includes LR and RF for the
reduced feature families, including the reviewer-requested Embedding + History
baseline, and the companion figure shows the best of LR/RF per feature set.

Outputs:
  - revision/figures/history_supervised_baselines.csv
  - revision/figures/history_supervised_best_summary.csv
  - revision/figures/history_supervised_best.png/pdf

Usage:
  uv run python analysis-script/history_supervised_baselines.py
  uv run python analysis-script/history_supervised_baselines.py --plot-only
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from revision_utils import (
    COLORS,
    DOMAINS,
    RATING_MAPS,
    align_features_labels,
    apply_repo_plot_style,
    compute_all_metrics,
    extract_demographic_features,
    extract_features,
    extract_labels,
    figures_path,
    load_canonical_data,
    save_figure,
)
from text_baselines import _load_embeddings, _match_embeddings
from filter_duplicates import get_duplicate_signatures

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
ALL_HISTORY_DOMAINS = ["content", "design", "coping", "quitting"]

FEATURE_SET_ORDER = [
    "Demographics",
    "Embedding",
    "Embedding + Demo",
    "Avg History Score",
    "Embedding + History",
    "Demographics + History + Message Embedding",
]

FEATURE_SET_SPECS = {
    "Demographics": {"demo": True, "history": False, "embedding": False},
    "Embedding": {"demo": False, "history": False, "embedding": True},
    "Embedding + Demo": {"demo": True, "history": False, "embedding": True},
    "Avg History Score": {"demo": False, "history": True, "embedding": False},
    "Embedding + History": {"demo": False, "history": True, "embedding": True},
    "Demographics + History + Message Embedding": {"demo": True, "history": True, "embedding": True},
}

FEATURE_SET_COLORS = {
    "Demographics": COLORS["Random Forest"],
    "Embedding": COLORS["GPT-4o-mini"],
    "Embedding + Demo": "#56B4E9",
    "Avg History Score": COLORS["GPT-5"],
    "Embedding + History": COLORS["DeepSeek-R1"],
    "Demographics + History + Message Embedding": COLORS["DeepSeek-R1"],
}

CLASSIFIER_LABELS = {"lr": "LR", "rf": "RF"}
METRICS_FOR_PLOT = ["Accuracy", "F1"]


def _response_ids(records: list[dict]) -> np.ndarray:
    return np.array([row["response_id"] for row in records])


def _normalize_message(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _make_item_key_from_record(row: dict) -> str:
    ratings = row.get("ratings", {})
    rating_parts = []
    for domain in ["content", "design", "coping", "quitting"]:
        rating_parts.append(str(ratings.get(domain, "NA")))
    return "||".join([
        str(row.get("response_id", "")),
        _normalize_message(row.get("input_message", "")),
        *rating_parts,
    ])


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


def _clip_round(values: np.ndarray) -> np.ndarray:
    return np.clip(np.round(values).astype(int), 1, 5)


def _build_history_lookup(records: list[dict]) -> dict[str, list[tuple[int, dict]]]:
    lookup: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for idx, row in enumerate(records):
        lookup[row["response_id"]].append((idx, row))
    return lookup


def _rating_values(history_rows: list[dict], domain: str) -> np.ndarray:
    vals = []
    for row in history_rows:
        rating_text = row.get("ratings", {}).get(domain)
        rating_num = RATING_MAPS[domain].get(rating_text)
        if rating_num is not None:
            vals.append(rating_num)
    return np.asarray(vals, dtype=float)


def _history_feature_row(history_rows: list[dict]) -> dict[str, float]:
    feat: dict[str, float] = {}
    overall_vals = []
    for domain in ALL_HISTORY_DOMAINS:
        vals = _rating_values(history_rows, domain)
        if len(vals):
            overall_vals.extend(vals.tolist())

    if overall_vals:
        overall_arr = np.asarray(overall_vals, dtype=float)
        feat["avg_history_overall"] = float(np.mean(overall_arr))
    else:
        feat["avg_history_overall"] = 0.0

    return feat


def extract_history_features(
    records: list[dict],
    history_lookup: dict[str, list[tuple[int, dict]]],
    exclude_self: bool,
) -> pd.DataFrame:
    rows = []
    for idx, row in enumerate(records):
        candidates = history_lookup.get(row["response_id"], [])
        if exclude_self:
            history_rows = [
                hist_row
                for hist_idx, hist_row in candidates
                if hist_idx != idx and hist_row["input_message"] != row["input_message"]
            ]
        else:
            history_rows = [
                hist_row for _, hist_row in candidates
                if hist_row["input_message"] != row["input_message"]
            ]
        rows.append(_history_feature_row(history_rows))
    return pd.DataFrame(rows).fillna(0.0)


def build_feature_matrix(
    records: list[dict],
    spec: dict[str, bool],
    demo_df: pd.DataFrame,
    history_df: pd.DataFrame,
    emb_matrix: np.ndarray,
    emb_lookup: dict[tuple[str, str], int],
) -> tuple[np.ndarray, np.ndarray]:
    if spec["embedding"]:
        emb_array, valid_indices = _match_embeddings(records, emb_matrix, emb_lookup)
        valid_indices = np.asarray(valid_indices, dtype=int)
    else:
        emb_array = None
        valid_indices = np.arange(len(records), dtype=int)

    pieces = []
    if spec["embedding"]:
        pieces.append(np.asarray(emb_array, dtype=float))
    if spec["demo"]:
        pieces.append(np.asarray(demo_df.iloc[valid_indices].values, dtype=float))
    if spec["history"]:
        pieces.append(np.asarray(history_df.iloc[valid_indices].values, dtype=float))

    if not pieces:
        raise ValueError("At least one feature family must be enabled.")

    X = pieces[0] if len(pieces) == 1 else np.hstack(pieces)
    return np.asarray(X, dtype=float), valid_indices


def fit_predict_dense(
    train_data: list[dict],
    test_data: list[dict],
    domain: str,
    feature_set: str,
    classifier_type: str,
    train_demo: pd.DataFrame,
    test_demo: pd.DataFrame,
    train_history: pd.DataFrame,
    test_history: pd.DataFrame,
    emb_matrix: np.ndarray,
    emb_lookup: dict[tuple[str, str], int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    spec = FEATURE_SET_SPECS[feature_set]
    y_train = extract_labels(train_data, domain)
    y_test = extract_labels(test_data, domain)

    X_train, valid_train_idx = build_feature_matrix(
        train_data, spec, train_demo, train_history, emb_matrix, emb_lookup
    )
    X_test, valid_test_idx = build_feature_matrix(
        test_data, spec, test_demo, test_history, emb_matrix, emb_lookup
    )

    y_tr_all = y_train[valid_train_idx]
    y_te_all = y_test[valid_test_idx]
    valid_label_train = ~np.isnan(y_tr_all)
    valid_label_test = ~np.isnan(y_te_all)

    X_tr = X_train[valid_label_train]
    X_te = X_test[valid_label_test]
    y_tr = y_tr_all[valid_label_train].astype(int)
    gt = y_te_all[valid_label_test].astype(int)
    rids = _response_ids(test_data)[valid_test_idx][valid_label_test]

    if classifier_type == "lr":
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    elif classifier_type == "rf":
        clf = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=100,
        )
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    clf.fit(X_tr, y_tr)
    pred = _clip_round(clf.predict(X_te).astype(float))
    valid_record_indices = valid_test_idx[valid_label_test]
    valid_records = [test_data[int(i)] for i in valid_record_indices]
    return gt, pred, rids, valid_records


def run_all_history_baselines() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading canonical 70/30 digital-twin data...")
    train_data, test_data = load_canonical_data("7030", "digital_twin")
    print(f"  Train: {len(train_data)} items, Test: {len(test_data)} items")

    dup_keys = _duplicate_item_keys()
    original_test_n = len(test_data)
    test_data = [row for row in test_data if _make_item_key_from_record(row) not in dup_keys]
    print(f"  Removed {original_test_n - len(test_data)} exact train/test duplicate test items")

    print("Loading embeddings...")
    emb_matrix, emb_lookup = _load_embeddings()
    print(f"  Embeddings: {emb_matrix.shape}")

    print("Building feature tables...")
    train_history_lookup = _build_history_lookup(train_data)
    train_demo = extract_demographic_features(train_data)
    test_demo = extract_demographic_features(test_data)
    train_demo, test_demo = align_features_labels(train_demo, test_demo)
    train_history = extract_history_features(train_data, train_history_lookup, exclude_self=True)
    test_history = extract_history_features(test_data, train_history_lookup, exclude_self=False)

    results = []
    prediction_rows = []
    for feature_set in FEATURE_SET_ORDER:
        for classifier_type in ["lr", "rf"]:
            for domain in DOMAINS:
                print(f"  Running: {feature_set} / {CLASSIFIER_LABELS[classifier_type]} / {domain}...")
                gt, pred, rids, valid_records = fit_predict_dense(
                    train_data=train_data,
                    test_data=test_data,
                    domain=domain,
                    feature_set=feature_set,
                    classifier_type=classifier_type,
                    train_demo=train_demo,
                    test_demo=test_demo,
                    train_history=train_history,
                    test_history=test_history,
                    emb_matrix=emb_matrix,
                    emb_lookup=emb_lookup,
                )
                metrics = compute_all_metrics(gt, pred, rids)
                results.append({
                    "Feature_Set": feature_set,
                    "Classifier": CLASSIFIER_LABELS[classifier_type],
                    "Domain": domain.capitalize(),
                    **metrics,
                })
                for record, gt_val, pred_val in zip(valid_records, gt, pred):
                    prediction_rows.append({
                        "Feature_Set": feature_set,
                        "Classifier": CLASSIFIER_LABELS[classifier_type],
                        "Domain": domain.capitalize(),
                        "response_id": record["response_id"],
                        "input_message": record["input_message"],
                        "Item_Key": _make_item_key_from_record(record),
                        "Ground_Truth_Num": int(gt_val),
                        "Predicted_Num": int(pred_val),
                    })
                print(
                    f"    Accuracy={metrics['Accuracy']:.3f}  "
                    f"F1={metrics['F1']:.3f}  "
                    f"QWK={metrics['QWK']:.3f}"
                )

    return pd.DataFrame(results), pd.DataFrame(prediction_rows)


def summarize_best_results(results_df: pd.DataFrame) -> pd.DataFrame:
    best_rows = []
    for domain in ["Content", "Coping", "Quitting"]:
        for metric in METRICS_FOR_PLOT:
            panel_df = results_df[results_df["Domain"] == domain]
            for feature_set in FEATURE_SET_ORDER:
                subset = panel_df[panel_df["Feature_Set"] == feature_set]
                if subset.empty:
                    continue
                best_idx = subset[metric].astype(float).idxmax()
                row = subset.loc[best_idx]
                best_rows.append({
                    "Domain": domain,
                    "Metric": metric,
                    "Feature_Set": feature_set,
                    "Winning_Classifier": row["Classifier"],
                    "Value": float(row[metric]),
                })
    return pd.DataFrame(best_rows)


def plot_best_summary(best_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    for row_idx, metric in enumerate(METRICS_FOR_PLOT):
        for col_idx, domain in enumerate(["Content", "Coping", "Quitting"]):
            ax = axes[row_idx, col_idx]
            panel = best_df[(best_df["Metric"] == metric) & (best_df["Domain"] == domain)].copy()
            panel = panel.sort_values("Value", ascending=False).reset_index(drop=True)

            y_pos = np.arange(len(panel))
            colors = [FEATURE_SET_COLORS[name] for name in panel["Feature_Set"]]
            bars = ax.barh(
                y_pos,
                panel["Value"],
                color=colors,
                edgecolor="#4D4D4D",
                linewidth=1.2,
                alpha=0.9,
            )
            ax.set_yticks(y_pos)
            ax.set_yticklabels(panel["Feature_Set"], fontsize=12, fontweight="bold")
            ax.invert_yaxis()
            ax.set_title(f"{domain} | {metric}", fontsize=17, fontweight="bold")
            ax.set_xlabel(metric, fontsize=16, fontweight="bold")

            xmax = max(0.5, float(panel["Value"].max()) * 1.18)
            ax.set_xlim(0, xmax)

            for bar, (_, row) in zip(bars, panel.iterrows()):
                x = float(bar.get_width())
                y = bar.get_y() + bar.get_height() / 2
                ax.text(
                    x + xmax * 0.015,
                    y,
                    f"{x:.3f} ({row['Winning_Classifier']})",
                    va="center",
                    ha="left",
                    fontsize=11,
                    fontweight="bold",
                    color="#2B2B2B",
                )

    apply_repo_plot_style(fig, axes)
    for ax in axes.ravel():
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", visible=False)

    fig.suptitle(
        "Reduced Supervised Baselines on the 70/30 PP Split",
        fontsize=19,
        fontweight="bold",
    )
    save_figure(fig, figures_path("history_supervised_best"))


def parse_args():
    parser = argparse.ArgumentParser(description="Build supervised history baseline artifacts.")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Reload saved baseline CSVs and regenerate history_supervised_best.png/pdf only.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_path = f"{figures_path('history_supervised_baselines')}.csv"
    prediction_path = f"{figures_path('history_supervised_predictions')}.csv"
    best_path = f"{figures_path('history_supervised_best_summary')}.csv"

    if args.plot_only:
        if os.path.exists(best_path):
            best_df = pd.read_csv(best_path)
        elif os.path.exists(results_path):
            results_df = pd.read_csv(results_path)
            best_df = summarize_best_results(results_df)
            best_df.to_csv(best_path, index=False)
            print(f"Saved best summary: {best_path}")
        else:
            raise FileNotFoundError(
                f"Missing cached baseline source CSV: {best_path} or {results_path}"
            )
        print(f"Reloaded best summary: {best_path}")
        plot_best_summary(best_df)
        return

    results_df, prediction_df = run_all_history_baselines()
    results_df.to_csv(results_path, index=False)
    print(f"Saved metrics: {results_path}")

    prediction_df.to_csv(prediction_path, index=False)
    print(f"Saved predictions: {prediction_path}")

    best_df = summarize_best_results(results_df)
    best_df.to_csv(best_path, index=False)
    print(f"Saved best summary: {best_path}")

    plot_best_summary(best_df)


if __name__ == "__main__":
    main()
