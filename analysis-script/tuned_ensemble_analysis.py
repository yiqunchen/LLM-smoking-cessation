#!/usr/bin/env python3
"""Tune RF + LLM-DT late-fusion ensembles without evaluating on tuning rows.

This is a diagnostic analysis, distinct from the prompt-based Hybrid RF+DT runs.
For each k and domain, it aligns saved RF predictions with saved Grok DT10 LLM
predictions, then performs participant-level cross-fitting:

  - split participants into folds
  - tune RF feature set and RF/LLM blend weight on the other folds
  - evaluate the chosen ensemble on the held-out fold

Outputs:
  revision/figures/tuned_ensemble_summary.csv
  revision/figures/tuned_ensemble_per_class.csv
  revision/figures/tuned_ensemble_complementarity.csv
  revision/figures/tuned_ensemble_choices.csv
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_utils import DOMAINS, RATING_MAPS, compute_all_metrics, figures_path  # noqa: E402


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RF_PRED_CSV = figures_path("lc_dt10_rf_predictions") + ".csv"
GROK_DIR = os.path.join(PROJECT_ROOT, "results_manuscript_x-ai_grok-4-fast")

K_VALUES = [1, 3]
N_FOLDS = 5
WEIGHT_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
RANDOM_STATE = 42


@dataclass(frozen=True)
class Choice:
    feature_set: str
    w_rf: float
    tune_score: float


def _grok_path(k: int) -> str:
    return os.path.join(GROK_DIR, f"digital_twin_dt10_k{k}.json")


def _load_llm_predictions(k: int) -> pd.DataFrame:
    """Return long-form LLM predictions keyed by response/message/domain."""
    with open(_grok_path(k)) as f:
        data = json.load(f)

    rows = []
    for item in data.values():
        if not isinstance(item, dict) or item.get("predicted_content") == "ERROR":
            continue
        rid = item.get("response_id")
        msg = item.get("input_message")
        if not rid or not msg:
            continue
        for domain in DOMAINS:
            pred_label = item.get(f"predicted_{domain}")
            pred_num = RATING_MAPS[domain].get(pred_label)
            gt_label = item.get(f"ground_truth_{domain}")
            gt_num = RATING_MAPS[domain].get(gt_label)
            if pred_num is None or gt_num is None:
                continue
            rows.append(
                {
                    "k_train": k,
                    "response_id": rid,
                    "input_message": msg,
                    "domain": domain,
                    "ground_truth_num": int(gt_num),
                    "llm_pred": int(pred_num),
                }
            )
    return pd.DataFrame(rows)


def _load_aligned(k: int) -> pd.DataFrame:
    rf = pd.read_csv(RF_PRED_CSV)
    rf = rf[rf["k_train"] == k].copy()
    rf = rf.rename(columns={"predicted_num": "rf_pred"})
    llm = _load_llm_predictions(k)
    keys = ["k_train", "response_id", "input_message", "domain", "ground_truth_num"]
    df = rf.merge(
        llm[keys + ["llm_pred"]],
        on=keys,
        how="inner",
        validate="many_to_one",
    )
    return df


def _blend(rf_pred: np.ndarray, llm_pred: np.ndarray, w_rf: float) -> np.ndarray:
    pred = np.rint(w_rf * rf_pred + (1.0 - w_rf) * llm_pred)
    return np.clip(pred, 1, 5).astype(int)


def _score(y: np.ndarray, pred: np.ndarray, objective: str) -> float:
    if objective == "accuracy":
        return float(accuracy_score(y, pred))
    if objective == "macro_f1":
        return float(f1_score(y, pred, average="macro", zero_division=0))
    if objective == "qwk":
        return float(cohen_kappa_score(y, pred, weights="quadratic"))
    raise ValueError(f"Unknown objective: {objective}")


def _assign_folds(response_ids: np.ndarray) -> dict[str, int]:
    rng = np.random.default_rng(RANDOM_STATE)
    unique = np.array(sorted(pd.unique(response_ids)))
    rng.shuffle(unique)
    return {rid: i % N_FOLDS for i, rid in enumerate(unique)}


def _tune_choice(train_df: pd.DataFrame, objective: str) -> Choice:
    best = Choice(feature_set="", w_rf=0.0, tune_score=-np.inf)
    for feature_set, sub in train_df.groupby("feature_set", sort=True):
        y = sub["ground_truth_num"].to_numpy(int)
        rf_pred = sub["rf_pred"].to_numpy(int)
        llm_pred = sub["llm_pred"].to_numpy(int)
        for w_rf in WEIGHT_GRID:
            pred = _blend(rf_pred, llm_pred, float(w_rf))
            score = _score(y, pred, objective)
            if score > best.tune_score:
                best = Choice(feature_set=feature_set, w_rf=float(w_rf), tune_score=score)
    return best


def _tune_rf_choice(train_df: pd.DataFrame, objective: str) -> Choice:
    best = Choice(feature_set="", w_rf=1.0, tune_score=-np.inf)
    for feature_set, sub in train_df.groupby("feature_set", sort=True):
        y = sub["ground_truth_num"].to_numpy(int)
        pred = sub["rf_pred"].to_numpy(int)
        score = _score(y, pred, objective)
        if score > best.tune_score:
            best = Choice(feature_set=feature_set, w_rf=1.0, tune_score=score)
    return best


def _crossfit(k: int, objective: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _load_aligned(k)
    fold_map = _assign_folds(df["response_id"].to_numpy())
    df["fold"] = df["response_id"].map(fold_map)

    pred_rows = []
    choice_rows = []
    for domain in DOMAINS:
        dom = df[df["domain"] == domain].copy()
        for fold in range(N_FOLDS):
            train = dom[dom["fold"] != fold]
            eval_all = dom[dom["fold"] == fold]
            ens_choice = _tune_choice(train, objective)
            rf_choice = _tune_rf_choice(train, objective)

            eval_ens = eval_all[eval_all["feature_set"] == ens_choice.feature_set].copy()
            eval_rf = eval_all[eval_all["feature_set"] == rf_choice.feature_set].copy()
            eval_llm = eval_all.drop_duplicates(
                ["response_id", "input_message", "domain", "ground_truth_num"]
            ).copy()

            ens_pred = _blend(
                eval_ens["rf_pred"].to_numpy(int),
                eval_ens["llm_pred"].to_numpy(int),
                ens_choice.w_rf,
            )
            rf_pred = eval_rf["rf_pred"].to_numpy(int)

            pred_rows.extend(
                {
                    "k_train": k,
                    "objective": objective,
                    "fold": fold,
                    "domain": domain,
                    "method": "Tuned ensemble",
                    "feature_set": ens_choice.feature_set,
                    "w_rf": ens_choice.w_rf,
                    "response_id": r.response_id,
                    "input_message": r.input_message,
                    "ground_truth_num": int(r.ground_truth_num),
                    "predicted_num": int(p),
                    "rf_pred_for_ensemble": int(r.rf_pred),
                    "llm_pred": int(r.llm_pred),
                }
                for r, p in zip(eval_ens.itertuples(index=False), ens_pred)
            )
            pred_rows.extend(
                {
                    "k_train": k,
                    "objective": objective,
                    "fold": fold,
                    "domain": domain,
                    "method": "RF selected",
                    "feature_set": rf_choice.feature_set,
                    "w_rf": 1.0,
                    "response_id": r.response_id,
                    "input_message": r.input_message,
                    "ground_truth_num": int(r.ground_truth_num),
                    "predicted_num": int(p),
                    "rf_pred_for_ensemble": int(p),
                    "llm_pred": int(r.llm_pred),
                }
                for r, p in zip(eval_rf.itertuples(index=False), rf_pred)
            )
            pred_rows.extend(
                {
                    "k_train": k,
                    "objective": objective,
                    "fold": fold,
                    "domain": domain,
                    "method": "LLM only",
                    "feature_set": "LLM",
                    "w_rf": 0.0,
                    "response_id": r.response_id,
                    "input_message": r.input_message,
                    "ground_truth_num": int(r.ground_truth_num),
                    "predicted_num": int(r.llm_pred),
                    "rf_pred_for_ensemble": np.nan,
                    "llm_pred": int(r.llm_pred),
                }
                for r in eval_llm.itertuples(index=False)
            )
            choice_rows.append(
                {
                    "k_train": k,
                    "objective": objective,
                    "domain": domain,
                    "fold": fold,
                    "ensemble_feature_set": ens_choice.feature_set,
                    "ensemble_w_rf": ens_choice.w_rf,
                    "ensemble_tune_score": ens_choice.tune_score,
                    "rf_feature_set": rf_choice.feature_set,
                    "rf_tune_score": rf_choice.tune_score,
                    "eval_n": int(
                        eval_llm.drop_duplicates(["response_id", "input_message"]).shape[0]
                    ),
                }
            )

    return pd.DataFrame(pred_rows), pd.DataFrame(choice_rows)


def _summarize(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    class_rows = []
    comp_rows = []

    group_cols = ["k_train", "objective", "domain", "method"]
    for key, sub in preds.groupby(group_cols, sort=True):
        k, objective, domain, method = key
        y = sub["ground_truth_num"].to_numpy(int)
        p = sub["predicted_num"].to_numpy(int)
        rids = sub["response_id"].to_numpy()
        metrics = compute_all_metrics(y, p, rids)
        summary_rows.append(
            {
                "k_train": k,
                "objective": objective,
                "domain": domain,
                "method": method,
                **metrics,
            }
        )
        for cls in range(1, 6):
            mask = y == cls
            class_rows.append(
                {
                    "k_train": k,
                    "objective": objective,
                    "domain": domain,
                    "method": method,
                    "class": cls,
                    "n": int(mask.sum()),
                    "accuracy": float(np.mean(p[mask] == y[mask])) if np.any(mask) else np.nan,
                    "mean_pred": float(np.mean(p[mask])) if np.any(mask) else np.nan,
                }
            )

    for (k, objective, domain), sub in preds.groupby(
        ["k_train", "objective", "domain"], sort=True
    ):
        wide = sub.pivot_table(
            index=["response_id", "input_message", "ground_truth_num"],
            columns="method",
            values="predicted_num",
            aggfunc="first",
        ).reset_index()
        if "RF selected" not in wide or "LLM only" not in wide:
            continue
        y = wide["ground_truth_num"].to_numpy(int)
        rf = wide["RF selected"].to_numpy(int)
        llm = wide["LLM only"].to_numpy(int)
        ens = wide["Tuned ensemble"].to_numpy(int) if "Tuned ensemble" in wide else None
        rf_ok = rf == y
        llm_ok = llm == y
        row = {
            "k_train": k,
            "objective": objective,
            "domain": domain,
            "N": int(len(wide)),
            "both_correct": int(np.sum(rf_ok & llm_ok)),
            "rf_only_correct": int(np.sum(rf_ok & ~llm_ok)),
            "llm_only_correct": int(np.sum(~rf_ok & llm_ok)),
            "neither_correct": int(np.sum(~rf_ok & ~llm_ok)),
            "llm_rescue_rate_when_rf_wrong": float(np.mean(llm_ok[~rf_ok])) if np.any(~rf_ok) else np.nan,
            "rf_rescue_rate_when_llm_wrong": float(np.mean(rf_ok[~llm_ok])) if np.any(~llm_ok) else np.nan,
        }
        if ens is not None:
            ens_ok = ens == y
            row["ensemble_correct"] = int(np.sum(ens_ok))
            row["ensemble_rescues_rf_wrong"] = int(np.sum(~rf_ok & ens_ok))
            row["ensemble_loses_rf_correct"] = int(np.sum(rf_ok & ~ens_ok))
        comp_rows.append(row)

    return pd.DataFrame(summary_rows), pd.DataFrame(class_rows), pd.DataFrame(comp_rows)


def _choice_summary(choices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, sub in choices.groupby(["k_train", "objective", "domain"], sort=True):
        k, objective, domain = key
        weights = Counter(sub["ensemble_w_rf"])
        ens_features = Counter(sub["ensemble_feature_set"])
        rf_features = Counter(sub["rf_feature_set"])
        rows.append(
            {
                "k_train": k,
                "objective": objective,
                "domain": domain,
                "ensemble_w_rf_by_fold": "; ".join(
                    f"{w:g}:{n}" for w, n in sorted(weights.items())
                ),
                "ensemble_feature_set_by_fold": "; ".join(
                    f"{name}:{n}" for name, n in ens_features.most_common()
                ),
                "rf_feature_set_by_fold": "; ".join(
                    f"{name}:{n}" for name, n in rf_features.most_common()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    all_preds = []
    all_choices = []
    for k in K_VALUES:
        if not os.path.exists(_grok_path(k)):
            print(f"Skipping k={k}: missing {_grok_path(k)}")
            continue
        for objective in ["accuracy", "macro_f1", "qwk"]:
            preds, choices = _crossfit(k, objective)
            all_preds.append(preds)
            all_choices.append(choices)

    preds = pd.concat(all_preds, ignore_index=True)
    choices = pd.concat(all_choices, ignore_index=True)
    summary, per_class, comp = _summarize(preds)
    choice_summary = _choice_summary(choices)

    out_pred = figures_path("tuned_ensemble_predictions") + ".csv"
    out_summary = figures_path("tuned_ensemble_summary") + ".csv"
    out_class = figures_path("tuned_ensemble_per_class") + ".csv"
    out_comp = figures_path("tuned_ensemble_complementarity") + ".csv"
    out_choices = figures_path("tuned_ensemble_choices") + ".csv"

    preds.to_csv(out_pred, index=False)
    summary.to_csv(out_summary, index=False)
    per_class.to_csv(out_class, index=False)
    comp.to_csv(out_comp, index=False)
    choice_summary.to_csv(out_choices, index=False)

    print(f"Saved: {out_summary}")
    print(f"Saved: {out_class}")
    print(f"Saved: {out_comp}")
    print(f"Saved: {out_choices}")

    primary = summary[(summary["k_train"] == 1) & (summary["objective"] == "accuracy")]
    print("\nPrimary k=1, accuracy-tuned summary:")
    print(
        primary.pivot_table(
            index=["domain", "method"], values=["Accuracy", "F1", "QWK", "N"], aggfunc="first"
        ).round(3).to_string()
    )


if __name__ == "__main__":
    main()
