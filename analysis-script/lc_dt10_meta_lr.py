#!/usr/bin/env python3
"""Stacked-LR meta-learner: fuse RF and LLM-PP probability vectors.

For each (k_train, feature_set, domain):
  - re-fit the RF with the SAME training data as `lc_dt10_rf.py`,
    but capture `predict_proba` on the test set (5-class soft outputs)
  - read the matching LLM-PP probability vector from the Grok JSON
  - build aligned features: [rf_p1..rf_p5, llm_p1..llm_p5] = 10-dim
  - target: rating 1..5 (5-class)
  - GroupKFold by `response_id`: for each held-out user, the LR head is
    fit on all other users' (rf_probs, llm_probs, label) and evaluated
    on the held-out user's rows.  This is the strongest held-out scheme
    we can implement without re-running the LLM on training rows
    (the LLM was only scored on test rows in this experiment, with the
    canonical train set consumed as in-context anchors).
  - aggregate out-of-fold predictions and compute the standard metrics

This is the proper "soft" fusion the rounded-mean router was unable to
do.  Output format matches `lc_dt10_ensemble_k{k}.csv` so the existing
progress-summary code can pick it up by adding a new "Stacked LR" row to
the ensemble of routers.

Outputs:
    revision/figures/lc_dt10_meta_lr_k{k}.csv
    (and rows are also appended to `lc_dt10_ensemble_k{k}.csv` under the
    label `Stacked LR` so the figure picks them up.)

Usage:
    uv run python analysis-script/lc_dt10_meta_lr.py            # k=1, 3, 7
    uv run python analysis-script/lc_dt10_meta_lr.py 3          # single k
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from revision_utils import (  # noqa: E402
    DOMAINS,
    RATING_MAPS,
    compute_all_metrics,
    extract_labels,
    figures_path,
)
from lc_dt10_rf import FEATURE_SETS, _load_split, build_features  # noqa: E402

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RANDOM_STATE = 42
N_FOLDS = 5
RATING_CLASSES = np.arange(1, 6)  # 1..5


def grok_path(k: int) -> str:
    return os.path.join(
        PROJECT_ROOT, "results_manuscript_x-ai_grok-4-fast",
        f"digital_twin_dt10_k{k}.json",
    )


# Map the textual rating keys in the LLM JSON's probability dicts to ints.
PROB_KEY_MAP = {
    "content": {"Very poor": 1, "Poor": 2, "Acceptable": 3, "Good": 4, "Very good": 5},
    "coping":  {"Not at all helpful": 1, "Somewhat helpful": 2,
                 "Moderately helpful": 3, "Very helpful": 4, "Extremely helpful": 5},
    "quitting": {"Not at all helpful": 1, "Somewhat helpful": 2,
                 "Moderately helpful": 3, "Very helpful": 4, "Extremely helpful": 5},
}


def load_llm_probs(k: int) -> dict[tuple[str, str, str], np.ndarray]:
    """Returns (response_id, input_message, domain) -> length-5 prob vector."""
    with open(grok_path(k)) as f:
        data = json.load(f)
    out: dict[tuple[str, str, str], np.ndarray] = {}
    for v in data.values():
        if not isinstance(v, dict):
            continue
        rid = v.get("response_id")
        msg = v.get("input_message")
        if rid is None or msg is None:
            continue
        for dom in DOMAINS:
            probs = v.get(f"predicted_{dom}_probabilities")
            if not probs:
                continue
            vec = np.zeros(5, dtype=float)
            for label_str, p in probs.items():
                idx = PROB_KEY_MAP[dom].get(label_str)
                if idx is not None:
                    vec[idx - 1] = float(p)
            s = vec.sum()
            if s > 0:
                vec = vec / s
            out[(rid, msg, dom)] = vec
    return out


def fit_rf_proba(X_tr, y_tr, X_te) -> np.ndarray:
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)
    classes = rf.classes_  # may be a subset of 1..5 if some classes absent
    raw = rf.predict_proba(X_te)
    out = np.zeros((raw.shape[0], 5), dtype=float)
    for i, c in enumerate(classes):
        if 1 <= int(c) <= 5:
            out[:, int(c) - 1] = raw[:, i]
    return out


def conf_gated_predict(rf_probs: np.ndarray, llm_probs: np.ndarray,
                        gate: str = "entropy") -> np.ndarray:
    """Per-row routing: pick the predictor whose probability vector is more
    confident, then take its argmax. No training; this is a stateless rule.

    gate = 'entropy' : pick the one with lower entropy.
    gate = 'maxp'    : pick the one with higher max-probability.
    """
    eps = 1e-12
    if gate == "entropy":
        h_rf = -np.sum(rf_probs * np.log(rf_probs + eps), axis=1)
        h_llm = -np.sum(llm_probs * np.log(llm_probs + eps), axis=1)
        use_rf = h_rf <= h_llm
    elif gate == "maxp":
        use_rf = np.max(rf_probs, axis=1) >= np.max(llm_probs, axis=1)
    else:
        raise ValueError(gate)
    rf_arg = np.argmax(rf_probs, axis=1) + 1  # 1..5
    llm_arg = np.argmax(llm_probs, axis=1) + 1
    return np.where(use_rf, rf_arg, llm_arg).astype(int)


def stacked_lr_cv(X_meta, y, groups, n_folds=N_FOLDS):
    """Out-of-fold predictions from multinomial LR meta-learner.

    Folds split on `groups` (response_id), so the LR fitting fold for a
    given user contains zero rows from that user.  The fold count is
    capped at the number of unique groups.
    """
    preds = np.zeros(len(y), dtype=int)
    if len(np.unique(y)) < 2:
        return None
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return None
    folds = min(n_folds, n_groups)
    gkf = GroupKFold(n_splits=folds)
    for tr_idx, te_idx in gkf.split(X_meta, y, groups=groups):
        if len(np.unique(y[tr_idx])) < 2:
            preds[te_idx] = int(np.bincount(y[tr_idx]).argmax())
            continue
        # multinomial is the default in sklearn ≥1.5; lbfgs supports it.
        lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        lr.fit(X_meta[tr_idx], y[tr_idx])
        proba = lr.predict_proba(X_meta[te_idx])
        cls = lr.classes_
        argmax = np.argmax(proba, axis=1)
        preds[te_idx] = cls[argmax]
    return preds


def run_k(k: int) -> pd.DataFrame:
    print(f"\n=== k_train = {k} ===")
    train, test = _load_split(k)
    llm_probs = load_llm_probs(k)
    print(f"  train rows = {len(train)}  |  test rows = {len(test)}  "
          f"|  LLM prob keys = {len(llm_probs)}")

    rows = []
    for fs_name, spec in FEATURE_SETS.items():
        X_tr, idx_tr, X_te, idx_te = build_features(train, test, spec)
        for domain in DOMAINS:
            y_tr_full = extract_labels(train, domain)[idx_tr]
            y_te_full = extract_labels(test, domain)[idx_te]
            valid_tr = ~np.isnan(y_tr_full)
            valid_te = ~np.isnan(y_te_full)
            X_tr_v = X_tr[valid_tr]; y_tr_v = y_tr_full[valid_tr].astype(int)
            X_te_v = X_te[valid_te]; y_te_v = y_te_full[valid_te].astype(int)
            test_idx_valid = idx_te[valid_te]

            if len(np.unique(y_tr_v)) < 2 or len(y_te_v) == 0:
                continue

            rf_probs = fit_rf_proba(X_tr_v, y_tr_v, X_te_v)

            # Align LLM probs row-by-row with the test rows we kept.
            llm_arr = np.zeros((len(y_te_v), 5), dtype=float)
            keep = np.ones(len(y_te_v), dtype=bool)
            rids = []
            for i, raw_idx in enumerate(test_idx_valid):
                rec = test[int(raw_idx)]
                key = (rec["response_id"], rec["input_message"], domain)
                vec = llm_probs.get(key)
                if vec is None:
                    keep[i] = False
                    rids.append(None)
                else:
                    llm_arr[i] = vec
                    rids.append(rec["response_id"])

            rf_kept = rf_probs[keep]
            llm_kept = llm_arr[keep]
            y_kept = y_te_v[keep]
            rids_kept = np.array([r for r, k_ in zip(rids, keep) if k_])

            if len(y_kept) == 0:
                continue

            X_meta = np.hstack([rf_kept, llm_kept])
            pred_lr = stacked_lr_cv(X_meta, y_kept, groups=rids_kept)
            pred_cg_ent = conf_gated_predict(rf_kept, llm_kept, gate="entropy")
            pred_cg_max = conf_gated_predict(rf_kept, llm_kept, gate="maxp")

            for ens_name, pred in [
                ("Stacked LR", pred_lr),
                ("Conf-Gated (entropy)", pred_cg_ent),
                ("Conf-Gated (maxp)", pred_cg_max),
            ]:
                if pred is None:
                    continue
                metrics = compute_all_metrics(y_kept, pred, rids_kept)
                rows.append({
                    "k_train": k, "feature_set": fs_name,
                    "ensemble": ens_name, "domain": domain,
                    "N": int(metrics["N"]),
                    "Accuracy": metrics["Accuracy"],
                    "F1": metrics["F1"],
                    "Kappa": metrics["Kappa"],
                    "QWK": metrics["QWK"],
                    "Directional Accuracy": metrics.get("Directional Accuracy"),
                    "Directional Macro-F1": metrics.get("Directional Macro-F1"),
                    "Spearman_Rho": metrics.get("Spearman_Rho"),
                })
                print(f"    {fs_name:23s}  {domain:9s}  {ens_name:22s}  "
                      f"acc={metrics['Accuracy']:.3f}  f1={metrics['F1']:.3f}  "
                      f"qwk={metrics['QWK']:.3f}  N={metrics['N']}")

    return pd.DataFrame(rows)


def merge_into_ensemble_csv(df_lr: pd.DataFrame, k: int) -> None:
    """Append `Stacked LR` rows to the existing ensemble file (de-duping)."""
    ens_path = figures_path(f"lc_dt10_ensemble_k{k}") + ".csv"
    if not os.path.exists(ens_path):
        df_lr.to_csv(ens_path, index=False)
        print(f"  wrote {ens_path} (created from scratch)")
        return
    existing = pd.read_csv(ens_path)
    drop_names = {"Stacked LR", "Conf-Gated (entropy)", "Conf-Gated (maxp)"}
    existing = existing[~existing["ensemble"].isin(drop_names)]
    combined = pd.concat([existing, df_lr], ignore_index=True)
    combined.to_csv(ens_path, index=False)
    print(f"  merged meta-learner rows into {ens_path}  "
          f"({len(combined)} total rows)")


def main():
    args = sys.argv[1:]
    ks = [int(a) for a in args] if args else [1, 3, 7]
    for k in ks:
        df_lr = run_k(k)
        out_path = figures_path(f"lc_dt10_meta_lr_k{k}") + ".csv"
        df_lr.to_csv(out_path, index=False)
        print(f"  saved {out_path}  ({len(df_lr)} rows)")
        merge_into_ensemble_csv(df_lr, k)


if __name__ == "__main__":
    main()
