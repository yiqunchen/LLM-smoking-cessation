#!/usr/bin/env python3
"""User-disjoint and double-unseen evaluation: RF vs Generic-LLM.

LLM-PP / Digital-Twin is N/A here by construction — there are no anchor
messages for unseen users, so the only LLM mode that's defined is the
zero-shot generic prompt.  We compare:

  - Supervised RF, trained ONLY on the user-disjoint train fold of
    participant_3070 (90 users, 275 rows), evaluated on the unseen
    211-user test fold (641 rows). Three feature sets: Demographics,
    Embedding, Embedding+Demo. (History features are not defined for
    unseen users, so we drop the *History feature sets.)

  - Generic-LLM zero-shot, the same five models as in the main grid,
    on the same 641 rows.

Two cells:
  (A) unseen-user only      — all 641 test rows
  (B) unseen-user + unseen-message (double-unseen) — the 62 rows whose
       input_message is also absent from participant_3070_train.

Outputs:
    revision/figures/unseen_participant/metrics.{md,csv}
    revision/figures/unseen_participant/per_row.csv
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "analysis-script"))

from revision_utils import (  # noqa: E402
    DOMAINS,
    align_features_labels,
    extract_features,
    extract_labels,
)
from text_baselines import _load_embeddings, _match_embeddings  # noqa: E402

OUT_DIR = PROJECT_ROOT / "revision" / "figures" / "unseen_participant"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_P = PROJECT_ROOT / "data_splits" / "canonical" / "train_participant_3070.json"
TEST_P  = PROJECT_ROOT / "data_splits" / "canonical" / "test_participant_3070.json"

LLM_MODELS = {
    "GPT-4o-mini":     PROJECT_ROOT / "results_manuscript_gpt-4o-mini",
    "GPT-5":           PROJECT_ROOT / "results_manuscript_gpt-5",
    "Gemini-2.5-Pro":  PROJECT_ROOT / "results_manuscript_gemini-2.5-pro",
    "Grok-4-Fast":     PROJECT_ROOT / "results_manuscript_x-ai_grok-4-fast",
    "DeepSeek-R1":     PROJECT_ROOT / "results_manuscript_deepseek_deepseek-r1-0528",
}

RATING_MAPS = {
    "content":  {"Very poor":1,"Poor":2,"Acceptable":3,"Good":4,"Very good":5},
    "coping":   {"Not at all helpful":1,"Somewhat helpful":2,
                  "Moderately helpful":3,"Very helpful":4,"Extremely helpful":5,
                  "Not Helpful":1},
    "quitting": {"Not at all helpful":1,"Somewhat helpful":2,
                  "Moderately helpful":3,"Very helpful":4,"Extremely helpful":5},
}

FEATURE_SETS = {
    "Demographics":   {"demo": True,  "embedding": False},
    "Embedding":      {"demo": False, "embedding": True},
    "Embedding+Demo": {"demo": True,  "embedding": True},
}
RANDOM_STATE = 42


def metrics(gts, preds):
    return {
        "Accuracy": accuracy_score(gts, preds),
        "F1":       f1_score(gts, preds, average="macro",
                              labels=[1,2,3,4,5], zero_division=0),
        "QWK":      cohen_kappa_score(gts, preds, weights="quadratic",
                                       labels=[1,2,3,4,5]),
    }


def build_features(train, test, spec):
    pieces_tr, pieces_te = [], []
    if spec["embedding"]:
        emb_matrix, emb_lookup = _load_embeddings()
        emb_tr_arr, idx_tr = _match_embeddings(train, emb_matrix, emb_lookup)
        emb_te_arr, idx_te = _match_embeddings(test, emb_matrix, emb_lookup)
        idx_tr = np.asarray(idx_tr, dtype=int)
        idx_te = np.asarray(idx_te, dtype=int)
        pieces_tr.append(np.asarray(emb_tr_arr, dtype=float))
        pieces_te.append(np.asarray(emb_te_arr, dtype=float))
    else:
        idx_tr = np.arange(len(train), dtype=int)
        idx_te = np.arange(len(test), dtype=int)

    if spec["demo"]:
        demo_tr = extract_features(train); demo_te = extract_features(test)
        demo_tr, demo_te = align_features_labels(demo_tr, demo_te)
        pieces_tr.append(np.asarray(demo_tr.iloc[idx_tr].values, dtype=float))
        pieces_te.append(np.asarray(demo_te.iloc[idx_te].values, dtype=float))

    X_tr = pieces_tr[0] if len(pieces_tr) == 1 else np.hstack(pieces_tr)
    X_te = pieces_te[0] if len(pieces_te) == 1 else np.hstack(pieces_te)
    return X_tr, idx_tr, X_te, idx_te


def fit_rf_predict(train, test, fs_name) -> dict[str, dict[tuple, int]]:
    """Returns domain -> {(rid, msg) -> predicted_int}."""
    spec = FEATURE_SETS[fs_name]
    X_tr, idx_tr, X_te, idx_te = build_features(train, test, spec)
    out = {dom: {} for dom in DOMAINS}
    for domain in DOMAINS:
        y_tr_full = extract_labels(train, domain)[idx_tr]
        y_te_full = extract_labels(test, domain)[idx_te]
        valid_tr = ~np.isnan(y_tr_full)
        valid_te = ~np.isnan(y_te_full)
        X_tr_v = X_tr[valid_tr]; y_tr_v = y_tr_full[valid_tr].astype(int)
        X_te_v = X_te[valid_te]
        if len(np.unique(y_tr_v)) < 2:
            continue
        rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                     random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_tr_v, y_tr_v)
        pred = np.clip(np.round(rf.predict(X_te_v)).astype(int), 1, 5)
        # Map back to (rid, msg) keys.
        idx_te_kept = idx_te[valid_te]
        for i, raw_idx in enumerate(idx_te_kept):
            rec = test[int(raw_idx)]
            out[domain][(rec["response_id"], rec["input_message"])] = int(pred[i])
    return out


def load_generic_llm_zero_shot(model_dir: Path) -> dict[tuple, dict[str, int]]:
    """(rid, msg) -> domain -> predicted int from generic_llm_1_zero_shot.json."""
    path = model_dir / "generic_llm_1_zero_shot.json"
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    out = {}
    for v in d.values():
        if not isinstance(v, dict):
            continue
        rid = v.get("response_id"); msg = v.get("input_message")
        if not rid or msg is None:
            continue
        rec = {}
        for dom in DOMAINS:
            p = RATING_MAPS[dom].get(v.get(f"predicted_{dom}"))
            if p is not None:
                rec[dom] = int(p)
        if rec:
            out[(rid, msg)] = rec
    return out


def gt_lookup(records) -> dict[tuple, dict[str, int]]:
    out = {}
    for r in records:
        key = (r["response_id"], r["input_message"])
        rec = {}
        for dom in DOMAINS:
            v = r.get("ratings", {}).get(dom)
            n = RATING_MAPS[dom].get(v)
            if n is not None:
                rec[dom] = int(n)
        if rec:
            out[key] = rec
    return out


def evaluate_on_keys(rf_preds_per_fs, llm_preds_per_model, gt, keys):
    """Strict shared-key evaluation. We restrict to rows where (a) every RF
    feature set has a prediction, (b) every LLM model has a prediction, and
    (c) ground truth is present.  Returns one DataFrame; every row reports
    metrics on the same N for that domain."""
    rows = []
    for dom in DOMAINS:
        # Build the intersection.
        shared = set(keys)
        for fs_name, rf_per_dom in rf_preds_per_fs.items():
            shared &= set(rf_per_dom[dom].keys())
        for model, preds_per_key in llm_preds_per_model.items():
            shared &= {k for k in preds_per_key if dom in preds_per_key[k]}
        shared &= {k for k in gt if dom in gt[k]}
        if not shared:
            continue
        ks = sorted(shared)
        gts = np.array([gt[k][dom] for k in ks])
        for fs_name, rf_per_dom in rf_preds_per_fs.items():
            preds = np.array([rf_per_dom[dom][k] for k in ks])
            m = metrics(gts, preds)
            rows.append({"method": "RF", "config": fs_name, "domain": dom,
                          "N": len(ks), **{k: round(v, 3) for k, v in m.items()}})
        for model, preds_per_key in llm_preds_per_model.items():
            preds = np.array([preds_per_key[k][dom] for k in ks])
            m = metrics(gts, preds)
            rows.append({"method": "Generic-LLM", "config": model, "domain": dom,
                          "N": len(ks), **{k: round(v, 3) for k, v in m.items()}})
    return pd.DataFrame(rows)


def write_metrics_md(df_single: pd.DataFrame, df_double: pd.DataFrame, n_single, n_double):
    md = ["# User-disjoint evaluation: RF vs Generic-LLM (zero-shot)\n",
          "Test fold = participant_3070 (211 unseen users, 641 rows). LLM-PP / ",
          "Digital-Twin is N/A here because unseen users have no anchor messages.\n",
          "Two cells:\n",
          f"- **(A) unseen-user only**: all {n_single} test rows.",
          f"- **(B) double-unseen** (unseen user + unseen message): {n_double} test rows ",
          "whose input_message also did NOT appear in the participant_3070 train fold.\n"]

    def emit_block(df: pd.DataFrame, title: str):
        md.append(f"## {title}\n")
        for dom in DOMAINS:
            sub = df[df["domain"] == dom].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("Accuracy", ascending=False)
            md.append(f"### {dom.title()}\n")
            cols = ["method", "config", "N", "Accuracy", "F1", "QWK"]
            md.append("| " + " | ".join(cols) + " |")
            md.append("|" + "|".join(["---"]*len(cols)) + "|")
            for _, r in sub.iterrows():
                md.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
            md.append("")

    emit_block(df_single, "(A) Unseen user — 211 held-out users, all 641 rows")
    emit_block(df_double, "(B) Double-unseen — unseen user AND unseen message")
    (OUT_DIR / "metrics.md").write_text("\n".join(md))


def main():
    with open(TRAIN_P) as f: train = json.load(f)
    with open(TEST_P) as f: test = json.load(f)
    print(f"train rows={len(train)}  test rows={len(test)}  "
          f"users_train={len({r['response_id'] for r in train})}  "
          f"users_test={len({r['response_id'] for r in test})}")

    msgs_train = {r["input_message"] for r in train}
    keys_all = [(r["response_id"], r["input_message"]) for r in test]
    keys_double = [(r["response_id"], r["input_message"]) for r in test
                    if r["input_message"] not in msgs_train]
    print(f"  all keys: {len(keys_all)}; double-unseen keys: {len(keys_double)}")

    rf_preds_per_fs = {}
    for fs in FEATURE_SETS:
        print(f"  fitting RF [{fs}]…")
        rf_preds_per_fs[fs] = fit_rf_predict(train, test, fs)

    llm_preds_per_model = {}
    for model, mdir in LLM_MODELS.items():
        preds = load_generic_llm_zero_shot(mdir)
        if preds:
            llm_preds_per_model[model] = preds
            print(f"  loaded LLM[{model}]: {len(preds)} keys")
        else:
            print(f"  LLM[{model}]: missing zero-shot file — skip")

    gt = gt_lookup(test)
    print(f"  ground truth keys: {len(gt)}")

    df_single = evaluate_on_keys(rf_preds_per_fs, llm_preds_per_model, gt, keys_all)
    df_double = evaluate_on_keys(rf_preds_per_fs, llm_preds_per_model, gt, keys_double)

    df_single.to_csv(OUT_DIR / "metrics_unseen_user.csv", index=False)
    df_double.to_csv(OUT_DIR / "metrics_double_unseen.csv", index=False)
    write_metrics_md(df_single, df_double, len(keys_all), len(keys_double))
    print(f"\nwrote {(OUT_DIR / 'metrics.md').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
