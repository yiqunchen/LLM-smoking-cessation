#!/usr/bin/env python3
"""RF augmented with RAG-derived features on participant_3070 unseen-user split.

Idea
----
The LLM RAG run gave each unseen test user the rated history of K=3
demographically-similar seen users.  This script feeds the SAME retrieved
information to the RF baseline, as numeric features:

  rag_mean_content, rag_mean_coping, rag_mean_quitting
  rag_std_content,  rag_std_coping,  rag_std_quitting
  rag_n_messages   (count of rated messages aggregated across the K users)

Train-time retrieval excludes the row's own user (leave-one-user-out within
train) so the feature distribution at train time matches the test-time
distribution.

Evaluated against the SAME 274-row strict shared subset used in
`unseen_participant_eval.py`, with three RF feature sets:

  - Demographics-only  (no RAG)
  - Demographics + RAG-features
  - RAG-features only

Output
------
  revision/figures/unseen_participant/rag_features_rf.{md,csv}
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
    DOMAINS, align_features_labels, extract_features, extract_labels,
)
from run_rag_unseen_grok import fit_train_test_vectors, topk_similar  # noqa: E402
from text_baselines import _load_embeddings, _match_embeddings  # noqa: E402

PROJECT_ROOT_OUT = PROJECT_ROOT / "revision" / "figures" / "unseen_participant"
PROJECT_ROOT_OUT.mkdir(parents=True, exist_ok=True)
GROK_DIR = PROJECT_ROOT / "results_manuscript_x-ai_grok-4-fast"

# Default split (kept for backwards compatibility); CLI args override.
TRAIN_P = PROJECT_ROOT / "data_splits" / "canonical" / "train_participant_3070.json"
TEST_P  = PROJECT_ROOT / "data_splits" / "canonical" / "test_participant_3070.json"
OUT_DIR = PROJECT_ROOT_OUT

K_RETRIEVE = 3
M_PER_USER = 4   # max retrieved messages per anchor user (for direct features)
PAD_VALUE  = -1.0
RANDOM_STATE = 42

RATING_MAPS = {
    "content":  {"Very poor":1,"Poor":2,"Acceptable":3,"Good":4,"Very good":5},
    "coping":   {"Not at all helpful":1,"Somewhat helpful":2,"Moderately helpful":3,
                  "Very helpful":4,"Extremely helpful":5,"Not Helpful":1},
    "quitting": {"Not at all helpful":1,"Somewhat helpful":2,"Moderately helpful":3,
                  "Very helpful":4,"Extremely helpful":5},
}


def metrics(g, p):
    return {
        "Accuracy": accuracy_score(g, p),
        "F1":       f1_score(g, p, average="macro", labels=[1,2,3,4,5], zero_division=0),
        "QWK":      cohen_kappa_score(g, p, weights="quadratic", labels=[1,2,3,4,5]),
    }


def user_rating_stats(records: list) -> dict[str, dict[str, list[int]]]:
    """response_id -> domain -> list of rating ints across that user's messages."""
    out: dict[str, dict[str, list[int]]] = {}
    for r in records:
        rid = r["response_id"]
        ratings = r.get("ratings", {}) or {}
        out.setdefault(rid, {d: [] for d in DOMAINS})
        for dom in DOMAINS:
            v = ratings.get(dom)
            n = RATING_MAPS[dom].get(v)
            if n is not None:
                out[rid][dom].append(int(n))
    return out


def aggregate_rag(stats: dict, anchor_rids: list[str]) -> np.ndarray:
    """For one row, compute the 7-dim RAG feature vector from K anchor users.

    Layout: [mean_content, mean_coping, mean_quitting,
             std_content,  std_coping,  std_quitting,
             total_n_messages]
    Missing values fill with the global mean of (3, 3, 3, 1, 1, 1, k).
    """
    pooled = {d: [] for d in DOMAINS}
    for rid in anchor_rids:
        if rid not in stats:
            continue
        for d in DOMAINS:
            pooled[d].extend(stats[rid][d])
    feats = []
    for d in DOMAINS:
        feats.append(float(np.mean(pooled[d])) if pooled[d] else 3.0)
    for d in DOMAINS:
        feats.append(float(np.std(pooled[d])) if pooled[d] else 1.0)
    feats.append(float(sum(len(pooled[d]) for d in DOMAINS) / len(DOMAINS)))
    return np.array(feats, dtype=float)


def build_user_anchor_maps(train: list, test: list, k: int):
    """Returns (user_anchors_tr, user_anchors_te), each rid -> [anchor_rids…]."""
    X_tr_user, rids_tr_users, X_te_user, rids_te_users = fit_train_test_vectors(train, test)
    Tn = X_tr_user / (np.linalg.norm(X_tr_user, axis=1, keepdims=True) + 1e-12)
    sim_tr = Tn @ Tn.T
    np.fill_diagonal(sim_tr, -np.inf)  # exclude self at train time
    nn_tr = np.argsort(-sim_tr, axis=1)[:, :k]
    nn_te = topk_similar(X_te_user, X_tr_user, k=k)
    user_anchors_tr = {
        rids_tr_users[i]: [rids_tr_users[j] for j in nn_tr[i]]
        for i in range(len(rids_tr_users))
    }
    user_anchors_te = {
        rids_te_users[i]: [rids_tr_users[j] for j in nn_te[i]]
        for i in range(len(rids_te_users))
    }
    return user_anchors_tr, user_anchors_te


def _msg_emb_lookup(records: list):
    """Return a dict: input_message → embedding vector, plus the matrix for
    fallback lookups (covers messages we didn't see in `records`)."""
    emb_matrix, emb_lookup = _load_embeddings()
    arr, idx_kept = _match_embeddings(records, emb_matrix, emb_lookup)
    by_msg = {}
    for i, raw_idx in enumerate(idx_kept):
        msg = records[int(raw_idx)]["input_message"]
        by_msg[msg] = arr[i]
    return by_msg, emb_matrix, emb_lookup


def _embed(msg: str, by_msg: dict, emb_matrix, emb_lookup) -> np.ndarray | None:
    """Best-effort message embedding (cached if seen, else fallback to global lookup)."""
    if msg in by_msg:
        return by_msg[msg]
    # Fallback: try the global embedding lookup table (returns None if absent).
    fake = [{"input_message": msg, "response_id": ""}]
    arr, idx = _match_embeddings(fake, emb_matrix, emb_lookup)
    if not idx:
        return None
    return np.asarray(arr[0], dtype=float)


def _build_user_to_messages_local(records):
    out = {}
    for r in records:
        out.setdefault(r["response_id"], []).append(r)
    return out


def build_rag_direct_features(train: list, test: list, k: int):
    """Per-row [K × M × 4] = 48-dim direct retrieval features.

    For each of K anchors (in similarity order to the test/train user),
    sort that anchor's rated messages by *message embedding similarity*
    to the row's own input message; take top M; emit
    (content, coping, quitting, msg_sim) for each slot.  Pad missing slots
    with PAD_VALUE.

    Same retrieval scheme as the LLM RAG run, with leave-one-user-out at
    train time.
    """
    user_anchors_tr, user_anchors_te = build_user_anchor_maps(train, test, k)
    user_to_msgs = _build_user_to_messages_local(train)
    by_msg, emb_matrix, emb_lookup = _msg_emb_lookup(train + test)

    def feats_for_row(rec: dict, anchors: list[str]) -> np.ndarray:
        test_msg = rec["input_message"]
        test_emb = _embed(test_msg, by_msg, emb_matrix, emb_lookup)
        slots = []
        for anchor_rid in anchors[:k]:
            msgs = user_to_msgs.get(anchor_rid, [])
            # Sort the anchor's messages by similarity to test_msg.
            sims = []
            for m in msgs:
                a_emb = _embed(m["input_message"], by_msg, emb_matrix, emb_lookup)
                if a_emb is None or test_emb is None:
                    sims.append(-1.0)
                else:
                    s = float(a_emb @ test_emb /
                              ((np.linalg.norm(a_emb) * np.linalg.norm(test_emb)) + 1e-12))
                    sims.append(s)
            order = sorted(range(len(msgs)), key=lambda i: -sims[i])
            chosen = order[:M_PER_USER]
            for i in chosen:
                ratings = msgs[i].get("ratings", {}) or {}
                rc = RATING_MAPS["content"].get(ratings.get("content"), PAD_VALUE)
                rk = RATING_MAPS["coping"].get(ratings.get("coping"), PAD_VALUE)
                rq = RATING_MAPS["quitting"].get(ratings.get("quitting"), PAD_VALUE)
                slots.append([rc, rk, rq, sims[i]])
            for _ in range(M_PER_USER - len(chosen)):
                slots.append([PAD_VALUE, PAD_VALUE, PAD_VALUE, PAD_VALUE])
        if len(slots) < k * M_PER_USER:
            slots.extend([[PAD_VALUE] * 4] * (k * M_PER_USER - len(slots)))
        return np.array(slots, dtype=float).flatten()

    X_tr = np.vstack([
        feats_for_row(rec, user_anchors_tr.get(rec["response_id"], []))
        for rec in train
    ])
    X_te = np.vstack([
        feats_for_row(rec, user_anchors_te.get(rec["response_id"], []))
        for rec in test
    ])
    return X_tr, X_te


def build_rag_features(train: list, test: list, k: int):
    """Returns RAG feature matrices for train and test, one row per ROW of
    the input list (NOT per user).  At train time we remove the row's own
    user from candidates, at test time we use all train users."""
    X_tr_user, rids_tr_users, X_te_user, rids_te_users = fit_train_test_vectors(train, test)
    rid_to_idx_tr = {r: i for i, r in enumerate(rids_tr_users)}
    rid_to_idx_te = {r: i for i, r in enumerate(rids_te_users)}

    # Pre-compute pairwise cosine sims within train and from test→train.
    Tn = X_tr_user / (np.linalg.norm(X_tr_user, axis=1, keepdims=True) + 1e-12)
    sim_tr = Tn @ Tn.T  # (n_users_tr, n_users_tr)
    np.fill_diagonal(sim_tr, -np.inf)  # exclude self at train time
    nn_tr = np.argsort(-sim_tr, axis=1)[:, :k]  # (n_users_tr, k)
    nn_te = topk_similar(X_te_user, X_tr_user, k=k)  # (n_users_te, k)

    stats = user_rating_stats(train)

    # Map each train user to its retrieved K seen-user ids (other train users).
    user_anchors_tr = {
        rids_tr_users[i]: [rids_tr_users[j] for j in nn_tr[i]]
        for i in range(len(rids_tr_users))
    }
    user_anchors_te = {
        rids_te_users[i]: [rids_tr_users[j] for j in nn_te[i]]
        for i in range(len(rids_te_users))
    }

    # Per-row feature matrices (train rows → train anchors; test rows → train anchors).
    X_rag_tr = np.vstack([
        aggregate_rag(stats, user_anchors_tr.get(r["response_id"], []))
        for r in train
    ])
    X_rag_te = np.vstack([
        aggregate_rag(stats, user_anchors_te.get(r["response_id"], []))
        for r in test
    ])
    return X_rag_tr, X_rag_te


def build_demo_features(train, test):
    df_tr = extract_features(train); df_te = extract_features(test)
    df_tr, df_te = align_features_labels(df_tr, df_te)
    return df_tr.values.astype(float), df_te.values.astype(float)


def fit_eval_rf(X_tr, y_tr_per_dom, X_te, y_te_per_dom, test_rids, valid_keys):
    """Returns dict[domain] -> dict[(rid,msg) -> predicted int]."""
    preds = {dom: {} for dom in DOMAINS}
    for dom in DOMAINS:
        y_tr = y_tr_per_dom[dom]
        y_te = y_te_per_dom[dom]
        v_tr = ~np.isnan(y_tr)
        v_te = ~np.isnan(y_te)
        if len(np.unique(y_tr[v_tr].astype(int))) < 2:
            continue
        rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                     random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_tr[v_tr], y_tr[v_tr].astype(int))
        out = np.clip(np.round(rf.predict(X_te[v_te])).astype(int), 1, 5)
        for i, raw in enumerate(np.where(v_te)[0]):
            preds[dom][test_rids[raw]] = int(out[i])
    return preds


def load_llm_pred(path: Path) -> dict[tuple, dict[str, int]]:
    if not path.exists():
        return {}
    with open(path) as f: d = json.load(f)
    out = {}
    for v in d.values():
        if not isinstance(v, dict): continue
        rid = v.get("response_id"); msg = v.get("input_message")
        if not rid or msg is None: continue
        rec = {}
        for dom in DOMAINS:
            p = RATING_MAPS[dom].get(v.get(f"predicted_{dom}"))
            if p is not None:
                rec[dom] = int(p)
        if rec:
            out[(rid, msg)] = rec
    return out


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train-path", default=str(TRAIN_P))
    p.add_argument("--test-path",  default=str(TEST_P))
    p.add_argument("--rag-llm-json", default=str(GROK_DIR / "rag_unseen_p3070_k3.json"),
                    help="JSON with RAG-Grok predictions on the test set")
    p.add_argument("--zero-llm-json", default=str(GROK_DIR / "generic_llm_1_zero_shot.json"),
                    help="JSON with zero-shot Grok predictions (for the optional footnote)")
    p.add_argument("--out-tag", default="3070",
                    help="filename tag for the markdown / csv outputs")
    args = p.parse_args()

    global OUT_DIR
    OUT_DIR = PROJECT_ROOT_OUT
    train = json.load(open(args.train_path))
    test  = json.load(open(args.test_path))
    print(f"train rows={len(train)} test rows={len(test)}")
    print(f"RAG-LLM JSON:  {args.rag_llm_json}")
    print(f"zero-LLM JSON: {args.zero_llm_json}")

    # Features
    X_demo_tr, X_demo_te = build_demo_features(train, test)
    X_rag_tr,  X_rag_te  = build_rag_features(train, test, k=K_RETRIEVE)
    print(f"  building direct (per-message) RAG features…")
    X_dir_tr, X_dir_te   = build_rag_direct_features(train, test, k=K_RETRIEVE)
    print(f"  demographics dim:        {X_demo_tr.shape[1]}")
    print(f"  RAG-aggregate dim:       {X_rag_tr.shape[1]}")
    print(f"  RAG-direct (per-msg) dim:{X_dir_tr.shape[1]}")

    # Labels
    y_tr_per_dom = {d: extract_labels(train, d) for d in DOMAINS}
    y_te_per_dom = {d: extract_labels(test, d)  for d in DOMAINS}

    test_rids = [(r["response_id"], r["input_message"]) for r in test]

    # Three RF flavours
    feature_specs = {
        "RF — Demographics only": (X_demo_tr, X_demo_te),
        "RF — Demo + RAG-aggregate": (
            np.hstack([X_demo_tr, X_rag_tr]),
            np.hstack([X_demo_te, X_rag_te]),
        ),
        "RF — Demo + RAG-direct (per-msg)": (
            np.hstack([X_demo_tr, X_dir_tr]),
            np.hstack([X_demo_te, X_dir_te]),
        ),
        "RF — RAG-direct only (per-msg)": (X_dir_tr, X_dir_te),
        "RF — RAG-aggregate only": (X_rag_tr, X_rag_te),
    }

    rf_preds = {}
    for label, (Xt, Xe) in feature_specs.items():
        print(f"  fitting [{label}] (X_tr.shape={Xt.shape}, X_te.shape={Xe.shape}) …")
        rf_preds[label] = fit_eval_rf(Xt, y_tr_per_dom, Xe, y_te_per_dom,
                                       test_rids, set())

    # Pull existing LLM predictions
    rag_llm  = load_llm_pred(Path(args.rag_llm_json))
    zero_llm = load_llm_pred(Path(args.zero_llm_json))
    print(f"  RAG-LLM rows: {len(rag_llm)}  |  zero-shot LLM rows: {len(zero_llm)}")

    # Build ground truth
    gt = {}
    for r in test:
        key = (r["response_id"], r["input_message"])
        rec = {}
        for dom in DOMAINS:
            v = (r.get("ratings") or {}).get(dom)
            n = RATING_MAPS[dom].get(v)
            if n is not None:
                rec[dom] = int(n)
        if rec:
            gt[key] = rec

    # Headline intersection: RF flavours + RAG-LLM + ground truth (zero-shot
    # is OPTIONAL — we don't want its smaller coverage to cap our N).
    msgs_train = {r["input_message"] for r in train}
    rows = []
    for dom in DOMAINS:
        keys_all = set(gt.keys())
        keys_all &= {k for k in gt if dom in gt[k]}
        for label, preds_per_dom in rf_preds.items():
            keys_all &= set(preds_per_dom[dom].keys())
        keys_all &= {k for k in rag_llm if dom in rag_llm[k]}
        if not keys_all:
            continue

        # Split into single-unseen (msg in train) vs double-unseen.
        keys_single = sorted(k for k in keys_all if k[1]     in msgs_train)
        keys_double = sorted(k for k in keys_all if k[1] not in msgs_train)
        cells = [
            ("All unseen-user", sorted(keys_all)),
            ("Single-unseen (msg seen in train)", keys_single),
            ("Double-unseen (msg also new)", keys_double),
        ]

        for cell_name, ks in cells:
            if not ks:
                continue
            gts = np.array([gt[k][dom] for k in ks])
            for label, preds_per_dom in rf_preds.items():
                preds = np.array([preds_per_dom[dom][k] for k in ks])
                m = metrics(gts, preds)
                rows.append({"cell": cell_name, "method": label, "domain": dom,
                              "N": len(ks), **{k: round(v, 3) for k, v in m.items()}})
            # RAG-LLM (always in headline intersection)
            preds = np.array([rag_llm[k][dom] for k in ks])
            m = metrics(gts, preds)
            rows.append({"cell": cell_name, "method": "Grok RAG (k=3)", "domain": dom,
                          "N": len(ks), **{k: round(v, 3) for k, v in m.items()}})
            # Zero-shot LLM only on the subset where it exists
            zs_ks = [k for k in ks if k in zero_llm and dom in zero_llm[k]]
            if zs_ks:
                gts_zs = np.array([gt[k][dom] for k in zs_ks])
                preds_zs = np.array([zero_llm[k][dom] for k in zs_ks])
                m = metrics(gts_zs, preds_zs)
                rows.append({"cell": cell_name + " [zero-shot subset]",
                              "method": "Grok zero-shot",
                              "domain": dom, "N": len(zs_ks),
                              **{k: round(v, 3) for k, v in m.items()}})
    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / f"rag_features_rf_{args.out_tag}.csv"
    df.to_csv(out_csv, index=False)

    md = ["# RF + RAG features vs zero-shot / RAG LLM — split by unseen-message status\n",
          "Split: **participant_3070** (30% train / 70% test, user-disjoint).\n",
          "  - **Train**: 90 users, 275 rows (119 unique messages).",
          "  - **Test**: 211 users, 641 rows; 270 of those have shared LLM/RF predictions.",
          "    - of which 244 are *single-unseen* (test message text WAS seen in train, rated by other users)",
          "    - and 26 are *double-unseen* (test message ALSO never seen in train).\n",
          f"RAG retrieval: top-K={K_RETRIEVE} demographically-similar past users; ",
          f"M={M_PER_USER} most-similar messages per anchor (sorted by message embedding cosine to test message).\n",
          "All rows in a cell share the same N (apples-to-apples within the cell).\n"]
    cell_order = ["All unseen-user",
                   "Single-unseen (msg seen in train)",
                   "Double-unseen (msg also new)"]
    for cell in cell_order:
        md.append(f"## Cell: {cell}\n")
        for dom in DOMAINS:
            sub = df[(df["cell"] == cell) & (df["domain"] == dom)] \
                    .sort_values("Accuracy", ascending=False)
            if sub.empty:
                continue
            md.append(f"### {dom.title()}\n")
            cols = ["method", "N", "Accuracy", "F1", "QWK"]
            md.append("| " + " | ".join(cols) + " |")
            md.append("|" + "|".join(["---"]*len(cols)) + "|")
            for _, r in sub.iterrows():
                md.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
            md.append("")
    out_md = OUT_DIR / f"rag_features_rf_{args.out_tag}.md"
    out_md.write_text("\n".join(md))
    print(f"wrote {out_md.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
