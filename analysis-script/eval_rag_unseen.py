#!/usr/bin/env python3
"""Compare RAG (retrieved-fewshot) Grok against zero-shot Grok and RF on the
participant_3070 unseen-user evaluation, restricted to the 270-row strict
shared-key intersection.

Inputs
------
  results_manuscript_x-ai_grok-4-fast/rag_unseen_p3070_k3.json
  results_manuscript_x-ai_grok-4-fast/generic_llm_1_zero_shot.json   (Grok zero-shot)
  data_splits/canonical/{train,test}_participant_3070.json           (RF train/eval)

Output
------
  revision/figures/unseen_participant/rag_vs_zero_shot.{md,csv}
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "analysis-script"))

# Reuse the unseen-participant RF refit + helpers.
from unseen_participant_eval import (  # noqa: E402
    DOMAINS, FEATURE_SETS, RATING_MAPS, fit_rf_predict, gt_lookup, metrics
)

OUT_DIR = PROJECT_ROOT / "revision" / "figures" / "unseen_participant"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_P = PROJECT_ROOT / "data_splits" / "canonical" / "train_participant_3070.json"
TEST_P  = PROJECT_ROOT / "data_splits" / "canonical" / "test_participant_3070.json"
GROK_DIR = PROJECT_ROOT / "results_manuscript_x-ai_grok-4-fast"


def load_llm_pred(path: Path) -> dict[tuple, dict[str, int]]:
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


def main():
    train = json.load(open(TRAIN_P))
    test = json.load(open(TEST_P))

    rag = load_llm_pred(GROK_DIR / "rag_unseen_p3070_k3.json")
    zero = load_llm_pred(GROK_DIR / "generic_llm_1_zero_shot.json")
    print(f"RAG rows: {len(rag)}; zero-shot rows: {len(zero)}")

    # RF on the same train/test fold.
    rf_per_fs = {fs: fit_rf_predict(train, test, fs) for fs in FEATURE_SETS}

    gt = gt_lookup(test)

    # Strict shared intersection per domain.
    def metrics_on_dom(dom: str, label: str, pred_dict):
        keys_all = set(rag.keys()) & set(zero.keys()) & set(gt.keys())
        for fs_name, rf_per_dom in rf_per_fs.items():
            keys_all &= set(rf_per_dom[dom].keys())
        keys_all &= {k for k in pred_dict if dom in pred_dict[k]}
        keys_all &= {k for k in gt if dom in gt[k]}
        keys_all &= {k for k in zero if dom in zero[k]}
        keys_all &= {k for k in rag if dom in rag[k]}
        ks = sorted(keys_all)
        if not ks:
            return None
        gts = np.array([gt[k][dom] for k in ks])
        preds = np.array([pred_dict[k][dom] for k in ks])
        m = metrics(gts, preds)
        return {"method": label, "domain": dom, "N": len(ks),
                **{k: round(v, 3) for k, v in m.items()}}

    rows = []
    for dom in DOMAINS:
        rows.append(metrics_on_dom(dom, "Grok zero-shot", zero))
        rows.append(metrics_on_dom(dom, "Grok RAG (k=3 sim users)", rag))
        for fs in FEATURE_SETS:
            rows.append(metrics_on_dom(dom, f"RF — {fs}", rf_per_fs[fs][dom]))
    rows = [r for r in rows if r]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "rag_vs_zero_shot.csv", index=False)

    md = ["# Unseen-user evaluation: RAG-Grok vs zero-shot Grok vs RF\n",
          "All rows below are evaluated on the SAME shared-key intersection ",
          "of (a) Grok zero-shot, (b) Grok RAG (k=3 retrieved similar past "
          "participants), (c) every RF feature set, and (d) ground truth.\n"]
    for dom in DOMAINS:
        sub = df[df["domain"] == dom].sort_values("Accuracy", ascending=False)
        md.append(f"## {dom.title()}\n")
        cols = ["method", "N", "Accuracy", "F1", "QWK"]
        md.append("| " + " | ".join(cols) + " |")
        md.append("|" + "|".join(["---"]*len(cols)) + "|")
        for _, r in sub.iterrows():
            md.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        md.append("")
    (OUT_DIR / "rag_vs_zero_shot.md").write_text("\n".join(md))
    print(f"wrote {(OUT_DIR/'rag_vs_zero_shot.md').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
