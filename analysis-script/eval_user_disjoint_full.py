#!/usr/bin/env python3
"""End-to-end grader for the user-disjoint full-message eval (30/70 + 10/90).

Steps 2–4 of the pipeline kicked off in the chat:
  Step 2: re-run RF flavours on both splits (free).
  Step 3: ingest the RAG-Grok JSONs (already produced by the background runs).
  Step 4: optional zero-shot footnote on the rows that already have zero-shot
          Grok coverage (from existing files; no new API calls).

Outputs
-------
    revision/figures/unseen_participant/headline_3070full.{md,csv}
    revision/figures/unseen_participant/headline_1090full.{md,csv}
    revision/figures/unseen_participant/headline_combined.md
    revision/figures/unseen_participant/zero_shot_footnote.md   (subset where zero-shot exists)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "analysis-script"))
OUT_DIR = PROJECT_ROOT / "revision" / "figures" / "unseen_participant"
OUT_DIR.mkdir(parents=True, exist_ok=True)
GROK_DIR = PROJECT_ROOT / "results_manuscript_x-ai_grok-4-fast"

DOMAINS = ["content", "coping", "quitting"]
RATING_MAPS = {
    "content":  {"Very poor":1,"Poor":2,"Acceptable":3,"Good":4,"Very good":5},
    "coping":   {"Not at all helpful":1,"Somewhat helpful":2,"Moderately helpful":3,
                  "Very helpful":4,"Extremely helpful":5,"Not Helpful":1},
    "quitting": {"Not at all helpful":1,"Somewhat helpful":2,"Moderately helpful":3,
                  "Very helpful":4,"Extremely helpful":5},
}

SPLITS = [
    {"tag": "3070full", "label": "30/70 user-disjoint (90 train / 211 test users)",
     "train": "data_splits/canonical/train_user_disjoint_3070_full.json",
     "test":  "data_splits/canonical/test_user_disjoint_3070_full.json",
     "rag_llm": "results_manuscript_x-ai_grok-4-fast/rag_unseen_3070full_k3.json"},
    {"tag": "1090full", "label": "10/90 user-disjoint (30 train / 271 test users)",
     "train": "data_splits/canonical/train_user_disjoint_1090_full.json",
     "test":  "data_splits/canonical/test_user_disjoint_1090_full.json",
     "rag_llm": "results_manuscript_x-ai_grok-4-fast/rag_unseen_1090full_k3.json"},
]


def metrics(g, p):
    return {
        "Accuracy": accuracy_score(g, p),
        "F1":       f1_score(g, p, average="macro", labels=[1,2,3,4,5], zero_division=0),
        "QWK":      cohen_kappa_score(g, p, weights="quadratic", labels=[1,2,3,4,5]),
    }


def step2_run_rf(split: dict) -> Path:
    """Invoke rag_features_rf_unseen.py with the new split paths."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "analysis-script" / "rag_features_rf_unseen.py"),
        "--train-path", str(PROJECT_ROOT / split["train"]),
        "--test-path",  str(PROJECT_ROOT / split["test"]),
        "--rag-llm-json", str(PROJECT_ROOT / split["rag_llm"]),
        "--out-tag", split["tag"],
    ]
    print(f"\n[step 2] RF flavours on {split['label']}")
    subprocess.run(cmd, check=True)
    return OUT_DIR / f"rag_features_rf_{split['tag']}.csv"


def step4_zero_shot_footnote():
    """Use the union of all existing zero-shot Grok JSONs to grade RAG vs zero-shot
    on the rows where both predictions exist.  No new API calls."""
    # Union of all zero-shot Grok JSONs.
    keys = {}
    for p in (GROK_DIR).glob("*zero*"):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        for v in d.values():
            if not isinstance(v, dict):
                continue
            rid = v.get("response_id"); msg = v.get("input_message")
            if not rid or msg is None:
                continue
            for dom in DOMAINS:
                pr = RATING_MAPS[dom].get(v.get(f"predicted_{dom}"))
                gt = RATING_MAPS[dom].get(v.get(f"ground_truth_{dom}"))
                if pr is None or gt is None:
                    continue
                keys[(rid, msg, dom)] = (gt, pr)

    md = ["# Zero-shot footnote: subset where zero-shot Grok already exists\n",
          "Used as a sanity check only; the headline tables use the full RAG-vs-RF ",
          "comparison on the user-disjoint test sets.\n",
          "Zero-shot rows are pulled from the union of every `*zero*` JSON in ",
          "`results_manuscript_x-ai_grok-4-fast/`.  For each split, we grade ",
          "RAG-Grok vs Grok zero-shot on the strict shared-key intersection.\n"]
    for split in SPLITS:
        rag_path = PROJECT_ROOT / split["rag_llm"]
        if not rag_path.exists():
            md.append(f"## {split['label']}\n_RAG run not finished yet — skipping._\n")
            continue
        rag = {}
        for v in json.load(open(rag_path)).values():
            if not isinstance(v, dict): continue
            rid = v.get("response_id"); msg = v.get("input_message")
            if not rid or msg is None: continue
            for dom in DOMAINS:
                pr = RATING_MAPS[dom].get(v.get(f"predicted_{dom}"))
                gt = RATING_MAPS[dom].get(v.get(f"ground_truth_{dom}"))
                if pr is not None and gt is not None:
                    rag[(rid, msg, dom)] = (gt, pr)
        shared = set(rag) & set(keys)
        md.append(f"## {split['label']}\n")
        md.append(f"Shared (RAG ∩ zero-shot): {len(shared)} (rid,msg,domain) triples\n")
        md.append("| Domain | N | zero-shot Acc | RAG Acc | Δ | zero-shot F1 | RAG F1 | Δ | zero-shot QWK | RAG QWK | Δ |")
        md.append("|" + "|".join(["---"] * 11) + "|")
        for dom in DOMAINS:
            ks = [k for k in shared if k[2] == dom]
            if not ks: continue
            g  = np.array([rag[k][0] for k in ks])
            pr = np.array([rag[k][1] for k in ks])
            pz = np.array([keys[k][1] for k in ks])
            mr = metrics(g, pr); mz = metrics(g, pz)
            md.append(
                f"| {dom} | {len(ks)} | "
                f"{mz['Accuracy']:.3f} | {mr['Accuracy']:.3f} | {mr['Accuracy']-mz['Accuracy']:+.3f} | "
                f"{mz['F1']:.3f} | {mr['F1']:.3f} | {mr['F1']-mz['F1']:+.3f} | "
                f"{mz['QWK']:.3f} | {mr['QWK']:.3f} | {mr['QWK']-mz['QWK']:+.3f} |"
            )
        md.append("")
    (OUT_DIR / "zero_shot_footnote.md").write_text("\n".join(md))
    print(f"\n[step 4] wrote {(OUT_DIR / 'zero_shot_footnote.md').relative_to(PROJECT_ROOT)}")


def step3_combined_summary(csvs: list[Path]):
    """Side-by-side 30/70 vs 10/90 mean-across-domains table."""
    md = ["# Headline: RF vs RAG-Grok on user-disjoint full-message splits\n",
          "Two splits run on the SAME nested user partition:\n",
          "  - **30/70**: 90 train users / 211 test users (898 train / 2107 test rows)",
          "  - **10/90**: 30 train users / 271 test users (300 train / 2705 test rows)\n",
          "All metrics here are evaluated on the FULL test fold of each split. ",
          "RF flavours and RAG-Grok share the same N within each row's "
          "(domain, cell) cell.\n",
          "## Mean across domains, all-unseen-user cell\n",
          "| Method | 30/70 Acc | 10/90 Acc | 30/70 F1 | 10/90 F1 | 30/70 QWK | 10/90 QWK |",
          "|" + "|".join(["---"] * 7) + "|"]
    by_split = {}
    for csv_path in csvs:
        tag = csv_path.stem.replace("rag_features_rf_", "")
        by_split[tag] = pd.read_csv(csv_path)

    methods = sorted({m for df in by_split.values() for m in df["method"].unique()})
    for method in methods:
        cells = []
        for metric in ("Accuracy", "F1", "QWK"):
            for tag in ("3070full", "1090full"):
                df = by_split.get(tag)
                if df is None:
                    cells.append("—"); continue
                sub = df[(df["method"] == method) & (df["cell"] == "All unseen-user")]
                if sub.empty:
                    cells.append("—"); continue
                v = sub[metric].mean()
                cells.append(f"{v:.3f}")
        md.append(f"| {method} | " + " | ".join(cells) + " |")
    md.append("")
    (OUT_DIR / "headline_combined.md").write_text("\n".join(md))
    print(f"[step 3] wrote {(OUT_DIR / 'headline_combined.md').relative_to(PROJECT_ROOT)}")


def main():
    csvs = []
    for split in SPLITS:
        rag_path = PROJECT_ROOT / split["rag_llm"]
        if not rag_path.exists():
            print(f"[skip] {split['label']}: RAG run not yet finished "
                  f"({rag_path.name})")
            continue
        csv_path = step2_run_rf(split)
        csvs.append(csv_path)

    if csvs:
        step3_combined_summary(csvs)
    step4_zero_shot_footnote()


if __name__ == "__main__":
    main()
