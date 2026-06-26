#!/usr/bin/env python3
"""Build per-participant k-of-10 digital-twin splits.

Each of 301 participants rated 10 messages in the survey. The canonical
1090/3070/.../9010 splits operate on only 3 of those 10 (the
"evaluation pool" stored in ``processed_llm_data.json``); the other 7
("history pool") sit in ``digitalTwin_msg.xlsx``. That asymmetry made
the existing splits effectively span 0–2 prior messages per participant
with floor-rounding artifacts.

This script unifies the two halves into the **full 10-message dataset**
(from the feedback CSV) and builds clean splits where each participant
has exactly k ∈ {1, 3, 5, 7} messages in train and (10 − k) in test.

Outputs:
    data_splits/canonical/train_dt10_k{k}.json
    data_splits/canonical/test_dt10_k{k}.json
    data_splits/canonical/metadata_dt10_k{k}.json

Record schema matches existing canonical splits so downstream tooling
(extract_features, extract_labels, etc.) keeps working unchanged:
    {response_id, input_message, image_path, metadata, ratings}

Usage:
    uv run python analysis-script/build_dt10_splits.py
"""
from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FB_CSV = os.path.join(
    PROJECT_ROOT, "archive", "data",
    "Message testing data with participant characteristics_02.27.csv",
)
PROCESSED = os.path.join(PROJECT_ROOT, "archive", "data", "processed_llm_data.json")
OUT_DIR = os.path.join(PROJECT_ROOT, "data_splits", "canonical")

K_VALUES = [1, 3, 5, 7]
SEED = 202509

CONTENT_RATING_MAP = {1: "Very poor", 2: "Poor", 3: "Acceptable",
                      4: "Good", 5: "Very good"}
HELPFUL_RATING_MAP = {1: "Not at all helpful", 2: "Somewhat helpful",
                      3: "Moderately helpful", 4: "Very helpful",
                      5: "Extremely helpful"}


def _ratings_dict(row: pd.Series) -> dict | None:
    """Convert the four numeric rating columns to the canonical text format.

    Returns None if any of the four ratings is missing — those rows are
    dropped from both train and test.
    """
    out = {}
    for col, mapper, label in [
        ("content",         CONTENT_RATING_MAP,  "content"),
        ("design",          CONTENT_RATING_MAP,  "design"),
        ("support_coping",  HELPFUL_RATING_MAP,  "coping"),
        ("support_quitting",HELPFUL_RATING_MAP,  "quitting"),
    ]:
        v = row.get(col)
        if pd.isna(v):
            return None
        try:
            v_int = int(round(float(v)))
        except (TypeError, ValueError):
            return None
        if v_int not in mapper:
            return None
        out[label] = mapper[v_int]
    return out


def _participant_metadata(processed: list) -> dict[str, dict]:
    """Pull one demographics dict per participant from
    ``processed_llm_data.json``. Image ID is per-message, so we strip it
    here — it gets re-added per row from the feedback CSV.
    """
    out: dict[str, dict] = {}
    for r in processed:
        rid = r.get("response_id")
        if rid is None or rid in out:
            continue
        meta = dict(r.get("metadata", {}) or {})
        meta.pop("Image ID", None)
        out[rid] = meta
    return out


def build_record(row: pd.Series, base_meta: dict) -> dict | None:
    ratings = _ratings_dict(row)
    if ratings is None:
        return None
    meta = dict(base_meta)
    photo_no = row.get("photo_no")
    if pd.notna(photo_no):
        meta["Image ID"] = str(photo_no)
    image_path = (
        f"data/downloaded_smoke_images\\{photo_no}.png"
        if pd.notna(photo_no) else ""
    )
    return {
        "response_id": row["response_id"],
        "input_message": str(row["message"]).strip(),
        "image_path": image_path,
        "metadata": meta,
        "ratings": ratings,
    }


def main():
    print("Loading feedback CSV (3,010 rows expected)…")
    fb = pd.read_csv(FB_CSV)
    print(f"  {fb.shape[0]} rows, {fb['response_id'].nunique()} participants")

    print("Loading processed_llm_data.json for demographics…")
    with open(PROCESSED) as f:
        processed = json.load(f)
    pid_meta = _participant_metadata(processed)
    print(f"  {len(pid_meta)} participants with demographics")

    # Build all 3,010 records (drop any with missing ratings)
    print("Building records…")
    records: list[dict] = []
    rid_to_records: dict[str, list[dict]] = {}
    dropped = 0
    for _, row in fb.iterrows():
        rid = row["response_id"]
        if rid not in pid_meta:
            dropped += 1
            continue
        rec = build_record(row, pid_meta[rid])
        if rec is None:
            dropped += 1
            continue
        records.append(rec)
        rid_to_records.setdefault(rid, []).append(rec)

    msg_counts = Counter(len(v) for v in rid_to_records.values())
    print(f"  {len(records)} rated rows kept, {dropped} dropped")
    print(f"  msgs/pp distribution: {sorted(msg_counts.items())}")

    os.makedirs(OUT_DIR, exist_ok=True)

    rng = np.random.RandomState(SEED)

    for k in K_VALUES:
        train_data: list[dict] = []
        test_data: list[dict] = []
        per_pp_stats = []

        for rid, recs in rid_to_records.items():
            n = len(recs)
            order = rng.permutation(n)
            k_eff = min(k, n - 1)  # at least one held out for test
            train_idx = order[:k_eff]
            test_idx = order[k_eff:]
            train_data.extend(recs[i] for i in train_idx)
            test_data.extend(recs[i] for i in test_idx)
            per_pp_stats.append({
                "response_id": rid,
                "n_total": n,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            })

        train_path = os.path.join(OUT_DIR, f"train_dt10_k{k}.json")
        test_path = os.path.join(OUT_DIR, f"test_dt10_k{k}.json")
        meta_path = os.path.join(OUT_DIR, f"metadata_dt10_k{k}.json")
        with open(train_path, "w") as f:
            json.dump(train_data, f, indent=2)
        with open(test_path, "w") as f:
            json.dump(test_data, f, indent=2)
        with open(meta_path, "w") as f:
            json.dump({
                "split_type": "k_train_per_participant_out_of_10",
                "k_train": k,
                "seed": SEED,
                "total_participants": len(rid_to_records),
                "train_samples": len(train_data),
                "test_samples": len(test_data),
                "per_participant": per_pp_stats,
            }, f, indent=2)

        avg_train = len(train_data) / len(rid_to_records)
        avg_test  = len(test_data) / len(rid_to_records)
        print(f"  k={k}: train={len(train_data)} ({avg_train:.2f}/pp), "
              f"test={len(test_data)} ({avg_test:.2f}/pp)")

    print("\nDone.")
    print("Saved into:", OUT_DIR)


if __name__ == "__main__":
    main()
