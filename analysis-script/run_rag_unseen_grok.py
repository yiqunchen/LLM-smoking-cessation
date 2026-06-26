#!/usr/bin/env python3
"""RAG few-shot for unseen users: retrieve K demographically-similar SEEN
users' rated messages and inject them as in-context anchors for Grok-4-Fast.

Why
---
LLM-PP / Digital-Twin is undefined on participant-disjoint splits because
unseen users have no anchor messages.  This script provides a "next best"
personalization signal: for each unseen test user we pull the rated history
of the K most demographically-similar SEEN users (from the training fold)
and use those as the in-context examples.

Setup
-----
  - Train fold:  data_splits/canonical/train_participant_3070.json
                 (90 seen users, 275 labelled messages)
  - Test fold:   data_splits/canonical/test_participant_3070.json
                 (211 unseen users, 641 messages — ground truth held out)
  - Encoder:     extract_features() → demographics as numeric vector,
                 standardized; cosine similarity → top-K seen users.
  - K_RETRIEVE:  3 (default)
  - Per-row prompt: K seen-user blocks; each block lists up to
                 MAX_MSGS_PER_USER (default 4) of that seen user's
                 (message, ratings) tuples.

Output
------
  results_manuscript_x-ai_grok-4-fast/rag_unseen_p3070_k{K}.json
  Same JSON schema as digital_twin_*.json files.

Usage
-----
  OPENROUTER_API_KEY=... uv run python analysis-script/run_rag_unseen_grok.py [--limit N] [--k K]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import signal
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "analysis-script"))

from revision_utils import extract_features, align_features_labels  # noqa: E402

CANONICAL_DIR = PROJECT_ROOT / "data_splits" / "canonical"
OUT_DIR = PROJECT_ROOT / "results_manuscript_x-ai_grok-4-fast"
MODEL = "x-ai/grok-4-fast"

K_DEFAULT = 3
MAX_MSGS_PER_USER = 4  # CLI override available via --m-per-user

shutting_down = False
def _sig_handler(*_):
    global shutting_down
    shutting_down = True
signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)


# ---------------------- retrieval --------------------------------------

def build_user_profile_vectors(records: list) -> tuple[np.ndarray, list[str]]:
    """One row per UNIQUE response_id.  Uses demographics (extract_features).
    Returns (N_users × D matrix, list of response_ids)."""
    df = extract_features(records)
    rids = [r.get("response_id") for r in records]
    df["__rid__"] = rids
    one_per_user = df.drop_duplicates(subset=["__rid__"], keep="first")
    user_rids = one_per_user["__rid__"].tolist()
    one_per_user = one_per_user.drop(columns=["__rid__"])
    return one_per_user.values.astype(float), user_rids


def fit_train_test_vectors(train: list, test: list):
    """Align demographic columns across train and test."""
    df_tr_full = extract_features(train)
    df_te_full = extract_features(test)
    df_tr_full, df_te_full = align_features_labels(df_tr_full, df_te_full)

    rids_tr = [r["response_id"] for r in train]
    rids_te = [r["response_id"] for r in test]
    df_tr_full["__rid__"] = rids_tr
    df_te_full["__rid__"] = rids_te

    user_tr = df_tr_full.drop_duplicates(subset=["__rid__"], keep="first")
    user_te = df_te_full.drop_duplicates(subset=["__rid__"], keep="first")
    user_rids_tr = user_tr["__rid__"].tolist()
    user_rids_te = user_te["__rid__"].tolist()
    X_tr = user_tr.drop(columns=["__rid__"]).values.astype(float)
    X_te = user_te.drop(columns=["__rid__"]).values.astype(float)

    # Standardize on train statistics (z-score) to give equal weight to
    # all demographic features regardless of native scale.
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0) + 1e-8
    X_tr = (X_tr - mu) / sd
    X_te = (X_te - mu) / sd
    return X_tr, user_rids_tr, X_te, user_rids_te


def topk_similar(X_te: np.ndarray, X_tr: np.ndarray, k: int) -> np.ndarray:
    """Cosine similarity → indices of top-K rows of X_tr per X_te row."""
    Tn = X_te / (np.linalg.norm(X_te, axis=1, keepdims=True) + 1e-12)
    Sn = X_tr / (np.linalg.norm(X_tr, axis=1, keepdims=True) + 1e-12)
    sims = Tn @ Sn.T  # (n_te, n_tr)
    return np.argsort(-sims, axis=1)[:, :k]


def build_user_to_messages(records: list) -> dict[str, list[dict]]:
    """response_id -> list of {input_message, ratings, ...}."""
    out: dict[str, list[dict]] = {}
    for r in records:
        out.setdefault(r["response_id"], []).append(r)
    return out


# ---------------------- prompt -----------------------------------------

def _format_metadata(meta: dict) -> str:
    if not isinstance(meta, dict):
        return ""
    out = []
    for k, v in meta.items():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        if str(v).lower() == "nan":
            continue
        out.append(f"- {k}: {v}")
    return "\n".join(out) + ("\n" if out else "")


def _format_anchor_block(anchor_users: list[tuple[str, list[dict]]]) -> str:
    """anchor_users = [(seen_user_rid, [up to MAX_MSGS_PER_USER rated messages]), …]."""
    if not anchor_users:
        return "(no anchor users retrieved)\n"
    chunks = []
    for i, (rid, msgs) in enumerate(anchor_users, start=1):
        body = []
        for m in msgs[:MAX_MSGS_PER_USER]:
            text = str(m.get("input_message", "")).strip()
            ratings = m.get("ratings", {}) or {}
            rating_lines = [
                f"  content: {ratings.get('content','')}",
                f"  coping: {ratings.get('coping','')}",
                f"  quitting: {ratings.get('quitting','')}",
            ]
            body.append(f"Message:\n{text}\nRatings:\n" + "\n".join(rating_lines))
        chunks.append(
            f"### Similar past participant {i} (id={rid})\n"
            + "\n\n".join(body) + "\n"
        )
    return "\n---\n".join(chunks) + "\n"


def build_rag_prompt(test_record: dict, anchor_users: list[tuple[str, list[dict]]]) -> str:
    test_msg = str(test_record.get("input_message", "")).strip()
    metadata_text = _format_metadata(test_record.get("metadata", {}))
    anchors_text = _format_anchor_block(anchor_users)

    return f"""
You are an AI assistant predicting how a NEW participant will rate a smoking-cessation message.

You do NOT have any prior ratings from this exact participant.  Instead you are given the rated history of {len(anchor_users)} demographically similar past participants.  Treat their ratings as a soft prior — adjust for the new participant's metadata where appropriate.

---
### RATING DIMENSIONS
1. **content** – How would you rate the content (that is, the words and meaning) of this message?
2. **coping** – How helpful would this message be to support coping with a smoking urge or craving?
3. **quitting** – How helpful would this message be to support quitting or reducing smoking?

### Allowed rating categories
**content** → Very poor · Poor · Acceptable · Good · Very good
**coping / quitting** → Not at all helpful · Somewhat helpful · Moderately helpful · Very helpful · Extremely helpful

---
### INPUTS
New participant metadata:
{metadata_text}
Rated history from {len(anchor_users)} demographically similar past participants:
{anchors_text}
Message to rate:
\\"{test_msg}\\"

---
### OUTPUT FORMAT
Return **exactly** this JSON object (no extra keys, no markdown):

{{
  "response_id": "{test_record.get('response_id', '')}",
  "input_message": "<echo back the new message text>",
  "predicted_content": "<one of: Very poor, Poor, Acceptable, Good, Very good>",
  "predicted_coping": "<one of: Not at all helpful, Somewhat helpful, Moderately helpful, Very helpful, Extremely helpful>",
  "predicted_quitting": "<one of: Not at all helpful, Somewhat helpful, Moderately helpful, Very helpful, Extremely helpful>",
  "predicted_content_probabilities": {{ "Very poor": <p>, "Poor": <p>, "Acceptable": <p>, "Good": <p>, "Very good": <p> }},
  "predicted_coping_probabilities": {{ "Not at all helpful": <p>, "Somewhat helpful": <p>, "Moderately helpful": <p>, "Very helpful": <p>, "Extremely helpful": <p> }},
  "predicted_quitting_probabilities": {{ "Not at all helpful": <p>, "Somewhat helpful": <p>, "Moderately helpful": <p>, "Very helpful": <p>, "Extremely helpful": <p> }},
  "explanation": "<one sentence referencing how the similar past participants' ratings informed your prediction>"
}}
"""


# ---------------------- LLM call ---------------------------------------

async def _query(client: AsyncOpenAI, prompt: str, sem, temperature: float,
                  max_retries: int = 5) -> str:
    async with sem:
        for attempt in range(max_retries):
            try:
                completion = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=temperature,
                )
                return completion.choices[0].message.content
            except Exception:
                wait = (2 ** attempt) + random.uniform(0, 1)
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait)
                else:
                    raise


async def _query_one(qid: str, test_rec: dict, anchors,
                       client: AsyncOpenAI, sem, temperature: float) -> dict:
    if shutting_down:
        return {}
    prompt = build_rag_prompt(test_rec, anchors)
    ratings = test_rec.get("ratings", {}) or {}
    base = {
        "response_id": test_rec.get("response_id", ""),
        "input_message": test_rec.get("input_message", ""),
        "metadata": test_rec.get("metadata", {}),
        "ground_truth_content": ratings.get("content", ""),
        "ground_truth_coping": ratings.get("coping", ""),
        "ground_truth_quitting": ratings.get("quitting", ""),
        "anchor_user_ids": [u for u, _ in anchors],
        "n_anchor_users": len(anchors),
    }
    try:
        raw = await _query(client, prompt, sem, temperature)
        parsed = json.loads(raw)
    except Exception as e:
        return {qid: {**base,
                       "predicted_content": "ERROR", "predicted_coping": "ERROR",
                       "predicted_quitting": "ERROR", "explanation": "ERROR",
                       "error": str(e)}}
    return {qid: {**base,
                   "predicted_content":  parsed.get("predicted_content", ""),
                   "predicted_coping":   parsed.get("predicted_coping", ""),
                   "predicted_quitting": parsed.get("predicted_quitting", ""),
                   "predicted_content_probabilities":
                       parsed.get("predicted_content_probabilities", {}),
                   "predicted_coping_probabilities":
                       parsed.get("predicted_coping_probabilities", {}),
                   "predicted_quitting_probabilities":
                       parsed.get("predicted_quitting_probabilities", {}),
                   "explanation":         parsed.get("explanation", "")}}


# ---------------------- main -------------------------------------------

async def run(k: int, limit: int | None, max_concurrent: int,
              temperature: float, restrict_to_existing_llm: bool,
              train_path: str | None, test_path: str | None,
              output_tag: str | None):
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY not set", file=sys.stderr); sys.exit(1)

    train_path = train_path or str(CANONICAL_DIR / "train_participant_3070.json")
    test_path  = test_path  or str(CANONICAL_DIR / "test_participant_3070.json")
    train = json.load(open(train_path))
    test = json.load(open(test_path))
    print(f"  train: {train_path}\n  test:  {test_path}")
    print(f"loaded train={len(train)}  test={len(test)}")

    # Optionally restrict test rows to those that already have a Generic-LLM
    # zero-shot prediction (so we can run a strict head-to-head later).
    if restrict_to_existing_llm:
        existing_path = (PROJECT_ROOT / "results_manuscript_x-ai_grok-4-fast"
                          / "generic_llm_1_zero_shot.json")
        with open(existing_path) as f:
            existing = json.load(f)
        existing_keys = set()
        for v in existing.values():
            if isinstance(v, dict):
                rid = v.get("response_id"); msg = v.get("input_message")
                if rid and msg is not None:
                    existing_keys.add((rid, msg))
        before = len(test)
        test = [r for r in test
                 if (r["response_id"], r["input_message"]) in existing_keys]
        print(f"  restricted to rows with existing zero-shot LLM: "
              f"{before} -> {len(test)}")

    if limit:
        test = test[:limit]
        print(f"  --limit applied: now {len(test)} test rows")

    # Build retrieval.
    X_tr, user_rids_tr, X_te, user_rids_te = fit_train_test_vectors(train, test)
    rid_to_idx_tr = {r: i for i, r in enumerate(user_rids_tr)}
    print(f"profile-vector dim: {X_tr.shape[1]} | "
          f"unique users (train) = {len(user_rids_tr)} | "
          f"unique users (test) = {len(user_rids_te)}")

    user_to_msgs_tr = build_user_to_messages(train)
    nn_idx = topk_similar(X_te, X_tr, k=k)  # (n_te_users, k)
    user_anchors = {}
    for i, te_rid in enumerate(user_rids_te):
        anchor_rids = [user_rids_tr[j] for j in nn_idx[i]]
        user_anchors[te_rid] = [(rid, user_to_msgs_tr.get(rid, [])) for rid in anchor_rids]

    # Build per-row work items.
    work = []
    for i, rec in enumerate(test):
        anchors = user_anchors[rec["response_id"]]
        work.append((str(i), rec, anchors))
    print(f"queued {len(work)} test rows for LLM")

    # Save sample prompts for inspection BEFORE calling LLM.
    sample_path = OUT_DIR / f"rag_unseen_p3070_k{k}_sample_prompts.txt"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(sample_path, "w") as f:
        for qid, rec, anchors in work[:3]:
            f.write(f"==== qid={qid} ====\n")
            f.write(build_rag_prompt(rec, anchors))
            f.write("\n\n")
    print(f"  wrote 3 sample prompts to {sample_path.relative_to(PROJECT_ROOT)}")

    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    sem = asyncio.Semaphore(max_concurrent)

    tag = output_tag or "p3070"
    out_path = OUT_DIR / f"rag_unseen_{tag}_k{k}.json"
    results: dict[str, dict] = {}
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        print(f"  resuming: {len(results)} rows already done")

    todo = [(qid, rec, anchors) for qid, rec, anchors in work
             if qid not in results]
    if not todo:
        print("  nothing to do.")
        return

    tasks = [_query_one(qid, rec, anchors, client, sem, temperature)
             for qid, rec, anchors in todo]
    pbar = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"RAG k={k}")
    for fut in pbar:
        partial = await fut
        results.update(partial)
        if len(results) % 25 == 0 or len(results) == len(work):
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {len(results)} rows to {out_path.relative_to(PROJECT_ROOT)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=K_DEFAULT,
                    help="number of similar past users to retrieve")
    p.add_argument("--limit", type=int, default=None,
                    help="cap on # of test rows to score (smoke testing)")
    p.add_argument("--max-concurrent", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--restrict-to-existing-llm", action="store_true",
                    default=True,
                    help="only score rows already covered by zero-shot LLM (default)")
    p.add_argument("--all-rows", action="store_true",
                    help="override --restrict-to-existing-llm and score ALL rows")
    p.add_argument("--train-path", default=None,
                    help="override default train file path")
    p.add_argument("--test-path", default=None,
                    help="override default test file path")
    p.add_argument("--output-tag", default=None,
                    help="filename tag (default 'p3070')")
    p.add_argument("--m-per-user", type=int, default=None,
                    help="max retrieved messages per anchor user (default 4)")
    args = p.parse_args()
    if args.m_per_user is not None:
        global MAX_MSGS_PER_USER
        MAX_MSGS_PER_USER = args.m_per_user
        print(f"[override] MAX_MSGS_PER_USER = {MAX_MSGS_PER_USER}")
    asyncio.run(run(
        k=args.k, limit=args.limit,
        max_concurrent=args.max_concurrent, temperature=args.temperature,
        restrict_to_existing_llm=not args.all_rows,
        train_path=args.train_path, test_path=args.test_path,
        output_tag=args.output_tag,
    ))


if __name__ == "__main__":
    main()
