#!/usr/bin/env python3
"""Zero-shot Grok-4-Fast scoring for the user-disjoint evaluation.

No retrieval, no anchors — just demographics + the test message.  Used as
the LLM baseline for the unseen-user RAG comparison.

Output schema matches the other digital-twin / RAG JSONs so downstream
analysis works unchanged.

Usage:
    OPENROUTER_API_KEY=... uv run python analysis-script/run_zeroshot_grok.py \
        --test-path data_splits/canonical/test_user_disjoint_1090_full.json \
        --output-name zero_shot_unseen_1090full
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

import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "results_manuscript_x-ai_grok-4-fast"
MODEL = "x-ai/grok-4-fast"

shutting_down = False
def _sig_handler(*_):
    global shutting_down
    shutting_down = True
signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)


def _format_metadata(meta: dict) -> str:
    out = []
    for k, v in (meta or {}).items():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        if str(v).lower() == "nan":
            continue
        out.append(f"- {k}: {v}")
    return "\n".join(out) + ("\n" if out else "")


def build_prompt(test_record: dict) -> str:
    test_msg = str(test_record.get("input_message", "")).strip()
    metadata_text = _format_metadata(test_record.get("metadata", {}))
    return f"""
You are an AI assistant predicting how a participant will rate a smoking-cessation message.

You have NOT seen any prior ratings from this participant.  Use only their demographic / behavioral profile and the message text.

---
### RATING DIMENSIONS
1. **content** – How would you rate the content (the words and meaning) of this message?
2. **coping** – How helpful would this message be to support coping with a smoking urge?
3. **quitting** – How helpful would this message be to support quitting / reducing smoking?

### Allowed rating categories
**content** → Very poor · Poor · Acceptable · Good · Very good
**coping / quitting** → Not at all helpful · Somewhat helpful · Moderately helpful · Very helpful · Extremely helpful

---
### INPUTS
Participant metadata:
{metadata_text}
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
  "explanation": "<one sentence>"
}}
"""


async def _query(client, prompt, sem, temperature, max_retries=5):
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


async def _query_one(qid, rec, client, sem, temperature):
    if shutting_down:
        return {}
    prompt = build_prompt(rec)
    ratings = rec.get("ratings", {}) or {}
    base = {
        "response_id": rec.get("response_id", ""),
        "input_message": rec.get("input_message", ""),
        "metadata": rec.get("metadata", {}),
        "ground_truth_content": ratings.get("content", ""),
        "ground_truth_coping":  ratings.get("coping", ""),
        "ground_truth_quitting": ratings.get("quitting", ""),
    }
    try:
        raw = await _query(client, prompt, sem, temperature)
        parsed = json.loads(raw)
    except Exception as e:
        return {qid: {**base, "predicted_content": "ERROR",
                       "predicted_coping": "ERROR", "predicted_quitting": "ERROR",
                       "explanation": "ERROR", "error": str(e)}}
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
                   "explanation": parsed.get("explanation", "")}}


async def run(test_path: str, output_name: str, max_concurrent: int,
              temperature: float, limit: int | None):
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY not set", file=sys.stderr); sys.exit(1)
    test = json.load(open(test_path))
    if limit:
        test = test[:limit]
    print(f"queued {len(test)} test rows from {test_path}")

    out_path = OUT_DIR / f"{output_name}.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        print(f"  resuming: {len(results)} rows already done")

    todo = [(str(i), rec) for i, rec in enumerate(test) if str(i) not in results]
    if not todo:
        print("  nothing to do."); return

    client = AsyncOpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                          base_url="https://openrouter.ai/api/v1")
    sem = asyncio.Semaphore(max_concurrent)
    tasks = [_query_one(qid, rec, client, sem, temperature) for qid, rec in todo]
    pbar = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="zero-shot")
    for fut in pbar:
        partial = await fut
        results.update(partial)
        if len(results) % 50 == 0 or len(results) == len(test):
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {len(results)} rows to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-path", required=True)
    p.add_argument("--output-name", required=True,
                    help="filename stem (no extension); written to results_manuscript_x-ai_grok-4-fast/")
    p.add_argument("--max-concurrent", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    asyncio.run(run(args.test_path, args.output_name,
                     args.max_concurrent, args.temperature, args.limit))


if __name__ == "__main__":
    main()
