#!/usr/bin/env python3
"""Re-run LLM Digital Twin (cbtact-style) for Grok-4-Fast on every canonical
digital-twin split, but with a FIXED prompt that honors the canonical
profile (i.e. uses ``data['profile_messages']`` from the canonical train
file rather than pulling all 7 Excel rows).

Why
---
The original ``generate_digital_twin_cbtact_prompt`` reads each
participant's full message history from ``digitalTwin_msg.xlsx`` regardless
of split, which makes the cross-split learning-curve plot meaningless for
LLM-DT (the LLM sees the same profile every time). This script gives one
LLM the canonical-split-respecting profile and traces the real curve.

What
----
For each split in {1090, 3070, 5050, 7030, 9010}:
  1. Load ``train_digital_twin_{split}.json`` and
     ``test_digital_twin_{split}.json``.
  2. For every participant, build their canonical profile (list of
     {message, ratings, l_category}) from the train split's rows.
  3. For every test row, build the cbtact-style prompt using only that
     participant's canonical profile (which is empty at split 1090).
     The test message text is NEVER copied into the past-message block.
  4. Call Grok-4-Fast (``x-ai/grok-4-fast`` via OpenRouter) with JSON
     response format.
  5. Save predictions to
     ``results_manuscript_x-ai_grok-4-fast/digital_twin_4_cbtact_canonical_{split}.json``.

Cost
----
Grok-4-Fast at ~$0.0003 / call × ~3,027 test rows ≈ $1–$2 total.

Usage
-----
    OPENROUTER_API_KEY=... uv run python analysis-script/run_dt_canonical_grok.py
    # Optional: --splits 1090 3070  to run a subset
    # Optional: --max-concurrent 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import signal
import sys
from typing import Optional

import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CANONICAL_DIR = os.path.join(PROJECT_ROOT, "data_splits", "canonical")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results_manuscript_x-ai_grok-4-fast")
FEEDBACK_CSV = os.path.join(
    PROJECT_ROOT, "archive", "data",
    "Message testing data with participant characteristics_02.27.csv",
)

MODEL = "x-ai/grok-4-fast"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_CONCURRENT = 8
DEFAULT_CHECKPOINT_INTERVAL = 25

ALL_SPLITS = ["1090", "3070", "5050", "7030", "9010"]

shutting_down = False


def _signal_handler(signum, frame):  # pragma: no cover
    global shutting_down
    print("\nShutdown requested - finishing in-flight calls then saving.")
    shutting_down = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ---------------------------------------------------------------------------
# Feedback CSV: image_id -> l_category lookup
# ---------------------------------------------------------------------------

def _build_l_category_lookup() -> dict[str, str]:
    """photo_no (== Image ID) -> l_category. Distinct image ids share one
    category, so a single dict is enough."""
    fb = pd.read_csv(FEEDBACK_CSV)
    if "photo_no" not in fb.columns or "l_category" not in fb.columns:
        return {}
    fb = fb[["photo_no", "l_category"]].dropna()
    fb["photo_no"] = fb["photo_no"].astype(str)
    fb["l_category"] = fb["l_category"].astype(str)
    # Deduplicate
    return dict(zip(fb["photo_no"], fb["l_category"]))


# ---------------------------------------------------------------------------
# Profile assembly per canonical split
# ---------------------------------------------------------------------------

def _profile_entry(rec: dict, lcat_lookup: dict[str, str]) -> dict:
    image_id = (rec.get("metadata") or {}).get("Image ID")
    return {
        "input_message": rec.get("input_message", ""),
        "ratings": rec.get("ratings", {}) or {},
        "l_category": lcat_lookup.get(str(image_id), "") if image_id else "",
    }


def _attach_canonical_profiles(test_items: list[dict],
                                 train_items: list[dict],
                                 lcat_lookup: dict[str, str]) -> None:
    """Attach a 'profile_messages' list to each test item — every other
    canonical-train message belonging to the same participant.
    """
    by_pid: dict[str, list[dict]] = {}
    for r in train_items:
        rid = str(r.get("response_id", ""))
        if not rid:
            continue
        by_pid.setdefault(rid, []).append(_profile_entry(r, lcat_lookup))

    for r in test_items:
        rid = str(r.get("response_id", ""))
        r["profile_messages"] = list(by_pid.get(rid, []))
        r["test_l_category"] = lcat_lookup.get(
            str((r.get("metadata") or {}).get("Image ID", "")), "")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def _format_profile_messages(profile: list[dict]) -> str:
    if not profile:
        return "(no past rated messages from this participant)\n"
    out = []
    for pm in profile:
        msg = str(pm.get("input_message", "")).strip()
        ratings = pm.get("ratings", {}) or {}
        ratings_lines = []
        for k in ["content", "design", "coping", "quitting"]:
            if k in ratings and ratings[k] is not None:
                ratings_lines.append(f"{k}: {ratings[k]}")
        l_cat = pm.get("l_category", "")
        block = f"Past message:\n{msg}\nRatings:\n" + "\n".join(ratings_lines)
        if l_cat:
            block += f"\nMessage type:\n{l_cat}"
        out.append(block + "\n\n---\n")
    return "".join(out)


def _format_metadata(metadata: dict) -> str:
    if not isinstance(metadata, dict):
        return ""
    out = []
    for key, value in metadata.items():
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        if str(value).lower() == "nan":
            continue
        out.append(f"- {key}: {value}")
    return "\n".join(out) + ("\n" if out else "")


def build_prompt(data: dict) -> str:
    metadata_text = _format_metadata(data.get("metadata", {}))
    profile_block = _format_profile_messages(data.get("profile_messages", []))
    msg_type = data.get("test_l_category", "")
    msg_type_text = f"\nMessage type: {msg_type}\n" if msg_type else ""

    test_msg = str(data.get("input_message", "")).strip()

    prompt = f"""
You are an AI assistant simulating this participant. Your task is to predict how the participant will rate a new smoking-cessation support message, using their participant metadata (which includes their characteristics) and their past rated messages.
Base your prediction on how similar the new message is to the participant's previously rated messages, if any. Remain consistent with the participant's prior ratings and stated characteristics, as if you are that person.
When predicting coping and quitting, also use the message type (Acceptance vs Distraction).
Be sure to carefully follow all provided instructions for formatting your answer to the new message.
---
Instructions:
### RATING DIMENSIONS
1. **content** – How would you rate the content (that is, the words and meaning) of this message?
2. **design** – How would you rate the design (that is, how the message looks) of this message?
3. **coping** – How helpful would this message be to support you in coping with a smoking urge or craving?
4. **quitting** – How helpful would this message be to support you in quitting or reducing smoking?

### Allowed rating categories
**content / design** → Very poor · Poor · Acceptable · Good · Very good
**coping / quitting** → Not at all helpful · Somewhat helpful · Moderately helpful · Very helpful · Extremely helpful

### INPUTS
Here is the message provided to the participant to be rated:

\\"{test_msg}\\"
{msg_type_text}

Participant metadata:
{metadata_text}
Past rated messages from this participant (canonical profile, may be empty at low training fractions):
{profile_block}
---
### OUTPUT FORMAT
Return **exactly** this JSON object:

{{
  "response_id": "{data.get('response_id', '')}",
  "input_message": "<echo back the new message text>",
  "predicted_content": "<one of: Very poor, Poor, Acceptable, Good, Very good>",
  "predicted_design": "<one of: Very poor, Poor, Acceptable, Good, Very good>",
  "predicted_coping": "<one of: Not at all helpful, Somewhat helpful, Moderately helpful, Very helpful, Extremely helpful>",
  "predicted_quitting": "<one of: Not at all helpful, Somewhat helpful, Moderately helpful, Very helpful, Extremely helpful>",
  "predicted_content_probabilities": {{ "Very poor": <p1>, "Poor": <p2>, "Acceptable": <p3>, "Good": <p4>, "Very good": <p5> }},
  "predicted_design_probabilities": {{ "Very poor": <p1>, "Poor": <p2>, "Acceptable": <p3>, "Good": <p4>, "Very good": <p5> }},
  "predicted_coping_probabilities": {{ "Not at all helpful": <p1>, "Somewhat helpful": <p2>, "Moderately helpful": <p3>, "Very helpful": <p4>, "Extremely helpful": <p5> }},
  "predicted_quitting_probabilities": {{ "Not at all helpful": <p1>, "Somewhat helpful": <p2>, "Moderately helpful": <p3>, "Very helpful": <p4>, "Extremely helpful": <p5> }},
  "explanation": "<short rationale referencing the profile and demographics>"
}}
"""
    return prompt


# ---------------------------------------------------------------------------
# Async caller
# ---------------------------------------------------------------------------

async def _get_response(client: AsyncOpenAI, prompt: str, semaphore,
                          temperature: float, max_retries: int = 5) -> str:
    async with semaphore:
        for attempt in range(max_retries):
            try:
                completion = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=temperature,
                )
                return completion.choices[0].message.content
            except Exception as e:
                wait = (2 ** attempt) + random.uniform(0, 1)
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait)
                else:
                    raise


async def _query_one(qid: str, data: dict, client: AsyncOpenAI, semaphore,
                       temperature: float) -> dict:
    if shutting_down:
        return {}
    prompt = build_prompt(data)
    try:
        raw = await _get_response(client, prompt, semaphore, temperature)
        parsed = json.loads(raw)
    except Exception as e:  # pragma: no cover
        ratings = data.get("ratings", {}) or {}
        return {qid: {
            "response_id": data.get("response_id", "UNKNOWN"),
            "input_message": data.get("input_message", "UNKNOWN"),
            "metadata": data.get("metadata", {}),
            "ground_truth_content": ratings.get("content", "ERROR"),
            "ground_truth_design": ratings.get("design", "ERROR"),
            "ground_truth_coping": ratings.get("coping", "ERROR"),
            "ground_truth_quitting": ratings.get("quitting", "ERROR"),
            "predicted_content": "ERROR",
            "predicted_design": "ERROR",
            "predicted_coping": "ERROR",
            "predicted_quitting": "ERROR",
            "explanation": "ERROR",
            "error": str(e),
            "n_profile_messages": len(data.get("profile_messages", [])),
        }}

    ratings = data.get("ratings", {}) or {}
    return {qid: {
        "response_id": data.get("response_id", "UNKNOWN"),
        "input_message": data.get("input_message", "UNKNOWN"),
        "metadata": data.get("metadata", {}),
        "ground_truth_content": ratings.get("content", ""),
        "ground_truth_design": ratings.get("design", ""),
        "ground_truth_coping": ratings.get("coping", ""),
        "ground_truth_quitting": ratings.get("quitting", ""),
        "predicted_content": parsed.get("predicted_content", ""),
        "predicted_design": parsed.get("predicted_design", ""),
        "predicted_coping": parsed.get("predicted_coping", ""),
        "predicted_quitting": parsed.get("predicted_quitting", ""),
        "predicted_content_probabilities": parsed.get("predicted_content_probabilities", {}),
        "predicted_design_probabilities": parsed.get("predicted_design_probabilities", {}),
        "predicted_coping_probabilities": parsed.get("predicted_coping_probabilities", {}),
        "predicted_quitting_probabilities": parsed.get("predicted_quitting_probabilities", {}),
        "explanation": parsed.get("explanation", ""),
        "n_profile_messages": len(data.get("profile_messages", [])),
    }}


async def run_split(split: str, lcat_lookup: dict[str, str],
                     client: AsyncOpenAI, max_concurrent: int,
                     temperature: float, checkpoint_interval: int) -> None:
    train_path = os.path.join(CANONICAL_DIR, f"train_digital_twin_{split}.json")
    test_path = os.path.join(CANONICAL_DIR, f"test_digital_twin_{split}.json")
    with open(train_path) as f:
        train = json.load(f)
    with open(test_path) as f:
        test = json.load(f)

    _attach_canonical_profiles(test, train, lcat_lookup)
    avg_profile = sum(len(r.get("profile_messages", [])) for r in test) / max(1, len(test))
    print(f"[{split}] train={len(train)}  test={len(test)}  avg profile/test row = {avg_profile:.2f}")

    out_path = os.path.join(
        OUTPUT_DIR, f"digital_twin_4_cbtact_canonical_{split}.json"
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results: dict[str, dict] = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        print(f"  loaded checkpoint: {len(results)} already done")

    pending = {str(i): r for i, r in enumerate(test) if str(i) not in results}
    if not pending:
        print(f"  all {len(test)} already done")
        return

    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        asyncio.create_task(_query_one(qid, data, client, semaphore, temperature))
        for qid, data in pending.items()
    ]

    pbar = tqdm(total=len(tasks), desc=f"split {split}")
    completed = 0
    for task in asyncio.as_completed(tasks):
        if shutting_down:
            for t in tasks:
                if not t.done():
                    t.cancel()
            break
        result = await task
        if result:
            results.update(result)
            completed += 1
            pbar.update(1)
            if completed % checkpoint_interval == 0:
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)
    pbar.close()

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  saved: {out_path}  ({len(results)} rows)")


# ---------------------------------------------------------------------------

async def main_async(args):
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    print("Building image_id -> l_category lookup…")
    lcat = _build_l_category_lookup()
    print(f"  {len(lcat)} image_id mappings")

    splits = args.splits or ALL_SPLITS
    for split in splits:
        if shutting_down:
            break
        await run_split(
            split, lcat, client,
            args.max_concurrent, args.temperature, args.checkpoint_interval,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+", default=None,
                   choices=ALL_SPLITS, help="Subset of splits to run")
    p.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--checkpoint-interval", type=int,
                   default=DEFAULT_CHECKPOINT_INTERVAL)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
