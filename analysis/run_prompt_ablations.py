#!/usr/bin/env python3
"""Run the Reviewer 3 PP ablations on the canonical dt10-k7 split.

The runner intentionally uses the same 2,107/898 participant-level split as
the retained PP benchmark.  It never reads the retired 70/30 artefacts.

Conditions (each is evaluated for all 898 held-out ratings):
  * pp_cbtact: metadata + seven past messages + their ratings and CBT/ACT
    category labels; rerun as a same-model comparison baseline.
  * full_pp_no_cbtact: metadata + seven past messages + their ratings;
    no CBT/ACT category labels or instructions.
  * history_ratings_only: seven past message texts and their ratings; no
    participant characteristics.
  * history_text_only: seven past message texts; no ratings or participant
    characteristics.

The script is resumable.  Successful rows are checkpointed; failed requests
remain pending for the next invocation.  A manifest records the input hashes
and exact prompt specification alongside the outputs.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
from datetime import datetime, timezone

import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data/splits" / "canonical"
TRAIN_PATH = SPLIT_DIR / "train_dt10_k7.json"
TEST_PATH = SPLIT_DIR / "test_dt10_k7.json"
METADATA_PATH = SPLIT_DIR / "metadata_dt10_k7.json"
OUTPUT_DIR = ROOT / "results/prompt_ablations/grok-4.3"
MODEL = "x-ai/grok-4.3"
FEEDBACK_PATH = ROOT / "data" / "raw" / "Message testing data with participant characteristics_02.27.csv"
CONDITIONS = ("pp_cbtact", "full_pp_no_cbtact", "history_ratings_only", "history_text_only")
RATING_ORDER = ("content", "coping", "quitting")
ALLOWED_LABELS = {
    "content": {"Very poor", "Poor", "Acceptable", "Good", "Very good"},
    "coping": {"Not at all helpful", "Somewhat helpful", "Moderately helpful", "Very helpful", "Extremely helpful"},
    "quitting": {"Not at all helpful", "Somewhat helpful", "Moderately helpful", "Very helpful", "Extremely helpful"},
}
DEFAULT_CONCURRENCY = 6
DEFAULT_CHECKPOINT_INTERVAL = 25
MAX_OUTPUT_TOKENS = 300
REASONING_EFFORT: str | None = None
shutting_down = False


def _signal_handler(_signum, _frame):  # pragma: no cover - signal timing
    global shutting_down
    print("\nShutdown requested; saving completed rows and stopping new requests.")
    shutting_down = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def format_metadata(metadata: dict) -> str:
    rows = []
    for key, value in (metadata or {}).items():
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        if str(value).strip().lower() == "nan":
            continue
        rows.append(f"- {key}: {value}")
    return "\n".join(rows) or "(No participant characteristics available.)"


def format_history(profile: list[dict], include_ratings: bool, include_categories: bool = False) -> str:
    if not profile:
        return "(No prior messages available.)"
    blocks = []
    for index, previous in enumerate(profile, start=1):
        message = str(previous.get("input_message", "")).strip()
        block = f"Past message {index}:\n{message}"
        if include_ratings:
            ratings = previous.get("ratings") or {}
            lines = [
                f"{domain}: {ratings[domain]}"
                for domain in RATING_ORDER
                if ratings.get(domain) is not None
            ]
            block += "\nRatings:\n" + ("\n".join(lines) or "(missing)")
        if include_categories and previous.get("l_category"):
            block += f"\nMessage type: {previous['l_category']}"
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)


def build_prompt(item: dict, condition: str) -> str:
    """Create the explicitly specified prompt component ablation."""
    test_message = str(item.get("input_message", "")).strip()
    profile = item.get("profile_messages", [])
    metadata = format_metadata(item.get("metadata", {}))
    if condition == "pp_cbtact":
        context = (
            "Participant characteristics:\n"
            f"{metadata}\n\n"
            "Seven past rated messages from this participant:\n"
            f"{format_history(profile, include_ratings=True, include_categories=True)}\n\n"
            f"New-message type: {item.get('test_l_category') or '(not available)'}"
        )
        instruction = (
            "Use the participant characteristics and past ratings to make an "
            "individual-level prediction. When predicting coping and quitting, "
            "also use the provided message type (Acceptance or Distraction)."
        )
    elif condition == "full_pp_no_cbtact":
        context = (
            "Participant characteristics:\n"
            f"{metadata}\n\n"
            "Seven past rated messages from this participant:\n"
            f"{format_history(profile, include_ratings=True)}"
        )
        instruction = (
            "Use the participant characteristics and past ratings to make an "
            "individual-level prediction."
        )
    elif condition == "history_ratings_only":
        context = (
            "Seven past rated messages from this participant:\n"
            f"{format_history(profile, include_ratings=True)}"
        )
        instruction = (
            "Use only the past messages and ratings below. Do not infer or use "
            "participant characteristics."
        )
    elif condition == "history_text_only":
        context = (
            "Seven past messages from this participant:\n"
            f"{format_history(profile, include_ratings=False)}"
        )
        instruction = (
            "Use only the past message texts below. Do not infer or use ratings "
            "or participant characteristics."
        )
    else:  # defensive: parser constrains this in normal use
        raise ValueError(f"Unknown condition: {condition}")

    return f"""You are an AI assistant simulating how one participant would rate a new smoking-cessation support message.
{instruction}

Rating dimensions:
1. content — words and meaning
2. coping — helpfulness for coping with an urge or craving
3. quitting — helpfulness for quitting or reducing smoking

Allowed categories:
- content: Very poor, Poor, Acceptable, Good, Very good
- coping/quitting: Not at all helpful, Somewhat helpful, Moderately helpful, Very helpful, Extremely helpful

New message to rate:
"{test_message}"

{context}

Return exactly one JSON object with this schema and no Markdown:
{{
  "response_id": "{item.get('response_id', '')}",
  "predicted_content": "one allowed content category",
  "predicted_coping": "one allowed coping category",
  "predicted_quitting": "one allowed quitting category",
  "explanation": "brief rationale"
}}
"""


def category_lookup() -> dict[str, str]:
    feedback = pd.read_csv(FEEDBACK_PATH)
    columns = {column.lower(): column for column in feedback.columns}
    image_column = columns.get("photo_no")
    category_column = columns.get("l_category")
    if not image_column or not category_column:
        raise ValueError("Feedback CSV lacks photo_no/l_category fields needed for pp_cbtact")
    return {
        str(row[image_column]): str(row[category_column])
        for _, row in feedback[[image_column, category_column]].dropna().iterrows()
    }


def attach_profiles(train: list[dict], test: list[dict], categories: dict[str, str]) -> None:
    by_participant: dict[str, list[dict]] = {}
    for record in train:
        participant = str(record.get("response_id", ""))
        if participant:
            by_participant.setdefault(participant, []).append({
                "input_message": record.get("input_message", ""),
                "ratings": record.get("ratings", {}) or {},
                "l_category": categories.get(str((record.get("metadata") or {}).get("Image ID", "")), ""),
            })
    for record in test:
        record["profile_messages"] = list(
            by_participant.get(str(record.get("response_id", "")), [])
        )
        record["test_l_category"] = categories.get(
            str((record.get("metadata") or {}).get("Image ID", "")), ""
        )


def output_path(condition: str) -> Path:
    return OUTPUT_DIR / f"{condition}_dt10_k7.json"


def load_checkpoint(condition: str) -> dict[str, dict]:
    path = output_path(condition)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint is not a row dictionary: {path}")
    return payload


def save_checkpoint(condition: str, results: dict[str, dict]) -> None:
    path = output_path(condition)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_record(item: dict, parsed: dict, condition: str) -> dict:
    for domain, allowed in ALLOWED_LABELS.items():
        prediction = str(parsed.get(f"predicted_{domain}", "")).strip()
        if prediction not in allowed:
            raise ValueError(
                f"Invalid predicted_{domain}={prediction!r}; expected one of {sorted(allowed)}"
            )
    ratings = item.get("ratings", {}) or {}
    return {
        "response_id": item.get("response_id", ""),
        "input_message": item.get("input_message", ""),
        "metadata": item.get("metadata", {}),
        "ground_truth_content": ratings.get("content", ""),
        "ground_truth_coping": ratings.get("coping", ""),
        "ground_truth_quitting": ratings.get("quitting", ""),
        "predicted_content": parsed.get("predicted_content", ""),
        "predicted_coping": parsed.get("predicted_coping", ""),
        "predicted_quitting": parsed.get("predicted_quitting", ""),
        "explanation": parsed.get("explanation", ""),
        "condition": condition,
        "n_profile_messages": len(item.get("profile_messages", [])),
    }


def parse_response_content(content: object) -> dict:
    """Accept JSON-object responses with optional Markdown code fences."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Model returned no final JSON content")
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(text[start:end + 1])


async def call_one(client, semaphore, qid: str, item: dict, condition: str,
                   retries: int) -> tuple[str, dict | None, str | None]:
    prompt = build_prompt(item, condition)
    async with semaphore:
        for attempt in range(retries):
            try:
                request = {
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if MAX_OUTPUT_TOKENS is not None:
                    request["max_tokens"] = MAX_OUTPUT_TOKENS
                # The Gemini OpenRouter route has returned empty content for
                # json_object requests.  Its prompt still requires an exact
                # JSON object and parse_response_content validates it; omit
                # only the incompatible transport-level schema flag.
                if "gemini" not in MODEL.lower():
                    request["response_format"] = {"type": "json_object"}
                if REASONING_EFFORT and any(token in MODEL.lower() for token in ("gpt-5", "deepseek-r1", "gemini")):
                    # This repository's OpenAI SDK predates a typed
                    # ``reasoning`` parameter; OpenRouter receives it via
                    # the supported pass-through request body.  Gemini 2.5
                    # Pro always thinks; without a bounded effort its hidden
                    # reasoning consumes ``max_tokens`` and the final JSON
                    # content comes back empty.
                    request["extra_body"] = {"reasoning": {"effort": REASONING_EFFORT}}
                # Mirrors the original evaluator: GPT-5 rejects the standard
                # temperature parameter, while the other model families use a
                # deterministic zero-temperature prompt.
                if "gpt-5" not in MODEL.lower():
                    request["temperature"] = 0.0
                completion = await client.chat.completions.create(**request)
                parsed = parse_response_content(completion.choices[0].message.content)
                return qid, make_record(item, parsed, condition), None
            except Exception as exc:  # records remain pending after final retry
                if attempt == retries - 1:
                    return qid, None, f"{type(exc).__name__}: {exc}"
                await asyncio.sleep((2 ** attempt) + random.uniform(0, 0.5))
    return qid, None, "interrupted"


def write_manifest(test: list[dict], conditions: list[str], args) -> None:
    """Record split hashes and the exact settings each condition was run with.

    Condition workers share one output directory, so the manifest is merged
    rather than replaced: the split/prompt block is rewritten (it is identical
    for every worker) and ``runs[<condition>]`` records this worker's settings
    without touching other conditions' entries.  The write is atomic through a
    per-process temporary file, so concurrent workers cannot collide.
    """
    manifest_path = OUTPUT_DIR / "manifest_dt10_k7.json"
    existing: dict = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).isoformat()
    runs = dict(existing.get("runs") or {})
    for condition in conditions:
        runs[condition] = {
            "started_utc": now,
            "max_concurrent": args.max_concurrent,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "reasoning_effort": REASONING_EFFORT,
            "checkpoint_interval": args.checkpoint_interval,
            "max_retries": args.max_retries,
            "resumed": output_path(condition).exists(),
        }
    manifest = {
        "created_utc": existing.get("created_utc", now),
        "updated_utc": now,
        "model": MODEL,
        "split": "dt10_k7",
        "train_path": str(TRAIN_PATH.relative_to(ROOT)),
        "test_path": str(TEST_PATH.relative_to(ROOT)),
        "train_sha256": sha256(TRAIN_PATH),
        "test_sha256": sha256(TEST_PATH),
        "metadata_sha256": sha256(METADATA_PATH),
        "n_test": len(test),
        "canonical_metadata": metadata,
        "conditions": {
            "pp_cbtact": "metadata + seven history texts + their ratings + CBT/ACT message-type labels; same-model baseline",
            "full_pp_no_cbtact": "metadata + seven history texts + their ratings; no CBT/ACT labels/instructions",
            "history_ratings_only": "seven history texts + their ratings; no metadata",
            "history_text_only": "seven history texts only; no history ratings or metadata",
        },
        "requested_conditions": sorted(runs, key=list(CONDITIONS).index),
        "runs": runs,
    }
    if existing.get("notes"):
        manifest["notes"] = existing["notes"]
    temporary = manifest_path.with_name(f"manifest_dt10_k7.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)


async def run_condition(client, test: list[dict], condition: str, args) -> None:
    results = load_checkpoint(condition)
    expected = {str(index) for index in range(len(test))}
    unexpected = set(results) - expected
    if unexpected:
        raise ValueError(f"{condition} checkpoint has unexpected row keys")
    pending = {str(index): item for index, item in enumerate(test) if str(index) not in results}
    print(f"[{condition}] completed={len(results)} pending={len(pending)}", flush=True)
    if not pending:
        return

    semaphore = asyncio.Semaphore(args.max_concurrent)
    tasks = [
        asyncio.create_task(call_one(client, semaphore, qid, item, condition, args.max_retries))
        for qid, item in pending.items()
    ]
    errors: list[str] = []
    completed_since_save = 0
    progress = tqdm(total=len(tasks), desc=condition, dynamic_ncols=True)
    for future in asyncio.as_completed(tasks):
        if shutting_down:
            for task in tasks:
                if not task.done():
                    task.cancel()
            break
        qid, record, error = await future
        progress.update(1)
        if record is not None:
            results[qid] = record
            completed_since_save += 1
            if completed_since_save >= args.checkpoint_interval:
                save_checkpoint(condition, results)
                completed_since_save = 0
        elif error:
            errors.append(f"row={qid}: {error}")
    progress.close()
    save_checkpoint(condition, results)
    if errors:
        error_path = OUTPUT_DIR / f"{condition}_dt10_k7_errors.log"
        error_path.write_text("\n".join(errors) + "\n", encoding="utf-8")
        print(f"[{condition}] saved {len(results)} rows; {len(errors)} rows remain pending", flush=True)
    else:
        print(f"[{condition}] complete: {len(results)}/{len(test)} rows", flush=True)


async def preflight(client, item: dict, condition: str, show: bool = False, retries: int = 1) -> dict:
    """Fail before the batch if the selected OpenRouter model is unavailable."""
    qid, record, error = await call_one(
        client, asyncio.Semaphore(1), "preflight", item, condition, retries=retries
    )
    if error or record is None:
        raise SystemExit(f"OpenRouter preflight failed for {MODEL}: {error}")
    print(f"OpenRouter preflight passed for {MODEL} ({condition}).", flush=True)
    if show:
        for domain in RATING_ORDER:
            print(f"  {domain:9s} truth={record[f'ground_truth_{domain}']!r:22} predicted={record[f'predicted_{domain}']!r}", flush=True)
        print(f"  explanation: {str(record.get('explanation', ''))[:160]}", flush=True)
    return record


async def main_async(args) -> None:
    train = json.loads(TRAIN_PATH.read_text(encoding="utf-8"))
    test = json.loads(TEST_PATH.read_text(encoding="utf-8"))
    if len(train) != 2107 or len(test) != 898:
        raise SystemExit("Canonical dt10-k7 files do not have the expected 2,107/898 rows")
    attach_profiles(train, test, category_lookup())
    profile_sizes = {len(item["profile_messages"]) for item in test}
    if profile_sizes != {7}:
        raise SystemExit(f"Expected exactly 7 history messages per test row; got {profile_sizes}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Canonical split: train={len(train)}, test={len(test)}, profiles={profile_sizes}", flush=True)
    if args.validate_only:
        for condition in args.conditions:
            prompt = build_prompt(test[0], condition)
            if '"predicted_design"' in prompt or "design — how the message looks" in prompt:
                raise SystemExit("Prompt unexpectedly requests a Design rating")
            if condition == "history_text_only" and "Ratings:" in prompt:
                raise SystemExit("history_text_only prompt unexpectedly includes ratings")
            if condition not in {"full_pp_no_cbtact", "pp_cbtact"} and "Participant characteristics:" in prompt:
                raise SystemExit(f"{condition} prompt unexpectedly includes metadata")
            if condition != "pp_cbtact" and ("Message type:" in prompt or "Acceptance" in prompt or "Distraction" in prompt):
                raise SystemExit(f"{condition} prompt unexpectedly includes CBT/ACT information")
        print("Validation passed: dt10-k7 profiles and prompt exclusions are correct.")
        return
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    # Bound provider stalls so a resumable checkpoint run retries a missing
    # row instead of keeping a semaphore slot indefinitely after a network
    # interruption.  Retry policy is managed explicitly in ``call_one``.
    client = AsyncOpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", timeout=120.0, max_retries=0)
    if args.preflight_only:
        # One real API call per requested condition on canonical row 0; nothing is written.
        for condition in args.conditions:
            await preflight(client, test[0], condition, show=True, retries=3)
        return
    await preflight(client, test[0], args.conditions[0])
    write_manifest(test, args.conditions, args)
    for condition in args.conditions:
        if shutting_down:
            break
        await run_condition(client, test, condition, args)


def main() -> None:
    global MODEL, OUTPUT_DIR, MAX_OUTPUT_TOKENS, REASONING_EFFORT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL, help="OpenRouter model identifier")
    parser.add_argument(
        "--output-dir", default=str(OUTPUT_DIR.relative_to(ROOT)),
        help="result directory relative to the repository root",
    )
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), help="explicit effort for GPT-5, DeepSeek-R1, and Gemini 2.5 Pro")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument(
        "--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS,
        help="maximum completion tokens; use 0 to follow the provider default",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--validate-only", action="store_true", help="validate split and prompt exclusions without API calls")
    parser.add_argument("--preflight-only", action="store_true",
                        help="send one real request per requested condition (canonical row 0), print the parsed prediction, write nothing")
    args = parser.parse_args()
    if args.max_concurrent < 1 or args.max_output_tokens < 0 or args.checkpoint_interval < 1 or args.max_retries < 1:
        parser.error("concurrency, checkpoint interval, and retries must be positive; output tokens may be 0")
    candidate_dir = (ROOT / args.output_dir).resolve()
    if ROOT not in candidate_dir.parents:
        parser.error("--output-dir must stay within the repository root")
    MODEL = args.model
    OUTPUT_DIR = candidate_dir
    MAX_OUTPUT_TOKENS = args.max_output_tokens or None
    REASONING_EFFORT = args.reasoning_effort
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
