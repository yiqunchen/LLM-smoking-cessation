#!/usr/bin/env python
"""Run GEPA optimization for digital twin rating tasks."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Sequence, Tuple

import dspy
from dspy.teleprompt.gepa import GEPA
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
from tqdm import tqdm

CD_MAP = {
    "Very poor": 1,
    "Poor": 2,
    "Acceptable": 3,
    "Good": 4,
    "Very good": 5,
}
CQ_MAP = {
    "Not at all helpful": 1,
    "Slightly helpful": 2,
    "Moderately helpful": 3,
    "Very helpful": 4,
    "Extremely helpful": 5,
}
CD_INV = {v: k for k, v in CD_MAP.items()}
CQ_INV = {v: k for k, v in CQ_MAP.items()}
DEFAULT_DIMENSIONS = ["content", "design", "coping", "quitting"]
DEFAULT_INSTRUCTIONS = (
    "You are simulating the participant described in persona_json."
    " Use the persona to adopt their goals, concerns, and style."
    " Read message_text carefully and focus on the specified dimension."
    " Answer the question with a single digit from 1-5 in the rating field."
    " Provide concise reasoning in the rationale field before the rating."
    " Never output words like 'Rating:' or additional text in the rating field."
)
REASONING_TAGS = ("/o1", "/o3", "/o4", "/o5", "gpt-5")
OPENROUTER_PREFIX = "openrouter/"


class DigitalTwinSignature(dspy.Signature):
    persona_json = dspy.InputField(desc="JSON persona for the participant to roleplay")
    message_text = dspy.InputField(desc="Smoking cessation message text to evaluate")
    dimension = dspy.InputField(desc="Target dimension: content/design/coping/quitting")
    question = dspy.InputField(desc="Rating question with scale guidance")
    rationale = dspy.OutputField(desc="Short reasoning that references persona goals")
    rating = dspy.OutputField(desc="Final score as a digit 1-5 with no extra text")
    instructions: ClassVar[str] = DEFAULT_INSTRUCTIONS


class DigitalTwinModule(dspy.Module):
    def __init__(self, instructions: str | None = None):
        super().__init__()
        signature_cls = DigitalTwinSignature
        if instructions:
            signature_cls = type(
                "CustomDigitalTwinSignature",
                (DigitalTwinSignature,),
                {
                    "__annotations__": {"instructions": ClassVar[str]},
                    "instructions": instructions,
                },
            )
        self.rate = dspy.ChainOfThought(signature_cls)

    def forward(self, persona_json: str, message_text: str, dimension: str, question: str):
        result = self.rate(
            persona_json=persona_json,
            message_text=message_text,
            dimension=dimension,
            question=question,
        )
        cleaned = extract_rating(result.rating)
        if cleaned is not None:
            result.rating = cleaned
        return result


def extract_rating(text: Any) -> str | None:
    if text is None:
        return None
    if isinstance(text, (int, float)):
        text = str(int(text))
    if not isinstance(text, str):
        text = str(text)
    match = re.search(r"\b([1-5])\b", text)
    if not match:
        return None
    return match.group(1)


def load_personas(personas_dir: str) -> Dict[str, Any]:
    personas_path = Path(personas_dir)
    agg_path = personas_path / "personas.json"
    if agg_path.exists():
        with open(agg_path, "r") as handle:
            data = json.load(handle)
            return {item["participant_id"]: item for item in data}
    personas = {}
    for file in personas_path.glob("*.json"):
        with open(file, "r") as handle:
            item = json.load(handle)
            personas[item["participant_id"]] = item
    return personas


def label_to_int(dimension: str, label_text: str | None) -> int | None:
    if label_text is None:
        return None
    if dimension in ("content", "design"):
        return CD_MAP.get(label_text)
    return CQ_MAP.get(label_text)


def label_to_text(dimension: str, label_value: int) -> str:
    if dimension in ("content", "design"):
        return CD_INV[label_value]
    return CQ_INV[label_value]


def build_question(dimension: str) -> str:
    if dimension in ("content", "design"):
        return (
            "Rate the {dim} of this message. Respond with 1-5 only where 1=Very poor, 2=Poor, "
            "3=Acceptable, 4=Good, 5=Very good."
        ).format(dim=dimension.upper())
    return (
        "How helpful is this message for {dim}? Respond with 1-5 only where 1=Not at all helpful, "
        "2=Slightly helpful, 3=Moderately helpful, 4=Very helpful, 5=Extremely helpful."
    ).format(dim=dimension.upper())


def build_records(
    data_json: str,
    personas_dir: str,
    dimensions: Sequence[str],
) -> List[Dict[str, Any]]:
    with open(data_json, "r") as handle:
        data = json.load(handle)
    personas = load_personas(personas_dir)
    records: List[Dict[str, Any]] = []
    for item in data:
        participant = item.get("response_id")
        persona = personas.get(participant)
        if not persona:
            continue
        persona_json = json.dumps(persona, ensure_ascii=False)
        message = item.get("input_message")
        ratings = item.get("ratings", {})
        for dimension in dimensions:
            gold_label = label_to_int(dimension, ratings.get(dimension))
            if gold_label is None:
                continue
            records.append(
                {
                    "participant_id": participant,
                    "persona_json": persona_json,
                    "message_text": message,
                    "dimension": dimension,
                    "question": build_question(dimension),
                    "label": str(gold_label),
                    "label_text": label_to_text(dimension, gold_label),
                }
            )
    return records


def split_records(
    records: Sequence[Dict[str, Any]],
    train_size: int,
    val_size: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    by_message: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        key = rec.get("message_text") or rec.get("input_message")
        if key is None:
            raise ValueError("Record is missing message text for grouping.")
        by_message.setdefault(key, []).append(rec)

    message_items = list(by_message.items())
    rng.shuffle(message_items)

    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []

    for message, recs in message_items:
        if len(train_records) < train_size:
            train_records.extend(recs)
        elif len(val_records) < val_size:
            val_records.extend(recs)
        else:
            break

    return train_records, val_records


def to_examples(records: Iterable[Dict[str, Any]]) -> List[dspy.Example]:
    examples: List[dspy.Example] = []
    for rec in records:
        example = dspy.Example(**rec).with_inputs(
            "persona_json",
            "message_text",
            "dimension",
            "question",
        )
        examples.append(example)
    return examples


def evaluate_module(
    module: DigitalTwinModule,
    dataset: Sequence[dspy.Example],
    desc: str,
) -> Tuple[float, List[Dict[str, Any]]]:
    records = list(dataset)
    results: List[Dict[str, Any]] = []
    total = len(records)
    correct = 0
    iterator = tqdm(records, desc=desc, leave=False)
    for example in iterator:
        inputs = dict(example.inputs().items())
        gold = example["label"]
        prediction = module(**inputs)
        predicted = extract_rating(getattr(prediction, "rating", None))
        is_correct = int(predicted == gold)
        correct += is_correct
        results.append(
            {
                "participant_id": example.get("participant_id"),
                "dimension": inputs["dimension"],
                "gold": gold,
                "gold_text": example.get("label_text"),
                "predicted": predicted,
                "rationale": getattr(prediction, "rationale", None),
                "success": bool(is_correct),
            }
        )
    accuracy = correct / total if total else 0.0
    return accuracy, results


def gepa_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: Any | None = None,
    pred_name: str | None = None,
    pred_trace: Any | None = None,
) -> ScoreWithFeedback:
    gold = example["label"]
    gold_text = example.get("label_text")
    predicted = extract_rating(getattr(prediction, "rating", None))
    if predicted is None:
        return ScoreWithFeedback(
            score=0.0,
            feedback=(
                "The rating field must contain only a single digit between 1 and 5. "
                "Re-read the instructions and output just that digit."
            ),
        )
    if predicted == gold:
        return ScoreWithFeedback(
            score=1.0,
            feedback=f"Correct rating {gold} ({gold_text}). Keep following the persona context."
        )
    question = example["question"]
    persona_hint = example.get("persona_json", "")[:240]
    feedback = (
        f"Incorrect rating {predicted}. The correct rating is {gold} ({gold_text})."
        f" Use the persona context and question guidance: {question}."
        f" Persona snippet: {persona_hint}."
        " Return only the digit 1-5 in the rating field."
    )
    return ScoreWithFeedback(score=0.0, feedback=feedback)


def infer_defaults(model: str, temperature: float | None, max_tokens: int | None) -> Tuple[float, int]:
    lower = model.lower()
    is_reasoning = any(tag in lower for tag in REASONING_TAGS)
    temp = temperature if temperature is not None else (1.0 if is_reasoning else 0.2)
    max_tok = max_tokens if max_tokens is not None else (32000 if is_reasoning else 4096)
    return temp, max_tok


def resolve_openrouter_headers(
    referer: Optional[str],
    title: Optional[str],
) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    referer = referer or os.getenv("OPENROUTER_HTTP_REFERER")
    title = title or os.getenv("OPENROUTER_X_TITLE")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    if not referer:
        raise ValueError(
            "OpenRouter requires an HTTP Referer. Provide --http-referer or set OPENROUTER_HTTP_REFERER."
        )
    headers.setdefault("X-Source", "digital-twin-gepa")
    return headers


def build_lm(
    model: str,
    temperature: float,
    max_tokens: int,
    api_base: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
) -> dspy.LM:
    kwargs: Dict[str, Any] = {}
    if api_base:
        kwargs["api_base"] = api_base
    if extra_headers:
        kwargs["extra_headers"] = extra_headers
    return dspy.LM(model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)


def sanitize_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GEPA for digital twin ratings")
    parser.add_argument("--data-json", default="data/processed_llm_data.json")
    parser.add_argument("--personas-dir", default="digital-twin/personas")
    parser.add_argument(
        "--dimensions",
        nargs="+",
        default=DEFAULT_DIMENSIONS,
        help="Dimensions to include (subset of content design coping quitting)",
    )
    parser.add_argument("--train-size", type=int, default=320)
    parser.add_argument("--val-size", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--student-temperature", type=float)
    parser.add_argument("--student-max-tokens", type=int)
    parser.add_argument("--reflection-model", required=True)
    parser.add_argument("--reflection-temperature", type=float)
    parser.add_argument("--reflection-max-tokens", type=int)
    parser.add_argument("--student-api-base", type=str)
    parser.add_argument("--reflection-api-base", type=str)
    parser.add_argument("--student-http-referer", type=str)
    parser.add_argument("--student-http-title", type=str)
    parser.add_argument("--reflection-http-referer", type=str)
    parser.add_argument("--reflection-http-title", type=str)
    parser.add_argument("--max-full-evals", type=int, default=6)
    parser.add_argument("--reflection-minibatch-size", type=int, default=4)
    parser.add_argument("--max-merge-invocations", type=int, default=5)
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--log-dir", default="digital-twin/gepa_logs")
    parser.add_argument("--output-dir", default="digital-twin/gepa_results")
    parser.add_argument("--instructions-path", type=str)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dimensions = [dim.lower() for dim in args.dimensions]
    records = build_records(args.data_json, args.personas_dir, dimensions)
    if not records:
        raise ValueError("No records found for the requested configuration.")
    train_records, val_records = split_records(records, args.train_size, args.val_size, args.seed)
    trainset = to_examples(train_records)
    valset = to_examples(val_records)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    instructions = DEFAULT_INSTRUCTIONS
    if args.instructions_path:
        with open(args.instructions_path, "r") as handle:
            instructions = handle.read().strip()

    run_tag = args.run_name or f"{sanitize_name(args.student_model)}__{sanitize_name(args.reflection_model)}"
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    summary_path = output_dir / f"gepa_summary_{run_tag}_{timestamp}.json"

    summary: Dict[str, Any] = {
        "student_model": args.student_model,
        "reflection_model": args.reflection_model,
        "train_size": len(trainset),
        "val_size": len(valset),
        "dimensions": dimensions,
        "log_dir": str(log_dir),
        "instructions_initial": instructions,
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        summary["note"] = "Dry run requested; skipped LM evaluation and GEPA compile."
        with open(summary_path, "w") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Wrote summary to {summary_path}")
        return

    student_temp, student_max_tokens = infer_defaults(
        args.student_model, args.student_temperature, args.student_max_tokens
    )
    reflection_temp, reflection_max_tokens = infer_defaults(
        args.reflection_model, args.reflection_temperature, args.reflection_max_tokens
    )

    student_headers: Optional[Dict[str, str]] = None
    if args.student_model.startswith(OPENROUTER_PREFIX):
        student_headers = resolve_openrouter_headers(
            args.student_http_referer, args.student_http_title
        )
        summary["student_headers"] = student_headers

    reflection_headers: Optional[Dict[str, str]] = None
    if args.reflection_model.startswith(OPENROUTER_PREFIX):
        reflection_headers = resolve_openrouter_headers(
            args.reflection_http_referer, args.reflection_http_title
        )
        summary["reflection_headers"] = reflection_headers

    dspy.configure(
        lm=build_lm(
            model=args.student_model,
            temperature=student_temp,
            max_tokens=student_max_tokens,
            api_base=args.student_api_base,
            extra_headers=student_headers,
        )
    )
    student_module = DigitalTwinModule(instructions=instructions)

    baseline_train_acc, _ = evaluate_module(student_module, trainset, desc="baseline-train")
    baseline_val_acc, baseline_details = evaluate_module(student_module, valset, desc="baseline-val")
    summary["baseline_train_accuracy"] = baseline_train_acc
    summary["baseline_val_accuracy"] = baseline_val_acc

    reflection_lm = build_lm(
        model=args.reflection_model,
        temperature=reflection_temp,
        max_tokens=reflection_max_tokens,
        api_base=args.reflection_api_base,
        extra_headers=reflection_headers,
    )
    gepa_runner = GEPA(
        metric=gepa_metric,
        reflection_lm=reflection_lm,
        max_full_evals=args.max_full_evals,
        reflection_minibatch_size=args.reflection_minibatch_size,
        use_merge=not args.skip_merge,
        max_merge_invocations=args.max_merge_invocations,
        log_dir=str(log_dir / f"{run_tag}_{timestamp}"),
        track_stats=True,
        track_best_outputs=True,
    )

    try:
        optimized_module = gepa_runner.compile(
            student_module,
            trainset=trainset,
            valset=valset,
        )
    except Exception as err:  # noqa: BLE001
        summary["gepa_error"] = str(err)
        summary["baseline_examples"] = baseline_details[:25]
        with open(summary_path, "w") as handle:
            json.dump(summary, handle, indent=2)
        raise

    optimized_train_acc, _ = evaluate_module(optimized_module, trainset, desc="optimized-train")
    optimized_val_acc, optimized_details = evaluate_module(optimized_module, valset, desc="optimized-val")
    summary["optimized_train_accuracy"] = optimized_train_acc
    summary["optimized_val_accuracy"] = optimized_val_acc
    summary["instructions_optimized"] = optimized_module.rate.signature.instructions

    if hasattr(optimized_module, "detailed_results") and optimized_module.detailed_results is not None:
        summary["gepa_detailed"] = optimized_module.detailed_results.to_dict()

    summary["baseline_examples"] = baseline_details[:25]
    summary["optimized_examples"] = optimized_details[:25]

    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
