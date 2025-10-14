#!/usr/bin/env python3
"""Export high/low rated training messages per dimension for rule crafting.

This utility reads the processed dataset, restricts to the message-training split,
and writes summaries that can be handed to an LLM (or a human) to derive heuristic
rules. The test split is never touched.
"""
import argparse
import json
import os
from typing import Dict, Optional

import pandas as pd

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
DIMENSIONS = ["content", "design", "coping", "quitting"]


def load_dataset(data_json: str) -> pd.DataFrame:
    with open(data_json, "r") as f:
        data = json.load(f)
    rows = []
    for item in data:
        pid = item["response_id"]
        msg = item["input_message"]
        ratings = item.get("ratings", {}) or {}
        for dim in ["content", "design"]:
            label = CD_MAP.get(ratings.get(dim))
            if label is not None:
                rows.append({
                    "participant_id": pid,
                    "input_message": msg,
                    "dimension": dim,
                    "label": label,
                })
        for dim in ["coping", "quitting"]:
            label = CQ_MAP.get(ratings.get(dim))
            if label is not None:
                rows.append({
                    "participant_id": pid,
                    "input_message": msg,
                    "dimension": dim,
                    "label": label,
                })
    return pd.DataFrame(rows)


def load_split_list(path: str) -> set[str]:
    with open(path, "r") as f:
        return {line.strip() for line in f if line.strip()}


def summarize_examples(df: pd.DataFrame, top_k: int) -> Dict[str, Dict[str, list[dict]]]:
    summaries: Dict[str, Dict[str, list[dict]]] = {}
    for dim in DIMENSIONS:
        sub = df[df["dimension"] == dim]
        grouped = sub.groupby("input_message")["label"].agg(["mean", "count"])
        grouped = grouped.rename(columns={"mean": "mean_label", "count": "n"})
        if grouped.empty:
            summaries[dim] = {"high_examples": [], "low_examples": []}
            continue
        high = grouped.sort_values(["mean_label", "n"], ascending=[False, False]).head(top_k)
        low = grouped.sort_values(["mean_label", "n"], ascending=[True, False]).head(top_k)
        def pack(rows):
            out = []
            for msg, row in rows.iterrows():
                out.append({
                    "input_message": msg,
                    "mean_label": round(float(row["mean_label"]), 3),
                    "num_ratings": int(row["n"]),
                })
            return out
        summaries[dim] = {
            "high_examples": pack(high),
            "low_examples": pack(low),
        }
    return summaries


def write_outputs(summaries: dict, out_json: str, out_md: Optional[str], top_k: int) -> None:
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    payload = {"top_k": top_k, "dimensions": summaries}
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    if out_md:
        os.makedirs(os.path.dirname(out_md), exist_ok=True)
        lines = ["# Training Examples for Rule Crafting", "", f"Top K per bucket: {top_k}", ""]
        for dim, payload in summaries.items():
            lines.append(f"## {dim.title()}")
            lines.append("")
            lines.append("### High-rated messages")
            lines.append("")
            for ex in payload["high_examples"]:
                lines.append(f"- Mean {ex['mean_label']} (n={ex['num_ratings']}): {ex['input_message']}")
            if not payload["high_examples"]:
                lines.append("- (no data)")
            lines.append("")
            lines.append("### Low-rated messages")
            lines.append("")
            for ex in payload["low_examples"]:
                lines.append(f"- Mean {ex['mean_label']} (n={ex['num_ratings']}): {ex['input_message']}")
            if not payload["low_examples"]:
                lines.append("- (no data)")
            lines.append("")
        with open(out_md, "w") as f:
            f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-json", default="data/processed_llm_data.json")
    parser.add_argument("--messages-train", default="digital-twin/splits/messages_train.txt")
    parser.add_argument("--top-k", type=int, default=12, help="Examples per polarity")
    parser.add_argument("--out-json", default="digital-twin/rules/train_examples.json")
    parser.add_argument("--out-md", default="digital-twin/rules/train_examples.md")
    args = parser.parse_args()

    df = load_dataset(args.data_json)
    train_msgs = load_split_list(args.messages_train)
    train_df = df[df["input_message"].isin(train_msgs)].copy()
    summaries = summarize_examples(train_df, args.top_k)
    write_outputs(summaries, args.out_json, args.out_md, args.top_k)
    print(f"Wrote summaries to {args.out_json}")
    if args.out_md:
        print(f"Wrote markdown prompt scaffold to {args.out_md}")


if __name__ == "__main__":
    main()
