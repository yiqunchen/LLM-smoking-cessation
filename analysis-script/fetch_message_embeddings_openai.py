#!/usr/bin/env python3
"""
Fetch message embeddings from OpenAI for local message datasets.

Outputs:
1) Pickle file with keys expected by existing code:
   {"input_messages": [...], "embeddings": np.ndarray, "model": str, "created_at": str}
2) Optional CSV with one row per message and emb_0000... columns.

Usage example:
  export OPENAI_API_KEY=...
  python analysis-script/fetch_message_embeddings_openai.py \
    --input-json data/processed_llm_data.json \
    --output-pkl message_embeddings.pkl \
    --output-csv archive/data/message_embeddings.csv \
    --model text-embedding-3-large
"""

import argparse
import json
import os
import pickle
import time
from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd
from openai import OpenAI


def load_unique_messages(input_json: str, message_key: str) -> List[str]:
    with open(input_json, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {input_json}")

    seen = set()
    messages: List[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        msg = item.get(message_key)
        if not isinstance(msg, str):
            continue
        msg = msg.strip()
        if not msg:
            continue
        if msg not in seen:
            seen.add(msg)
            messages.append(msg)
    return messages


def embed_messages(
    client: OpenAI,
    messages: List[str],
    model: str,
    batch_size: int,
    max_retries: int,
    sleep_seconds: float,
) -> np.ndarray:
    vectors: List[List[float]] = []
    total = len(messages)
    for i in range(0, total, batch_size):
        batch = messages[i:i + batch_size]
        attempt = 0
        while True:
            try:
                resp = client.embeddings.create(model=model, input=batch)
                batch_vecs = [d.embedding for d in resp.data]
                if len(batch_vecs) != len(batch):
                    raise RuntimeError(
                        f"Embedding count mismatch in batch {i // batch_size}: "
                        f"expected {len(batch)}, got {len(batch_vecs)}"
                    )
                vectors.extend(batch_vecs)
                print(f"Embedded {min(i + len(batch), total)}/{total}")
                break
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    raise RuntimeError(
                        f"Failed after {max_retries} retries in batch starting at index {i}"
                    ) from e
                wait = sleep_seconds * attempt
                print(f"Retry {attempt}/{max_retries} for batch {i // batch_size} after error: {e}")
                time.sleep(wait)
    return np.asarray(vectors, dtype=np.float32)


def save_pickle(output_pkl: str, messages: List[str], embeddings: np.ndarray, model: str) -> None:
    os.makedirs(os.path.dirname(output_pkl) or ".", exist_ok=True)
    payload = {
        "input_messages": messages,
        "embeddings": embeddings,
        "model": model,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_pkl, "wb") as f:
        pickle.dump(payload, f)


def save_csv(output_csv: str, messages: List[str], embeddings: np.ndarray) -> None:
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    n_dims = embeddings.shape[1]
    cols = [f"emb_{i:04d}" for i in range(n_dims)]
    emb_df = pd.DataFrame(embeddings, columns=cols)
    df = pd.concat([pd.DataFrame({"input_message": messages}), emb_df], axis=1)
    df.to_csv(output_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch message embeddings from OpenAI.")
    parser.add_argument("--input-json", default="data/processed_llm_data.json")
    parser.add_argument("--message-key", default="input_message")
    parser.add_argument("--output-pkl", default="message_embeddings.pkl")
    parser.add_argument("--output-csv", default="archive/data/message_embeddings.csv")
    parser.add_argument("--model", default="text-embedding-3-large")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=1.5)
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    messages = load_unique_messages(args.input_json, args.message_key)
    if not messages:
        raise ValueError("No valid messages found.")
    print(f"Loaded {len(messages)} unique messages from {args.input_json}")

    client = OpenAI(api_key=api_key)
    embeddings = embed_messages(
        client=client,
        messages=messages,
        model=args.model,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
    )
    print(f"Embedding matrix shape: {embeddings.shape}")

    save_pickle(args.output_pkl, messages, embeddings, args.model)
    print(f"Saved PKL: {args.output_pkl}")

    if args.output_csv:
        save_csv(args.output_csv, messages, embeddings)
        print(f"Saved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()

