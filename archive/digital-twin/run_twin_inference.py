#!/usr/bin/env python3
import os
import re
import json
import argparse
import asyncio
from typing import Dict, Any, List, Optional

import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from tqdm.auto import tqdm

try:
    from openai import AsyncOpenAI
except Exception as e:
    AsyncOpenAI = None


SYSTEM_PROMPT = (
    "You are simulating a specific participant’s survey response. "
    "Read the persona and message carefully. If fewshot_examples are provided, treat them as examples for the SAME dimension you are asked about; "
    "use them as guidance but do not copy labels blindly. Answer with a single integer 1–5 using the exact scale for the specified dimension. "
    "Output only the integer."
)


def build_user_prompt(rec: Dict[str, Any]) -> str:
    persona_json = rec.get('persona_json', '')
    msg_text = rec.get('message_text', '')
    img_tags = rec.get('image_tags')
    question = rec.get('question', '')
    fewshot = rec.get('fewshot')
    blocks = [
        f"persona_json:\n{persona_json}",
        f"message_text:\n{msg_text}",
    ]
    if img_tags:
        blocks.append(f"image_tags: {img_tags}")
    if fewshot:
        blocks.append("fewshot_examples:")
        if isinstance(fewshot, list):
            for ex in fewshot:
                em = ex.get('example_message','')
                ed = ex.get('example_dimension','')
                el = ex.get('example_label','')
                blocks.append(f"- dim: {ed}; label: {el}; msg: {em}")
    blocks.append(f"question:\n{question}")
    blocks.append("Respond with a single integer 1–5. No explanations.")
    return "\n\n".join(blocks)


def parse_int_1_to_5(text: str) -> int:
    m = re.search(r"([1-5])", text)
    if not m:
        return 3
    return int(m.group(1))


async def worker(client: AsyncOpenAI, queue: asyncio.Queue, results: List[Dict[str, Any]], model: str, temperature: float):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        rec = item
        try:
            user_prompt = build_user_prompt(rec)
            resp = await client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = resp.choices[0].message.content.strip()
            pred = parse_int_1_to_5(text)
        except Exception as e:
            pred = 3
        results.append({
            'participant_id': rec.get('participant_id'),
            'input_message': rec.get('input_message'),
            'dimension': rec.get('dimension'),
            'predicted': pred,
        })
        queue.task_done()


async def run_async(
    prompts_dir: str,
    out_csv: str,
    model: str,
    temperature: float,
    concurrency: int,
    max_records: int,
    resume: bool,
    api_provider: str,
    api_key_env: Optional[str],
    api_base: Optional[str],
    default_headers: Optional[Dict[str, str]],
):
    if AsyncOpenAI is None:
        raise RuntimeError("openai>=1.0.0 is required. pip install openai")
    env_var = api_key_env or ('OPENAI_API_KEY' if api_provider == 'openai' else 'OPENROUTER_API_KEY')
    api_key = os.getenv(env_var)
    if not api_key:
        raise RuntimeError(f"API key missing. Set the environment variable {env_var}.")

    client_kwargs: Dict[str, Any] = {'api_key': api_key}
    if api_base:
        client_kwargs['base_url'] = api_base
    if default_headers:
        client_kwargs['default_headers'] = default_headers

    client = AsyncOpenAI(**client_kwargs)

    files = [os.path.join(prompts_dir, f) for f in os.listdir(prompts_dir) if f.endswith('.jsonl')]
    records: List[Dict[str, Any]] = []

    # Resume support: if out_csv exists, load completed keys to skip
    completed: set = set()
    if resume and os.path.exists(out_csv):
        try:
            df_existing = pd.read_csv(out_csv)
            for _, row in df_existing.iterrows():
                completed.add((row['participant_id'], row['input_message'], row['dimension']))
            print(f"Resume: found {len(completed)} completed predictions in {out_csv}")
        except Exception as e:
            print(f"Resume disabled (failed to read existing CSV): {e}")
            completed = set()
    for fp in files:
        with open(fp, 'r') as f:
            for i, line in enumerate(f):
                try:
                    rec = json.loads(line)
                    key = (rec.get('participant_id'), rec.get('input_message'), rec.get('dimension'))
                    if completed and key in completed:
                        continue
                    records.append(rec)
                except Exception:
                    continue
                if 0 < max_records <= len(records):
                    break
        if 0 < max_records <= len(records):
            break

    queue: asyncio.Queue = asyncio.Queue()
    for rec in records:
        await queue.put(rec)
    for _ in range(concurrency):
        await queue.put(None)

    results: List[Dict[str, Any]] = []
    tasks = [asyncio.create_task(worker(client, queue, results, model, temperature)) for _ in range(concurrency)]

    # Progress bar: wait for queue to drain with tqdm
    pbar_total = len(records)
    with tqdm(total=pbar_total, desc='LLM Twin Inference') as pbar:
        prev_done = 0
        while any(not t.done() for t in tasks):
            done_now = len(results)
            pbar.update(max(0, done_now - prev_done))
            prev_done = done_now
            await asyncio.sleep(0.2)
        # Final flush
        done_now = len(results)
        pbar.update(max(0, done_now - prev_done))

    await asyncio.gather(*tasks)

    # Write CSV
    # Merge with existing if resume
    df_new = pd.DataFrame(results)
    if resume and os.path.exists(out_csv):
        try:
            df_old = pd.read_csv(out_csv)
            df = pd.concat([df_old, df_new], ignore_index=True)
            df.drop_duplicates(subset=['participant_id','input_message','dimension'], keep='last', inplace=True)
        except Exception:
            df = df_new
    else:
        df = df_new
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(df)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompts-dir', type=str, default='digital-twin/prompts')
    ap.add_argument('--out', type=str, default=None, help='Optional explicit output CSV path')
    ap.add_argument('--model', type=str, default='gpt-4o-mini')
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--concurrency', type=int, default=8)
    ap.add_argument('--max-records', type=int, default=0, help='Optional cap for quick runs')
    ap.add_argument('--split-type', type=str, default=None, help='Optional split tag to include in auto filename (e.g., msg or part)')
    ap.add_argument('--resume', action='store_true', help='Resume from existing CSV (skip completed prompts)')
    ap.add_argument('--api-provider', choices=['openai', 'openrouter'], default='openai', help='LLM API provider to use')
    ap.add_argument('--api-key-env', type=str, default=None, help='Override environment variable name for API key')
    ap.add_argument('--api-base', type=str, default=None, help='Override API base URL (defaults per provider)')
    ap.add_argument('--http-referer', type=str, default=None, help='Optional HTTP Referer header (required by some providers such as OpenRouter)')
    ap.add_argument('--http-title', type=str, default=None, help='Optional HTTP X-Title header for OpenRouter')
    args = ap.parse_args()

    # Auto-generate output name if not provided: predictions_{model}_temp_{T}.csv
    out_path = args.out
    if out_path is None:
        model_tag = args.model.lower()
        # strip common prefixes and sanitize
        for pref in ('openai/', 'gpt-', 'gpt_'):
            if model_tag.startswith(pref):
                model_tag = model_tag[len(pref):]
                break
        model_tag = model_tag.replace('/', '_').replace('-', '_').replace('.', '_')
        temp_str = f"{args.temperature:g}"
        split_tag = ''
        if args.split_type:
            split_tag = f"{args.split_type}_"
        out_path = os.path.join('digital-twin', f"predictions_{split_tag}{model_tag}_temp_{temp_str}.csv")

    # Normalize to absolute path and ensure parent dir exists
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    default_headers: Optional[Dict[str, str]] = None
    if args.api_provider == 'openrouter':
        default_headers = {}
        referer = args.http_referer or os.getenv('OPENROUTER_HTTP_REFERER')
        title = args.http_title or os.getenv('OPENROUTER_X_TITLE')
        if referer:
            default_headers['HTTP-Referer'] = referer
        if title:
            default_headers['X-Title'] = title
        if not referer:
            raise ValueError('OpenRouter requires an HTTP Referer. Provide --http-referer or set OPENROUTER_HTTP_REFERER.')
        if not title:
            default_headers.setdefault('X-Title', 'LLM Smoking Twin')
    elif args.http_referer or args.http_title:
        raise ValueError('HTTP Referer/Title headers are only valid when using --api-provider openrouter')

    api_base = args.api_base
    if api_base is None and args.api_provider == 'openrouter':
        api_base = 'https://openrouter.ai/api/v1'

    asyncio.run(
        run_async(
            args.prompts_dir,
            out_path,
            args.model,
            args.temperature,
            args.concurrency,
            args.max_records,
            args.resume,
            api_provider=args.api_provider,
            api_key_env=args.api_key_env,
            api_base=api_base,
            default_headers=default_headers,
        )
    )


if __name__ == '__main__':
    main()
