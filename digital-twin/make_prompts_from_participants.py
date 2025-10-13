#!/usr/bin/env python3
import os
import json
import argparse


def write_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-json', type=str, default='data/processed_llm_data.json')
    ap.add_argument('--personas-dir', type=str, default='digital-twin/personas')
    ap.add_argument('--participant-list', type=str, required=True, help='Text file: one participant_id per line')
    ap.add_argument('--out-dir', type=str, required=True, help='Output directory for prompts JSONL')
    args = ap.parse_args()

    # Load personas by id
    pid_to_persona = {}
    agg_path = os.path.join(args.personas_dir, 'personas.json')
    if os.path.exists(agg_path):
        with open(agg_path, 'r') as f:
            for p in json.load(f):
                pid_to_persona[p['participant_id']] = p
    else:
        for fn in os.listdir(args.personas_dir):
            if not fn.endswith('.json'):
                continue
            with open(os.path.join(args.personas_dir, fn), 'r') as f:
                p = json.load(f)
                pid_to_persona[p['participant_id']] = p

    with open(args.data_json, 'r') as f:
        data = json.load(f)
    keep_p = set(open(args.participant_list, 'r').read().splitlines())

    dims = ['content', 'design', 'coping', 'quitting']
    buckets = {d: [] for d in dims}

    for item in data:
        pid = item['response_id']
        if pid not in keep_p:
            continue
        persona = pid_to_persona.get(pid)
        if not persona:
            continue
        persona_json = json.dumps(persona, ensure_ascii=False)
        msg = item.get('input_message')
        img_id = item.get('metadata', {}).get('Image ID')
        image_tags = f"ImageID:{img_id}" if img_id is not None else None
        for dim in dims:
            if dim in ['content','design']:
                scale = '1=Very poor, 2=Poor, 3=Acceptable, 4=Good, 5=Very good'
                question = f"Rate the {dim.upper()} of this message. Respond with 1–5 (" + scale + ") only."
            else:
                scale = '1=Not at all helpful, 2=Slightly helpful, 3=Moderately helpful, 4=Very helpful, 5=Extremely helpful'
                question = f"How helpful is this message for {dim.upper()}? Respond with 1–5 (" + scale + ") only."
            buckets[dim].append({
                'participant_id': pid,
                'input_message': msg,
                'dimension': dim,
                'persona_json': persona_json,
                'message_text': msg,
                'image_tags': image_tags,
                'question': question,
            })

    os.makedirs(args.out_dir, exist_ok=True)
    for dim, recs in buckets.items():
        write_jsonl(recs, os.path.join(args.out_dir, f'{dim}.jsonl'))
    print(f"Wrote prompts for {sum(len(v) for v in buckets.values())} rows to {args.out_dir}")


if __name__ == '__main__':
    main()

