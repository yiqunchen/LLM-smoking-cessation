#!/usr/bin/env python3
import os
import json
import argparse
from sklearn.model_selection import train_test_split


def write_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')


def build_prompts_for_subset(data, personas_by_id, keep_messages):
    dims = ['content', 'design', 'coping', 'quitting']
    buckets = {d: [] for d in dims}
    for item in data:
        pid = item['response_id']
        msg = item.get('input_message')
        if msg not in keep_messages:
            continue
        persona = personas_by_id.get(pid)
        if not persona:
            continue
        persona_json = json.dumps(persona, ensure_ascii=False)
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
    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-json', type=str, default='data/processed_llm_data.json')
    ap.add_argument('--personas-dir', type=str, default='digital-twin/personas')
    ap.add_argument('--out-dir', type=str, default='digital-twin/prompts_test')
    ap.add_argument('--splits-dir', type=str, default='digital-twin/splits')
    ap.add_argument('--test-size', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    # Load data and personas
    with open(args.data_json, 'r') as f:
        data = json.load(f)
    personas_by_id = {}
    agg_path = os.path.join(args.personas_dir, 'personas.json')
    if os.path.exists(agg_path):
        with open(agg_path, 'r') as f:
            for p in json.load(f):
                personas_by_id[p['participant_id']] = p
    else:
        for fn in os.listdir(args.personas_dir):
            if not fn.endswith('.json'): continue
            with open(os.path.join(args.personas_dir, fn), 'r') as f:
                p = json.load(f)
                personas_by_id[p['participant_id']] = p

    # Message-based split
    uniq_messages = sorted({d['input_message'] for d in data})
    train_msgs, test_msgs = train_test_split(uniq_messages, test_size=args.test_size, random_state=args.seed)

    # Save split lists
    os.makedirs(args.splits_dir, exist_ok=True)
    with open(os.path.join(args.splits_dir, 'messages_train.txt'), 'w') as f:
        f.write('\n'.join(train_msgs))
    with open(os.path.join(args.splits_dir, 'messages_test.txt'), 'w') as f:
        f.write('\n'.join(test_msgs))

    # Prepare filter: remove any persona history entries whose input_message is in TEST messages
    test_set = set(test_msgs)

    def filtered_persona(persona):
        p = dict(persona)
        hist = p.get('history', {})
        rated = hist.get('rated_messages', []) or []
        rated_f = [rm for rm in rated if rm.get('input_message') not in test_set]
        p['history'] = {'rated_messages': rated_f}
        return p

    # Build test-only prompts with filtered persona JSON (no overlap with test messages)
    dims = ['content', 'design', 'coping', 'quitting']
    buckets = {d: [] for d in dims}
    for item in data:
        if item['input_message'] not in test_set:
            continue
        pid = item['response_id']
        persona = personas_by_id.get(pid)
        if not persona:
            continue
        persona_json = json.dumps(filtered_persona(persona), ensure_ascii=False)
        message_text = item.get('input_message')
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
                'input_message': message_text,
                'dimension': dim,
                'persona_json': persona_json,
                'message_text': message_text,
                'image_tags': image_tags,
                'question': question,
            })

    for dim, recs in buckets.items():
        write_jsonl(recs, os.path.join(args.out_dir, f'{dim}.jsonl'))

    print(f"Wrote test-only prompts to {args.out_dir}")


if __name__ == '__main__':
    main()
