#!/usr/bin/env python3
import os, json, argparse, random


CD_MAP = {'Very poor':1,'Poor':2,'Acceptable':3,'Good':4,'Very good':5}
CQ_MAP = {'Not at all helpful':1,'Slightly helpful':2,'Moderately helpful':3,'Very helpful':4,'Extremely helpful':5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-json', default='data/processed_llm_data.json')
    ap.add_argument('--personas-dir', default='digital-twin/personas')
    ap.add_argument('--train-participants', required=True)
    ap.add_argument('--test-participants', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--fewshot-k', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # Load personas (we will drop history for test)
    pid2p = {}
    agg = os.path.join(args.personas_dir, 'personas.json')
    if os.path.exists(agg):
        for p in json.load(open(agg)):
            pid2p[p['participant_id']] = p
    else:
        for fn in os.listdir(args.personas_dir):
            if fn.endswith('.json'):
                p = json.load(open(os.path.join(args.personas_dir, fn)))
                pid2p[p['participant_id']] = p

    data = json.load(open(args.data_json))
    train_p = set(open(args.train_participants).read().splitlines())
    test_p = set(open(args.test_participants).read().splitlines())

    dims = ['content','design','coping','quitting']

    # Build pool of few-shot examples from TRAIN participants (not test)
    pool = []  # list of dicts: {example_message, example_dimension, example_label}
    for item in data:
        pid = item['response_id']
        if pid not in train_p:
            continue
        msg = item.get('input_message')
        r = item.get('ratings', {})
        for dim in dims:
            label = r.get(dim)
            if label is None:
                continue
            v = (CD_MAP if dim in ['content','design'] else CQ_MAP).get(label)
            if v is None:
                continue
            pool.append({'example_message': msg, 'example_dimension': dim, 'example_label': v})

    # Build test prompts with few-shot examples (drawn from pool)
    buckets = {d: [] for d in dims}
    for item in data:
        pid = item['response_id']
        if pid not in test_p:
            continue
        persona = pid2p.get(pid)
        if not persona:
            continue
        # Drop history entirely for test participants
        pf = dict(persona)
        pf['history'] = {'rated_messages': []}
        persona_json = json.dumps(pf, ensure_ascii=False)

        msg = item.get('input_message')
        img_id = item.get('metadata', {}).get('Image ID')
        image_tags = f"ImageID:{img_id}" if img_id is not None else None

        # sample up to K few-shot examples from pool
        few = random.sample(pool, k=min(args.fewshot_k, len(pool))) if pool else []

        for dim in dims:
            if dim in ['content','design']:
                scale = '1=Very poor, 2=Poor, 3=Acceptable, 4=Good, 5=Very good'
                question = f"Rate the {dim.upper()} of this message. Respond with 1–5 (" + scale + ") only."
            else:
                scale = '1=Not at all helpful, 2=Slightly helpful, 3=Moderately helpful, 4=Very helpful, 5=Extremely helpful'
                question = f"How helpful is this message for {dim.upper()}? Respond with 1–5 (" + scale + ") only."
            few_dim = [ex for ex in few if ex.get('example_dimension') == dim][: args.fewshot_k]
            buckets[dim].append({
                'participant_id': pid,
                'input_message': msg,
                'dimension': dim,
                'persona_json': persona_json,
                'message_text': msg,
                'image_tags': image_tags,
                'question': question,
                'fewshot': few_dim
            })

    os.makedirs(args.out_dir, exist_ok=True)
    for d, recs in buckets.items():
        with open(os.path.join(args.out_dir, f'{d}.jsonl'), 'w') as f:
            for r in recs:
                f.write(json.dumps(r) + '\n')
    print(f"Wrote participant few-shot test prompts to {args.out_dir}")

if __name__ == '__main__':
    main()
