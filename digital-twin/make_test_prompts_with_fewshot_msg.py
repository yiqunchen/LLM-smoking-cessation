#!/usr/bin/env python3
import os, json, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-json', default='data/processed_llm_data.json')
    ap.add_argument('--personas-dir', default='digital-twin/personas')
    ap.add_argument('--train-messages', required=True)
    ap.add_argument('--test-messages', required=True)
    ap.add_argument('--out-dir', required=True)
    # Default: include all available train examples (can be large)
    ap.add_argument('--fewshot-k', type=int, default=1000000)
    args = ap.parse_args()

    # Load personas
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
    train_set = set(open(args.train_messages).read().splitlines())
    test_set = set(open(args.test_messages).read().splitlines())

    dims = ['content','design','coping','quitting']
    buckets = {d: [] for d in dims}

    # Build index of (participant -> list of (msg, ratings)) for train messages
    per_pid_train = {}
    for item in data:
        pid = item['response_id']
        msg = item.get('input_message')
        if msg not in train_set:
            continue
        r = item.get('ratings', {})
        per_pid_train.setdefault(pid, []).append((msg, r))

    for item in data:
        pid = item['response_id']
        msg = item.get('input_message')
        if msg not in test_set:
            continue
        persona = pid2p.get(pid)
        if not persona:
            continue
        # persona history filtered to train only (no test overlap)
        hist = persona.get('history', {}).get('rated_messages', [])
        hist_train = [rm for rm in hist if rm.get('input_message') in train_set]
        persona_f = dict(persona)
        persona_f['history'] = {'rated_messages': hist_train}
        persona_json = json.dumps(persona_f, ensure_ascii=False)

        # few-shot examples from this participant's train ratings
        fewshot = []
        for (m, rr) in per_pid_train.get(pid, [])[: args.fewshot_k]:
            for dim in dims:
                label = rr.get(dim)
                if label is None:
                    continue
                # map to 1-5
                cd = {'Very poor':1,'Poor':2,'Acceptable':3,'Good':4,'Very good':5}
                cq = {'Not at all helpful':1,'Slightly helpful':2,'Moderately helpful':3,'Very helpful':4,'Extremely helpful':5}
                v = (cd if dim in ['content','design'] else cq).get(label)
                if v is None: continue
                fewshot.append({'example_message': m, 'example_dimension': dim, 'example_label': v})

        img_id = item.get('metadata', {}).get('Image ID')
        image_tags = f"ImageID:{img_id}" if img_id is not None else None
        for dim in dims:
            if dim in ['content','design']:
                scale = '1=Very poor, 2=Poor, 3=Acceptable, 4=Good, 5=Very good'
                question = f"Rate the {dim.upper()} of this message. Respond with 1–5 (" + scale + ") only."
            else:
                scale = '1=Not at all helpful, 2=Slightly helpful, 3=Moderately helpful, 4=Very helpful, 5=Extremely helpful'
                question = f"How helpful is this message for {dim.upper()}? Respond with 1–5 (" + scale + ") only."
            few_dim = [ex for ex in fewshot if ex.get('example_dimension') == dim][: args.fewshot_k]
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
    print(f"Wrote few-shot test prompts to {args.out_dir}")

if __name__ == '__main__':
    main()
