import os
import json
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--personas-dir', type=str, default='digital-twin/personas')
    ap.add_argument('--data-json', type=str, default='data/processed_llm_data.json')
    ap.add_argument('--out-dir', type=str, default='digital-twin/prompts')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load personas by id
    pid_to_persona = {}
    agg_path = os.path.join(args.personas_dir, 'personas.json')
    if os.path.exists(agg_path):
        with open(agg_path, 'r') as f:
            for p in json.load(f):
                pid_to_persona[p['participant_id']] = p
    else:
        # fallback to reading files
        for fn in os.listdir(args.personas_dir):
            if not fn.endswith('.json'): continue
            with open(os.path.join(args.personas_dir, fn), 'r') as f:
                p = json.load(f)
                pid_to_persona[p['participant_id']] = p

    with open(args.data_json, 'r') as f:
        data = json.load(f)

    dims = ['content', 'design', 'coping', 'quitting']
    writers = {d: open(os.path.join(args.out_dir, f'{d}.jsonl'), 'w') for d in dims}

    for item in data:
        pid = item['response_id']
        persona = pid_to_persona.get(pid)
        if not persona:
            continue
        persona_json = json.dumps(persona, ensure_ascii=False)
        message_text = item.get('input_message')
        # Minimal image tags from metadata (if present)
        img_id = item.get('metadata', {}).get('Image ID')
        image_tags = f"ImageID:{img_id}" if img_id is not None else None

        for dim in dims:
            if dim in ['content','design']:
                scale = '1=Very poor, 2=Poor, 3=Acceptable, 4=Good, 5=Very good'
                question = f"Rate the {dim.upper()} of this message. Respond with 1–5 (" + scale + ") only."
            else:
                scale = '1=Not at all helpful, 2=Slightly helpful, 3=Moderately helpful, 4=Very helpful, 5=Extremely helpful'
                question = f"How helpful is this message for {dim.upper()}? Respond with 1–5 (" + scale + ") only."

            rec = {
                'participant_id': pid,
                'input_message': message_text,
                'dimension': dim,
                'persona_json': persona_json,
                'message_text': message_text,
                'image_tags': image_tags,
                'question': question,
            }
            writers[dim].write(json.dumps(rec) + '\n')

    for w in writers.values():
        w.close()
    print(f"Wrote prompts to {args.out_dir}")


if __name__ == '__main__':
    main()

