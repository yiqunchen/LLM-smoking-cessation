import os
import json
import argparse


SELECT_KEYS = [
    # demographics
    'age_years', 'gender_identity', 'race_ethnicity', 'is_hispanic_latino',
    'education_level', 'household_income',
    # smoking behavior
    'smoking_status', 'cigs_per_day', 'days_smoked_past_30d', 'time_to_first_cig',
    'quit_motivation_level', 'social_support_to_quit', 'quit_attempts_count', 'quit_intention',
    'friends_smoke_level', 'household_smokers',
    # psychosocial scales
    'pain_blocks_valued_life', 'fear_of_feelings', 'worry_about_control',
    'memories_block_fulfillment', 'emotions_cause_problems', 'others_handle_life_better',
    'worry_blocks_success',
]


def build_summary_text(meta: dict) -> str:
    parts = []
    age = meta.get('age_years'); gender = meta.get('gender_identity'); status = meta.get('smoking_status')
    if age: parts.append(f"Age {age}")
    if gender: parts.append(str(gender))
    if status: parts.append(str(status))
    qmot = meta.get('quit_motivation_level'); sup = meta.get('social_support_to_quit')
    if qmot: parts.append(f"Motivation: {qmot}")
    if sup: parts.append(f"Support: {sup}")
    cpd = meta.get('cigs_per_day'); ttf = meta.get('time_to_first_cig')
    if cpd: parts.append(f"CPD: {cpd}")
    if ttf: parts.append(f"TTF: {ttf}")
    return ", ".join(map(str, parts)) or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=str, default='data/processed_llm_data.json')
    ap.add_argument('--out-dir', type=str, default='digital-twin/personas')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.input, 'r') as f:
        data = json.load(f)

    personas = {}
    for item in data:
        pid = item['response_id']
        meta = item.get('metadata', {})
        persona = personas.get(pid)
        if persona is None:
            persona = {
                'participant_id': pid,
                'demographics': {},
                'smoking_behavior': {},
                'psychosocial': {},
                'history': {'rated_messages': []},
                'summary_text': None,
            }
        # assign fields
        for k, v in meta.items():
            if k in ['age_years','gender_identity','race_ethnicity','is_hispanic_latino','education_level','household_income']:
                persona['demographics'][k] = v
            elif k in ['smoking_status','cigs_per_day','days_smoked_past_30d','time_to_first_cig','quit_motivation_level','social_support_to_quit','quit_attempts_count','quit_intention','friends_smoke_level','household_smokers']:
                persona['smoking_behavior'][k] = v
            elif k in ['pain_blocks_valued_life','fear_of_feelings','worry_about_control','memories_block_fulfillment','emotions_cause_problems','others_handle_life_better','worry_blocks_success']:
                persona['psychosocial'][k] = v
        # history (optional, compact)
        ratings = item.get('ratings', {})
        img_id = meta.get('Image ID')
        persona['history']['rated_messages'].append({
            'input_message': item.get('input_message'),
            'ImageID': img_id,
            'ratings': ratings,
        })
        persona['summary_text'] = build_summary_text(meta)
        personas[pid] = persona

    # write individual files and aggregate
    agg = []
    for pid, p in personas.items():
        agg.append(p)
        with open(os.path.join(args.out_dir, f'{pid}.json'), 'w') as f:
            json.dump(p, f, indent=2)
    with open(os.path.join(args.out_dir, 'personas.json'), 'w') as f:
        json.dump(agg, f, indent=2)

    print(f"Wrote {len(agg)} personas to {args.out_dir}")


if __name__ == '__main__':
    main()

