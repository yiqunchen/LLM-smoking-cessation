import os, json, glob
from typing import Dict, Any

RERUN_LIST = 'results_manuscript/rerun_list.txt'

# Derive checkpoint filename from result filename
def checkpoint_name_for(result_filename: str) -> str:
    # generic_llm_N_*.json -> checkpoints/checkpoint_generic_llm_N.json
    if result_filename.startswith('generic_llm_'):
        parts = result_filename.split('_')  # ['generic', 'llm', 'N', ...]
        if len(parts) >= 3 and parts[2].isdigit():
            n = parts[2]
            return f'checkpoints/checkpoint_generic_llm_{n}.json'
    # digital_twin_M_*_<split>.json -> checkpoints/checkpoint_digital_twin_M_<split>.json
    if result_filename.startswith('digital_twin_'):
        base = result_filename.replace('digital_twin_', '')  # e.g., '4_cbtact_1090.json'
        # split key is last token before .json
        try:
            main, _ = os.path.splitext(base)
            # main like '4_cbtact_1090' or '1_full_7030'
            tokens = main.split('_')
            if len(tokens) >= 2:
                m = tokens[0]  # method index (e.g., '4')
                split = tokens[-1] if tokens[-1].isdigit() else '7030'
                return f'checkpoints/checkpoint_digital_twin_{m}_{split}.json'
        except Exception:
            pass
    return None


def load_json(path: str) -> Any:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def save_json(path: str, data: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def is_bad_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return True
    for k in ('predicted_content','predicted_design','predicted_coping','predicted_quitting'):
        if item.get(k) == 'ERROR':
            return True
    return False


def clean_file(result_dir: str, result_filename: str) -> Dict[str,int]:
    result_path = os.path.join(result_dir, result_filename)
    checkpoint_rel = checkpoint_name_for(result_filename)
    checkpoint_path = os.path.join(result_dir, checkpoint_rel) if checkpoint_rel else None

    stats = {'removed_from_result': 0, 'removed_from_checkpoint': 0}

    # Clean result file if present
    data = load_json(result_path)
    if isinstance(data, dict):
        keys_to_delete = [k for k, v in data.items() if is_bad_item(v)]
        if keys_to_delete:
            for k in keys_to_delete:
                data.pop(k, None)
            save_json(result_path, data)
            stats['removed_from_result'] = len(keys_to_delete)

    # Clean checkpoint file if present
    if checkpoint_path:
        ckp = load_json(checkpoint_path)
        if isinstance(ckp, dict):
            keys_to_delete = [k for k, v in ckp.items() if is_bad_item(v)]
            if keys_to_delete:
                for k in keys_to_delete:
                    ckp.pop(k, None)
                save_json(checkpoint_path, ckp)
                stats['removed_from_checkpoint'] = len(keys_to_delete)

    return stats


def main():
    # Build tasks to clean from rerun_list if exists, else from scanning dirs
    tasks = []
    if os.path.exists(RERUN_LIST):
        with open(RERUN_LIST, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines:
            # model::filename::status::reasons
            parts = line.split('::')
            if len(parts) >= 2:
                model, filename = parts[0], parts[1]
                tasks.append((f'results_manuscript_{model}', filename))
    else:
        # Fallback: scan all result dirs and pick known files
        for rdir in glob.glob('results_manuscript_*'):
            for filename in os.listdir(rdir):
                if not filename.endswith('.json'):
                    continue
                if filename.startswith(('generic_llm_', 'digital_twin_')):
                    tasks.append((rdir, filename))

    # Execute cleaning
    summary = []
    for rdir, fname in tasks:
        stats = clean_file(rdir, fname)
        if stats['removed_from_result'] or stats['removed_from_checkpoint']:
            summary.append({'dir': rdir, 'file': fname, **stats})
            print(f"Cleaned {rdir}/{fname}: -result {stats['removed_from_result']} -checkpoint {stats['removed_from_checkpoint']}")

    # Save summary
    os.makedirs('results_manuscript', exist_ok=True)
    save_json('results_manuscript/cleanup_summary.json', summary)
    print("\nSaved cleanup summary to results_manuscript/cleanup_summary.json")
    if not summary:
        print("No bad entries found to remove.")

if __name__ == '__main__':
    main()
