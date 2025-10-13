import os, json, glob

CANON = {
  'participant_7030': 'data_splits/canonical/test_participant_7030.json',
  'digital_1090': 'data_splits/canonical/test_digital_twin_1090.json',
  'digital_3070': 'data_splits/canonical/test_digital_twin_3070.json',
  'digital_7030': 'data_splits/canonical/test_digital_twin_7030.json',
  'digital_9010': 'data_splits/canonical/test_digital_twin_9010.json',
}

def load_expected_counts():
    counts = {}
    for key, path in CANON.items():
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            counts[key] = len(data) if isinstance(data, list) else len(list(data))
        except Exception as e:
            counts[key] = None
    return counts

# Map result filenames to split key for expected count
FILE_SPECS = {
  # Generic LLM (participant 70/30)
  'generic_llm_1_zero_shot.json': 'participant_7030',
  'generic_llm_2_zero_shot_select.json': 'participant_7030',
  'generic_llm_3_few_shot.json': 'participant_7030',
  'generic_llm_4_few_shot_select.json': 'participant_7030',
  'generic_llm_5_continuous.json': 'participant_7030',
  'generic_llm_6_continuous_select.json': 'participant_7030',
  # Digital twin (message-within-participant)
  'digital_twin_1_full_7030.json': 'digital_7030',
  'digital_twin_2_select_7030.json': 'digital_7030',
  'digital_twin_3_feedback_7030.json': 'digital_7030',
  'digital_twin_4_cbtact_1090.json': 'digital_1090',
  'digital_twin_4_cbtact_3070.json': 'digital_3070',
  'digital_twin_4_cbtact_7030.json': 'digital_7030',
  'digital_twin_4_cbtact_9010.json': 'digital_9010',
}

def count_valid_entries(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {'total': 0, 'valid': 0, 'errors': 0}
        total = len(data)
        errors = 0
        for _, item in data.items():
            if not isinstance(item, dict):
                errors += 1
                continue
            if any(item.get(k) == 'ERROR' for k in [
                'predicted_content','predicted_design','predicted_coping','predicted_quitting']):
                errors += 1
        valid = total - errors
        return {'total': total, 'valid': valid, 'errors': errors}
    except Exception:
        return {'total': 0, 'valid': 0, 'errors': 0}


def main():
    expected = load_expected_counts()
    report = []

    result_dirs = sorted(glob.glob('results_manuscript_*'))
    for rdir in result_dirs:
        model_name = rdir.replace('results_manuscript_', '')
        for fname, split_key in FILE_SPECS.items():
            fpath = os.path.join(rdir, fname)
            exp = expected.get(split_key)
            exists = os.path.exists(fpath)
            stats = count_valid_entries(fpath) if exists else {'total': 0, 'valid': 0, 'errors': 0}
            status = 'ok'
            reasons = []
            if not exists:
                status = 'missing'
                reasons.append('file_missing')
            else:
                if exp is not None and stats['valid'] != exp:
                    status = 'incomplete'
                    reasons.append(f"valid={stats['valid']} expected={exp}")
                if stats['errors'] > 0:
                    if status == 'ok':
                        status = 'has_errors'
                    reasons.append(f"errors={stats['errors']}")
            report.append({
                'model': model_name,
                'file': fpath,
                'split_key': split_key,
                'expected': exp,
                'total': stats['total'],
                'valid': stats['valid'],
                'errors': stats['errors'],
                'status': status,
                'reasons': reasons,
            })

    # Save a machine-readable report and print a concise summary
    os.makedirs('results_manuscript', exist_ok=True)
    with open('results_manuscript/verification_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    # Print summary and rerun suggestions
    print("\n=== Verification Summary ===")
    problems = [r for r in report if r['status'] in ('missing','incomplete','has_errors')]
    if not problems:
        print('All models/files complete and clean!')
        return
    # Group by model
    from collections import defaultdict
    by_model = defaultdict(list)
    for r in problems:
        by_model[r['model']].append(r)
    for model, items in by_model.items():
        print(f"\nModel: {model}")
        for r in items:
            print(f"  - {os.path.basename(r['file'])}: {r['status']} ({', '.join(r['reasons'])})")
    
    # Create a rerun list file (just filenames; pipeline can be rerun per model)
    with open('results_manuscript/rerun_list.txt', 'w') as f:
        for r in problems:
            f.write(f"{r['model']}::{os.path.basename(r['file'])}::{r['status']}::{','.join(r['reasons'])}\n")
    print("\nSaved detailed report to results_manuscript/verification_report.json")
    print("Saved rerun list to results_manuscript/rerun_list.txt")

if __name__ == '__main__':
    main()
