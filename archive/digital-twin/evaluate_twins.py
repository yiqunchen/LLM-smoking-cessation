import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score
from scipy.stats import spearmanr, kendalltau


def load_truth(data_json: str) -> pd.DataFrame:
    with open(data_json, 'r') as f:
        data = json.load(f)
    rows = []
    cd_map = {'Very poor':1,'Poor':2,'Acceptable':3,'Good':4,'Very good':5}
    cq_map = {'Not at all helpful':1,'Slightly helpful':2,'Moderately helpful':3,'Very helpful':4,'Extremely helpful':5}
    for item in data:
        pid = item['response_id']; msg = item['input_message']; r = item.get('ratings', {})
        for dim in ['content','design']:
            val = cd_map.get(r.get(dim))
            if val is not None:
                rows.append({'participant_id': pid, 'input_message': msg, 'dimension': dim, 'label': val})
        for dim in ['coping','quitting']:
            val = cq_map.get(r.get(dim))
            if val is not None:
                rows.append({'participant_id': pid, 'input_message': msg, 'dimension': dim, 'label': val})
    return pd.DataFrame(rows)


def test_retest_ceiling(df: pd.DataFrame) -> dict:
    # If exact repeats exist (same participant_id + input_message rated twice), compute self‑agreement.
    # Otherwise, return N/A. This dataset likely has single ratings per (pid,msg), so ceiling may be N/A.
    key_cols = ['participant_id','input_message','dimension']
    counts = df.groupby(key_cols).size().reset_index(name='n')
    repeats = counts[counts['n'] > 1]
    if repeats.empty:
        return {d: None for d in df['dimension'].unique()}
    # For repeats, compute majority agreement rate per (pid,msg,dim), then average per dim.
    acc_by_dim = {}
    for dim in df['dimension'].unique():
        sub = df[df['dimension']==dim]
        sub_counts = sub.groupby(key_cols + ['label']).size().reset_index(name='cnt')
        maj = sub_counts.sort_values(['participant_id','input_message','cnt'], ascending=[True,True,False])\
                         .drop_duplicates(['participant_id','input_message'])
        total = sub.groupby(['participant_id','input_message']).size().shape[0]
        if total == 0:
            acc_by_dim[dim] = None
        else:
            acc_by_dim[dim] = float((maj['cnt'] / sub.groupby(['participant_id','input_message']).size().values).mean())
    return acc_by_dim


def pairwise_metrics(pred_df: pd.DataFrame, truth_df: pd.DataFrame) -> dict:
    # Convert to per‑participant rankings and compute pairwise accuracy and Kendall‑τ per participant.
    out = {}
    for dim in truth_df['dimension'].unique():
        psub = pred_df[pred_df['dimension']==dim]
        tsub = truth_df[truth_df['dimension']==dim]
        merged = psub.merge(tsub, on=['participant_id','input_message','dimension'], suffixes=('_pred','_true'))
        by_pid = merged.groupby('participant_id')
        pair_accs = []; taus = []
        for pid, g in by_pid:
            if g.shape[0] < 3:
                continue
            # pairwise accuracy
            correct = 0; total = 0
            vals_p = g['label_pred'].values
            vals_t = g['label_true'].values
            for i in range(len(vals_p)):
                for j in range(i+1, len(vals_p)):
                    total += 1
                    pred_pair = np.sign(vals_p[i] - vals_p[j])
                    true_pair = np.sign(vals_t[i] - vals_t[j])
                    if pred_pair == true_pair:
                        correct += 1
            if total>0:
                pair_accs.append(correct/total)
            # Kendall tau on rankings
            try:
                tau, _ = kendalltau(vals_p, vals_t)
                taus.append(tau)
            except Exception:
                pass
        out[dim] = {
            'pairwise_accuracy_mean': float(np.mean(pair_accs)) if pair_accs else None,
            'kendall_tau_mean': float(np.mean(taus)) if taus else None,
            'n_participants': int(len(pair_accs))
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-csv', type=str, required=True, help='CSV: participant_id,input_message,dimension,predicted (1–5)')
    ap.add_argument('--data-json', type=str, default='data/processed_llm_data.json')
    ap.add_argument('--out-dir', type=str, default='digital-twin/eval')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    truth = load_truth(args.data_json)
    preds = pd.read_csv(args.pred_csv)
    preds = preds.rename(columns={'predicted':'label'})
    preds['label'] = preds['label'].astype(int)

    merged = preds.merge(truth, on=['participant_id','input_message','dimension'], suffixes=('_pred','_true'))

    # Direct metrics by dimension
    lines = []
    lines.append('# Digital Twin Evaluation Summary')
    dims = ['content','design','coping','quitting']
    header = ['Dimension','Accuracy','Cohen_kappa','Spearman_r','Ceiling','Acc/Ceiling']
    lines.append('\n| ' + ' | '.join(header) + ' |')
    lines.append('| ' + ' | '.join(['---']*len(header)) + ' |')

    ceilings = test_retest_ceiling(truth)
    accs = []; kappas = []; rhos = []
    for dim in dims:
        sub = merged[merged['dimension']==dim]
        if sub.empty:
            lines.append(f"| {dim} | NA | NA | NA | {ceilings.get(dim)} | NA |")
            continue
        acc = accuracy_score(sub['label_true'], sub['label_pred'])
        kap = cohen_kappa_score(sub['label_true'], sub['label_pred'])
        try:
            rho, _ = spearmanr(sub['label_true'], sub['label_pred'])
        except Exception:
            rho = np.nan
        accs.append(acc); kappas.append(kap); rhos.append(rho)
        ceil = ceilings.get(dim)
        ratio = (acc/ceil) if ceil and ceil>0 else None
        lines.append(f"| {dim} | {acc:.3f} | {kap:.3f} | {('%.3f'%rho) if not np.isnan(rho) else 'NA'} | {ceil if ceil else 'NA'} | {('%.3f'%ratio) if ratio else 'NA'} |")

    # Overall
    if accs:
        lines.append(f"\n**Overall Accuracy:** {np.mean(accs):.3f}; **κ:** {np.mean(kappas):.3f}; **ρ:** {np.nanmean(rhos):.3f}")

    # Ranking metrics
    rmetrics = pairwise_metrics(preds.rename(columns={'label':'label_pred'}), truth.rename(columns={'label':'label_true'}))
    lines.append('\n## Ranking Metrics (per-participant)')
    lines.append('\n| Dimension | Pairwise Acc (mean) | Kendall τ (mean) | N participants |')
    lines.append('| --- | --- | --- | --- |')
    for dim in dims:
        rm = rmetrics.get(dim, {})
        lines.append(f"| {dim} | {rm.get('pairwise_accuracy_mean','NA')} | {rm.get('kendall_tau_mean','NA')} | {rm.get('n_participants','NA')} |")

    out_md = os.path.join(args.out_dir, 'twin_eval_summary.md')
    with open(out_md, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Wrote {out_md}")


if __name__ == '__main__':
    main()

