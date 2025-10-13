#!/usr/bin/env python3
import os
import json
import argparse
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score
from scipy.stats import spearmanr


CD_MAP = {'Very poor':1,'Poor':2,'Acceptable':3,'Good':4,'Very good':5}
CQ_MAP = {'Not at all helpful':1,'Slightly helpful':2,'Moderately helpful':3,'Very helpful':4,'Extremely helpful':5}
DIMENSIONS = ['content','design','coping','quitting']


def load_data(json_path: str) -> List[dict]:
    with open(json_path, 'r') as f:
        return json.load(f)


def build_participant_features(items: List[dict]) -> pd.DataFrame:
    # Mirror a compact subset of comprehensive_model_comparison participant features
    numeric = [
        'age_years','days_smoked_past_30d','cigs_per_day','quit_attempts_count',
        'pain_blocks_valued_life','fear_of_feelings','worry_about_control',
        'memories_block_fulfillment','emotions_cause_problems','others_handle_life_better',
        'worry_blocks_success'
    ]
    ordinal_maps = {
        'quit_motivation_level': {
            'Not at all motivated': 1, 'Slightly motivated': 2, 'Moderately motivated': 3,
            'Very motivated': 4, 'Extremely motivated': 5
        },
        'social_support_to_quit': {
            'Not at all supportive': 1, 'Slightly supportive': 2, 'Moderately supportive': 3,
            'Very supportive': 4, 'Extremely supportive': 5
        },
        'time_to_first_cig': {
            'Within 5 minutes': 5, '6-30 minutes': 4, '31-60 minutes': 3, 'After 60 minutes': 2,
            "I don't smoke every day": 1
        },
        'friends_smoke_level': {'None':0,'A few':1,'Some':2,'Most':3,'All':4}
    }
    categoricals = ['gender_identity','race_ethnicity','is_hispanic_latino','quit_intention','household_smokers','quit_attempt_past_year']

    rows = []
    for it in items:
        pid = it['response_id']
        meta = it.get('metadata', {})
        row = {'participant_id': pid}
        for k in numeric:
            v = meta.get(k)
            try:
                row[k] = float(v)
            except Exception:
                row[k] = 0.0
        for k, m in ordinal_maps.items():
            row[k] = m.get(meta.get(k), 0)
        for k in categoricals:
            row[k] = str(meta.get(k) or 'missing')
        rows.append(row)
    df = pd.DataFrame(rows)
    # one-hot top 5 for categoricals
    for k in categoricals:
        top = df[k].value_counts().head(5).index
        for cat in top:
            df[f'{k}_{cat}'] = (df[k] == cat).astype(int)
        df.drop(columns=[k], inplace=True)
    return df


def get_labels(items: List[dict]) -> pd.DataFrame:
    rows = []
    for it in items:
        pid = it['response_id']; msg = it['input_message']; r = it.get('ratings', {})
        for dim in ['content','design']:
            val = CD_MAP.get(r.get(dim))
            if val is not None:
                rows.append({'participant_id': pid, 'input_message': msg, 'dimension': dim, 'label': val})
        for dim in ['coping','quitting']:
            val = CQ_MAP.get(r.get(dim))
            if val is not None:
                rows.append({'participant_id': pid, 'input_message': msg, 'dimension': dim, 'label': val})
    return pd.DataFrame(rows)


def load_split(root: str, split_type: str):
    if split_type == 'msg':
        test_list = os.path.join(root, 'digital-twin/splits/messages_test.txt')
        keys = set(open(test_list, 'r').read().splitlines())
        return ('input_message', keys)
    else:
        test_list = os.path.join(root, 'digital-twin/splits/participants_test.txt')
        keys = set(open(test_list, 'r').read().splitlines())
        return ('participant_id', keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-csv', type=str, required=True, help='digital-twin predictions CSV (participant_id,input_message,dimension,predicted)')
    ap.add_argument('--data-json', type=str, default='data/processed_llm_data.json')
    ap.add_argument('--split-type', type=str, choices=['msg','part'], required=True)
    ap.add_argument('--out-dir', type=str, default='digital-twin/eval_meta_fusion')
    ap.add_argument('--train-pred-csv', type=str, default=None, help='Optional Twin predictions CSV for TRAIN split to select alpha on train')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    root = os.getcwd()
    data = load_data(args.data_json)
    feats_df = build_participant_features(data)
    labels_df = get_labels(data)

    key_name, test_keys = load_split(root, args.split_type)
    # Build label+participant features table
    base = labels_df.merge(feats_df, on='participant_id', how='left')
    # Split into train vs test according to split keys (before merging predictions)
    is_test = base[key_name].isin(test_keys)
    train = base[~is_test].copy()
    test_base = base[is_test].copy()
    # Merge Twin predictions into TEST only
    preds = pd.read_csv(args.pred_csv)
    preds = preds.rename(columns={'predicted':'twin_pred'})
    test = test_base.merge(preds, on=['participant_id','input_message','dimension'], how='inner')

    # Prepare participant-only feature matrices (exclude labels and keys)
    drop_cols = {'participant_id','input_message','dimension','label','twin_pred'}
    feat_cols = [c for c in base.columns if c not in drop_cols]
    if len(feat_cols) == 0:
        Xtr_part = np.zeros((train.shape[0], 0))
        Xte_part = np.zeros((test.shape[0], 0))
    else:
        scaler = StandardScaler()
        if not train.empty:
            Xtr_part = scaler.fit_transform(train[feat_cols])
            Xte_part = scaler.transform(test[feat_cols]) if not test.empty else np.zeros((0, len(feat_cols)))
        else:
            # No train rows: leave participant features as zeros to avoid NotFittedError
            Xtr_part = np.zeros((0, len(feat_cols)))
            Xte_part = np.zeros((test.shape[0], len(feat_cols)))

    # Train per-dimension participant-only RF and fuse with Twin predictions on test
    lines = []
    lines.append('# Meta-Fusion (Participant-only RF + Twin prediction; alpha chosen on TRAIN if available)')
    lines.append(f'Predictions: {os.path.abspath(args.pred_csv)}')
    lines.append(f'Split: {"message" if args.split_type=="msg" else "participant"}-based')
    lines.append('')
    lines.append('| Dimension | Accuracy | Cohen κ | Spearman ρ |')
    lines.append('| --- | --- | --- | --- |')

    for dim in DIMENSIONS:
        tr_d = train[train['dimension']==dim]
        te_d = test[test['dimension']==dim]
        if tr_d.empty or te_d.empty:
            lines.append(f'| {dim} | NA | NA | NA |')
            continue
        ytr = tr_d['label'].values.astype(int)
        yte = te_d['label'].values.astype(int)
        # aligned participant features
        Xtr_d = Xtr_part[train['dimension']==dim] if Xtr_part.size else np.zeros((0, len(feat_cols)))
        Xte_d = Xte_part[test['dimension']==dim] if Xte_part.size else np.zeros((0, len(feat_cols)))

        # Train participant-only RF (if any train rows), else skip to Twin baseline
        rf = None
        if Xtr_d.shape[0] >= 10:  # require minimal samples
            rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
            rf.fit(Xtr_d, ytr)
            proba_part = np.zeros((Xte_d.shape[0], 5))
            if Xte_d.shape[0] > 0:
                # map RF classes to 1..5 proba slots
                proba = rf.predict_proba(Xte_d)
                for cls_idx, cls in enumerate(rf.classes_):
                    if 1 <= int(cls) <= 5:
                        proba_part[:, int(cls)-1] = proba[:, cls_idx]
        else:
            proba_part = np.zeros((te_d.shape[0], 5))

        # Build Twin one-hot distributions on test
        twin_pred = te_d['twin_pred'].values.astype(int)
        proba_twin = np.zeros((te_d.shape[0], 5))
        for i, cls in enumerate(twin_pred):
            if 1 <= cls <= 5:
                proba_twin[i, cls-1] = 1.0

        # Choose alpha on TRAIN if Twin predictions for TRAIN are provided; else use fixed 0.5
        best_alpha = 0.5
        if args.train_pred_csv is not None and rf is not None and Xtr_d.shape[0] > 0:
            # Load train twin predictions and merge for this dimension
            try:
                train_preds = pd.read_csv(args.train_pred_csv)
                train_preds = train_preds.rename(columns={'predicted':'twin_pred'})
                tr_merge = tr_d.merge(train_preds, on=['participant_id','input_message','dimension'], how='inner')
                if not tr_merge.empty:
                    # Participant proba on train
                    Xtr_d_aligned = Xtr_part[train['dimension']==dim]
                    proba_part_tr = np.zeros((Xtr_d_aligned.shape[0], 5))
                    if Xtr_d_aligned.shape[0] > 0:
                        proba = rf.predict_proba(Xtr_d_aligned)
                        for cls_idx, cls in enumerate(rf.classes_):
                            if 1 <= int(cls) <= 5:
                                proba_part_tr[:, int(cls)-1] = proba[:, cls_idx]
                    twin_pred_tr = tr_merge['twin_pred'].values.astype(int)
                    # align lengths (inner merge may reduce rows); rebuild aligned arrays
                    mask_align = train['dimension']==dim
                    # map (pid,msg) to index
                    idx_map = {(pid, msg): i for i,(pid,msg) in enumerate(zip(tr_d['participant_id'], tr_d['input_message']))}
                    sel_idx = [idx_map.get((r['participant_id'], r['input_message'])) for _, r in tr_merge.iterrows()]
                    sel_idx = [i for i in sel_idx if i is not None]
                    if sel_idx:
                        P_part_tr = proba_part_tr[sel_idx]
                        P_twin_tr = np.zeros((len(sel_idx), 5))
                        for i, cls in enumerate(tr_merge['twin_pred'].values.astype(int)):
                            if 1 <= cls <= 5:
                                P_twin_tr[i, cls-1] = 1.0
                        ytr_aligned = tr_merge['label'].values.astype(int)
                        # grid search alpha on TRAIN merge
                        best_acc = -1.0
                        for alpha in np.linspace(0.0, 1.0, 11):
                            P = alpha*P_part_tr + (1-alpha)*P_twin_tr
                            yhat = 1 + np.argmax(P, axis=1)
                            acc = accuracy_score(ytr_aligned, yhat)
                            if acc > best_acc:
                                best_acc = acc
                                best_alpha = alpha
            except Exception:
                pass

        # Evaluate on tst_idx
        n = te_d.shape[0]
        if n > 0:
            P_test = best_alpha*proba_part + (1-best_alpha)*proba_twin
            yhat_test = 1 + np.argmax(P_test, axis=1)
            acc = accuracy_score(yte, yhat_test)
            kap = cohen_kappa_score(yte, yhat_test)
            try:
                rho, _ = spearmanr(yte, yhat_test)
                if np.isnan(rho): rho = 0.0
            except Exception:
                rho = 0.0
            lines.append(f'| {dim} | {acc:.3f} | {kap:.3f} | {rho:.3f} |')
        else:
            lines.append(f'| {dim} | NA | NA | NA |')

    with open(os.path.join(args.out_dir, 'meta_fusion_summary.md'), 'w') as f:
        f.write('\n'.join(lines))
    print(f"Wrote {os.path.join(args.out_dir, 'meta_fusion_summary.md')}")


if __name__ == '__main__':
    main()
