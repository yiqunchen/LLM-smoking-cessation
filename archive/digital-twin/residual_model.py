import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score


def ordinal_to_int(v, dim):
    cd = {'Very poor':1,'Poor':2,'Acceptable':3,'Good':4,'Very good':5}
    cq = {'Not at all helpful':1,'Slightly helpful':2,'Moderately helpful':3,'Very helpful':4,'Extremely helpful':5}
    return (cd if dim in ['content','design'] else cq).get(v, None)


def load_dataset(data_json: str, emb_csv: str) -> pd.DataFrame:
    with open(data_json, 'r') as f:
        data = json.load(f)
    emb = pd.read_csv(emb_csv)
    cols = [c for c in emb.columns if c.startswith('emb_')]
    df = []
    for item in data:
        pid = item['response_id']; msg = item['input_message']; meta = item.get('metadata', {})
        mid = meta.get('Image ID')
        ratings = item.get('ratings', {})
        row = {
            'participant_id': pid,
            'input_message': msg,
            'message_id': mid,
            'content': ordinal_to_int(ratings.get('content'), 'content'),
            'design': ordinal_to_int(ratings.get('design'), 'design'),
            'coping': ordinal_to_int(ratings.get('coping'), 'coping'),
            'quitting': ordinal_to_int(ratings.get('quitting'), 'quitting'),
        }
        df.append(row)
    df = pd.DataFrame(df)
    # merge embeddings
    if 'message_id' in emb.columns:
        merged = df.merge(emb, left_on='message_id', right_on='message_id', how='left')
    else:
        merged = df
    return merged, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-json', type=str, default='data/processed_llm_data.json')
    ap.add_argument('--emb-csv', type=str, default='data/message_embeddings.csv')
    ap.add_argument('--out-dir', type=str, default='digital-twin/eval')
    ap.add_argument('--pca', type=int, default=32)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df, emb_cols = load_dataset(args.data_json, args.emb_csv)
    dims = ['content','design','coping','quitting']

    # Per‑participant baseline: mean rating per participant and dimension
    baseline = df.groupby('participant_id')[dims].mean().reset_index()
    dfb = df.merge(baseline, on='participant_id', suffixes=('', '_base'))
    for d in dims:
        dfb[d + '_resid'] = dfb[d] - dfb[d + '_base']

    # Embedding PCA
    X = dfb[emb_cols].fillna(0.0).values if emb_cols else None
    if X is None or X.shape[1] == 0:
        print('No embeddings available — residual model skipped.')
        return
    k = min(args.pca, X.shape[1])
    pca = PCA(n_components=k, random_state=42)
    Z = pca.fit_transform(X)

    # Fit residuals with Ridge per dimension (simple baseline)
    results = []
    for d in dims:
        y = dfb[d + '_resid'].values
        mask = ~np.isnan(y)
        if mask.sum() < 20:
            results.append({'dimension': d, 'r2': None})
            continue
        mdl = Ridge(alpha=1.0, random_state=42)
        mdl.fit(Z[mask], y[mask])
        r2 = mdl.score(Z[mask], y[mask])
        results.append({'dimension': d, 'r2': float(r2)})

    # Simple diagnostic: reconstruct predicted ratings = baseline + residual_hat; compute 5‑class accuracy by rounding
    diag = []
    for d in dims:
        y = dfb[d].values
        y_base = dfb[d + '_base'].values
        # reuse the same Ridge fit by re‑fitting here (for brevity)
        mask = ~np.isnan(y)
        mdl = Ridge(alpha=1.0, random_state=42)
        mdl.fit(Z[mask], (y - y_base)[mask])
        yhat = y_base + mdl.predict(Z)
        yhat_cls = np.clip(np.rint(yhat), 1, 5)
        acc = accuracy_score(y[mask], yhat_cls[mask])
        diag.append({'dimension': d, 'acc_from_residual_fusion': float(acc)})

    out = os.path.join(args.out_dir, 'residual_model_summary.json')
    with open(out, 'w') as f:
        json.dump({'residual_r2': results, 'fusion_accuracy': diag}, f, indent=2)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()

