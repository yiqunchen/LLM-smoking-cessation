#!/usr/bin/env python3
import os
import re
import argparse


def parse_table(md_path):
    metrics = {}
    if not os.path.exists(md_path):
        return metrics
    with open(md_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Match rows like: | content | 0.412 | 0.123 | 0.085 |
            m = re.match(r"^\|\s*(content|design|coping|quitting)\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|", line, re.I)
            if m:
                dim = m.group(1).lower()
                acc = m.group(2).strip()
                kap = m.group(3).strip()
                rho = m.group(4).strip()
                # coerce NA to None, numeric strings to float strings kept for printing
                metrics[dim] = {'acc': acc, 'kappa': kap, 'rho': rho}
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-dir', required=True, help='Directory containing twin_eval_summary.md and meta_fusion_summary.md')
    ap.add_argument('--out', default=None, help='Optional output markdown path (default: fused_vs_twin.md in eval-dir)')
    args = ap.parse_args()

    twin_md = os.path.join(args.eval_dir, 'twin_eval_summary.md')
    fusion_md = os.path.join(args.eval_dir, 'meta_fusion_summary.md')

    twin = parse_table(twin_md)
    fusion = parse_table(fusion_md)

    dims = ['content','design','coping','quitting']
    lines = []
    lines.append('# Fusion vs. Twin (Direct 5-Class)')
    lines.append(f'Eval dir: {os.path.abspath(args.eval_dir)}')
    lines.append('')
    lines.append('| Dimension | Twin Acc | Fusion Acc | ΔAcc | Twin κ | Fusion κ | Twin ρ | Fusion ρ |')
    lines.append('| --- | ---:| ---:| ---:| ---:| ---:| ---:| ---:|')
    for d in dims:
        t = twin.get(d, {})
        f = fusion.get(d, {})
        t_acc = t.get('acc', 'NA'); f_acc = f.get('acc', 'NA')
        # compute delta if numeric
        try:
            delta = float(f_acc) - float(t_acc)
            delta_str = f"{delta:.3f}"
        except Exception:
            delta_str = 'NA'
        row = f"| {d} | {t_acc} | {f_acc} | {delta_str} | {t.get('kappa','NA')} | {f.get('kappa','NA')} | {t.get('rho','NA')} | {f.get('rho','NA')} |"
        lines.append(row)

    out_path = args.out or os.path.join(args.eval_dir, 'fused_vs_twin.md')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()

