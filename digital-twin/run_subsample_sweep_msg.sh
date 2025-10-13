#!/usr/bin/env bash
set -euo pipefail

# Config
MODEL=${MODEL:-gpt-4o-mini}
TEMP=${TEMP:-0.5}
CONCURRENCY=${CONCURRENCY:-8}
SEED=${SEED:-42}
RATIOS=(0.75 0.5 0.25 0.1 0.05)

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
SPLIT_DIR="${ROOT_DIR}/digital-twin/splits"
TEST_LIST="${SPLIT_DIR}/messages_test.txt"
TRAIN_LIST="${SPLIT_DIR}/messages_train.txt"

if [ ! -f "${TEST_LIST}" ] || [ ! -f "${TRAIN_LIST}" ]; then
  echo "ERROR: Split files not found. Run run_digital_twin_pipeline.sh first to create message-based split."
  exit 1
fi

N_TEST=$(wc -l < "${TEST_LIST}")
MODEL_TAG=$(echo "${MODEL}" | sed -E 's#^openai/##; s#^gpt[-_]##; s#[./-]#_#g')
TEST_PRED="${ROOT_DIR}/digital-twin/predictions_msg_${MODEL_TAG}_temp_${TEMP}.csv"
if [ ! -f "${TEST_PRED}" ]; then
  echo "ERROR: Test predictions not found: ${TEST_PRED}. Run the main pipeline first."
  exit 1
fi

OUT_BASE="${ROOT_DIR}/digital-twin/eval_message_split/sweep"
mkdir -p "${OUT_BASE}"

echo "Fixed test set with ${N_TEST} messages. Sweeping train subset sizes relative to test: ${RATIOS[*]}"

for R in "${RATIOS[@]}"; do
  # Compute subset size: ceil(R * N_TEST)
  PY_ARG=$(python - << PY
import math
print(math.ceil(${R} * ${N_TEST}))
PY
)
  K=${PY_ARG}
  TAG=$(echo ${R} | sed 's/\./_/g')
  SUB_LIST="${SPLIT_DIR}/messages_train_sub_${TAG}.txt"
  head -n ${K} "${TRAIN_LIST}" > "${SUB_LIST}"
  echo "\n[Subset ${R}:1] Using ${K} train messages -> ${SUB_LIST}"

  # Generate train-only prompts for this subset
  PROM_TRAIN_DIR="${ROOT_DIR}/digital-twin/prompts_train_sub_${TAG}"
  if ls "${PROM_TRAIN_DIR}"/*.jsonl >/dev/null 2>&1; then
    echo "  Prompts exist: ${PROM_TRAIN_DIR}"
  else
    python "${ROOT_DIR}/digital-twin/make_prompts_from_list.py" \
      --data-json "${ROOT_DIR}/data/processed_llm_data.json" \
      --personas-dir "${ROOT_DIR}/digital-twin/personas" \
      --message-list "${SUB_LIST}" \
      --out-dir "${PROM_TRAIN_DIR}"
  fi

  # Run Twin inference on this train subset
  TRAIN_PRED="${ROOT_DIR}/digital-twin/predictions_msg_${MODEL_TAG}_temp_${TEMP}_train_${TAG}.csv"
  python "${ROOT_DIR}/digital-twin/run_twin_inference.py" \
    --model ${MODEL} \
    --prompts-dir "${PROM_TRAIN_DIR}" \
    --concurrency ${CONCURRENCY} \
    --temperature ${TEMP} \
    --split-type msg \
    --out "${TRAIN_PRED}" \
    --resume

  # Twin inference on TEST with few-shot from TRAIN subset (to vary Twin with ratio)
  PROM_TEST_FS_DIR="${ROOT_DIR}/digital-twin/prompts_test_fewshot_${TAG}"
  python "${ROOT_DIR}/digital-twin/make_test_prompts_with_fewshot_msg.py" \
    --data-json "${ROOT_DIR}/data/processed_llm_data.json" \
    --personas-dir "${ROOT_DIR}/digital-twin/personas" \
    --train-messages "${SUB_LIST}" \
    --test-messages "${TEST_LIST}" \
    --out-dir "${PROM_TEST_FS_DIR}"

  TEST_PRED_FS="${ROOT_DIR}/digital-twin/predictions_msg_${MODEL_TAG}_temp_${TEMP}_test_${TAG}.csv"
  python "${ROOT_DIR}/digital-twin/run_twin_inference.py" \
    --model ${MODEL} \
    --prompts-dir "${PROM_TEST_FS_DIR}" \
    --concurrency ${CONCURRENCY} \
    --temperature ${TEMP} \
    --split-type msg \
    --out "${TEST_PRED_FS}" \
    --resume

  # Evaluate Twin metrics per ratio and Meta-fusion with alpha chosen on TRAIN
  SWEEP_DIR="${OUT_BASE}/${TAG}"
  mkdir -p "${SWEEP_DIR}"
  python "${ROOT_DIR}/digital-twin/evaluate_twins.py" \
    --pred-csv "${TEST_PRED_FS}" \
    --data-json "${ROOT_DIR}/data/processed_llm_data.json" \
    --out-dir "${SWEEP_DIR}"
  python "${ROOT_DIR}/digital-twin/meta_fusion_supervised.py" \
    --pred-csv "${TEST_PRED_FS}" \
    --train-pred-csv "${TRAIN_PRED}" \
    --data-json "${ROOT_DIR}/data/processed_llm_data.json" \
    --split-type msg \
    --out-dir "${SWEEP_DIR}"
done

# Aggregate and plot
python - << 'PY'
import os, re, json
import pandas as pd
import matplotlib.pyplot as plt
base = 'digital-twin/eval_message_split/sweep'
tags = sorted([d for d in os.listdir(base) if re.match(r'^\d+_\d+$', d)])
rows = []
def parse_table(md):
    out = {}
    if not os.path.exists(md):
        return out
    for line in open(md):
        line=line.strip()
        m=re.match(r'^\|\s*(content|design|coping|quitting)\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|',line,re.I)
        if m:
            out[m.group(1).lower()]={'acc':m.group(2).strip(),'kappa':m.group(3).strip(),'rho':m.group(4).strip()}
    return out
for tag in tags:
    fusion = parse_table(os.path.join(base, tag, 'meta_fusion_summary.md'))
    twin = parse_table(os.path.join(base, tag, 'twin_eval_summary.md'))
    # record overall accuracy across dimensions (skip design)
    def tofloat(x):
        try: return float(x)
        except: return None
    twin_acc = pd.Series({k:tofloat(v['acc']) for k,v in twin.items() if k!='design'}).mean()
    twin_kap = pd.Series({k:tofloat(v['kappa']) for k,v in twin.items() if k!='design'}).mean()
    twin_rho = pd.Series({k:tofloat(v['rho']) for k,v in twin.items() if k!='design'}).mean()
    fus_acc = pd.Series({k:tofloat(v['acc']) for k,v in fusion.items() if k!='design'}).mean()
    fus_kap = pd.Series({k:tofloat(v['kappa']) for k,v in fusion.items() if k!='design'}).mean()
    fus_rho = pd.Series({k:tofloat(v['rho']) for k,v in fusion.items() if k!='design'}).mean()
    rows.append({'tag': tag, 'ratio': float(tag.replace('_','.')),
                 'twin_acc': twin_acc, 'fusion_acc': fus_acc,
                 'twin_kappa': twin_kap, 'fusion_kappa': fus_kap,
                 'twin_rho': twin_rho, 'fusion_rho': fus_rho})
df = pd.DataFrame(rows).sort_values('ratio')
# Plots
os.makedirs('digital-twin/eval_message_split/sweep', exist_ok=True)
def plot_metric(col_twin, col_fus, ylabel, fname, ylim=(0.2,0.7)):
    plt.figure(figsize=(7,4))
    plt.plot(df['ratio'], df[col_twin], '-o', label='Twin only')
    plt.plot(df['ratio'], df[col_fus], '-o', label='Fusion (RF+Twin)')
    plt.xlabel('Train:Test message ratio (fixed test set)')
    plt.ylabel(ylabel)
    if ylim: plt.ylim(*ylim)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_png = f'digital-twin/eval_message_split/sweep/{fname}'
    plt.tight_layout(); plt.savefig(out_png, dpi=200)
    return out_png

acc_png = plot_metric('twin_acc','fusion_acc','Accuracy (avg content/coping/quitting)','sweep_plot_acc.png',(0.2,0.7))
kappa_png = plot_metric('twin_kappa','fusion_kappa','Cohen kappa (avg)','sweep_plot_kappa.png',None)
rho_png = plot_metric('twin_rho','fusion_rho','Spearman rho (avg)','sweep_plot_rho.png',None)

# Markdown summary
md_lines = ['# Train Subsample Sweep (Message Split)', '', f'Plots:', f'- {acc_png}', f'- {kappa_png}', f'- {rho_png}',
            '', '| Ratio | Twin Acc | Fusion Acc | Twin κ | Fusion κ | Twin ρ | Fusion ρ |',
            '| --- | --- | --- | --- | --- | --- | --- |']
for _,r in df.iterrows():
    md_lines.append(f"| {r['ratio']:.2f} | {r['twin_acc']:.3f} | {r['fusion_acc']:.3f} | {r['twin_kappa']:.3f} | {r['fusion_kappa']:.3f} | {r['twin_rho']:.3f} | {r['fusion_rho']:.3f} |")
with open('digital-twin/eval_message_split/sweep/sweep_summary.md','w') as f:
    f.write('\n'.join(md_lines))
print('Wrote sweep plot and summary under digital-twin/eval_message_split/sweep')
PY

echo "\nDone. See digital-twin/eval_message_split/sweep for plots and summary."
