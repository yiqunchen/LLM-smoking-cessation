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
TEST_LIST="${SPLIT_DIR}/participants_test.txt"
TRAIN_LIST="${SPLIT_DIR}/participants_train.txt"

if [ ! -f "${TEST_LIST}" ] || [ ! -f "${TRAIN_LIST}" ]; then
  echo "ERROR: Participant split files not found. Run run_digital_twin_pipeline_participant.sh first to create participant-based split."
  exit 1
fi

N_TEST=$(wc -l < "${TEST_LIST}")
MODEL_TAG=$(echo "${MODEL}" | sed -E 's#^openai/##; s#^gpt[-_]##; s#[./-]#_#g')
TEST_PRED="${ROOT_DIR}/digital-twin/predictions_part_${MODEL_TAG}_temp_${TEMP}.csv"
if [ ! -f "${TEST_PRED}" ]; then
  echo "ERROR: Test predictions not found: ${TEST_PRED}. Run the participant pipeline first."
  exit 1
fi

OUT_BASE="${ROOT_DIR}/digital-twin/eval_participant_split/sweep"
mkdir -p "${OUT_BASE}"

echo "Fixed test participants with ${N_TEST} IDs. Sweeping train subset sizes relative to test: ${RATIOS[*]}"

for R in "${RATIOS[@]}"; do
  PY_ARG=$(python - << PY
import math
print(math.ceil(${R} * ${N_TEST}))
PY
)
  K=${PY_ARG}
  TAG=$(echo ${R} | sed 's/\./_/g')
  SUB_LIST="${SPLIT_DIR}/participants_train_sub_${TAG}.txt"
  head -n ${K} "${TRAIN_LIST}" > "${SUB_LIST}"
  echo "\n[Subset ${R}:1] Using ${K} train participants -> ${SUB_LIST}"

  # Generate train-only prompts for this subset of participants
  PROM_TRAIN_DIR="${ROOT_DIR}/digital-twin/prompts_train_part_${TAG}"
  if ls "${PROM_TRAIN_DIR}"/*.jsonl >/dev/null 2>&1; then
    echo "  Prompts exist: ${PROM_TRAIN_DIR}"
  else
    python "${ROOT_DIR}/digital-twin/make_prompts_from_participants.py" \
      --data-json "${ROOT_DIR}/data/processed_llm_data.json" \
      --personas-dir "${ROOT_DIR}/digital-twin/personas" \
      --participant-list "${SUB_LIST}" \
      --out-dir "${PROM_TRAIN_DIR}"
  fi

  # Run Twin inference on this train subset
  TRAIN_PRED="${ROOT_DIR}/digital-twin/predictions_part_${MODEL_TAG}_temp_${TEMP}_train_${TAG}.csv"
  python "${ROOT_DIR}/digital-twin/run_twin_inference.py" \
    --model ${MODEL} \
    --prompts-dir "${PROM_TRAIN_DIR}" \
    --concurrency ${CONCURRENCY} \
    --temperature ${TEMP} \
    --split-type part \
    --out "${TRAIN_PRED}" \
    --resume

  # Twin inference on TEST with few-shot from TRAIN participants (to vary Twin with ratio)
  PROM_TEST_FS_DIR="${ROOT_DIR}/digital-twin/prompts_test_part_fewshot_${TAG}"
  python "${ROOT_DIR}/digital-twin/make_test_prompts_with_fewshot_part.py" \
    --data-json "${ROOT_DIR}/data/processed_llm_data.json" \
    --personas-dir "${ROOT_DIR}/digital-twin/personas" \
    --train-participants "${SUB_LIST}" \
    --test-participants "${TEST_LIST}" \
    --out-dir "${PROM_TEST_FS_DIR}" \
    --fewshot-k 3 \
    --seed ${SEED}

  TEST_PRED_FS="${ROOT_DIR}/digital-twin/predictions_part_${MODEL_TAG}_temp_${TEMP}_test_${TAG}.csv"
  python "${ROOT_DIR}/digital-twin/run_twin_inference.py" \
    --model ${MODEL} \
    --prompts-dir "${PROM_TEST_FS_DIR}" \
    --concurrency ${CONCURRENCY} \
    --temperature ${TEMP} \
    --split-type part \
    --out "${TEST_PRED_FS}" \
    --resume

  # Evaluate Twin metrics and Meta-fusion with alpha chosen on TRAIN
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
    --split-type part \
    --out-dir "${SWEEP_DIR}"
done

# Aggregate and plot
python - << 'PY'
import os, re, json
import pandas as pd
import matplotlib.pyplot as plt
base = 'digital-twin/eval_participant_split/sweep'
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
twin = parse_table('digital-twin/eval_participant_split/twin_eval_summary.md')
for tag in tags:
    fusion = parse_table(os.path.join(base, tag, 'meta_fusion_summary.md'))
    twin = parse_table(os.path.join(base, tag, 'twin_eval_summary.md'))
    def tofloat(x):
        try: return float(x)
        except: return None
    twin_overall = pd.Series({k:tofloat(v['acc']) for k,v in twin.items() if k!='design'}).mean()
    fusion_overall = pd.Series({k:tofloat(v['acc']) for k,v in fusion.items() if k!='design'}).mean()
    rows.append({'tag': tag, 'ratio': float(tag.replace('_','.')), 'twin_acc': twin_overall, 'fusion_acc': fusion_overall})
df = pd.DataFrame(rows).sort_values('ratio')
plt.figure(figsize=(7,4))
plt.plot(df['ratio'], df['twin_acc'], '-o', label='Twin only')
plt.plot(df['ratio'], df['fusion_acc'], '-o', label='Fusion (RF+Twin)')
plt.xlabel('Train:Test participant ratio (fixed test set)')
plt.ylabel('Accuracy (avg of content/coping/quitting)')
plt.ylim(0.2, 0.7)
plt.grid(True, alpha=0.3)
plt.legend()
out_png = 'digital-twin/eval_participant_split/sweep/sweep_plot.png'
os.makedirs(os.path.dirname(out_png), exist_ok=True)
plt.tight_layout(); plt.savefig(out_png, dpi=200)
md_lines = ['# Train Subsample Sweep (Participant Split)', '', f'Plot: {out_png}', '', '| Ratio | Twin Acc | Fusion Acc |', '| --- | --- | --- |']
for _,r in df.iterrows():
    md_lines.append(f"| {r['ratio']:.2f} | {r['twin_acc']:.3f} | {r['fusion_acc']:.3f} |")
with open('digital-twin/eval_participant_split/sweep/sweep_summary.md','w') as f:
    f.write('\n'.join(md_lines))
print('Wrote sweep plot and summary under digital-twin/eval_participant_split/sweep')
PY

echo "\nDone. See digital-twin/eval_participant_split/sweep for plots and summary."
