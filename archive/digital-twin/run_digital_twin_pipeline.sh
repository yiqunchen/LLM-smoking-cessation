#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root relative to this script location
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

# Configurable parameters
MODEL=${MODEL:-gpt-4o-mini}
TEMP=${TEMP:-0.5}
TEST_SIZE=${TEST_SIZE:-0.5}
SEED=${SEED:-42}
CONCURRENCY=${CONCURRENCY:-8}
EVAL_DIR="${ROOT_DIR}/digital-twin/eval_message_split"

echo "[1/5] Building personas (skip if exists)"
if [ -f "${ROOT_DIR}/digital-twin/personas/personas.json" ]; then
  echo "  personas/personas.json found; skipping build_personas.py"
else
  python "${ROOT_DIR}/digital-twin/build_personas.py" \
    --input "${ROOT_DIR}/data/processed_llm_data.json" \
    --out-dir "${ROOT_DIR}/digital-twin/personas"
fi

echo "[2/5] Creating message-based split and test-only prompts (skip if present)"
PROMPTS_DIR="${ROOT_DIR}/digital-twin/prompts_test"
if ls "${PROMPTS_DIR}"/*.jsonl >/dev/null 2>&1; then
  echo "  prompts_test/*.jsonl found; skipping make_prompts_split.py"
else
  python "${ROOT_DIR}/digital-twin/make_prompts_split.py" \
    --data-json "${ROOT_DIR}/data/processed_llm_data.json" \
    --personas-dir "${ROOT_DIR}/digital-twin/personas" \
    --out-dir "${PROMPTS_DIR}" \
    --splits-dir "${ROOT_DIR}/digital-twin/splits" \
    --test-size ${TEST_SIZE} \
    --seed ${SEED}
fi

# Construct output path to evaluate
MODEL_TAG=$(echo "${MODEL}" | sed -E 's#^openai/##; s#^gpt[-_]##; s#[./-]#_#g')
OUT_CSV="${ROOT_DIR}/digital-twin/predictions_msg_${MODEL_TAG}_temp_${TEMP}.csv"

echo "[3/5] Running async twin inference on TEST prompts (model=${MODEL}, temp=${TEMP}, concurrency=${CONCURRENCY})"
# Determine total prompts to process
TOTAL_PROMPTS=$(wc -l "${PROMPTS_DIR}"/*.jsonl | tail -1 | awk '{print $1}')
if [ -f "${OUT_CSV}" ]; then
  PRED_ROWS=$(wc -l < "${OUT_CSV}")
  # subtract header
  if [ "${PRED_ROWS}" -gt 0 ]; then PRED_ROWS=$((PRED_ROWS-1)); fi
else
  PRED_ROWS=0
fi
echo "  Prompts: ${TOTAL_PROMPTS} | Existing predictions: ${PRED_ROWS}"
if [ "${PRED_ROWS}" -ge "${TOTAL_PROMPTS}" ]; then
  echo "  All prompts already inferred; skipping inference."
else
  python "${ROOT_DIR}/digital-twin/run_twin_inference.py" \
    --model ${MODEL} \
    --prompts-dir "${PROMPTS_DIR}" \
    --concurrency ${CONCURRENCY} \
    --temperature ${TEMP} \
    --split-type msg \
    --out "${OUT_CSV}" \
    --resume
fi

echo "[4/5] Evaluating twins against ground truth (message split)"
python "${ROOT_DIR}/digital-twin/evaluate_twins.py" \
  --pred-csv "${OUT_CSV}" \
  --data-json "${ROOT_DIR}/data/processed_llm_data.json" \
  --out-dir "${EVAL_DIR}"

echo "[5/5] Residual fusion diagnostic (baseline + embedding residuals; message split)"
python "${ROOT_DIR}/digital-twin/residual_model.py" \
  --data-json "${ROOT_DIR}/data/processed_llm_data.json" \
  --emb-csv "${ROOT_DIR}/data/message_embeddings.csv" \
  --out-dir "${EVAL_DIR}" \
  --pca 32

echo "\nDone. Key artifacts:"
echo "- Predictions: ${OUT_CSV}"
echo "- Evaluation: ${EVAL_DIR}/twin_eval_summary.md"
echo "- Residual summary: ${EVAL_DIR}/residual_model_summary.json"
