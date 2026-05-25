#!/usr/bin/env bash
# Sweep concurrency levels 1/2/4 against a live vLLM endpoint and save raw results per level.
set -euo pipefail

cd "$(dirname "$0")/.."

set -a
source .env
set +a

RUN_ID="$(date +%F-%H%M%S)"
BASE_OUT_DIR="results/concurrency-sweep-${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

for CONC in 1 2 4; do
  OUT_DIR="${BASE_OUT_DIR}/conc_${CONC}"
  mkdir -p "${OUT_DIR}"

  echo "========================================="
  echo "Running concurrency = ${CONC}"
  echo "========================================="

  docker run --rm \
    --network host \
    --entrypoint bash \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e HF_HUB_DISABLE_XET=1 \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -v "$(pwd)/${OUT_DIR}:/workspace/results" \
    "${VLLM_IMAGE:-vllm/vllm-openai:v0.8.5}" \
    -lc "vllm bench serve \
      --endpoint-type openai-comp \
      --base-url http://127.0.0.1:${HOST_PORT:-8001} \
      --endpoint /v1/completions \
      --model ${MODEL_NAME:-CohereLabs/c4ai-command-r7b-12-2024} \
      --dataset-name random \
      --random-input-len 512 \
      --random-output-len 128 \
      --num-prompts 100 \
      --request-rate 1 \
      --max-concurrency ${CONC} \
      --save-result \
      --result-dir /workspace/results \
      --label conc_${CONC} \
      --percentile-metrics ttft,tpot,itl,e2el \
      --metric-percentiles 50,95,99"

done

echo "Saved results to ${BASE_OUT_DIR}"