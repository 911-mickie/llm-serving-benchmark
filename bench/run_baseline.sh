#!/usr/bin/env bash
# Run a single vLLM benchmark pass against a live serving endpoint and save raw results.
set -euo pipefail

cd "$(dirname "$0")/.."

set -a
source .env
set +a

RUN_ID="$(date +%F-%H%M%S)"
OUT_DIR="results/baseline-${RUN_ID}"
mkdir -p "${OUT_DIR}"

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
    --num-prompts 50 \
    --request-rate 1 \
    --max-concurrency 4 \
    --save-result \
    --result-dir /workspace/results \
    --label baseline \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99"

echo "Saved results to ${OUT_DIR}"
