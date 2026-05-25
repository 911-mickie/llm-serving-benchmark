#!/usr/bin/env bash
# Run a short smoke benchmark (10 requests) against a live SGLang endpoint to verify it is working.
set -euo pipefail

cd "$(dirname "$0")/.."

set -a
source .env.sglang
set +a

RUN_ID="$(date +%F-%H%M%S)"
OUT_DIR="results/sglang-smoke-${RUN_ID}"
mkdir -p "${OUT_DIR}"

docker run --rm \
  --network host \
  --entrypoint bash \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e HF_HUB_DISABLE_XET=1 \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "$(pwd)/${OUT_DIR}:/workspace/results" \
  "vllm/vllm-openai:v0.8.5" \
  -lc "vllm bench serve \
    --endpoint-type openai-comp \
    --base-url http://127.0.0.1:${SGLANG_HOST_PORT:-30000} \
    --endpoint /v1/completions \
    --model ${MODEL_NAME:-CohereLabs/c4ai-command-r7b-12-2024} \
    --dataset-name random \
    --random-input-len 128 \
    --random-output-len 32 \
    --num-prompts 10 \
    --request-rate 1 \
    --max-concurrency 1 \
    --save-result \
    --result-dir /workspace/results \
    --label sglang_smoke \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99"

echo "Saved results to ${OUT_DIR}"
