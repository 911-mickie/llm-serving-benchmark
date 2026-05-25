#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# ============================================================
# Qwen2.5-7B-Instruct-AWQ SGLang-only Concurrency Matrix
# Use this when SGLang AWQ is already up and you do NOT want
# to restart from vLLM.
# ============================================================

SGLANG_URL="http://127.0.0.1:30000"

TOKENIZER_ID="Qwen/Qwen2.5-7B-Instruct"

RESULT_ROOT="results/qwen-awq-engine-comparison"
MODEL_SLUG="qwen25-7b-instruct-awq"

WARMUP_REQUESTS=4
CONCURRENCIES=(1 2 4)

REST_SECONDS=5
BENCH_IMAGE="llm-serving-bench:local"

log() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

gpu_snapshot() {
  local label="$1"
  mkdir -p "${RESULT_ROOT}/gpu-logs"

  {
    echo
    echo "=============================="
    echo "${label}"
    date
    nvidia-smi
  } >> "${RESULT_ROOT}/gpu-logs/gpu_snapshots.log"
}

wait_for_models() {
  log "Checking SGLang OpenAI API at ${SGLANG_URL}/v1/models"

  for i in $(seq 1 60); do
    if curl -fsS "${SGLANG_URL}/v1/models" >/tmp/models_sglang_awq.json 2>/dev/null; then
      echo "SGLang AWQ is ready."
      cat /tmp/models_sglang_awq.json | jq
      return 0
    fi

    echo "Waiting for SGLang AWQ... attempt ${i}/60"
    sleep 5
  done

  echo "ERROR: SGLang AWQ is not reachable at ${SGLANG_URL}/v1/models"
  docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
  exit 1
}

get_model_id() {
  curl -s "${SGLANG_URL}/v1/models" | jq -r '.data[0].id'
}

requests_for_concurrency() {
  local concurrency="$1"

  case "${concurrency}" in
    1)
      echo 32
      ;;
    2)
      echo 48
      ;;
    4)
      echo 64
      ;;
    8)
      echo 96
      ;;
    *)
      echo 64
      ;;
  esac
}

run_bench() {
  local dataset="$1"
  local workload="$2"
  local concurrency="$3"
  local model_id="$4"

  local num_requests
  num_requests="$(requests_for_concurrency "${concurrency}")"

  local out_dir="${RESULT_ROOT}/sglang/${MODEL_SLUG}/${workload}/c${concurrency}"

  log "Running sglang | ${workload} | concurrency=${concurrency} | requests=${num_requests}"

  gpu_snapshot "BEFORE sglang ${workload} c${concurrency}"

  docker run --rm \
    --network host \
    --user "$(id -u):$(id -g)" \
    -v "$PWD:/workspace" \
    -v "$HOME/.cache/huggingface:/hf-cache" \
    -e HF_HOME=/hf-cache \
    "${BENCH_IMAGE}" \
    --engine sglang \
    --base-url "${SGLANG_URL}" \
    --model "${model_id}" \
    --tokenizer "${TOKENIZER_ID}" \
    --dataset "${dataset}" \
    --concurrency "${concurrency}" \
    --warmup-requests "${WARMUP_REQUESTS}" \
    --num-requests "${num_requests}" \
    --output-dir "${out_dir}"

  gpu_snapshot "AFTER sglang ${workload} c${concurrency}"

  echo "Sleeping ${REST_SECONDS}s before next experiment..."
  sleep "${REST_SECONDS}"
}

aggregate_results() {
  log "Aggregating results"

  docker run --rm \
    --network host \
    --user "$(id -u):$(id -g)" \
    -v "$PWD:/workspace" \
    --entrypoint python \
    "${BENCH_IMAGE}" \
    bench/aggregate_results.py \
    --results-dir "${RESULT_ROOT}" \
    --out-csv "${RESULT_ROOT}/summary.csv" \
    --out-md "${RESULT_ROOT}/summary.md"

  echo
  echo "Summary:"
  cat "${RESULT_ROOT}/summary.md"
}

# ============================================================
# Main
# ============================================================

log "Starting SGLang-only Qwen AWQ benchmark matrix"

mkdir -p "${RESULT_ROOT}"

wait_for_models

MODEL_ID="$(get_model_id)"

if [[ -z "${MODEL_ID}" || "${MODEL_ID}" == "null" ]]; then
  echo "ERROR: Could not read model id from ${SGLANG_URL}/v1/models"
  exit 1
fi

log "SGLang exposed model id: ${MODEL_ID}"

gpu_snapshot "INITIAL SGLANG-ONLY AWQ"

for concurrency in "${CONCURRENCIES[@]}"; do
  run_bench "datasets/chat_trace.jsonl" "chat_trace" "${concurrency}" "${MODEL_ID}"
done

for concurrency in "${CONCURRENCIES[@]}"; do
  run_bench "datasets/prefix_heavy.jsonl" "prefix_heavy" "${concurrency}" "${MODEL_ID}"
done

aggregate_results

log "Done. SGLang AWQ results saved under ${RESULT_ROOT}/sglang/${MODEL_SLUG}"
