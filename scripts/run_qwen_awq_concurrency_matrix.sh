#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# ============================================================
# Qwen2.5-7B-Instruct-AWQ vLLM vs SGLang Concurrency Matrix
# ============================================================

# ------------------------------
# Config
# ------------------------------

VLLM_ENV=".env.qwen-awq.vllm"
SGLANG_ENV=".env.qwen-awq.sglang"

VLLM_COMPOSE="docker-compose.yml"
SGLANG_COMPOSE="docker-compose.sglang.yml"

VLLM_CONTAINER="vllm-qwen25-7b-instruct-awq"
SGLANG_CONTAINER="sglang-qwen25-7b-instruct-awq"

VLLM_URL="http://127.0.0.1:8001"
SGLANG_URL="http://127.0.0.1:30000"

TOKENIZER_ID="Qwen/Qwen2.5-7B-Instruct"

RESULT_ROOT="results/qwen-awq-engine-comparison"
MODEL_SLUG="qwen25-7b-instruct-awq"

WARMUP_REQUESTS=4

# Keep this aligned with your server config.
# If MAX_NUM_SEQS / SGLANG_MAX_RUNNING_REQUESTS is 4,
# then 1,2,4 is the clean concurrency sweep.
CONCURRENCIES=(1 2 4)

# Optional later:
# CONCURRENCIES=(1 2 4 8)

REST_SECONDS=5

BENCH_IMAGE="llm-serving-bench:local"

# ------------------------------
# Helpers
# ------------------------------

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

stop_all_engines() {
  log "Stopping any existing vLLM/SGLang containers"

  docker compose --env-file "${VLLM_ENV}" -f "${VLLM_COMPOSE}" down || true
  docker compose --env-file "${SGLANG_ENV}" -f "${SGLANG_COMPOSE}" down || true

  # Stop known old containers too, just in case.
  docker stop vllm-command-r7b 2>/dev/null || true
  docker stop sglang-command-r7b 2>/dev/null || true
  docker stop vllm-qwen25-7b-instruct 2>/dev/null || true
  docker stop sglang-qwen25-7b-instruct 2>/dev/null || true
  docker stop "${VLLM_CONTAINER}" 2>/dev/null || true
  docker stop "${SGLANG_CONTAINER}" 2>/dev/null || true

  sleep "${REST_SECONDS}"
}

wait_for_models() {
  local base_url="$1"
  local engine_name="$2"

  log "Waiting for ${engine_name} OpenAI API at ${base_url}/v1/models"

  for i in $(seq 1 180); do
    if curl -fsS "${base_url}/v1/models" >/tmp/models_${engine_name}.json 2>/dev/null; then
      echo "${engine_name} is ready."
      cat /tmp/models_${engine_name}.json | jq
      return 0
    fi

    echo "Waiting for ${engine_name}... attempt ${i}/180"
    sleep 5
  done

  echo "ERROR: ${engine_name} did not become ready."
  docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

  echo
  echo "Last logs from ${engine_name}:"
  if [[ "${engine_name}" == "vllm" ]]; then
    docker logs --tail 100 "${VLLM_CONTAINER}" || true
  else
    docker logs --tail 100 "${SGLANG_CONTAINER}" || true
  fi

  exit 1
}

get_model_id() {
  local base_url="$1"
  curl -s "${base_url}/v1/models" | jq -r '.data[0].id'
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
  local engine="$1"
  local base_url="$2"
  local model_id="$3"
  local dataset="$4"
  local workload="$5"
  local concurrency="$6"

  local num_requests
  num_requests="$(requests_for_concurrency "${concurrency}")"

  local out_dir="${RESULT_ROOT}/${engine}/${MODEL_SLUG}/${workload}/c${concurrency}"

  log "Running ${engine} | ${workload} | concurrency=${concurrency} | requests=${num_requests}"

  gpu_snapshot "BEFORE ${engine} ${workload} c${concurrency}"

  docker run --rm \
    --network host \
    --user "$(id -u):$(id -g)" \
    -v "$PWD:/workspace" \
    -v "$HOME/.cache/huggingface:/hf-cache" \
    -e HF_HOME=/hf-cache \
    "${BENCH_IMAGE}" \
    --engine "${engine}" \
    --base-url "${base_url}" \
    --model "${model_id}" \
    --tokenizer "${TOKENIZER_ID}" \
    --dataset "${dataset}" \
    --concurrency "${concurrency}" \
    --warmup-requests "${WARMUP_REQUESTS}" \
    --num-requests "${num_requests}" \
    --output-dir "${out_dir}"

  gpu_snapshot "AFTER ${engine} ${workload} c${concurrency}"

  echo "Sleeping ${REST_SECONDS}s before next experiment..."
  sleep "${REST_SECONDS}"
}

run_engine_matrix() {
  local engine="$1"
  local base_url="$2"

  local model_id
  model_id="$(get_model_id "${base_url}")"

  if [[ -z "${model_id}" || "${model_id}" == "null" ]]; then
    echo "ERROR: Could not read model id from ${base_url}/v1/models"
    exit 1
  fi

  log "${engine} exposed model id: ${model_id}"

  for concurrency in "${CONCURRENCIES[@]}"; do
    run_bench "${engine}" "${base_url}" "${model_id}" \
      "datasets/chat_trace.jsonl" \
      "chat_trace" \
      "${concurrency}"
  done

  for concurrency in "${CONCURRENCIES[@]}"; do
    run_bench "${engine}" "${base_url}" "${model_id}" \
      "datasets/prefix_heavy.jsonl" \
      "prefix_heavy" \
      "${concurrency}"
  done
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

# ------------------------------
# Main
# ------------------------------

log "Starting Qwen AWQ engine comparison matrix"

mkdir -p "${RESULT_ROOT}"

gpu_snapshot "INITIAL"

stop_all_engines

log "Starting vLLM Qwen AWQ"
docker compose --env-file "${VLLM_ENV}" -f "${VLLM_COMPOSE}" up -d
wait_for_models "${VLLM_URL}" "vllm"
run_engine_matrix "vllm" "${VLLM_URL}"

log "Stopping vLLM before starting SGLang"
docker compose --env-file "${VLLM_ENV}" -f "${VLLM_COMPOSE}" down
sleep "${REST_SECONDS}"
gpu_snapshot "AFTER STOPPING VLLM"

log "Starting SGLang Qwen AWQ"
docker compose --env-file "${SGLANG_ENV}" -f "${SGLANG_COMPOSE}" up -d
wait_for_models "${SGLANG_URL}" "sglang"
run_engine_matrix "sglang" "${SGLANG_URL}"

log "Stopping SGLang"
docker compose --env-file "${SGLANG_ENV}" -f "${SGLANG_COMPOSE}" down
sleep "${REST_SECONDS}"
gpu_snapshot "AFTER STOPPING SGLANG"

aggregate_results

log "Done. Results saved to ${RESULT_ROOT}"