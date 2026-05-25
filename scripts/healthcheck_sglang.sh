#!/usr/bin/env bash
# Check SGLang health, models, and smoke completion endpoints; exits non-zero if any fail.
set -euo pipefail

cd "$(dirname "$0")/.."

set -a
source .env.sglang
set +a

PORT="${SGLANG_HOST_PORT:-30000}"
MODEL="${MODEL_NAME:-CohereLabs/c4ai-command-r7b-12-2024}"

echo "Checking SGLang health..."
curl -fsS "http://127.0.0.1:${PORT}/health"
echo
echo "Health endpoint passed."

echo "Checking OpenAI models endpoint..."
curl -fsS "http://127.0.0.1:${PORT}/v1/models"
echo
echo "Models endpoint passed."

echo "Checking basic completion..."
curl -fsS "http://127.0.0.1:${PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"prompt\": \"Explain GPU memory in one sentence.\",
    \"max_tokens\": 32,
    \"temperature\": 0
  }"

echo
echo "SGLang completion check passed."