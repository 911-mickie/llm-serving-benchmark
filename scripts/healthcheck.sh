#!/usr/bin/env bash
# Hit the /v1/models endpoint of a running serving container and print the response.
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8001}"

echo "Checking ${BASE_URL}/v1/models ..."
curl -s "${BASE_URL}/v1/models" | python3 -m json.tool
