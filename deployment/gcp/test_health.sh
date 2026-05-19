#!/usr/bin/env bash
# Usage: BASE_URL=https://your-service-xxx.run.app ./test_health.sh
set -euo pipefail
BASE_URL="${BASE_URL:-http://127.0.0.1:8080}"
curl -sS -f "${BASE_URL}/health" | python3 -m json.tool
echo "OK: ${BASE_URL}/health"
