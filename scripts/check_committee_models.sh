#!/usr/bin/env bash
# Verify that the committee-facing, unauthenticated /v1/models endpoint works.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${EXACT_ENV_FILE:-$PROJECT_ROOT/.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

PUBLIC_HOST="${EXACT_COMMITTEE_VLLM_HOST:-https://api.iamphuckhang.dev}"
MODELS_URL="${PUBLIC_HOST%/}/v1/models"
PYTHON_BIN="${EXACT_API_PYTHON_BIN:-$PROJECT_ROOT/exact/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

echo "Checking committee model endpoint: $MODELS_URL"
payload="$(curl --fail --silent --show-error --max-time 20 "$MODELS_URL")"

"$PYTHON_BIN" -c '
import json
import sys

payload = json.loads(sys.stdin.read())
assert payload.get("object") == "list", "object must be list"
models = payload.get("data")
assert isinstance(models, list) and models, "data must contain at least one model"
ids = [model.get("id") for model in models if isinstance(model, dict)]
assert len(ids) == len(models) and all(ids), "every model must contain an id"
print("Committee /v1/models reachable. Models: " + ", ".join(ids))
' <<< "$payload"
