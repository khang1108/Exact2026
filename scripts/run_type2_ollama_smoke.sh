#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_CONFIG="$ROOT_DIR/configs/type2_dataset_run.example.toml"

if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  CONFIG_PATH="$DEFAULT_CONFIG"
  OUTPUT_PATH="/tmp/type2_ollama_smoke.json"
  LIMIT="${1:-1}"
  OFFSET="${2:-0}"
else
  CONFIG_PATH="${1:-$DEFAULT_CONFIG}"
  OUTPUT_PATH="${2:-/tmp/type2_ollama_smoke.json}"
  LIMIT="${3:-1}"
  OFFSET="${4:-0}"
fi

PYTHONPATH="$ROOT_DIR/src" "$ROOT_DIR/venv/bin/python" -m exact.scripts.run_type2_monitor \
  --config "$CONFIG_PATH" \
  --limit "$LIMIT" \
  --offset "$OFFSET" \
  --output "$OUTPUT_PATH"
