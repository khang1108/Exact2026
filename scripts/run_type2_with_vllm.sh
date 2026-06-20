#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG_PATH="${1:-$PROJECT_ROOT/configs/type2_dataset_run.example.toml}"
LIMIT="${LIMIT:-10}"
OFFSET="${OFFSET:-0}"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"
BASE_URL="http://${HOST}:${PORT}/v1"
API_KEY="${VLLM_API_KEY:-EMPTY}"
MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
OUTPUT_PATH="${OUTPUT_PATH:-$PROJECT_ROOT/artifacts/predictions/type2/type2_vllm_run.json}"
LOG_PATH="${VLLM_LOG_PATH:-$PROJECT_ROOT/artifacts/logs/type2_vllm.log}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-180}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/venv/bin/python}"
VLLM_BIN="${VLLM_BIN:-}"
STARTED_VLLM=0
VLLM_PID=""

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

curl_ready() {
  curl -fsS -H "Authorization: Bearer $API_KEY" "$BASE_URL/models" >/dev/null 2>&1
}

cleanup() {
  local exit_code="${1:-$?}"
  trap - EXIT INT TERM
  if [[ "$STARTED_VLLM" == "1" && -n "$VLLM_PID" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
  exit "$exit_code"
}

trap 'cleanup $?' EXIT
trap 'cleanup 130' INT
trap 'cleanup 143' TERM

resolve_python() {
  if [[ -x "$PYTHON_BIN" ]]; then
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
    return 0
  fi
  log_error "Python interpreter not found. Set PYTHON_BIN=/path/to/python"
  exit 1
}

resolve_vllm_bin() {
  if [[ -n "$VLLM_BIN" && -x "$VLLM_BIN" ]]; then
    return 0
  fi
  if [[ -x "$PROJECT_ROOT/venv/bin/vllm" ]]; then
    VLLM_BIN="$PROJECT_ROOT/venv/bin/vllm"
    return 0
  fi
  if command -v vllm >/dev/null 2>&1; then
    VLLM_BIN="$(command -v vllm)"
    return 0
  fi
  log_error "vllm binary not found. Set VLLM_BIN=/path/to/vllm"
  exit 1
}

ensure_dirs() {
  mkdir -p "$(dirname "$OUTPUT_PATH")" "$(dirname "$LOG_PATH")"
}

load_model_from_config_if_present() {
  local configured_model
  configured_model="$($PYTHON_BIN -X utf8 - <<'PY' "$CONFIG_PATH"
import sys, tomllib
from pathlib import Path

path = Path(sys.argv[1])
try:
    with path.open('rb') as fh:
        config = tomllib.load(fh)
except FileNotFoundError:
    print("")
    raise SystemExit(0)
print(str(config.get('llm', {}).get('model') or ''))
PY
)"
  if [[ -n "$configured_model" && -z "${VLLM_MODEL:-}" ]]; then
    MODEL="$configured_model"
  fi
}

export_cuda_env() {
  export TRITON_LIBCUDA_PATH="${TRITON_LIBCUDA_PATH:-/usr/lib/x86_64-linux-gnu}"
  export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
}

preflight_checks() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log_error "nvidia-smi not found. This script requires an NVIDIA GPU host for vLLM."
    exit 1
  fi
  if ! nvidia-smi >/dev/null 2>&1; then
    log_error "nvidia-smi failed. GPU runtime is not healthy."
    exit 1
  fi
  if [[ ! -e "/usr/lib/x86_64-linux-gnu/libcuda.so.1" && ! -e "/lib/x86_64-linux-gnu/libcuda.so.1" ]]; then
    log_error "libcuda.so.1 not found in standard library paths."
    exit 1
  fi
  if ! command -v curl >/dev/null 2>&1; then
    log_error "curl is required. Please install curl."
    exit 1
  fi
}

ensure_model() {
  log_info "Ensuring model is cached: $MODEL"
  PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" "$PROJECT_ROOT/src/exact/scripts/pull_model.py" \
    --config "$CONFIG_PATH" \
    --model "$MODEL" \
    --force
}

wait_for_vllm() {
  local elapsed=0
  log_info "Waiting for vLLM at $BASE_URL/models"
  while true; do
    if curl_ready; then
      log_info "vLLM is ready"
      return 0
    fi
    if [[ -n "$VLLM_PID" ]] && ! kill -0 "$VLLM_PID" 2>/dev/null; then
      log_error "vLLM exited before becoming ready"
      tail -n 120 "$LOG_PATH" || true
      return 1
    fi
    if [[ "$elapsed" -ge "$WAIT_TIMEOUT" ]]; then
      log_error "Timed out waiting for vLLM after ${WAIT_TIMEOUT}s"
      tail -n 120 "$LOG_PATH" || true
      return 1
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
}

start_vllm() {
  if curl_ready; then
    log_warn "Reusing existing vLLM at $BASE_URL"
    return 0
  fi

  : >"$LOG_PATH"
  log_info "Starting vLLM"
  log_info "Model: $MODEL"
  log_info "Base URL: $BASE_URL"
  log_info "Log: $LOG_PATH"

  local -a cmd=(
    "$VLLM_BIN" serve "$MODEL"
    --host "$HOST"
    --port "$PORT"
    --api-key "$API_KEY"
    --served-model-name "$MODEL"
    --dtype auto
    --max-model-len 4096
    --gpu-memory-utilization 0.90
    --generation-config vllm
    --enable-prefix-caching
    --enforce-eager
  )

  printf '%s\n' "${cmd[*]}" >>"$LOG_PATH"
  "${cmd[@]}" >>"$LOG_PATH" 2>&1 &
  VLLM_PID="$!"
  STARTED_VLLM=1
  wait_for_vllm
}

run_type2() {
  log_info "Running Type 2 pipeline"
  PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -m exact.scripts.run_type2_vllm \
    --config "$CONFIG_PATH" \
    --limit "$LIMIT" \
    --offset "$OFFSET" \
    --output "$OUTPUT_PATH"
}

resolve_python
resolve_vllm_bin
ensure_dirs
load_model_from_config_if_present
export_cuda_env
preflight_checks

log_info "Project root: $PROJECT_ROOT"
log_info "Config:       $CONFIG_PATH"
log_info "Model:        $MODEL"
log_info "Offset:       $OFFSET"
log_info "Limit:        $LIMIT"
log_info "Output:       $OUTPUT_PATH"
log_info "Python:       $PYTHON_BIN"
log_info "vLLM bin:     $VLLM_BIN"

ensure_model
start_vllm
run_type2

log_info "Finished. Predictions written to $OUTPUT_PATH"
