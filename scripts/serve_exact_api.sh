#!/usr/bin/env bash
# Start the full EXACT stack: Type 1 parser vLLM → main vLLM → FastAPI.
#
# Parser launch method is chosen automatically:
#   device=cpu  → Docker (vllm/vllm-openai-cpu image)
#   device=auto/cuda → native vllm binary
#
# Main vLLM is skipped (warning only) if the binary is not found, since it
# may already be running on a GPU host or a separate terminal.
#
# Ctrl+C stops all processes started by this script cleanly.
#
# One-off overrides:
#   PARSER_DEVICE=cpu bash scripts/serve_exact_api.sh
#   VLLM_SKIP=1 bash scripts/serve_exact_api.sh   # skip main vLLM entirely

set -uo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${EXACT_ENV_FILE:-$PROJECT_ROOT/.env}"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  echo "[INFO]  Loaded configuration from $ENV_FILE"
else
  echo "[WARN]  No .env found at $ENV_FILE — using shell environment only"
fi

# ---------------------------------------------------------------------------
# Colors / logging
# ---------------------------------------------------------------------------

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

connect_host_for_bind() {
  local host="$1"
  [[ "$host" == "0.0.0.0" || "$host" == "::" ]] && host="127.0.0.1"
  printf '%s' "$host"
}

wait_for_tcp() {
  local name="$1" host="$2" port="$3" timeout="${4:-120}" pid="${5:-}" log_file="${6:-}" elapsed=0
  log_info "Waiting for $name at $host:$port (timeout ${timeout}s)..."
  while true; do
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      local process_status=0
      wait "$pid" 2>/dev/null || process_status=$?
      log_error "$name exited before becoming ready (status $process_status)."
      if [[ -n "$log_file" && -f "$log_file" ]]; then
        log_error "Last lines from $log_file:"
        tail -n 40 "$log_file" >&2
      fi
      return 1
    fi

    if "$PYTHON_BIN" -c \
      'import socket, sys; socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=1).close()' \
      "$host" "$port" 2>/dev/null; then
      echo ""
      log_info "$name ready after ${elapsed}s."
      return 0
    fi

    if [[ $elapsed -ge $timeout ]]; then
      log_error "$name not ready after ${timeout}s — check $LOG_DIR/"
      if [[ -n "$log_file" && -f "$log_file" ]]; then
        log_error "Last lines from $log_file:"
        tail -n 40 "$log_file" >&2
      fi
      return 1
    fi
    sleep 3; elapsed=$((elapsed + 3)); echo -n "."
  done
}

check_models_endpoint() {
  local url="$1"
  if ! "$PYTHON_BIN" -c '
import json
import sys
from urllib.request import urlopen

with urlopen(sys.argv[1], timeout=10) as response:
    payload = json.load(response)
assert payload.get("object") == "list"
assert isinstance(payload.get("data"), list) and payload["data"]
assert all(isinstance(model, dict) and model.get("id") for model in payload["data"])
' "$url"; then
    log_error "Committee model endpoint is not ready: $url"
    return 1
  fi
  log_info "Committee model endpoint ready: $url"
}

check_vllm_runtime() {
  local vllm_bin="$1" resolved_bin runtime_python
  resolved_bin="$(command -v "$vllm_bin" 2>/dev/null || true)"
  [[ -z "$resolved_bin" ]] && return 0

  runtime_python="$(dirname "$resolved_bin")/python"
  [[ ! -x "$runtime_python" ]] && return 0

  if ! PYTHONNOUSERSITE=1 "$runtime_python" - <<'PY'
import importlib.metadata
import importlib.util

import torch

vllm_version = importlib.metadata.version("vllm")
if vllm_version == "0.8.5" and not torch.__version__.startswith("2.6.0"):
    raise RuntimeError(
        f"vLLM 0.8.5 requires Torch 2.6.0, but found Torch {torch.__version__}"
    )

if importlib.util.find_spec("flashinfer") is not None:
    import flashinfer.sampling
PY
  then
    log_error "The vLLM Python environment has an incompatible Torch/FlashInfer installation."
    log_error "Repair it with: bash scripts/setup_vllm_env.sh"
    return 1
  fi
}

PIDS=()
SHUTDOWN_REASON="launcher exit"
cleanup() {
  local exit_code="${1:-$?}" index pid
  trap - INT TERM EXIT
  echo ""
  log_warn "Stopping all services (reason: $SHUTDOWN_REASON, status: $exit_code)..."
  # Stop dependents before their model servers: API, main vLLM, then parser.
  for (( index=${#PIDS[@]}-1; index>=0; index-- )); do
    pid="${PIDS[index]}"
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
  # Stop the parser Docker container if we started one.
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^exact-type1-parser-cpu$"; then
    log_info "Stopping Docker container exact-type1-parser-cpu..."
    docker stop exact-type1-parser-cpu >/dev/null 2>&1 || true
  fi
  log_info "Done."
  exit "$exit_code"
}
trap 'SHUTDOWN_REASON="received SIGINT"; cleanup 130' INT
trap 'SHUTDOWN_REASON="received SIGTERM"; cleanup 143' TERM
trap 'SHUTDOWN_REASON="received SIGHUP (terminal/SSH disconnected)"; cleanup 129' HUP
trap 'cleanup $?' EXIT

# ---------------------------------------------------------------------------
# API Python executable
# ---------------------------------------------------------------------------

PYTHON_BIN="${EXACT_API_PYTHON_BIN:-$PROJECT_ROOT/exact/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  log_error "Python not found: $PYTHON_BIN"
  exit 1
fi

# Keep the CUDA-heavy vLLM stack separate from the API environment. A shared
# environment is prone to Torch ABI conflicts from optional packages such as
# FlashInfer.
DEFAULT_VLLM_BIN="$PROJECT_ROOT/.venv-vllm/bin/vllm"

# ---------------------------------------------------------------------------
# 1. Type 1 parser vLLM
# ---------------------------------------------------------------------------

PARSER_MODEL="${EXACT_TYPE1_PARSER_SOURCE_MODEL:-Qwen/Qwen3-1.7B}"
PARSER_SERVED_NAME="${EXACT_TYPE1_PARSER_MODEL:-type1-parser}"
PARSER_HOST="${EXACT_TYPE1_PARSER_SERVER_HOST:-127.0.0.1}"
PARSER_CONNECT_HOST="$(connect_host_for_bind "$PARSER_HOST")"
PARSER_PORT="${EXACT_TYPE1_PARSER_SERVER_PORT:-8001}"
PARSER_API_KEY="${EXACT_TYPE1_PARSER_API_KEY:-exact-parser-token}"
PARSER_DEVICE="${EXACT_TYPE1_PARSER_SERVER_DEVICE:-auto}"
PARSER_MAX_MODEL_LEN="${EXACT_TYPE1_PARSER_SERVER_MAX_MODEL_LEN:-4096}"
PARSER_MAX_NUM_SEQS="${EXACT_TYPE1_PARSER_SERVER_MAX_NUM_SEQS:-64}"
PARSER_MAX_BATCHED_TOKENS="${EXACT_TYPE1_PARSER_SERVER_MAX_NUM_BATCHED_TOKENS:-8192}"
PARSER_GPU_MEM="${EXACT_TYPE1_PARSER_SERVER_GPU_MEMORY_UTILIZATION:-0.25}"
PARSER_DTYPE="${EXACT_TYPE1_PARSER_SERVER_DTYPE:-auto}"
PARSER_QUANTIZATION="${EXACT_TYPE1_PARSER_SERVER_QUANTIZATION:-}"
PARSER_TENSOR_PARALLEL="${EXACT_TYPE1_PARSER_SERVER_TENSOR_PARALLEL_SIZE:-1}"
PARSER_CUDA_VISIBLE_DEVICES="${EXACT_TYPE1_PARSER_SERVER_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
# Default to eager mode: CUDA-graph capture is unreliable on virtualized GPUs
# (e.g. ThunderCompute). Set =0 on a dedicated GPU host to re-enable graphs.
PARSER_ENFORCE_EAGER="${EXACT_TYPE1_PARSER_SERVER_ENFORCE_EAGER:-1}"
PARSER_CPU_KVCACHE="${EXACT_TYPE1_PARSER_SERVER_CPU_KVCACHE_SPACE:-1}"
PARSER_CPU_RESERVED="${EXACT_TYPE1_PARSER_SERVER_CPU_RESERVED_CPUS:-1}"
PARSER_CPU_IMAGE="${EXACT_TYPE1_PARSER_SERVER_CPU_IMAGE:-vllm/vllm-openai-cpu:latest-x86_64}"
# Model (1.4 GiB) + KV cache (1 GiB) + overhead → 4 GiB is the safe minimum.
PARSER_DOCKER_MEMORY="${EXACT_TYPE1_PARSER_SERVER_DOCKER_MEMORY:-4g}"

# LoRA adapter (optional). GPU-only; ignored in the Docker CPU path.
PARSER_LORA_ADAPTER="${EXACT_TYPE1_PARSER_LORA_ADAPTER:-}"
PARSER_LORA_MAX_RANK="${EXACT_TYPE1_PARSER_LORA_MAX_RANK:-64}"

if [[ "$PARSER_DEVICE" == "cpu" ]]; then
  # ---- Docker CPU path ----
  DOCKER_CONTEXT="${EXACT_TYPE1_PARSER_SERVER_DOCKER_CONTEXT:-}"
  docker_cmd=(docker)
  [[ -n "$DOCKER_CONTEXT" ]] && docker_cmd+=(--context "$DOCKER_CONTEXT")

  if ! "${docker_cmd[@]}" info >/dev/null 2>&1; then
    log_error "Docker is not accessible. Start Docker or set EXACT_TYPE1_PARSER_SERVER_DEVICE=auto."
    exit 1
  fi

  HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
  mkdir -p "$HF_CACHE"

  # Remove stale container from a previous interrupted run.
  "${docker_cmd[@]}" rm -f exact-type1-parser-cpu >/dev/null 2>&1 || true

  log_info "Starting Type 1 parser via Docker CPU image: $PARSER_MODEL → $PARSER_HOST:$PARSER_PORT (memory: $PARSER_DOCKER_MEMORY)"

  run_cmd=(
    "${docker_cmd[@]}" run --rm
    --name exact-type1-parser-cpu
    --security-opt seccomp=unconfined
    --cap-add SYS_NICE
    --memory "$PARSER_DOCKER_MEMORY"
    --shm-size=2g
    -p "${PARSER_PORT}:${PARSER_PORT}"
    -v "$HF_CACHE:/root/.cache/huggingface"
    -e "VLLM_CPU_KVCACHE_SPACE=$PARSER_CPU_KVCACHE"
    -e "VLLM_CPU_NUM_OF_RESERVED_CPU=$PARSER_CPU_RESERVED"
  )
  [[ -n "${HF_TOKEN:-}" ]] && run_cmd+=(-e "HF_TOKEN=$HF_TOKEN")
  run_cmd+=(
    "$PARSER_CPU_IMAGE"
    "$PARSER_MODEL"
    --served-model-name "$PARSER_SERVED_NAME"
    --host 0.0.0.0
    --port "$PARSER_PORT"
    --api-key "$PARSER_API_KEY"
    --dtype "$PARSER_DTYPE"
    --max-model-len "$PARSER_MAX_MODEL_LEN"
    --max-num-seqs "$PARSER_MAX_NUM_SEQS"
    --max-num-batched-tokens "$PARSER_MAX_BATCHED_TOKENS"
    --tensor-parallel-size 1
    --generation-config vllm
    --enforce-eager
  )

  "${run_cmd[@]}" >> "$LOG_DIR/parser.log" 2>&1 &
  PARSER_PID=$!
  PIDS+=("$PARSER_PID")
  log_info "Parser vLLM (Docker) started (PID $PARSER_PID) — log: logs/parser.log"

else
  # ---- Native vllm binary path ----
  PARSER_VLLM_BIN="${EXACT_TYPE1_PARSER_SERVER_VLLM_BIN:-}"
  if [[ -z "$PARSER_VLLM_BIN" ]]; then
    if [[ -n "${VLLM_BIN:-}" ]]; then
      PARSER_VLLM_BIN="$VLLM_BIN"
    elif [[ -x "$DEFAULT_VLLM_BIN" ]]; then
      PARSER_VLLM_BIN="$DEFAULT_VLLM_BIN"
    elif [[ -x "$PROJECT_ROOT/.venv-vllm-cpu/bin/vllm" ]]; then
      PARSER_VLLM_BIN="$PROJECT_ROOT/.venv-vllm-cpu/bin/vllm"
    else
      PARSER_VLLM_BIN="vllm"
    fi
  fi
  if ! command -v "$PARSER_VLLM_BIN" >/dev/null 2>&1; then
    log_error "vLLM binary not found: $PARSER_VLLM_BIN"
    log_error "Install vLLM, set EXACT_TYPE1_PARSER_SERVER_VLLM_BIN, or set EXACT_TYPE1_PARSER_SERVER_DEVICE=cpu to use Docker."
    exit 1
  fi
  check_vllm_runtime "$PARSER_VLLM_BIN" || exit 1

  log_info "Starting Type 1 parser vLLM: $PARSER_MODEL → $PARSER_HOST:$PARSER_PORT ($PARSER_DEVICE)"
  [[ -n "$PARSER_CUDA_VISIBLE_DEVICES" ]] && log_info "Parser GPU(s): $PARSER_CUDA_VISIBLE_DEVICES"
  [[ -n "$PARSER_LORA_ADAPTER" ]] && log_info "Parser LoRA adapter: $PARSER_LORA_ADAPTER (max rank: $PARSER_LORA_MAX_RANK)"

  # VLLM_PORT is reserved by vLLM as the base for internal coordination
  # sockets. The launcher also accepts it as the main HTTP-port override, so
  # do not pass an exported value into either vLLM child process; otherwise
  # vLLM may try to claim adjacent ports such as 8001/8002.
  parser_env=(env -u VLLM_PORT)
  [[ -n "$PARSER_CUDA_VISIBLE_DEVICES" ]] && parser_env+=(CUDA_VISIBLE_DEVICES="$PARSER_CUDA_VISIBLE_DEVICES")
  parser_cmd=(
    "${parser_env[@]}" "$PARSER_VLLM_BIN" serve "$PARSER_MODEL"
    --served-model-name "$PARSER_SERVED_NAME"
    --host "$PARSER_HOST" --port "$PARSER_PORT"
    --api-key "$PARSER_API_KEY"
    --dtype "$PARSER_DTYPE"
    --max-model-len "$PARSER_MAX_MODEL_LEN"
    --max-num-seqs "$PARSER_MAX_NUM_SEQS"
    --max-num-batched-tokens "$PARSER_MAX_BATCHED_TOKENS"
    --tensor-parallel-size "$PARSER_TENSOR_PARALLEL"
    --gpu-memory-utilization "$PARSER_GPU_MEM"
    --enable-prefix-caching
    --generation-config vllm
  )
  [[ -n "$PARSER_QUANTIZATION" ]] && parser_cmd+=(--quantization "$PARSER_QUANTIZATION")
  [[ "$PARSER_ENFORCE_EAGER" == "1" ]] && parser_cmd+=(--enforce-eager)

  # LoRA: add --enable-lora when an adapter path/repo is configured.
  # The adapter is registered under the same served-model-name so the
  # ParserClient needs no changes.
  if [[ -n "$PARSER_LORA_ADAPTER" ]]; then
    parser_cmd+=(
      --enable-lora
      --lora-modules "${PARSER_SERVED_NAME}=${PARSER_LORA_ADAPTER}"
      --max-lora-rank "$PARSER_LORA_MAX_RANK"
    )
  fi

  "${parser_cmd[@]}" >> "$LOG_DIR/parser.log" 2>&1 &
  PARSER_PID=$!
  PIDS+=("$PARSER_PID")
  log_info "Parser vLLM started (PID $PARSER_PID) — log: logs/parser.log"
fi

# Finish parser initialization before starting the second vLLM process. This
# avoids races while both processes allocate internal coordination ports.
wait_for_tcp \
  "Parser vLLM" \
  "$PARSER_CONNECT_HOST" \
  "$PARSER_PORT" \
  300 \
  "$PARSER_PID" \
  "$LOG_DIR/parser.log" || exit 1

# ---------------------------------------------------------------------------
# 2. Main vLLM (GPU, port 8000) — skipped if binary not found
# ---------------------------------------------------------------------------

VLLM_SKIP="${VLLM_SKIP:-0}"
if [[ -n "${VLLM_BIN:-}" ]]; then
  MAIN_VLLM_BIN="$VLLM_BIN"
elif [[ -x "$DEFAULT_VLLM_BIN" ]]; then
  MAIN_VLLM_BIN="$DEFAULT_VLLM_BIN"
else
  MAIN_VLLM_BIN="vllm"
fi

if [[ "$VLLM_SKIP" == "1" ]]; then
  log_warn "VLLM_SKIP=1 — skipping main vLLM (assumed already running)."
elif ! command -v "$MAIN_VLLM_BIN" >/dev/null 2>&1; then
  log_warn "Main vLLM binary not found ($MAIN_VLLM_BIN) — skipping."
  log_warn "If the main model is already running, set VLLM_SKIP=1 to suppress this warning."
  VLLM_SKIP=1
else
  check_vllm_runtime "$MAIN_VLLM_BIN" || exit 1
  VLLM_MODEL="${VLLM_MODEL:-${EXACT_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}}"
  VLLM_SERVED_NAME="${VLLM_SERVED_MODEL_NAME:-$VLLM_MODEL}"
  VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
  VLLM_CONNECT_HOST="$(connect_host_for_bind "$VLLM_HOST")"
  VLLM_PORT="${VLLM_PORT:-8000}"
  VLLM_API_KEY="${VLLM_API_KEY:-${EXACT_LLM_API_KEY:-exact-local-token}}"
  VLLM_DTYPE="${VLLM_DTYPE:-auto}"
  VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
  VLLM_GPU_MEM="${VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
  VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-}"
  VLLM_TENSOR_PARALLEL="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
  VLLM_CUDA_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
  VLLM_ENFORCE_EAGER_FLAG="${VLLM_ENFORCE_EAGER:-1}"

  log_info "Starting main vLLM: $VLLM_MODEL → $VLLM_HOST:$VLLM_PORT"
  [[ -n "$VLLM_CUDA_DEVICES" ]] && log_info "Main vLLM GPU(s): $VLLM_CUDA_DEVICES"

  main_vllm_env=(env -u VLLM_PORT)
  [[ -n "$VLLM_CUDA_DEVICES" ]] && main_vllm_env+=(CUDA_VISIBLE_DEVICES="$VLLM_CUDA_DEVICES")
  vllm_cmd=(
    "${main_vllm_env[@]}" "$MAIN_VLLM_BIN" serve "$VLLM_MODEL"
    --served-model-name "$VLLM_SERVED_NAME"
    --host "$VLLM_HOST" --port "$VLLM_PORT"
    --api-key "$VLLM_API_KEY"
    --dtype "$VLLM_DTYPE"
    --max-model-len "$VLLM_MAX_MODEL_LEN"
    --gpu-memory-utilization "$VLLM_GPU_MEM"
    --tensor-parallel-size "$VLLM_TENSOR_PARALLEL"
    --enable-prefix-caching
    --generation-config vllm
  )
  [[ -n "$VLLM_QUANTIZATION" ]] && vllm_cmd+=(--quantization "$VLLM_QUANTIZATION")
  [[ "$VLLM_ENFORCE_EAGER_FLAG" == "1" ]] && vllm_cmd+=(--enforce-eager)

  "${vllm_cmd[@]}" >> "$LOG_DIR/vllm.log" 2>&1 &
  VLLM_PID=$!
  PIDS+=("$VLLM_PID")
  log_info "Main vLLM started (PID $VLLM_PID) — log: logs/vllm.log"
fi

# ---------------------------------------------------------------------------
# 3. Wait for vLLM servers to be ready
# ---------------------------------------------------------------------------

if [[ "$VLLM_SKIP" != "1" ]]; then
  wait_for_tcp \
    "Main vLLM" \
    "$VLLM_CONNECT_HOST" \
    "$VLLM_PORT" \
    360 \
    "$VLLM_PID" \
    "$LOG_DIR/vllm.log" || exit 1
fi

# ---------------------------------------------------------------------------
# 4. EXACT API (foreground)
# ---------------------------------------------------------------------------

API_HOST="${API_HOST:-${EXACT_API_HOST:-0.0.0.0}}"
API_PORT="${API_PORT:-${EXACT_API_PORT:-8080}}"
API_CONNECT_HOST="$(connect_host_for_bind "$API_HOST")"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

log_info "Starting EXACT API: $API_HOST:$API_PORT"
# uvicorn only accepts lowercase log levels; EXACT_LOG_LEVEL may be uppercase.
API_LOG_LEVEL="${UVICORN_LOG_LEVEL:-${EXACT_LOG_LEVEL:-info}}"
API_LOG_LEVEL="${API_LOG_LEVEL,,}"
uvicorn_cmd=(
  "$PYTHON_BIN" -m uvicorn exact.app.main:app
  --host "$API_HOST"
  --port "$API_PORT"
  --log-level "$API_LOG_LEVEL"
)
if [[ "${UVICORN_RELOAD:-0}" == "1" ]]; then
  uvicorn_cmd+=(--reload --reload-dir "$PROJECT_ROOT/src")
  log_info "API hot-reload enabled (watching src/)"
fi
"${uvicorn_cmd[@]}" &
API_PID=$!
PIDS+=("$API_PID")

wait_for_tcp \
  "EXACT API" \
  "$API_CONNECT_HOST" \
  "$API_PORT" \
  60 \
  "$API_PID" \
  "$PROJECT_ROOT/outputs/logs/api.log" || exit 1

check_models_endpoint "http://${API_CONNECT_HOST}:${API_PORT}/v1/models" || exit 1

echo ""
log_info "All services ready."
echo -e "  ${GREEN}Parser vLLM:${NC}  http://${PARSER_HOST}:${PARSER_PORT}/v1"
[[ "$VLLM_SKIP" != "1" ]] && echo -e "  ${GREEN}Main vLLM:${NC}    http://${VLLM_HOST}:${VLLM_PORT}/v1"
echo -e "  ${GREEN}EXACT API:${NC}    http://${API_HOST}:${API_PORT}"
echo -e "  ${GREEN}Committee:${NC}    http://${API_CONNECT_HOST}:${API_PORT}/v1/models"
echo ""

# Keep the launcher attached to the directly-owned API process without polling
# any HTTP endpoint. vLLM wrapper PIDs are intentionally not supervised because
# some vLLM versions hand off to another process after startup.
API_STATUS=0
wait "$API_PID" || API_STATUS=$?
SHUTDOWN_REASON="EXACT API exited"
log_error "EXACT API exited unexpectedly (status $API_STATUS)."
[[ $API_STATUS -eq 0 ]] && API_STATUS=1
exit "$API_STATUS"
