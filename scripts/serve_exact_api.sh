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

wait_for_http() {
  local name="$1" url="$2" timeout="${3:-120}" elapsed=0
  log_info "Waiting for $name at $url (timeout ${timeout}s)..."
  until curl -sf "$url" -o /dev/null 2>/dev/null; do
    if [[ $elapsed -ge $timeout ]]; then
      log_error "$name not ready after ${timeout}s — check $LOG_DIR/"
      return 1
    fi
    sleep 3; elapsed=$((elapsed + 3)); echo -n "."
  done
  echo ""; log_info "$name ready after ${elapsed}s."
}

PIDS=()
cleanup() {
  echo ""
  log_warn "Stopping all services..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  # Stop the parser Docker container if we started one.
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^exact-type1-parser-cpu$"; then
    log_info "Stopping Docker container exact-type1-parser-cpu..."
    docker stop exact-type1-parser-cpu >/dev/null 2>&1 || true
  fi
  log_info "Done."
  exit 0
}
trap cleanup INT TERM

# ---------------------------------------------------------------------------
# API Python executable
# ---------------------------------------------------------------------------

PYTHON_BIN="${EXACT_API_PYTHON_BIN:-$PROJECT_ROOT/exact/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  log_error "Python not found: $PYTHON_BIN"
  exit 1
fi

# ---------------------------------------------------------------------------
# 1. Type 1 parser vLLM
# ---------------------------------------------------------------------------

PARSER_MODEL="${EXACT_TYPE1_PARSER_SOURCE_MODEL:-Qwen/Qwen3-1.7B}"
PARSER_SERVED_NAME="${EXACT_TYPE1_PARSER_MODEL:-type1-parser}"
PARSER_HOST="${EXACT_TYPE1_PARSER_SERVER_HOST:-127.0.0.1}"
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
PARSER_CPU_KVCACHE="${EXACT_TYPE1_PARSER_SERVER_CPU_KVCACHE_SPACE:-1}"
PARSER_CPU_RESERVED="${EXACT_TYPE1_PARSER_SERVER_CPU_RESERVED_CPUS:-1}"
PARSER_CPU_IMAGE="${EXACT_TYPE1_PARSER_SERVER_CPU_IMAGE:-vllm/vllm-openai-cpu:latest-x86_64}"
# Model (1.4 GiB) + KV cache (1 GiB) + overhead → 4 GiB is the safe minimum.
PARSER_DOCKER_MEMORY="${EXACT_TYPE1_PARSER_SERVER_DOCKER_MEMORY:-4g}"

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
    if [[ -x "$PROJECT_ROOT/.venv-vllm-cpu/bin/vllm" ]]; then
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

  log_info "Starting Type 1 parser vLLM: $PARSER_MODEL → $PARSER_HOST:$PARSER_PORT ($PARSER_DEVICE)"

  parser_cmd=(
    "$PARSER_VLLM_BIN" serve "$PARSER_MODEL"
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

  "${parser_cmd[@]}" >> "$LOG_DIR/parser.log" 2>&1 &
  PARSER_PID=$!
  PIDS+=("$PARSER_PID")
  log_info "Parser vLLM started (PID $PARSER_PID) — log: logs/parser.log"
fi

# ---------------------------------------------------------------------------
# 2. Main vLLM (GPU, port 8000) — skipped if binary not found
# ---------------------------------------------------------------------------

VLLM_SKIP="${VLLM_SKIP:-0}"
MAIN_VLLM_BIN="${VLLM_BIN:-vllm}"

if [[ "$VLLM_SKIP" == "1" ]]; then
  log_warn "VLLM_SKIP=1 — skipping main vLLM (assumed already running)."
elif ! command -v "$MAIN_VLLM_BIN" >/dev/null 2>&1; then
  log_warn "Main vLLM binary not found ($MAIN_VLLM_BIN) — skipping."
  log_warn "If the main model is already running, set VLLM_SKIP=1 to suppress this warning."
  VLLM_SKIP=1
else
  VLLM_MODEL="${VLLM_MODEL:-${EXACT_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}}"
  VLLM_SERVED_NAME="${VLLM_SERVED_MODEL_NAME:-$VLLM_MODEL}"
  VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
  VLLM_PORT="${VLLM_PORT:-8000}"
  VLLM_API_KEY="${VLLM_API_KEY:-${EXACT_LLM_API_KEY:-exact-local-token}}"
  VLLM_DTYPE="${VLLM_DTYPE:-auto}"
  VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
  VLLM_GPU_MEM="${VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
  VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-}"
  VLLM_TENSOR_PARALLEL="${VLLM_TENSOR_PARALLEL_SIZE:-1}"

  log_info "Starting main vLLM: $VLLM_MODEL → $VLLM_HOST:$VLLM_PORT"

  vllm_cmd=(
    "$MAIN_VLLM_BIN" serve "$VLLM_MODEL"
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

  "${vllm_cmd[@]}" >> "$LOG_DIR/vllm.log" 2>&1 &
  VLLM_PID=$!
  PIDS+=("$VLLM_PID")
  log_info "Main vLLM started (PID $VLLM_PID) — log: logs/vllm.log"
fi

# ---------------------------------------------------------------------------
# 3. Wait for vLLM servers to be ready
# ---------------------------------------------------------------------------

wait_for_http "Parser vLLM" "http://${PARSER_HOST}:${PARSER_PORT}/health" 300 || exit 1

if [[ "$VLLM_SKIP" != "1" ]]; then
  wait_for_http "Main vLLM" "http://${VLLM_HOST}:${VLLM_PORT}/health" 360 || exit 1
fi

# ---------------------------------------------------------------------------
# 4. EXACT API (foreground)
# ---------------------------------------------------------------------------

API_HOST="${API_HOST:-${EXACT_API_HOST:-0.0.0.0}}"
API_PORT="${API_PORT:-${EXACT_API_PORT:-8080}}"

echo ""
log_info "All services ready."
echo -e "  ${GREEN}Parser vLLM:${NC}  http://${PARSER_HOST}:${PARSER_PORT}/v1"
[[ "$VLLM_SKIP" != "1" ]] && echo -e "  ${GREEN}Main vLLM:${NC}    http://${VLLM_HOST}:${VLLM_PORT}/v1"
echo -e "  ${GREEN}EXACT API:${NC}    http://${API_HOST}:${API_PORT} (starting...)"
echo ""

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" -m uvicorn exact.app.main:app \
  --host "$API_HOST" \
  --port "$API_PORT" \
  --log-level "${UVICORN_LOG_LEVEL:-${EXACT_LOG_LEVEL:-info}}"
