#!/usr/bin/env bash
# =============================================================================
# start_all_tmux.sh — EXACT 2026 VM Startup (with tmux for real-time logs)
#
# Chạy 3 services (vLLM, EXACT API, Cloudflare Tunnel) trong tmux sessions
# có thể tách rời, sao cho mỗi cái có realtime log window riêng.
#
# Cách chạy:
#   bash scripts/start_all_tmux.sh
#   VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ bash scripts/start_all_tmux.sh
#
# Keybindings (khi attach tmux):
#   Ctrl+B N  — Next window
#   Ctrl+B P  — Previous window
#   Ctrl+B 0  — Go to window 0 (vLLM)
#   Ctrl+B 1  — Go to window 1 (API)
#   Ctrl+B 2  — Go to window 2 (Tunnel)
#   Ctrl+B D  — Detach (script vẫn chạy background)
#   Ctrl+B [  — Scroll mode (PgUp/PgDn, q để exit)
#
# Kill tmux session:
#   tmux kill-session -t exact
# =============================================================================

set -uo pipefail

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# Paths
VLLM_LOG="$LOG_DIR/vllm.log"
API_LOG="$LOG_DIR/api.log"
TUNNEL_LOG="$LOG_DIR/cloudflared.log"

API_PYTHON="$PROJECT_ROOT/exact/bin/python"
API_PIP="$PROJECT_ROOT/exact/bin/pip"
VLLM_VENV="$PROJECT_ROOT/.venv-vllm"
VLLM_PYTHON="$VLLM_VENV/bin/python"
VLLM_PIP="$VLLM_VENV/bin/pip"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

log_info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }
log_section() { echo -e "\n${BOLD}${BLUE}=== $* ===${NC}"; }

# Config
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-$VLLM_MODEL}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.90}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-awq_marlin}"
VLLM_API_KEY="${VLLM_API_KEY:-exact-local-token}"

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8080}"

CLOUDFLARE_TUNNEL_NAME="${CLOUDFLARE_TUNNEL_NAME:-}"

VLLM_WAIT_TIMEOUT="${VLLM_WAIT_TIMEOUT:-360}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-60}"

SESSION_NAME="exact"

# Cleanup: kill tmux session khi thoát
cleanup() {
    echo ""
    log_warn "Dừng tmux session..."
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
    log_info "Đã dừng."
    exit 0
}
trap cleanup INT TERM EXIT

cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Setup helpers (copy từ start_all.sh)
# ---------------------------------------------------------------------------

wait_for_http() {
    local name="$1"
    local url="$2"
    local timeout="${3:-$WAIT_TIMEOUT}"
    local elapsed=0

    log_info "Chờ $name sẵn sàng tại $url (timeout ${timeout}s)..."
    while ! curl -sf "$url" -o /dev/null 2>/dev/null; do
        if [[ $elapsed -ge $timeout ]]; then
            log_error "$name không sẵn sàng sau ${timeout}s. Kiểm tra tmux window."
            return 1
        fi
        sleep 3
        elapsed=$((elapsed + 3))
        echo -n "."
    done
    echo ""
    log_info "$name sẵn sàng sau ${elapsed}s."
}

pkg_installed() {
    local pip_bin="$1"
    local pkg="$2"
    "$pip_bin" show "$pkg" &>/dev/null
}

# ---------------------------------------------------------------------------
# Environment checks (copy từ start_all.sh)
# ---------------------------------------------------------------------------

log_section "Kiểm tra môi trường"

if [[ ! -x "$API_PYTHON" ]]; then
    log_error "Không tìm thấy Python tại $API_PYTHON"
    exit 1
fi
log_info "API Python: $($API_PYTHON --version) ($API_PYTHON)"

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "?")
    log_info "GPU: $GPU_NAME ($GPU_MEM)"
else
    log_warn "nvidia-smi không có"
fi

# ---------------------------------------------------------------------------
# Install packages (copy từ start_all.sh)
# ---------------------------------------------------------------------------

log_section "Kiểm tra package EXACT API"

MISSING_API=0
for pkg in fastapi uvicorn httpx pydantic-settings sympy; do
    if ! pkg_installed "$API_PIP" "$pkg"; then
        log_warn "Thiếu package: $pkg"
        MISSING_API=1
    fi
done

if [[ $MISSING_API -eq 1 ]]; then
    log_info "Cài đặt requirements-api.txt..."
    "$API_PIP" install --quiet -r "$PROJECT_ROOT/requirements-api.txt"
    "$API_PIP" install --quiet -e "$PROJECT_ROOT"
fi

log_section "Kiểm tra vLLM"

if [[ ! -x "$VLLM_PYTHON" ]]; then
    log_info "Tạo venv vLLM..."
    python3 -m venv "$VLLM_VENV"
    "$VLLM_PIP" install --quiet --upgrade pip
fi

if ! pkg_installed "$VLLM_PIP" vllm; then
    log_info "Cài đặt vLLM..."
    "$VLLM_PIP" install --quiet vllm
fi

log_section "Kiểm tra cloudflared"

CLOUDFLARED_BIN=""
if command -v cloudflared &>/dev/null; then
    CLOUDFLARED_BIN="$(command -v cloudflared)"
else
    CF_BIN="$PROJECT_ROOT/.venv-vllm/bin/cloudflared"
    if [[ ! -x "$CF_BIN" ]]; then
        log_info "Tải cloudflared binary..."
        ARCH=$(uname -m)
        case "$ARCH" in
            x86_64)  CF_ARCH="amd64" ;;
            aarch64) CF_ARCH="arm64" ;;
            *)       log_error "Architecture không hỗ trợ: $ARCH"; exit 1 ;;
        esac
        CF_URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${CF_ARCH}"
        curl -fsSL "$CF_URL" -o "$CF_BIN"
        chmod +x "$CF_BIN"
    fi
    CLOUDFLARED_BIN="$CF_BIN"
fi

# GPU memory check
if command -v nvidia-smi &>/dev/null; then
    FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
    FREE_PCT=$(( FREE_MIB * 100 / TOTAL_MIB ))
    log_info "GPU memory: ${FREE_MIB}/${TOTAL_MIB} MiB free (${FREE_PCT}%)"
    if [[ $FREE_PCT -lt 20 ]]; then
        log_error "GPU memory còn ${FREE_PCT}% free — vui lòng dọn dẹp."
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Create tmux session + windows
# ---------------------------------------------------------------------------

log_section "Tạo tmux session"

# Kill session cũ nếu có
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
sleep 1

# Tạo session với window đầu tiên (vLLM)
tmux new-session -d -s "$SESSION_NAME" -n "vllm" -c "$PROJECT_ROOT"

# Thêm 2 window nữa
tmux new-window -t "$SESSION_NAME" -n "api" -c "$PROJECT_ROOT"
tmux new-window -t "$SESSION_NAME" -n "tunnel" -c "$PROJECT_ROOT"

log_info "Tmux session '$SESSION_NAME' tạo xong."

# ---------------------------------------------------------------------------
# Window 0: vLLM
# ---------------------------------------------------------------------------

log_section "Khởi động vLLM"

VLLM_CMD="$VLLM_PYTHON -m vllm.entrypoints.openai.api_server"
VLLM_CMD="$VLLM_CMD --model $VLLM_MODEL"
VLLM_CMD="$VLLM_CMD --served-model-name $VLLM_SERVED_MODEL_NAME"
VLLM_CMD="$VLLM_CMD --host $VLLM_HOST"
VLLM_CMD="$VLLM_CMD --port $VLLM_PORT"
VLLM_CMD="$VLLM_CMD --dtype $VLLM_DTYPE"
VLLM_CMD="$VLLM_CMD --max-model-len $VLLM_MAX_MODEL_LEN"
VLLM_CMD="$VLLM_CMD --gpu-memory-utilization $VLLM_GPU_MEM"
VLLM_CMD="$VLLM_CMD --enable-prefix-caching"
VLLM_CMD="$VLLM_CMD --api-key $VLLM_API_KEY"
if [[ -n "$VLLM_QUANTIZATION" ]]; then
    VLLM_CMD="$VLLM_CMD --quantization $VLLM_QUANTIZATION"
fi
VLLM_CMD="$VLLM_CMD 2>&1 | tee -a $VLLM_LOG"

tmux send-keys -t "$SESSION_NAME:vllm" "$VLLM_CMD" Enter
log_info "vLLM window: $SESSION_NAME:vllm — log realtime"

# Chờ vLLM ready
wait_for_http "vLLM" "http://${VLLM_HOST}:${VLLM_PORT}/health" "$VLLM_WAIT_TIMEOUT" || {
    log_error "vLLM không start được."
    exit 1
}

# ---------------------------------------------------------------------------
# Window 1: EXACT API
# ---------------------------------------------------------------------------

log_section "Khởi động EXACT API"

API_CMD="EXACT_LLM_BASE_URL=http://${VLLM_HOST}:${VLLM_PORT}/v1"
API_CMD="$API_CMD EXACT_LLM_API_KEY=$VLLM_API_KEY"
API_CMD="$API_CMD EXACT_LLM_MODEL=$VLLM_MODEL"
API_CMD="$API_CMD PYTHONPATH=$PROJECT_ROOT/src"
API_CMD="$API_CMD $API_PYTHON -m uvicorn exact.app.main:app"
API_CMD="$API_CMD --host $API_HOST"
API_CMD="$API_CMD --port $API_PORT"
API_CMD="$API_CMD --log-level info"
API_CMD="$API_CMD 2>&1 | tee -a $API_LOG"

tmux send-keys -t "$SESSION_NAME:api" "$API_CMD" Enter
log_info "API window: $SESSION_NAME:api — log realtime"

# Chờ API ready
wait_for_http "EXACT API" "http://127.0.0.1:${API_PORT}/health" || {
    log_error "EXACT API không start được."
    exit 1
}

# ---------------------------------------------------------------------------
# Window 2: Cloudflare Tunnel
# ---------------------------------------------------------------------------

log_section "Khởi động Cloudflare Tunnel"

if [[ -n "$CLOUDFLARE_TUNNEL_NAME" ]]; then
    # Named tunnel routing is defined in ~/.cloudflared/config.yml — do NOT pass --url.
    TUNNEL_CMD="$CLOUDFLARED_BIN tunnel run $CLOUDFLARE_TUNNEL_NAME"
else
    TUNNEL_CMD="$CLOUDFLARED_BIN tunnel --url http://localhost:${API_PORT}"
fi
TUNNEL_CMD="$TUNNEL_CMD 2>&1 | tee -a $TUNNEL_LOG"

tmux send-keys -t "$SESSION_NAME:tunnel" "$TUNNEL_CMD" Enter
log_info "Tunnel window: $SESSION_NAME:tunnel — log realtime"

# Chờ tunnel URL nếu quick tunnel
if [[ -z "$CLOUDFLARE_TUNNEL_NAME" ]]; then
    log_info "Đang chờ tunnel URL..."
    for i in $(seq 1 30); do
        TUNNEL_URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" 2>/dev/null | head -1 || true)
        if [[ -n "$TUNNEL_URL" ]]; then
            echo -e "\n${GREEN}${BOLD}>>> Tunnel URL: $TUNNEL_URL${NC}\n"
            break
        fi
        sleep 2
    done
fi

# ---------------------------------------------------------------------------
# Done — show info and attach
# ---------------------------------------------------------------------------

log_section "Tất cả services đã khởi động"

echo ""
echo -e "  ${GREEN}vLLM:${NC}        http://${VLLM_HOST}:${VLLM_PORT}/v1"
echo -e "  ${GREEN}EXACT API:${NC}   http://localhost:${API_PORT}"
echo -e "  ${GREEN}Logs:${NC}        $LOG_DIR/"
echo ""
echo -e "  ${BOLD}Tmux windows:${NC}"
echo -e "    ${GREEN}tmux a -t $SESSION_NAME${NC}     Attach session"
echo -e "    ${GREEN}Ctrl+B N${NC}                  Next window"
echo -e "    ${GREEN}Ctrl+B P${NC}                  Prev window"
echo -e "    ${GREEN}Ctrl+B 0/1/2${NC}             Go to vLLM/API/Tunnel"
echo -e "    ${GREEN}Ctrl+B D${NC}                  Detach (keep running)"
echo -e "    ${GREEN}Ctrl+B [${NC}                  Scroll (PgUp/PgDn, q exit)"
echo ""

# Attach to the session
exec tmux attach-session -t "$SESSION_NAME"
