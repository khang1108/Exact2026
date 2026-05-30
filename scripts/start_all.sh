#!/usr/bin/env bash
# =============================================================================
# start_all.sh — EXACT 2026 VM Startup Script
#
# Khởi động toàn bộ stack theo đúng thứ tự:
#   1. vLLM model server     (127.0.0.1:8000)
#   2. EXACT FastAPI/uvicorn (0.0.0.0:8080)
#   3. Cloudflare Tunnel     (tunnel → localhost:8080)
#
# Script tự kiểm tra package còn thiếu và cài đặt trước khi start mỗi service.
# Ctrl+C sẽ dừng sạch tất cả process con.
#
# Cách chạy:
#   bash scripts/start_all.sh
#   VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ bash scripts/start_all.sh
# =============================================================================

set -uo pipefail

# ---------------------------------------------------------------------------
# 0. Paths và constants
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

VLLM_LOG="$LOG_DIR/vllm.log"
API_LOG="$LOG_DIR/api.log"
TUNNEL_LOG="$LOG_DIR/cloudflared.log"

# File lưu PIDs để cleanup khi thoát
PID_FILE="$LOG_DIR/start_all.pids"

# Venv cho EXACT API (exact/ là symlink tới system python + site-packages của project)
API_PYTHON="$PROJECT_ROOT/exact/bin/python"
API_PIP="$PROJECT_ROOT/exact/bin/pip"

# Venv riêng cho vLLM (tránh conflict dependencies)
VLLM_VENV="$PROJECT_ROOT/.venv-vllm"
VLLM_PYTHON="$VLLM_VENV/bin/python"
VLLM_PIP="$VLLM_VENV/bin/pip"

# ---------------------------------------------------------------------------
# 1. Config có thể override qua environment variable
# ---------------------------------------------------------------------------

# vLLM
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-$VLLM_MODEL}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.90}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-awq_marlin}"  # awq_marlin cho AWQ model
VLLM_API_KEY="${VLLM_API_KEY:-exact-local-token}"

# EXACT API
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8080}"

# Cloudflare Tunnel: dùng named tunnel nếu có, fallback về quick tunnel
# Named tunnel cần CLOUDFLARE_TUNNEL_NAME và file credential ~/.cloudflared/
# Quick tunnel (--url) tạo URL ngẫu nhiên mỗi lần, không cần setup trước
CLOUDFLARE_TUNNEL_NAME="${CLOUDFLARE_TUNNEL_NAME:-}"  # để trống → quick tunnel

# Timeout (giây) chờ mỗi service ready trước khi tiếp tục
WAIT_TIMEOUT="${WAIT_TIMEOUT:-120}"

# ---------------------------------------------------------------------------
# 2. Màu output
# ---------------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'  # No Color

log_info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }
log_section() { echo -e "\n${BOLD}${BLUE}=== $* ===${NC}"; }

# ---------------------------------------------------------------------------
# 3. Cleanup khi Ctrl+C hoặc script kết thúc
# ---------------------------------------------------------------------------

cleanup() {
    echo ""
    log_warn "Đang dừng tất cả services..."

    if [[ -f "$PID_FILE" ]]; then
        while IFS='=' read -r name pid; do
            if kill -0 "$pid" 2>/dev/null; then
                log_info "Dừng $name (PID $pid)..."
                kill "$pid" 2>/dev/null || true
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi

    log_info "Đã dừng tất cả."
    exit 0
}

trap cleanup INT TERM EXIT

# Xóa PID file cũ từ lần chạy trước
rm -f "$PID_FILE"

# ---------------------------------------------------------------------------
# 4. Hàm tiện ích
# ---------------------------------------------------------------------------

# Chờ một HTTP endpoint trả về 200, tối đa $WAIT_TIMEOUT giây
wait_for_http() {
    local name="$1"
    local url="$2"
    local elapsed=0

    log_info "Chờ $name sẵn sàng tại $url ..."
    while ! curl -sf "$url" -o /dev/null 2>/dev/null; do
        if [[ $elapsed -ge $WAIT_TIMEOUT ]]; then
            log_error "$name không sẵn sàng sau ${WAIT_TIMEOUT}s. Kiểm tra $LOG_DIR/"
            return 1
        fi
        sleep 3
        elapsed=$((elapsed + 3))
        echo -n "."
    done
    echo ""
    log_info "$name sẵn sàng sau ${elapsed}s."
}

# Ghi PID để cleanup sau
register_pid() {
    local name="$1"
    local pid="$2"
    echo "${name}=${pid}" >> "$PID_FILE"
}

# Kiểm tra một Python package đã cài chưa (trong venv chỉ định)
pkg_installed() {
    local pip_bin="$1"
    local pkg="$2"
    "$pip_bin" show "$pkg" &>/dev/null
}

# ---------------------------------------------------------------------------
# 5. Kiểm tra môi trường cơ bản
# ---------------------------------------------------------------------------

log_section "Kiểm tra môi trường"

cd "$PROJECT_ROOT"

# Python cho EXACT API
if [[ ! -x "$API_PYTHON" ]]; then
    log_error "Không tìm thấy Python tại $API_PYTHON"
    log_error "Tạo venv: python -m venv $PROJECT_ROOT/exact && $PROJECT_ROOT/exact/bin/pip install -r requirements-api.txt -e ."
    exit 1
fi
log_info "API Python: $($API_PYTHON --version) ($API_PYTHON)"

# GPU check (cảnh báo nếu không có, không abort)
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "?")
    log_info "GPU: $GPU_NAME ($GPU_MEM)"
else
    log_warn "nvidia-smi không có — vLLM sẽ chạy trên CPU (rất chậm)"
fi

# ---------------------------------------------------------------------------
# 6. Cài đặt package cho EXACT API nếu thiếu
# ---------------------------------------------------------------------------

log_section "Kiểm tra package EXACT API"

MISSING_API=0
for pkg in fastapi uvicorn openai httpx pydantic_settings z3 sympy; do
    if ! pkg_installed "$API_PIP" "$pkg"; then
        log_warn "Thiếu package: $pkg"
        MISSING_API=1
    fi
done

if [[ $MISSING_API -eq 1 ]]; then
    log_info "Cài đặt requirements-api.txt và project package..."
    "$API_PIP" install --quiet -r "$PROJECT_ROOT/requirements-api.txt"
    "$API_PIP" install --quiet -e "$PROJECT_ROOT"
    log_info "Cài đặt xong."
else
    log_info "Tất cả API package đã có sẵn."
fi

# ---------------------------------------------------------------------------
# 7. Setup venv và cài đặt vLLM nếu thiếu
# ---------------------------------------------------------------------------

log_section "Kiểm tra vLLM"

# Tạo venv riêng cho vLLM nếu chưa có
if [[ ! -x "$VLLM_PYTHON" ]]; then
    log_info "Tạo venv vLLM tại $VLLM_VENV ..."
    python3 -m venv "$VLLM_VENV"
    "$VLLM_PIP" install --quiet --upgrade pip
fi

if ! pkg_installed "$VLLM_PIP" vllm; then
    log_info "Cài đặt vLLM (có thể mất vài phút)..."
    "$VLLM_PIP" install --quiet vllm
    log_info "Cài đặt vLLM xong."
else
    VLLM_VERSION=$("$VLLM_PIP" show vllm 2>/dev/null | grep ^Version | awk '{print $2}')
    log_info "vLLM $VLLM_VERSION đã có sẵn."
fi

# ---------------------------------------------------------------------------
# 8. Kiểm tra và cài cloudflared
# ---------------------------------------------------------------------------

log_section "Kiểm tra cloudflared"

CLOUDFLARED_BIN=""
if command -v cloudflared &>/dev/null; then
    CLOUDFLARED_BIN="$(command -v cloudflared)"
    log_info "cloudflared: $($CLOUDFLARED_BIN --version 2>&1 | head -1)"
else
    # Tải cloudflared binary nếu chưa có
    CF_BIN="$PROJECT_ROOT/.venv-vllm/bin/cloudflared"
    if [[ ! -x "$CF_BIN" ]]; then
        log_info "Tải cloudflared binary..."
        ARCH=$(uname -m)
        case "$ARCH" in
            x86_64)  CF_ARCH="amd64" ;;
            aarch64) CF_ARCH="arm64" ;;
            *)        log_error "Architecture không hỗ trợ: $ARCH"; exit 1 ;;
        esac
        CF_URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${CF_ARCH}"
        curl -fsSL "$CF_URL" -o "$CF_BIN"
        chmod +x "$CF_BIN"
        log_info "Đã tải cloudflared vào $CF_BIN"
    fi
    CLOUDFLARED_BIN="$CF_BIN"
    log_info "cloudflared: $($CLOUDFLARED_BIN --version 2>&1 | head -1)"
fi

# ---------------------------------------------------------------------------
# 9. Start vLLM
# ---------------------------------------------------------------------------

log_section "Khởi động vLLM"
log_info "Model: $VLLM_MODEL"
log_info "Log: $VLLM_LOG"

# Build command array
VLLM_CMD=(
    "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server
    --model "$VLLM_MODEL"
    --served-model-name "$VLLM_SERVED_MODEL_NAME"
    --host "$VLLM_HOST"
    --port "$VLLM_PORT"
    --dtype "$VLLM_DTYPE"
    --max-model-len "$VLLM_MAX_MODEL_LEN"
    --gpu-memory-utilization "$VLLM_GPU_MEM"
    --enable-prefix-caching
    --api-key "$VLLM_API_KEY"
)

# Thêm quantization nếu được set
if [[ -n "$VLLM_QUANTIZATION" ]]; then
    VLLM_CMD+=(--quantization "$VLLM_QUANTIZATION")
fi

"${VLLM_CMD[@]}" >> "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
register_pid "vllm" "$VLLM_PID"
log_info "vLLM started (PID $VLLM_PID)"

# Chờ vLLM healthy trước khi start API
wait_for_http "vLLM" "http://${VLLM_HOST}:${VLLM_PORT}/health" || {
    log_error "vLLM không start được. Xem log: $VLLM_LOG"
    exit 1
}

# ---------------------------------------------------------------------------
# 10. Start EXACT API (uvicorn)
# ---------------------------------------------------------------------------

log_section "Khởi động EXACT API"
log_info "Endpoint: http://${API_HOST}:${API_PORT}"
log_info "Log: $API_LOG"

# Override LLM base URL để trỏ vào vLLM local (không dùng giá trị trong .env
# vì .env local có thể trỏ ra remote URL)
EXACT_LLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1" \
EXACT_LLM_API_KEY="$VLLM_API_KEY" \
EXACT_LLM_MODEL="$VLLM_MODEL" \
EXACT_API_HOST="$API_HOST" \
EXACT_API_PORT="$API_PORT" \
PYTHONPATH="$PROJECT_ROOT/src" \
"$API_PYTHON" -m uvicorn exact.app.main:app \
    --host "$API_HOST" \
    --port "$API_PORT" \
    --log-level info \
    >> "$API_LOG" 2>&1 &
API_PID=$!
register_pid "api" "$API_PID"
log_info "EXACT API started (PID $API_PID)"

# Chờ API ready
wait_for_http "EXACT API" "http://127.0.0.1:${API_PORT}/health" || {
    log_error "EXACT API không start được. Xem log: $API_LOG"
    exit 1
}

# ---------------------------------------------------------------------------
# 11. Start Cloudflare Tunnel
# ---------------------------------------------------------------------------

log_section "Khởi động Cloudflare Tunnel"
log_info "Log: $TUNNEL_LOG"

if [[ -n "$CLOUDFLARE_TUNNEL_NAME" ]]; then
    # Named tunnel — cần đã setup trước: cloudflared tunnel create <name>
    log_info "Dùng named tunnel: $CLOUDFLARE_TUNNEL_NAME"
    "$CLOUDFLARED_BIN" tunnel run "$CLOUDFLARE_TUNNEL_NAME" \
        >> "$TUNNEL_LOG" 2>&1 &
else
    # Quick tunnel — tạo URL ngẫu nhiên, in ra để biết URL
    log_warn "CLOUDFLARE_TUNNEL_NAME chưa set → dùng quick tunnel (URL random mỗi lần)"
    log_info "URL sẽ xuất hiện trong log: $TUNNEL_LOG"
    "$CLOUDFLARED_BIN" tunnel --url "http://localhost:${API_PORT}" \
        >> "$TUNNEL_LOG" 2>&1 &
fi

TUNNEL_PID=$!
register_pid "cloudflared" "$TUNNEL_PID"
log_info "cloudflared started (PID $TUNNEL_PID)"

# Với quick tunnel: đợi URL xuất hiện trong log và in ra
if [[ -z "$CLOUDFLARE_TUNNEL_NAME" ]]; then
    log_info "Đang chờ quick tunnel URL..."
    for i in $(seq 1 20); do
        TUNNEL_URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" 2>/dev/null | head -1 || true)
        if [[ -n "$TUNNEL_URL" ]]; then
            echo -e "\n${GREEN}${BOLD}>>> Tunnel URL: $TUNNEL_URL${NC}\n"
            break
        fi
        sleep 2
    done
fi

# ---------------------------------------------------------------------------
# 12. Tất cả đã chạy — theo dõi và giữ script sống
# ---------------------------------------------------------------------------

log_section "Tất cả services đã khởi động"
echo ""
echo -e "  ${GREEN}vLLM:${NC}        http://${VLLM_HOST}:${VLLM_PORT}/v1"
echo -e "  ${GREEN}EXACT API:${NC}   http://localhost:${API_PORT}"
echo -e "  ${GREEN}Tunnel:${NC}      $TUNNEL_URL (xem $TUNNEL_LOG)"
echo ""
echo -e "  Logs: $LOG_DIR/"
echo -e "  Ctrl+C để dừng tất cả.\n"

# Monitor: nếu bất kỳ process con nào chết, cảnh báo nhưng không abort
while true; do
    for entry in "vLLM=$VLLM_PID" "API=$API_PID" "Tunnel=$TUNNEL_PID"; do
        name="${entry%%=*}"
        pid="${entry##*=}"
        if ! kill -0 "$pid" 2>/dev/null; then
            log_warn "$name (PID $pid) đã thoát bất ngờ. Xem log."
        fi
    done
    sleep 10
done
