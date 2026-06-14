#!/usr/bin/env bash
# One-shot environment setup for the EXACT vLLM serving stack.
#
# Steps:
#   1. create a venv named `exact` when it does not already exist
#   2. activate it
#   3. install both root requirements files (requirements.txt + requirements-serve.txt)
#   4. install vllm==0.8.5
#   5. refresh the dynamic linker cache (sudo /sbin/ldconfig)
#   6. remove FlashInfer / TVM-FFI packages whose Torch ABI mismatch crashes
#      vLLM on import
#
# Run from the project root:  bash setup.sh

set -euo pipefail

# Work from this script's directory (the project root).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Never let a stray ~/.local install shadow the venv.
export PYTHONNOUSERSITE=1

VENV="exact"

# --- pick interpreter -----------------------------------------------------
if command -v python3.12 >/dev/null 2>&1; then
  PYTHON=python3.12
else
  PYTHON=python3
fi
echo "[setup] interpreter: $("$PYTHON" --version) ($PYTHON)"

# --- 1. create venv -------------------------------------------------------
if [[ -d "$VENV" ]]; then
  echo "[setup] keeping existing venv: $VENV"
else
  echo "[setup] creating venv: $VENV"
  "$PYTHON" -m venv "$VENV"
fi

# --- 2. activate ----------------------------------------------------------
# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "[setup] activated: $(python --version) at $(command -v python)"
python -m pip install --upgrade pip setuptools wheel

# --- 3. install both root requirements files ------------------------------
echo "[setup] installing requirements.txt"
python -m pip install -r requirements.txt
echo "[setup] installing requirements-serve.txt"
python -m pip install --no-cache-dir -r requirements-serve.txt

# --- 4. pin vLLM 0.8.5 ----------------------------------------------------
echo "[setup] installing vllm==0.8.5"
python -m pip install --no-cache-dir "vllm==0.8.5"

# --- 5. refresh dynamic linker cache --------------------------------------
echo "[setup] running ldconfig"
sudo /sbin/ldconfig

# --- 5b. libcuda.so dev symlink (required by Triton JIT on first inference) -
# The NVIDIA driver ships libcuda.so.1 but not the plain libcuda.so that the
# linker needs. Without it, Triton's cuda_utils compilation fails at runtime.
echo "[setup] ensuring libcuda.so symlink for Triton JIT"
LIBCUDA_TARGET=""
for candidate in \
    /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
    /usr/local/cuda/lib64/libcuda.so.1 \
    /usr/local/cuda/lib64/stubs/libcuda.so; do
  if [[ -f "$candidate" ]]; then
    LIBCUDA_TARGET="$candidate"
    break
  fi
done
if [[ -n "$LIBCUDA_TARGET" ]]; then
  sudo ln -sf "$LIBCUDA_TARGET" /usr/lib/x86_64-linux-gnu/libcuda.so
  echo "[setup] libcuda.so -> $LIBCUDA_TARGET"
else
  echo "[setup] WARNING: libcuda.so not found; Triton JIT will fail on first inference"
fi

# --- 6. remove FlashInfer / TVM-FFI (Torch ABI mismatch crashes vLLM) -----
echo "[setup] removing flashinfer / tvm-ffi packages"
python -m pip uninstall -y \
  flashinfer flashinfer-python tvm-ffi apache-tvm-ffi torch-c-dlpack-ext || true

# --- 7. cloudflared tunnel (exposes the EXACT API at api.iamphuckhang.dev) -
echo "[setup] installing cloudflared"
if ! command -v cloudflared >/dev/null 2>&1; then
  curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg \
    | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
  echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main" \
    | sudo tee /etc/apt/sources.list.d/cloudflared.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y cloudflared
else
  echo "[setup] cloudflared already installed: $(cloudflared --version)"
fi

CF_DIR="$HOME/.cloudflared"
CF_TUNNEL_ID="8aba8569-c29f-48fb-acc9-41f0dfa6463a"
mkdir -p "$CF_DIR"

echo "[setup] writing tunnel credentials: $CF_DIR/$CF_TUNNEL_ID.json"
cat > "$CF_DIR/$CF_TUNNEL_ID.json" <<'JSON'
{
    "AccountTag": "d65d503a55c7aaa86cb279e57131929c",
    "TunnelSecret": "Co7tzVUuKaWulSpFbKdeiltEuvhwUcf/qrKF1s5dHoA=",
    "TunnelID": "8aba8569-c29f-48fb-acc9-41f0dfa6463a",
    "Endpoint": ""
}
JSON
chmod 600 "$CF_DIR/$CF_TUNNEL_ID.json"

echo "[setup] writing config: $CF_DIR/config.yml"
cat > "$CF_DIR/config.yml" <<YAML
# cloudflared tunnel "vllm"
tunnel: $CF_TUNNEL_ID
credentials-file: $CF_DIR/$CF_TUNNEL_ID.json

ingress:
  - hostname: api.iamphuckhang.dev
    service: http://127.0.0.1:8080
  - service: http_status:404
YAML

echo "[setup] cloudflared config ready."

# --- 8. Copy example env into .env -----------------------------------------
ENV_FILE="$ROOT/.env"
cp -f "$ROOT/.env.example" "$ENV_FILE"

if [[ -f "$ENV_FILE" ]]; then
  echo "[setup] Success: .env created at $ENV_FILE."
else
  echo "[setup] WARNING: .env not found at $ENV_FILE — copy it to the project root first"
fi

# --- 9. Start cloudflared tunnel in background ---
mkdir -p "$ROOT/logs" "$ROOT/outputs/logs"
cloudflared tunnel run vllm >> "$ROOT/logs/cloudflared.log" 2>&1 &
echo "[setup] cloudflared started in background (PID $!) — log: logs/cloudflared.log"

# --- 9. Start EXACT API stack in background (uvicorn --reload watches src/) --
echo "[setup] launching EXACT API stack..."
UVICORN_RELOAD=1 bash "$ROOT/scripts/serve_exact_api.sh" >> "$ROOT/logs/serve_launcher.log" 2>&1 &
echo "[setup] serve stack started in background (PID $!) — launcher log: logs/serve_launcher.log"

# --- 10. Git watch loop: poll GitHub every 30s, pull on new commit -----------
echo "[setup] starting GitHub watch loop (interval: 30s)..."
(
  BRANCH="main"
  cd "$ROOT"
  while true; do
    git fetch origin "$BRANCH" -q 2>/dev/null
    LOCAL=$(git rev-parse HEAD 2>/dev/null)
    REMOTE=$(git rev-parse "origin/$BRANCH" 2>/dev/null)
    if [[ "$LOCAL" != "$REMOTE" ]]; then
      echo "[watch] $(date '+%H:%M:%S') new commit $REMOTE — pulling..."
      git pull origin "$BRANCH"
      echo "[watch] pull done. uvicorn --reload will restart API automatically."
    fi
    sleep 30
  done
) >> "$ROOT/logs/watch.log" 2>&1 &
echo "[setup] watch loop started (PID $!) — log: logs/watch.log"

# Touch so tail -f works even before services write their first line.
touch "$ROOT/logs/watch.log" "$ROOT/logs/parser.log" \
      "$ROOT/logs/vllm.log" "$ROOT/outputs/logs/api.log"

echo ""
echo "[setup] done. Venv: source $VENV/bin/activate"
echo "[setup] === Tailing live logs — Ctrl+C stops tailing; services keep running ==="
echo ""
exec tail -f \
  "$ROOT/logs/watch.log" \
  "$ROOT/logs/parser.log" \
  "$ROOT/logs/vllm.log" \
  "$ROOT/outputs/logs/api.log"
