#!/usr/bin/env bash
# One-shot environment setup for the EXACT vLLM serving stack.
#
# Steps:
#   1. create a venv named `exact`
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
  echo "[setup] removing existing venv: $VENV"
  rm -rf "$VENV"
fi
echo "[setup] creating venv: $VENV"
"$PYTHON" -m venv "$VENV"

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

# --- 6. remove FlashInfer / TVM-FFI (Torch ABI mismatch crashes vLLM) -----
echo "[setup] removing flashinfer / tvm-ffi packages"
python -m pip uninstall -y \
  flashinfer flashinfer-python tvm-ffi apache-tvm-ffi torch-c-dlpack-ext || true

echo ""
echo "[setup] done. Activate with:  source $VENV/bin/activate"
