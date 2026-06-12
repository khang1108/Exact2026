#!/usr/bin/env bash
# Create or repair the dedicated GPU vLLM environment used by serve_exact_api.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VLLM_VENV="${VLLM_VENV:-$PROJECT_ROOT/.venv-vllm}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VLLM_VENV"
fi

VLLM_PYTHON="$VLLM_VENV/bin/python"

export PYTHONNOUSERSITE=1

"$VLLM_PYTHON" -m pip install --upgrade pip

# FlashInfer is optional in vLLM 0.8.5. A wheel compiled against another Torch
# ABI crashes before the engine starts, so remove it and use vLLM's native
# sampler.
"$VLLM_PYTHON" -m pip uninstall -y \
  flashinfer \
  flashinfer-python \
  tvm-ffi \
  torch-c-dlpack-ext || true

"$VLLM_PYTHON" -m pip install --no-cache-dir --force-reinstall \
  -r "$PROJECT_ROOT/requirements-serve.txt"

# Re-run removal in case an existing dependency constraint retained an optional
# FlashInfer installation during repair.
"$VLLM_PYTHON" -m pip uninstall -y \
  flashinfer \
  flashinfer-python \
  tvm-ffi \
  torch-c-dlpack-ext || true

"$VLLM_PYTHON" - <<'PY'
import importlib.util

import torch
import transformers
import vllm

print(f"vLLM: {vllm.__version__}")
print(f"Torch: {torch.__version__} CUDA: {torch.version.cuda}")
print(f"Transformers: {transformers.__version__}")
print(f"FlashInfer installed: {importlib.util.find_spec('flashinfer') is not None}")
PY

echo "Dedicated vLLM environment ready: $VLLM_VENV"
