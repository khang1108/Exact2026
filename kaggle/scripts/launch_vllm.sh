#!/usr/bin/env bash
set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-$MODEL}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
DTYPE="${VLLM_DTYPE:-auto}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
QUANTIZATION="${VLLM_QUANTIZATION:-}"
API_KEY="${VLLM_API_KEY:-}"

cmd=(
  vllm serve "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --dtype "$DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --generation-config vllm
  --enable-prefix-caching
)

if [[ -n "$QUANTIZATION" ]]; then
  cmd+=(--quantization "$QUANTIZATION")
fi

if [[ -n "$API_KEY" ]]; then
  cmd+=(--api-key "$API_KEY")
fi

exec "${cmd[@]}"
