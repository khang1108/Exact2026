#!/usr/bin/env bash
# Serve the dedicated Type 1 parser model with vLLM.
#
# The parser handles short structured-output requests such as the 14 premises
# in one logic problem. The EXACT application sends those requests
# concurrently; vLLM continuously batches them on the GPU.
#
# By default the server binds to localhost for safety. Set PARSER_HOST=0.0.0.0
# only when the application runs on another machine, and restrict port 8001
# with a firewall to the application's private IP.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${EXACT_ENV_FILE:-$PROJECT_ROOT/.env}"

# Load deployment and client settings from the project's trusted local .env.
# PARSER_* shell variables take precedence for one-off launch overrides.
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  ENV_STATUS="Loaded parser configuration from $ENV_FILE"
else
  ENV_STATUS="No environment file found at $ENV_FILE; using defaults and shell overrides"
fi

# Source model loaded by vLLM and stable served name used by ParserClient.
PARSER_MODEL="${PARSER_MODEL:-${EXACT_TYPE1_PARSER_SOURCE_MODEL:-Qwen/Qwen3-1.7B}}"
PARSER_SERVED_MODEL_NAME="${PARSER_SERVED_MODEL_NAME:-${EXACT_TYPE1_PARSER_MODEL:-type1-parser}}"

# Network settings for the parser-only vLLM endpoint.
PARSER_HOST="${PARSER_HOST:-${EXACT_TYPE1_PARSER_SERVER_HOST:-127.0.0.1}}"
PARSER_PORT="${PARSER_PORT:-${EXACT_TYPE1_PARSER_SERVER_PORT:-8001}}"
PARSER_API_KEY="${PARSER_API_KEY:-${EXACT_TYPE1_PARSER_API_KEY:-exact-parser-token}}"

# Parser requests are short, so favor concurrent sequences over long context.
PARSER_MAX_MODEL_LEN="${PARSER_MAX_MODEL_LEN:-${EXACT_TYPE1_PARSER_SERVER_MAX_MODEL_LEN:-4096}}"
PARSER_MAX_NUM_SEQS="${PARSER_MAX_NUM_SEQS:-${EXACT_TYPE1_PARSER_SERVER_MAX_NUM_SEQS:-64}}"
PARSER_MAX_NUM_BATCHED_TOKENS="${PARSER_MAX_NUM_BATCHED_TOKENS:-${EXACT_TYPE1_PARSER_SERVER_MAX_NUM_BATCHED_TOKENS:-8192}}"

# Reserve only part of a 48 GB L40/A6000 when another reasoning model shares it.
PARSER_GPU_MEMORY_UTILIZATION="${PARSER_GPU_MEMORY_UTILIZATION:-${EXACT_TYPE1_PARSER_SERVER_GPU_MEMORY_UTILIZATION:-0.25}}"
PARSER_DTYPE="${PARSER_DTYPE:-${EXACT_TYPE1_PARSER_SERVER_DTYPE:-auto}}"
PARSER_QUANTIZATION="${PARSER_QUANTIZATION:-${EXACT_TYPE1_PARSER_SERVER_QUANTIZATION:-}}"
PARSER_TENSOR_PARALLEL_SIZE="${PARSER_TENSOR_PARALLEL_SIZE:-${EXACT_TYPE1_PARSER_SERVER_TENSOR_PARALLEL_SIZE:-1}}"
PARSER_DEVICE="${PARSER_DEVICE:-${EXACT_TYPE1_PARSER_SERVER_DEVICE:-auto}}"
PARSER_CPU_KVCACHE_SPACE="${PARSER_CPU_KVCACHE_SPACE:-${EXACT_TYPE1_PARSER_SERVER_CPU_KVCACHE_SPACE:-1}}"
PARSER_CPU_RESERVED_CPUS="${PARSER_CPU_RESERVED_CPUS:-${EXACT_TYPE1_PARSER_SERVER_CPU_RESERVED_CPUS:-1}}"

PARSER_VLLM_BIN="${PARSER_VLLM_BIN:-${EXACT_TYPE1_PARSER_SERVER_VLLM_BIN:-}}"

# LoRA adapter settings (optional)
PARSER_LORA_ADAPTER="${PARSER_LORA_ADAPTER:-${EXACT_TYPE1_PARSER_LORA_ADAPTER:-}}"
PARSER_LORA_MAX_RANK="${PARSER_LORA_MAX_RANK:-${EXACT_TYPE1_PARSER_LORA_MAX_RANK:-64}}"

# Eager mode: safer on virtualized GPUs (ThunderCompute etc.)
PARSER_ENFORCE_EAGER="${PARSER_ENFORCE_EAGER:-${EXACT_TYPE1_PARSER_SERVER_ENFORCE_EAGER:-1}}"
if [[ -z "$PARSER_VLLM_BIN" ]]; then
  if [[ -x "$PROJECT_ROOT/.venv-vllm-cpu/bin/vllm" ]]; then
    PARSER_VLLM_BIN="$PROJECT_ROOT/.venv-vllm-cpu/bin/vllm"
  else
    PARSER_VLLM_BIN="vllm"
  fi
fi
if ! command -v "$PARSER_VLLM_BIN" >/dev/null 2>&1; then
  echo "vLLM executable not found: $PARSER_VLLM_BIN" >&2
  echo "Install a vLLM CPU build or set EXACT_TYPE1_PARSER_SERVER_VLLM_BIN." >&2
  exit 1
fi

cmd=(
  "$PARSER_VLLM_BIN" serve "$PARSER_MODEL"
  --served-model-name "$PARSER_SERVED_MODEL_NAME"
  --host "$PARSER_HOST"
  --port "$PARSER_PORT"
  --api-key "$PARSER_API_KEY"
  --dtype "$PARSER_DTYPE"
  --max-model-len "$PARSER_MAX_MODEL_LEN"
  --max-num-seqs "$PARSER_MAX_NUM_SEQS"
  --max-num-batched-tokens "$PARSER_MAX_NUM_BATCHED_TOKENS"
  --tensor-parallel-size "$PARSER_TENSOR_PARALLEL_SIZE"
  --generation-config vllm
)

# CPU vLLM uses host RAM for KV cache and should not receive GPU-only flags.
if [[ "$PARSER_DEVICE" == "cpu" ]]; then
  export VLLM_CPU_KVCACHE_SPACE="$PARSER_CPU_KVCACHE_SPACE"
  export VLLM_CPU_NUM_OF_RESERVED_CPU="$PARSER_CPU_RESERVED_CPUS"
else
  cmd+=(--gpu-memory-utilization "$PARSER_GPU_MEMORY_UTILIZATION")
  cmd+=(--enable-prefix-caching)
fi

# Quantization is optional because supported formats depend on the model artifact.
if [[ -n "$PARSER_QUANTIZATION" ]]; then
  cmd+=(--quantization "$PARSER_QUANTIZATION")
fi

# LoRA adapter: enable only when EXACT_TYPE1_PARSER_LORA_ADAPTER is set.
# The adapter is served under the same name as the base model so that the
# ParserClient continues to work without any code changes.
if [[ -n "$PARSER_LORA_ADAPTER" ]]; then
  echo "[parser] LoRA adapter enabled: $PARSER_LORA_ADAPTER (max rank: $PARSER_LORA_MAX_RANK)"
  cmd+=(
    --enable-lora
    --lora-modules "${PARSER_SERVED_MODEL_NAME}=${PARSER_LORA_ADAPTER}"
    --max-lora-rank "$PARSER_LORA_MAX_RANK"
  )
fi

# Enforce eager mode (no CUDA graph capture) — safer on virtualised GPU hosts.
if [[ "$PARSER_ENFORCE_EAGER" == "1" ]]; then
  cmd+=(--enforce-eager)
fi

echo "$ENV_STATUS"
echo "Serving $PARSER_MODEL as $PARSER_SERVED_MODEL_NAME on $PARSER_HOST:$PARSER_PORT ($PARSER_DEVICE)"
[[ -n "$PARSER_LORA_ADAPTER" ]] && echo "  + LoRA adapter: $PARSER_LORA_ADAPTER"
exec "${cmd[@]}"
