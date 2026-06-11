#!/usr/bin/env bash
# Serve the lightweight Type 1 parser through the official vLLM CPU image.
#
# This launcher is intended for local parser integration tests on machines
# without CUDA. Production GPU hosts should use scripts/serve_type1_parser.sh.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${EXACT_ENV_FILE:-$PROJECT_ROOT/.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

MODEL="${EXACT_TYPE1_PARSER_SOURCE_MODEL:-Qwen/Qwen3-0.6B}"
SERVED_MODEL="${EXACT_TYPE1_PARSER_MODEL:-type1-parser}"
PORT="${EXACT_TYPE1_PARSER_SERVER_PORT:-8001}"
API_KEY="${EXACT_TYPE1_PARSER_API_KEY:-exact-parser-token}"
DTYPE="${EXACT_TYPE1_PARSER_SERVER_DTYPE:-float}"
MAX_MODEL_LEN="${EXACT_TYPE1_PARSER_SERVER_MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${EXACT_TYPE1_PARSER_SERVER_MAX_NUM_SEQS:-4}"
MAX_BATCHED_TOKENS="${EXACT_TYPE1_PARSER_SERVER_MAX_NUM_BATCHED_TOKENS:-2048}"
CPU_KVCACHE_SPACE="${EXACT_TYPE1_PARSER_SERVER_CPU_KVCACHE_SPACE:-1}"
CPU_RESERVED_CPUS="${EXACT_TYPE1_PARSER_SERVER_CPU_RESERVED_CPUS:-1}"
IMAGE="${EXACT_TYPE1_PARSER_SERVER_CPU_IMAGE:-vllm/vllm-openai-cpu:latest-x86_64}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
DOCKER_CONTEXT="${EXACT_TYPE1_PARSER_SERVER_DOCKER_CONTEXT:-}"

mkdir -p "$HF_CACHE"

docker_cmd=(docker)
if [[ -n "$DOCKER_CONTEXT" ]]; then
  docker_cmd+=(--context "$DOCKER_CONTEXT")
fi
if ! "${docker_cmd[@]}" info >/dev/null 2>&1; then
  echo "Cannot access Docker. Start Docker Desktop or grant access to the system Docker daemon." >&2
  echo "Selected context: ${DOCKER_CONTEXT:-current Docker context}" >&2
  exit 1
fi

cmd=(
  "${docker_cmd[@]}" run --rm
  --name exact-type1-parser-cpu
  --security-opt seccomp=unconfined
  --cap-add SYS_NICE
  --shm-size=2g
  -p "$PORT:$PORT"
  -v "$HF_CACHE:/root/.cache/huggingface"
  -e "VLLM_CPU_KVCACHE_SPACE=$CPU_KVCACHE_SPACE"
  -e "VLLM_CPU_NUM_OF_RESERVED_CPU=$CPU_RESERVED_CPUS"
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  cmd+=(-e "HF_TOKEN=$HF_TOKEN")
fi

cmd+=(
  "$IMAGE"
  "$MODEL"
  --served-model-name "$SERVED_MODEL"
  --host 0.0.0.0
  --port "$PORT"
  --api-key "$API_KEY"
  --dtype "$DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
  --tensor-parallel-size 1
  --generation-config vllm
)

echo "Serving $MODEL as $SERVED_MODEL through $IMAGE on localhost:$PORT"
exec "${cmd[@]}"
