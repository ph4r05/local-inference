#!/usr/bin/env bash
set -euo pipefail

export VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.20.0}"
export DOCKER_ENTRYPOINT="${DOCKER_ENTRYPOINT:-}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODEL_ID="inferRouter/Qwen3.6-27B-NVFP4"
REVISION="${REVISION:-main}"
start_vllm_model "${MODEL_ID}" \
  --revision "${REVISION}" \
  --reasoning-parser qwen3 \
  "$@"
wait_for_vllm "${MODEL_ID}"
