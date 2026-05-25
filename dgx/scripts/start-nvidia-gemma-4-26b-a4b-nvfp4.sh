#!/usr/bin/env bash
set -euo pipefail

export VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.20.0}"
export DOCKER_ENTRYPOINT="${DOCKER_ENTRYPOINT:-}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODEL_ID="nvidia/Gemma-4-26B-A4B-NVFP4"
start_vllm_model "${MODEL_ID}" \
  --kv-cache-dtype fp8 \
  "$@"
wait_for_vllm "${MODEL_ID}"
