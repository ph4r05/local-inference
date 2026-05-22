#!/usr/bin/env bash
set -euo pipefail

export VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.20.0}"
export DOCKER_ENTRYPOINT="${DOCKER_ENTRYPOINT:-}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODEL_ID="RedHatAI/Qwen3.6-35B-A3B-NVFP4"
start_vllm_model "${MODEL_ID}" \
  --reasoning-parser qwen3 \
  --moe_backend flashinfer_cutlass \
  "$@"
wait_for_vllm "${MODEL_ID}"
