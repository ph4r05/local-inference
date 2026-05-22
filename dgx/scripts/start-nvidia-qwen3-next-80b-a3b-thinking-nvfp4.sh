#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODEL_ID="nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4"
start_vllm_model "${MODEL_ID}" \
  --reasoning-parser qwen3 \
  --kv-cache-dtype fp8 \
  "$@"
wait_for_vllm "${MODEL_ID}"
