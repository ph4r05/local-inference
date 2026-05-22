#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODEL_ID="Qwen/Qwen3.6-35B-A3B-NVFP4"
start_vllm_model "${MODEL_ID}" \
  --reasoning-parser qwen3 \
  "$@"
wait_for_vllm "${MODEL_ID}"
