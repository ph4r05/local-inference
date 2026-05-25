#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODEL_ID="nvidia/Gemma-4-31B-IT-NVFP4"
start_vllm_model "${MODEL_ID}" \
  --kv-cache-dtype fp8 \
  "$@"
wait_for_vllm "${MODEL_ID}"
