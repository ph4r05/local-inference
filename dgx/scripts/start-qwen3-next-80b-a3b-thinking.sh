#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODEL_ID="Qwen/Qwen3-Next-80B-A3B-Thinking"
start_vllm_model "${MODEL_ID}" "$@"
wait_for_vllm "${MODEL_ID}"
