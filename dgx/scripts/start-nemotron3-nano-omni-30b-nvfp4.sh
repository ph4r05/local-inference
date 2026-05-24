#!/usr/bin/env bash
set -euo pipefail

export VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.20.0}"
export DOCKER_ENTRYPOINT="${DOCKER_ENTRYPOINT:-}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export VLLM_READY_SLEEP="${VLLM_READY_SLEEP:-10}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODEL_ID="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
start_vllm_model "${MODEL_ID}" \
  --limit-mm-per-prompt '{"video": 1, "image": 1, "audio": 1}' \
  --media-io-kwargs '{"video": {"fps": 2, "num_frames": 256}}' \
  --allowed-local-media-path / \
  --enable-prefix-caching \
  --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --kv-cache-dtype fp8 \
  "$@"
wait_for_vllm "${MODEL_ID}"
