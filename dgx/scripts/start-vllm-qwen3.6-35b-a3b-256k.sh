#!/usr/bin/env bash
set -euo pipefail

export VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.20.0}"
export DOCKER_ENTRYPOINT="${DOCKER_ENTRYPOINT:-}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODEL_ID="RedHatAI/Qwen3.6-35B-A3B-NVFP4"
start_vllm_model "${MODEL_ID}"   --reasoning-parser qwen3   --moe_backend flashinfer_cutlass   "$@"
wait_for_vllm "${MODEL_ID}"
