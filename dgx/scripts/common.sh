#!/usr/bin/env bash
set -euo pipefail

if [[ -f "${HOME}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${HOME}/.env"
  set +a
fi

BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-bench}"
LEGACY_CONTAINER_NAME="${LEGACY_CONTAINER_NAME:-vllm-nemotron}"
VLLM_IMAGE="${VLLM_IMAGE:-nvcr.io/nvidia/vllm:26.04-py3}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"
GPU_DEVICE="${GPU_DEVICE:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-${HOME}/.cache/tiktoken_cache}"
DOCKER_ENTRYPOINT="${DOCKER_ENTRYPOINT-__unset__}"
VLLM_START_TIMEOUT="${VLLM_START_TIMEOUT:-900}"
VLLM_READY_SLEEP="${VLLM_READY_SLEEP:-10}"
VLLM_STARTUP_LOAD_THRESHOLD="${VLLM_STARTUP_LOAD_THRESHOLD:-6}"
VLLM_STARTUP_LOAD_INTERVAL="${VLLM_STARTUP_LOAD_INTERVAL:-1}"
VLLM_STARTUP_SWAP_USED_GIB="${VLLM_STARTUP_SWAP_USED_GIB:-2}"
VLLM_DOCKER_MEMORY_LIMIT_GIB="${VLLM_DOCKER_MEMORY_LIMIT_GIB:-}"
VLLM_DOCKER_SWAP_LIMIT_GIB="${VLLM_DOCKER_SWAP_LIMIT_GIB:-}"
LOAD_WATCHDOG_PID=""
LOAD_WATCHDOG_STOP_FILE=""

safe_name() {
  printf '%s' "$1" | tr '/:[:space:]' '---' | tr -cd 'A-Za-z0-9._-'
}

stop_vllm_container() {
  stop_startup_load_watchdog
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  if [[ "${LEGACY_CONTAINER_NAME}" != "${CONTAINER_NAME}" ]]; then
    docker rm -f "${LEGACY_CONTAINER_NAME}" >/dev/null 2>&1 || true
  fi
}

start_startup_load_watchdog() {
  local threshold="${1:-${VLLM_STARTUP_LOAD_THRESHOLD}}"
  local interval="${2:-${VLLM_STARTUP_LOAD_INTERVAL}}"
  local swap_limit_gib="${3:-${VLLM_STARTUP_SWAP_USED_GIB}}"
  [[ -n "${threshold}" ]] || return 0
  awk -v value="${threshold}" 'BEGIN { exit !(value + 0 > 0) }' || return 0
  if [[ -n "${LOAD_WATCHDOG_PID}" ]] && kill -0 "${LOAD_WATCHDOG_PID}" >/dev/null 2>&1; then
    return 0
  fi

  LOAD_WATCHDOG_STOP_FILE="$(mktemp)"
  (
    while [[ ! -f "${LOAD_WATCHDOG_STOP_FILE}" ]]; do
      current_load="$(awk '{print $1}' /proc/loadavg)"
      current_swap="$(awk '/SwapTotal:/ { total=$2 } /SwapFree:/ { free=$2 } END { if (total == "") { print "" } else { printf "%.0f", (total - free) / 1024 / 1024 } }' /proc/meminfo)"
      if awk -v current="${current_load}" -v limit="${threshold}" 'BEGIN { exit !(current + 0 > limit + 0) }'; then
        echo "Startup load watchdog tripped: loadavg=${current_load} > ${threshold}; killing ${CONTAINER_NAME}" >&2
        docker kill "${CONTAINER_NAME}" >/dev/null 2>&1 || true
        exit 0
      fi
      if [[ -n "${current_swap}" ]] && awk -v current="${current_swap}" -v limit="${swap_limit_gib}" 'BEGIN { exit !(current + 0 > limit + 0) }'; then
        echo "Startup swap watchdog tripped: swap=${current_swap} GiB > ${swap_limit_gib} GiB; killing ${CONTAINER_NAME}" >&2
        docker kill "${CONTAINER_NAME}" >/dev/null 2>&1 || true
        exit 0
      fi
      sleep "${interval}"
    done
  ) &
  LOAD_WATCHDOG_PID="$!"
}

stop_startup_load_watchdog() {
  if [[ -n "${LOAD_WATCHDOG_STOP_FILE}" ]]; then
    rm -f "${LOAD_WATCHDOG_STOP_FILE}" >/dev/null 2>&1 || true
    LOAD_WATCHDOG_STOP_FILE=""
  fi
  if [[ -n "${LOAD_WATCHDOG_PID}" ]]; then
    kill "${LOAD_WATCHDOG_PID}" >/dev/null 2>&1 || true
    wait "${LOAD_WATCHDOG_PID}" >/dev/null 2>&1 || true
    LOAD_WATCHDOG_PID=""
  fi
}

wait_for_vllm() {
  local expected_model="$1"
  local deadline=$((SECONDS + VLLM_START_TIMEOUT))
  local url="http://127.0.0.1:${PORT}/v1/models"
  start_startup_load_watchdog

  while (( SECONDS < deadline )); do
    if ! docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" >/dev/null 2>&1; then
      stop_startup_load_watchdog
      echo "Container ${CONTAINER_NAME} is not present while waiting for ${expected_model}" >&2
      docker logs --tail 200 "${CONTAINER_NAME}" >&2 || true
      return 1
    fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || true)" != "true" ]]; then
      stop_startup_load_watchdog
      echo "Container ${CONTAINER_NAME} exited while waiting for ${expected_model}" >&2
      docker logs --tail 200 "${CONTAINER_NAME}" >&2 || true
      return 1
    fi
    if body="$(curl -fsS "${url}" 2>/dev/null)"; then
      if [[ "${body}" == *"${expected_model}"* ]]; then
        stop_startup_load_watchdog
        sleep "${VLLM_READY_SLEEP}"
        return 0
      fi
    fi
    sleep 5
  done

  stop_startup_load_watchdog
  echo "Timed out waiting for ${expected_model} at ${url}" >&2
  docker logs --tail 200 "${CONTAINER_NAME}" >&2 || true
  return 1
}

start_vllm_model() {
  local model_id="$1"
  shift
  local extra_args=("$@")
  local extra_env_args=()
  local docker_memory_args=()
  if [[ -n "${VLLM_DOCKER_MEMORY_LIMIT_GIB}" ]]; then
    docker_memory_args+=(--memory "${VLLM_DOCKER_MEMORY_LIMIT_GIB}g")
    if [[ -n "${VLLM_DOCKER_SWAP_LIMIT_GIB}" ]]; then
      local docker_memory_swap_gib
      docker_memory_swap_gib="$(awk -v mem="${VLLM_DOCKER_MEMORY_LIMIT_GIB}" -v swap="${VLLM_DOCKER_SWAP_LIMIT_GIB}" 'BEGIN { printf "%.0f", mem + swap }')"
      docker_memory_args+=(--memory-swap "${docker_memory_swap_gib}g")
    fi
  fi
  if [[ -n "${EXTRA_DOCKER_ENV:-}" ]]; then
    local env_name
    for env_name in ${EXTRA_DOCKER_ENV}; do
      extra_env_args+=("-e" "${env_name}=${!env_name:-}")
    done
  fi

  mkdir -p "${HF_CACHE_DIR}" "${TIKTOKEN_CACHE_DIR}"
  stop_vllm_container

  local entrypoint_args=()
  if [[ "${DOCKER_ENTRYPOINT}" != "__unset__" ]]; then
    entrypoint_args=(--entrypoint "${DOCKER_ENTRYPOINT}")
  fi

  docker run -d \
    --name "${CONTAINER_NAME}" \
    "${entrypoint_args[@]}" \
    --gpus "device=${GPU_DEVICE}" \
    --ipc=host \
    --shm-size=16g \
    -p "${PORT}:${PORT}" \
    -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VALUE}" \
    -e "HF_TOKEN=${HF_TOKEN:-}" \
    -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-}" \
    "${extra_env_args[@]}" \
    -e "TIKTOKEN_CACHE_DIR=/root/.cache/tiktoken_cache" \
    -e "TIKTOKEN_RS_CACHE_DIR=/root/.cache/tiktoken_cache" \
    -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
    -v "${TIKTOKEN_CACHE_DIR}:/root/.cache/tiktoken_cache" \
    "${VLLM_IMAGE}" \
    python3 -m vllm.entrypoints.openai.api_server \
      --model "${model_id}" \
      --host "${HOST}" \
      --port "${PORT}" \
      --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
      --trust-remote-code \
      --max-model-len "${MAX_MODEL_LEN}" \
      "${extra_args[@]}"
}
