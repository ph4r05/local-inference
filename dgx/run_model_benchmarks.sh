#!/usr/bin/env bash
set -euo pipefail

BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${BENCH_ROOT}/scripts/common.sh"

SUITE_CASES="${SUITE_CASES:-1:1,1:2,1:4,2:4,4:8,8:16,16:32,32:64}"
PROMPT_WORDS="${PROMPT_WORDS:-256}"
MAX_TOKENS="${MAX_TOKENS:-128}"
PROMPT_MODE="${PROMPT_MODE:-unique}"
WARMUP="${WARMUP:-1}"
RESOURCE_INTERVAL="${RESOURCE_INTERVAL:-1.0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-${BENCH_ROOT}/results/${RUN_ID}}"
STOP_AFTER_EACH="${STOP_AFTER_EACH:-1}"
SKIP_CONTEXTS="${SKIP_CONTEXTS:-}"
FILTER_SUITE_BY_VLLM_MAX="${FILTER_SUITE_BY_VLLM_MAX:-1}"
CONCURRENCY_SAFETY_FRACTION="${CONCURRENCY_SAFETY_FRACTION:-0.85}"
MAX_HOST_RAM_PCT="${MAX_HOST_RAM_PCT:-96.5}"
MAX_SWAP_USED_GIB="${MAX_SWAP_USED_GIB:-4.0}"
MAX_SWAP_GROWTH_GIB="${MAX_SWAP_GROWTH_GIB:-1.0}"
GUARD_GRACE_SAMPLES="${GUARD_GRACE_SAMPLES:-2}"

MODELS=(
  "nemotron3-nano-omni-30b-a3b-reasoning-nvfp4|nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4|${BENCH_ROOT}/scripts/start-nemotron3-nano-omni-30b-nvfp4.sh"
  "qwen3.6-35b-a3b-fp8|Qwen/Qwen3.6-35B-A3B-FP8|${BENCH_ROOT}/scripts/start-qwen3.6-35b-a3b-fp8.sh"
  "qwen3.6-35b-a3b-nvfp4|Qwen/Qwen3.6-35B-A3B-NVFP4|${BENCH_ROOT}/scripts/start-qwen3.6-35b-a3b-nvfp4.sh"
  "redhatai-qwen3.6-35b-a3b-nvfp4|RedHatAI/Qwen3.6-35B-A3B-NVFP4|${BENCH_ROOT}/scripts/start-redhatai-qwen3.6-35b-a3b-nvfp4.sh"
  "qwen3-next-80b-a3b-thinking|Qwen/Qwen3-Next-80B-A3B-Thinking|${BENCH_ROOT}/scripts/start-qwen3-next-80b-a3b-thinking.sh"
  "nvidia-qwen3-next-80b-a3b-thinking-nvfp4|nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4|${BENCH_ROOT}/scripts/start-nvidia-qwen3-next-80b-a3b-thinking-nvfp4.sh"
  "qwen3.6-27b-fp8|Qwen/Qwen3.6-27B-FP8|${BENCH_ROOT}/scripts/start-qwen3.6-27b-fp8.sh"
  "inferrouter-qwen3.6-27b-nvfp4|inferRouter/Qwen3.6-27B-NVFP4|${BENCH_ROOT}/scripts/start-inferrouter-qwen3.6-27b-nvfp4.sh"
  "qwen3.6-35b-a3b|Qwen/Qwen3.6-35B-A3B|${BENCH_ROOT}/scripts/start-qwen3.6-35b-a3b.sh"
)

if [[ "${INCLUDE_SUPER:-0}" == "1" ]]; then
  MODELS=(
    "nemotron3-super-120b-nvfp4|nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4|${BENCH_ROOT}/scripts/start-nemotron3-super-120b-nvfp4.sh"
    "${MODELS[@]}"
  )
fi

if [[ "${#}" -gt 0 ]]; then
  selected=("$@")
else
  selected=()
fi

mkdir -p "${RESULTS_ROOT}"

should_run() {
  local slug="$1"
  if [[ "${#selected[@]}" -eq 0 ]]; then
    return 0
  fi
  local wanted
  for wanted in "${selected[@]}"; do
    if [[ "${wanted}" == "${slug}" ]]; then
      return 0
    fi
  done
  return 1
}

context_label() {
  local value="$1"
  case "${value}" in
    32768) printf "ctx32k" ;;
    65536) printf "ctx64k" ;;
    131072) printf "ctx128k" ;;
    262144) printf "ctx256k" ;;
    524288) printf "ctx512k" ;;
    *)
      if (( value % 1024 == 0 )); then
        printf "ctx%dk" "$((value / 1024))"
      else
        printf "ctx%s" "${value}"
      fi
      ;;
  esac
}

read_vllm_capacity() {
  local logs_file="$1"
  sed -nE 's/.*Maximum concurrency for ([0-9,]+) tokens per request: ([0-9.]+)x.*/\1 \2/p' "${logs_file}" \
    | tail -n 1 \
    | tr -d ','
}

safe_concurrency_limit() {
  local max_concurrency="$1"
  awk -v max_conc="${max_concurrency}" -v fraction="${CONCURRENCY_SAFETY_FRACTION}" 'BEGIN { value = int(max_conc * fraction); if (value < 1) value = 1; print value }'
}

filter_suite_cases() {
  local raw_cases="$1"
  local max_concurrency="$2"
  local kept=()
  local skipped=()
  local item concurrency requests_count
  IFS=',' read -ra items <<<"${raw_cases}"
  for item in "${items[@]}"; do
    item="${item//[[:space:]]/}"
    [[ -z "${item}" ]] && continue
    concurrency="${item%%:*}"
    requests_count="${item#*:}"
    if (( concurrency <= max_concurrency )); then
      kept+=("${concurrency}:${requests_count}")
    else
      skipped+=("${concurrency}:${requests_count}")
    fi
  done
  FILTERED_SUITE_CASES="$(IFS=,; printf '%s' "${kept[*]}")"
  SKIPPED_SUITE_CASES="$(IFS=,; printf '%s' "${skipped[*]}")"
}

for spec in "${MODELS[@]}"; do
  IFS='|' read -r slug model_id start_script <<<"${spec}"
  ctx_label="$(context_label "${MAX_MODEL_LEN}")"
  if [[ ",${SKIP_CONTEXTS}," == *",${ctx_label},"* ]]; then
    echo "Skipping ${slug} at ${ctx_label} due SKIP_CONTEXTS=${SKIP_CONTEXTS}"
    continue
  fi
  if ! should_run "${slug}"; then
    continue
  fi

  echo "=== ${slug} :: ${model_id} ==="
  "${start_script}"

  output_dir="${RESULTS_ROOT}/${slug}-${ctx_label}-full"
  mkdir -p "${output_dir}"
  docker logs "${CONTAINER_NAME}" > "${output_dir}/startup.log" 2>&1 || true

  vllm_capacity="$(read_vllm_capacity "${output_dir}/startup.log")"
  vllm_max_model_len=""
  vllm_max_concurrency=""
  safe_max_concurrency=""
  effective_suite_cases="${SUITE_CASES}"
  skipped_suite_cases=""
  if [[ -n "${vllm_capacity}" ]]; then
    read -r vllm_max_model_len vllm_max_concurrency <<<"${vllm_capacity}"
    safe_max_concurrency="$(safe_concurrency_limit "${vllm_max_concurrency}")"
    if [[ "${FILTER_SUITE_BY_VLLM_MAX}" == "1" ]]; then
      filter_suite_cases "${SUITE_CASES}" "${safe_max_concurrency}"
      effective_suite_cases="${FILTERED_SUITE_CASES}"
      skipped_suite_cases="${SKIPPED_SUITE_CASES}"
    fi
  fi
  if [[ -z "${effective_suite_cases}" ]]; then
    echo "No suite cases remain after vLLM max concurrency filter for ${slug}" >&2
    continue
  fi

  vllm_metadata_args=()
  if [[ -n "${vllm_max_concurrency}" ]]; then
    vllm_metadata_args+=(--vllm-max-concurrency "${vllm_max_concurrency}")
  fi
  if [[ -n "${vllm_max_model_len}" ]]; then
    vllm_metadata_args+=(--vllm-max-model-len "${vllm_max_model_len}")
  fi

  cat > "${output_dir}/run-config.md" <<EOF
# Run Config

- Slug: ${slug}
- Model: ${model_id}
- Startup script: ${start_script}
- Container image: ${VLLM_IMAGE}
- Port: ${PORT}
- Tensor parallel size: ${TENSOR_PARALLEL_SIZE}
- MAX_MODEL_LEN / served context window: ${MAX_MODEL_LEN}
- Suite cases requested: ${SUITE_CASES}
- Suite cases effective: ${effective_suite_cases}
- Suite cases skipped by vLLM max concurrency guard: ${skipped_suite_cases}
- vLLM reported max model length: ${vllm_max_model_len}
- vLLM reported max full-context concurrency: ${vllm_max_concurrency}
- Concurrency safety fraction: ${CONCURRENCY_SAFETY_FRACTION}
- Safe max benchmark concurrency: ${safe_max_concurrency}
- Host RAM guard percent: ${MAX_HOST_RAM_PCT}
- Swap used guard GiB: ${MAX_SWAP_USED_GIB}
- Swap growth guard GiB: ${MAX_SWAP_GROWTH_GIB}
- Prompt mode: ${PROMPT_MODE}
- Prompt words: ${PROMPT_WORDS}
- Max output tokens: ${MAX_TOKENS}
- Warmup requests per case: ${WARMUP}
EOF
  python3 "${BENCH_ROOT}/benchmark_vllm_openai.py" \
    --model "${model_id}" \
    --suite \
    --suite-cases "${effective_suite_cases}" \
    --prompt-mode "${PROMPT_MODE}" \
    --prompt-words "${PROMPT_WORDS}" \
    --max-tokens "${MAX_TOKENS}" \
    --warmup "${WARMUP}" \
    --no-per-request-details \
    --resource-interval "${RESOURCE_INTERVAL}" \
    --process-match vllm \
    --max-host-ram-pct "${MAX_HOST_RAM_PCT}" \
    --max-swap-used-gib "${MAX_SWAP_USED_GIB}" \
    --max-swap-growth-gib "${MAX_SWAP_GROWTH_GIB}" \
    --guard-grace-samples "${GUARD_GRACE_SAMPLES}" \
    "${vllm_metadata_args[@]}" \
    --concurrency-safety-fraction "${CONCURRENCY_SAFETY_FRACTION}" \
    --output-dir "${output_dir}"

  if [[ "${STOP_AFTER_EACH}" == "1" ]]; then
    stop_vllm_container
  fi
 done

echo "Results written to ${RESULTS_ROOT}"
