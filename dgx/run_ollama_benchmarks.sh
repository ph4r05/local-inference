#!/usr/bin/env bash
set -euo pipefail

BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SUITE_CASES="${SUITE_CASES:-1:1,1:2,1:4,2:4,4:8,8:16,16:32,32:64}"
PROMPT_WORDS="${PROMPT_WORDS:-256}"
MAX_TOKENS="${MAX_TOKENS:-128}"
PROMPT_MODE="${PROMPT_MODE:-unique}"
WARMUP="${WARMUP:-1}"
RESOURCE_INTERVAL="${RESOURCE_INTERVAL:-1.0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-${BENCH_ROOT}/results/${RUN_ID}}"
MAX_HOST_RAM_PCT="${MAX_HOST_RAM_PCT:-0}"
MAX_SWAP_USED_GIB="${MAX_SWAP_USED_GIB:-4.0}"
MAX_SWAP_GROWTH_GIB="${MAX_SWAP_GROWTH_GIB:-1.0}"
GUARD_GRACE_SAMPLES="${GUARD_GRACE_SAMPLES:-2}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
OLLAMA_MAX_QUEUE="${OLLAMA_MAX_QUEUE:-512}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-${MAX_MODEL_LEN:-32768}}"

MODELS=(
  "ollama-qwen3.6-35b-a3b-q8_0|qwen3.6:35b-a3b-q8_0|${BENCH_ROOT}/scripts/start-ollama-qwen3.6-35b-a3b-q8_0.sh"
)

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
    32768) printf 'ctx32k' ;;
    65536) printf 'ctx64k' ;;
    131072) printf 'ctx128k' ;;
    262144) printf 'ctx256k' ;;
    524288) printf 'ctx512k' ;;
    *)
      if (( value % 1024 == 0 )); then
        printf 'ctx%dk' "$((value / 1024))"
      else
        printf 'ctx%s' "${value}"
      fi
      ;;
  esac
}

build_run_config_json() {
  local slug="$1"
  local model_id="$2"
  local start_script="$3"
  local suite_cases="$4"

  python3 "${BENCH_ROOT}/scripts/make_run_config.py" \
    --set backend=ollama \
    --set slug="${slug}" \
    --set model_id="${model_id}" \
    --set start_script="${start_script}" \
    --set server_url="${OLLAMA_BASE_URL}" \
    --set context_length="${CONTEXT_LENGTH}" \
    --set ollama_num_parallel="${OLLAMA_NUM_PARALLEL}" \
    --set ollama_max_queue="${OLLAMA_MAX_QUEUE}" \
    --set suite_cases="${suite_cases}" \
    --set prompt_mode="${PROMPT_MODE}" \
    --set prompt_words="${PROMPT_WORDS}" \
    --set max_tokens="${MAX_TOKENS}" \
    --set warmup="${WARMUP}" \
    --set resource_interval="${RESOURCE_INTERVAL}" \
    --set max_host_ram_pct="${MAX_HOST_RAM_PCT}" \
    --set max_swap_used_gib="${MAX_SWAP_USED_GIB}" \
    --set max_swap_growth_gib="${MAX_SWAP_GROWTH_GIB}" \
    --set guard_grace_samples="${GUARD_GRACE_SAMPLES}"
}

for spec in "${MODELS[@]}"; do
  IFS="|" read -r slug model_id start_script <<<"${spec}"
  ctx_label="$(context_label "${CONTEXT_LENGTH}")"
  if ! should_run "${slug}"; then
    continue
  fi

  echo "=== ${slug} :: ${model_id} ==="
  run_config_json="$(build_run_config_json "${slug}" "${model_id}" "${start_script}" "${SUITE_CASES}")"
  run_id="$(printf '%s' "${run_config_json}" | sha256sum | cut -c1-6)"

  output_dir="${RESULTS_ROOT}/${slug}-${ctx_label}-${run_id}-full"
  mkdir -p "${output_dir}"
  printf '%s\n' "${run_config_json}" > "${output_dir}/run-config.json"
  export OLLAMA_LOG_FILE="${output_dir}/startup.log"

  "${start_script}"

  cat > "${output_dir}/run-config.md" <<EOF
# Run Config

- Run ID: ${run_id}
- Backend: ollama
- Slug: ${slug}
- Model: ${model_id}
- Server URL: ${OLLAMA_BASE_URL}
- Run config JSON: ${output_dir}/run-config.json
- Context length / num_ctx: ${CONTEXT_LENGTH}
- OLLAMA_NUM_PARALLEL: ${OLLAMA_NUM_PARALLEL}
- OLLAMA_MAX_QUEUE: ${OLLAMA_MAX_QUEUE}
- Suite cases requested: ${SUITE_CASES}
- Prompt mode: ${PROMPT_MODE}
- Prompt words: ${PROMPT_WORDS}
- Max output tokens: ${MAX_TOKENS}
- Warmup requests per case: ${WARMUP}
- Host RAM guard percent: ${MAX_HOST_RAM_PCT}
- Swap used guard GiB: ${MAX_SWAP_USED_GIB}
- Swap growth guard GiB: ${MAX_SWAP_GROWTH_GIB}
EOF

  python3 "${BENCH_ROOT}/benchmark_ollama.py" \
    --base-url "${OLLAMA_BASE_URL}" \
    --model "${model_id}" \
    --suite \
    --suite-cases "${SUITE_CASES}" \
    --prompt-mode "${PROMPT_MODE}" \
    --prompt-words "${PROMPT_WORDS}" \
    --max-tokens "${MAX_TOKENS}" \
    --warmup "${WARMUP}" \
    --no-per-request-details \
    --resource-interval "${RESOURCE_INTERVAL}" \
    --process-match "ollama serve" \
    --max-host-ram-pct "${MAX_HOST_RAM_PCT}" \
    --max-swap-used-gib "${MAX_SWAP_USED_GIB}" \
    --max-swap-growth-gib "${MAX_SWAP_GROWTH_GIB}" \
    --guard-grace-samples "${GUARD_GRACE_SAMPLES}" \
    --ollama-num-parallel "${OLLAMA_NUM_PARALLEL}" \
    --ollama-max-queue "${OLLAMA_MAX_QUEUE}" \
    --context-length "${CONTEXT_LENGTH}" \
    --run-config-path "${output_dir}/run-config.json" \
    --output-dir "${output_dir}"

done

echo "Results written to ${RESULTS_ROOT}"
