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

for spec in "${MODELS[@]}"; do
  IFS='|' read -r slug model_id start_script <<<"${spec}"
  ctx_label="$(context_label "${CONTEXT_LENGTH}")"
  if ! should_run "${slug}"; then
    continue
  fi

  echo "=== ${slug} :: ${model_id} ==="

  output_dir="${RESULTS_ROOT}/${slug}-${ctx_label}-full"
  mkdir -p "${output_dir}"
  export OLLAMA_LOG_FILE="${output_dir}/startup.log"

  "${start_script}"

  cat > "${output_dir}/run-config.md" <<EOF
# Run Config

- Backend: ollama
- Slug: ${slug}
- Model: ${model_id}
- Server URL: ${OLLAMA_BASE_URL}
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
    --output-dir "${output_dir}"

done

echo "Results written to ${RESULTS_ROOT}"
