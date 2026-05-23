#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${OLLAMA_MODEL_ID:-qwen3.6:35b-a3b-q8_0}"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
OLLAMA_MAX_QUEUE="${OLLAMA_MAX_QUEUE:-512}"
OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-${MAX_MODEL_LEN:-32768}}"
LOG_FILE="${OLLAMA_LOG_FILE:-/tmp/ollama-${MODEL_ID//[:\/]/_}.log}"
: >"${LOG_FILE}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama command not found" >&2
  exit 1
fi

if ! curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
  echo "Starting ollama serve on ${OLLAMA_HOST}..." | tee -a "${LOG_FILE}"
  nohup env \
    OLLAMA_HOST="0.0.0.0:${OLLAMA_PORT}" \
    OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL}" \
    OLLAMA_MAX_QUEUE="${OLLAMA_MAX_QUEUE}" \
    OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH}" \
    ollama serve >"${LOG_FILE}" 2>&1 &
  sleep 3
fi

for _ in $(seq 1 60); do
  if curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

ollama pull "${MODEL_ID}"
