# Startup Failure Summary

- Model: `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`
- Context window / max model length: `262144` tokens (`ctx256k`)
- Startup script: `scripts/start-nemotron3-nano-omni-30b-nvfp4.sh`
- Image: `vllm/vllm-openai:v0.20.0`
- Startup env: `MAX_MODEL_LEN=262144`, `MAX_NUM_BATCHED_TOKENS=262144`, `VLLM_START_TIMEOUT=3600`
- Important startup flags: `--gpu-memory-utilization 0.70`, `--max-num-seqs 8`, `--max-num-batched-tokens 262144`, `--enable-prefix-caching`, `--reasoning-parser nemotron_v3`, `--tool-call-parser qwen3_coder`, `--kv-cache-dtype fp8`
- Outcome: failed during startup/autotune/module load before `/v1/models` became ready.
- Docker status observed after failure: `Exited (255)`.
- vLLM emitted before failure:
  - GPU KV cache size: `1,710,912` tokens
  - Maximum concurrency for `262,144` tokens per request: `30.47x`
  - Available KV cache memory: `24.49 GiB`
- Decision: skip `ctx256k` for this model/profile to avoid risking another host lockup.

Full logs: `startup.log`.
