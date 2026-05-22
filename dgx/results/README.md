# Benchmark Results

This directory contains the benchmark artifacts gathered against the local vLLM server on port `8000`.

## Full Benchmarks

| Model | Context | Status | Notes |
| --- | --- | --- | --- |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | `ctx32k` | done | Full suite completed. |
| `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` | `ctx32k` | done | Full suite completed. |
| `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` | `ctx128k` | done | Full suite completed. |
| `Qwen/Qwen3.6-27B-FP8` | `ctx32k` | done | Full suite completed. |
| `RedHatAI/Qwen3.6-35B-A3B-NVFP4` | `ctx32k` | done | Full suite completed. |
| `RedHatAI/Qwen3.6-35B-A3B-NVFP4` | `ctx128k` | done | Full suite completed. |
| `nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4` | `ctx32k` | done | Full suite completed. |
| `nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4` | `ctx128k` | done | Full suite completed. |
| `nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4` | `ctx256k` | done | Full suite completed. vLLM reported max full-context concurrency `20.30x`. |

## Validation Runs

| Model | Context | Status | Notes |
| --- | --- | --- | --- |
| `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` | `ctx32k` | done | Validation only. |
| `Qwen/Qwen3.6-27B-FP8` | `ctx32k` | done | Validation only. |
| `RedHatAI/Qwen3.6-35B-A3B-NVFP4` | `ctx32k` | done | Validation only. |
| `nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4` | `ctx32k` | done | Validation only. |

## Startup / Fit Failures

| Model | Context | Status | Notes |
| --- | --- | --- | --- |
| `Qwen/Qwen3-Next-80B-A3B-Thinking` | `ctx32k` | failed | OOM during startup; FlashInfer retry and reduced `ctx8192` also failed. |
| `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` | `ctx256k` | failed | Failed during startup/autotune/module load before `/v1/models` was ready. vLLM reported max full-context concurrency `30.47x`. |

## Notes

- Result folders are named with `ctx32k`, `ctx128k`, or `ctx256k` so the context window is explicit in the path.
- The benchmark suite records:
  - aggregate prompt and decode tok/s
  - per-request decode tok/s
  - host CPU, host RAM, swap, and process CPU/RSS
  - vLLM-reported maximum full-context concurrency from startup logs
- For long-context runs, the suite filters out cases above a safety fraction of the emitted vLLM concurrency limit so it does not overload the server.

