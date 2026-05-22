# Qwen3 Next 80B A3B Thinking FP8 - Failure Report

## Model

- Model: `Qwen/Qwen3-Next-80B-A3B-Thinking`
- Intended benchmark role: Qwen Next 80B Thinking baseline. No suitable NVFP4 variant found during lookup.
- Image: `nvcr.io/nvidia/vllm:26.04-py3`
- vLLM version from logs: `0.19.0+6bc3197f.nv26.04.48680843`
- GPU: single GPU, `nvidia-smi` reports about 121.69 GiB capacity in vLLM OOM logs.

## Default Attempt

Startup script:

```bash
./scripts/start-qwen3-next-80b-a3b-thinking.sh
```

Effective vLLM parameters:

```text
--model Qwen/Qwen3-Next-80B-A3B-Thinking
--host 0.0.0.0
--port 8000
--tensor-parallel-size 1
--trust-remote-code
--max-model-len 32768
```

Important log observations:

```text
Resolved architecture: Qwen3NextForCausalLM
Using max model len 32768
FlashInfer CUTLASS MoE is available but not enabled, consider setting VLLM_USE_FLASHINFER_MOE_FP16=1 to enable it.
Using TRITON backend for Unquantized MoE
```

The first attempt failed with CUDA OOM during model initialization:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB.
GPU 0 has a total capacity of 121.69 GiB of which 972.70 MiB is free.
Including non-PyTorch memory, this process has 117.55 GiB memory in use.
```

The repeated validation attempt exited with code `137` after reaching the same unquantized MoE path. `docker inspect` reported:

```text
OOMKilled=false ExitCode=137
```

This still supports the conclusion that the default single-GPU configuration does not fit reliably on this 121 GiB GPU.

## Next Attempts Planned

1. Retry with `VLLM_USE_FLASHINFER_MOE_FP16=1`.
2. If still failing, retry with reduced `MAX_MODEL_LEN`, starting at 8192.

## FlashInfer MoE Attempt

Startup command:

```bash
VLLM_USE_FLASHINFER_MOE_FP16=1 EXTRA_DOCKER_ENV=VLLM_USE_FLASHINFER_MOE_FP16 ./scripts/start-qwen3-next-80b-a3b-thinking.sh
```

The environment variable was successfully passed into the container. Logs changed from Triton MoE to FlashInfer MoE:

```text
Using FlashInfer CUTLASS backend for Unquantized MoE
```

However, the container still failed before readiness. `docker inspect` showed:

```text
OOMKilled=false ExitCode=137
```

The user manually killed the stuck/failed attempt after it consumed memory and did not become ready. Logs are saved in `docker-flashinfer.log`.

Conclusion: enabling FlashInfer MoE alone is not enough to fit or reliably initialize the model on the single 121 GiB GPU with `max_model_len=32768`.

## FlashInfer MoE + Reduced Context Attempt

Startup command:

```bash
VLLM_USE_FLASHINFER_MOE_FP16=1 EXTRA_DOCKER_ENV=VLLM_USE_FLASHINFER_MOE_FP16 MAX_MODEL_LEN=8192 ./scripts/start-qwen3-next-80b-a3b-thinking.sh
```

Effective vLLM config from logs:

```text
model='Qwen/Qwen3-Next-80B-A3B-Thinking'
max_seq_len=8192
Using FlashInfer CUTLASS backend for Unquantized MoE
```

The container still exited before readiness. `docker inspect` showed:

```text
OOMKilled=false ExitCode=137
```

Logs are saved in `docker-flashinfer-maxlen8192.log`.

## Final Decision

Abort further attempts for this model on the current single 121 GiB GPU setup. The model is too large in the available FP/BF16 form, even with FlashInfer MoE and reduced context. Revisit only if an FP8/NVFP4 quantized variant is found or if multi-GPU/tensor-parallel serving is available.

## Context Window

All attempts in this report used the startup-side served context window via `MAX_MODEL_LEN`:

- Default attempts: `MAX_MODEL_LEN=32768`
- Reduced-context attempt: `MAX_MODEL_LEN=8192`

The benchmark request prompt/output sizes are separate from `MAX_MODEL_LEN`. Larger context windows such as 200k or 500k can materially affect memory pressure, cache planning, max concurrency, and therefore throughput, even if individual benchmark prompts are much shorter.
