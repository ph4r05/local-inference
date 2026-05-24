# Model Benchmark Suite

This repo benchmarks local model servers, primarily vLLM on `http://127.0.0.1:8000` and Ollama on `http://127.0.0.1:11434`.

## Run One Model Manually

Start a model:

```bash
./scripts/start-qwen3.6-35b-a3b-fp8.sh
```

Run the benchmark suite against the running server:

```bash
python3 benchmark_vllm_openai.py \
  --model Qwen/Qwen3.6-35B-A3B-FP8 \
  --suite \
  --suite-cases 1:1,1:2,1:4,2:4,4:8,8:16,16:32,32:64 \
  --prompt-mode unique \
  --prompt-words 256 \
  --max-tokens 128 \
  --warmup 1 \
  --no-per-request-details \
  --output-dir results/manual-qwen3.6-35b-a3b-fp8
```

## Run All Configured Models

```bash
./run_model_benchmarks.sh
```

Run only selected model slugs:

```bash
./run_model_benchmarks.sh qwen3.6-35b-a3b-fp8 qwen3.6-27b-fp8
```

Include the already-tested 120B Nemotron Super model:

```bash
INCLUDE_SUPER=1 ./run_model_benchmarks.sh
```

## Run Ollama Benchmarks

```bash
./run_ollama_benchmarks.sh
```

Run only the Ollama Qwen 35B test:

```bash
./run_ollama_benchmarks.sh ollama-qwen3.6-35b-a3b-q8_0
```

Ollama request concurrency is controlled separately with `OLLAMA_NUM_PARALLEL`. The summary files use the same columns as the vLLM runs, and the shared concurrency/capacity column is populated from `OLLAMA_NUM_PARALLEL` for layout compatibility, so the token and resource numbers are directly comparable.

## Suite Case Meaning

`--suite-cases 32:64` means 64 total requests with at most 32 concurrent in-flight requests. It does not mean 32 * 64 requests.

## Useful Environment Variables

- `VLLM_IMAGE`: Docker image. Default: `nvcr.io/nvidia/vllm:26.04-py3`
- `CONTAINER_NAME`: benchmark container name. Default: `vllm-bench`
- `LEGACY_CONTAINER_NAME`: also stopped before a run. Default: `vllm-nemotron`
- `PORT`: API port. Default: `8000`
- `GPU_DEVICE`: Docker GPU device selector. Default: `0`
- `CUDA_VISIBLE_DEVICES_VALUE`: container CUDA visibility. Default: `0`
- `MAX_MODEL_LEN`: vLLM max model length. Default: `32768`
- `VLLM_GPU_MEMORY_UTILIZATION`: vLLM GPU memory utilization. Default: `0.85`
- `MAX_NUM_SEQS`: explicit vLLM `--max-num-seqs`. Default: auto-computed from `VLLM_TARGET_BATCH_CONTEXT_TOKENS / MAX_MODEL_LEN`.
- `VLLM_TARGET_BATCH_CONTEXT_TOKENS`: auto `MAX_NUM_SEQS` target token budget. Default: `262144`, so 32K context uses 8 seqs, 128K uses 2, and 256K+ uses 1.
- `VLLM_MAX_NUM_SEQS_CAP`: upper bound for auto `MAX_NUM_SEQS`. Default: `16`.
- `VLLM_MAX_NUM_BATCHED_TOKENS`: explicit vLLM `--max-num-batched-tokens`. Also accepts legacy `MAX_NUM_BATCHED_TOKENS`. Default: max of `MAX_MODEL_LEN` and `VLLM_TARGET_BATCH_CONTEXT_TOKENS`.
- `TENSOR_PARALLEL_SIZE`: vLLM tensor parallel size. Default: `1`
- `HF_TOKEN`: Hugging Face token for gated models.
- `SUITE_CASES`: default `1:1,1:2,1:4,2:4,4:8,8:16,16:32,32:64`
- `PROMPT_WORDS`: default `256`
- `MAX_TOKENS`: default `128`
- `OLLAMA_BASE_URL`: Ollama API base URL. Default: `http://127.0.0.1:11434`
- `OLLAMA_NUM_PARALLEL`: Ollama server-side parallel request limit. Default: `1`
- `OLLAMA_MAX_QUEUE`: Ollama queue limit. Default: `512`
- `CONTEXT_LENGTH`: Ollama context length / `num_ctx` used by the runner. Default: `32768`

## Configured Models

- `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`
- `Qwen/Qwen3.6-35B-A3B-FP8`
- `Qwen/Qwen3.6-35B-A3B-NVFP4`
- `RedHatAI/Qwen3.6-35B-A3B-NVFP4` third-party NVFP4 quantization
- `Qwen/Qwen3-Next-80B-A3B-Thinking`
- `nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4` official NVIDIA NVFP4 TensorRT-LLM checkpoint
- `Qwen/Qwen3.6-27B-FP8`
- `Qwen/Qwen3.6-35B-A3B`
- Optional: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`


## NVFP4 Variant Tracking

Use explicit result names that include model size and quantization. For example:

- `nemotron3-nano-omni-30b-a3b-reasoning-nvfp4-full`
- `redhatai-qwen3.6-35b-a3b-nvfp4-full`
- `qwen3.6-27b-fp8-full`

Known NVFP4 candidates found so far:

- `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` official NVIDIA.
- `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` official NVIDIA.
- `RedHatAI/Qwen3.6-35B-A3B-NVFP4` third-party quantization of `Qwen/Qwen3.6-35B-A3B`.
- `inferRouter/Qwen3.6-27B-NVFP4` is the selected Qwen 27B NVFP4 candidate for now. It had the strongest model-card/runtime guidance found and 2,198 downloads last month during lookup. Other candidates: `unsloth/Qwen3.6-27B-NVFP4`, `vrfai/Qwen3.6-27B-NVFP4`, `mmangkad/Qwen3.6-27B-NVFP4`. Prefer an official Qwen repo if one appears.

No public official `Qwen/Qwen3.6-35B-A3B-NVFP4` repo was found during testing; the direct startup script is kept only to document that attempted model ID.

Note: `nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4` is documented by NVIDIA for TensorRT-LLM, with example tensor parallel size 4. vLLM loading is experimental in this benchmark setup.


## Context Window Policy

Every startup script uses `MAX_MODEL_LEN` as the served context window. Default:

```bash
MAX_MODEL_LEN=32768
```

This is separate from the benchmark request shape:

- `--prompt-words` controls generated synthetic prompt length.
- `--max-tokens` controls maximum generated output tokens.
- `MAX_MODEL_LEN` controls the model server's maximum context window and KV/cache planning.
- `VLLM_GPU_MEMORY_UTILIZATION` controls how aggressively vLLM reserves GPU memory for weights, KV cache, and runtime buffers. The default is `0.85`, which is less aggressive than vLLM's typical high-throughput settings while still leaving useful cache headroom.
- Auto `MAX_NUM_SEQS` deliberately shrinks at larger context windows to avoid planning many full-context requests at once. Set `MAX_NUM_SEQS` explicitly when you want a high-concurrency serving test.

Record `MAX_MODEL_LEN` in result folder names or notes when comparing models. Performance can change materially at larger context windows such as 200k or 500k because vLLM may reserve/profile more KV cache capacity, use different scheduling limits, reduce maximum concurrency, or hit memory pressure. For fair apples-to-apples throughput comparisons, keep `MAX_MODEL_LEN` fixed across models. For long-context benchmarks, run a separate suite and name outputs explicitly, for example:

```bash
MAX_MODEL_LEN=200000 ./run_model_benchmarks.sh redhatai-qwen3.6-35b-a3b-nvfp4
```

Recommended result naming pattern:

```text
results/<model-size-quant>-ctx32k
results/<model-size-quant>-ctx200k
results/<model-size-quant>-ctx500k
```
