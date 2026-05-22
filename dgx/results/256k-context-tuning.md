# 256k Context Startup Tuning

This note covers how to start the benchmarked models at a `256k` context window without immediately blowing up swap or startup memory.

## What Matters

For startup, the knobs that matter are the ones that shape vLLM's preallocation and warmup footprint:

- `MAX_MODEL_LEN=262144`
- `MAX_NUM_BATCHED_TOKENS`
- `MAX_NUM_SEQS`
- `GPU_MEMORY_UTILIZATION`
- model-specific options like `--enable-prefix-caching`

Request concurrency mostly matters after the model is already loaded. It does not usually fix a startup failure.

## Safe Startup Pattern

Use a conservative load first, then widen it only after the model starts cleanly.

```bash
MAX_MODEL_LEN=262144 MAX_NUM_BATCHED_TOKENS=32768 MAX_NUM_SEQS=1 GPU_MEMORY_UTILIZATION=0.70 VLLM_STARTUP_LOAD_THRESHOLD=6 VLLM_STARTUP_SWAP_USED_GIB=2 ./scripts/start-<model>.sh
```

If startup succeeds, raise batching in steps:

1. `MAX_NUM_BATCHED_TOKENS=65536`
2. `MAX_NUM_BATCHED_TOKENS=131072`
3. `MAX_NUM_BATCHED_TOKENS=262144` only if the previous step stays stable

Keep `MAX_NUM_SEQS` low during load. `1` is the most conservative setting; `2` or `4` is only worth trying after the model already loads once.

## Model-Specific Guidance

### `nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4`

This is the only model in this set that is already proven at `256k` on this box.

Recommended launch approach:

```bash
MAX_MODEL_LEN=262144 ./scripts/start-nvidia-qwen3-next-80b-a3b-thinking-nvfp4.sh
```

If you need to reduce startup pressure further, lower batching first. Do not start by increasing concurrency.

### `RedHatAI/Qwen3.6-35B-A3B-NVFP4`

This model loaded cleanly at `128k`, but `256k` is more sensitive to startup memory geometry.

Recommended conservative launch:

```bash
MAX_MODEL_LEN=262144 MAX_NUM_BATCHED_TOKENS=32768 MAX_NUM_SEQS=1 GPU_MEMORY_UTILIZATION=0.70 ./scripts/start-redhatai-qwen3.6-35b-a3b-nvfp4.sh
```

If it loads, try `65536` batched tokens next. If it fails during model load, lower `MAX_NUM_BATCHED_TOKENS` before changing anything else.

### `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`

This one failed during `256k` startup/autotune/module load on this machine, so it is not a good candidate for an unconditional `256k` run.

If you still want to test it, keep the startup footprint very small:

```bash
MAX_MODEL_LEN=262144 MAX_NUM_BATCHED_TOKENS=16384 MAX_NUM_SEQS=1 GPU_MEMORY_UTILIZATION=0.70 ./scripts/start-nemotron3-nano-omni-30b-nvfp4.sh
```

Expect a failure unless the machine is otherwise very quiet. Treat this as an experiment, not a baseline.

## Benchmark Guarding

For the benchmark suite itself, keep the load guards separate from the startup guards:

- disable RAM aborts during the benchmark harness if you are prioritizing startup survival: `MAX_HOST_RAM_PCT=0`
- keep the benchmark swap ceiling active: `MAX_SWAP_USED_GIB=4`
- keep the startup-only watchdog active: `VLLM_STARTUP_LOAD_THRESHOLD=6`
- keep the startup-only swap ceiling active: `VLLM_STARTUP_SWAP_USED_GIB=2`

That setup lets startup fail fast if it starts to drift into swap or load pressure, without rejecting a healthy benchmark run later.

## Practical Order

1. Start with `MAX_MODEL_LEN=262144`, `MAX_NUM_BATCHED_TOKENS=32768`, `MAX_NUM_SEQS=1`.
2. Confirm `/v1/models` becomes ready.
3. If it loads, raise `MAX_NUM_BATCHED_TOKENS` gradually.
4. Only then run the benchmark suite, starting at low concurrency.
5. Use the vLLM-reported max full-context concurrency to filter the higher suite cases.
