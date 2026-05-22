# Run Config

- Slug: nemotron3-nano-omni-30b-a3b-reasoning-nvfp4
- Model: nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4
- Startup script: /home/ph4r05/benchmark/scripts/start-nemotron3-nano-omni-30b-nvfp4.sh
- Container image: vllm/vllm-openai:v0.20.0
- Port: 8000
- Tensor parallel size: 1
- MAX_MODEL_LEN / served context window: 131072
- Context label: ctx128k
- Suite cases: 1:1,1:2,1:4,2:4,4:8,8:16,16:32,32:64
- Prompt mode: unique
- Prompt words: 256
- Max output tokens: 128
- Warmup requests per case: 1
- Extra startup env: MAX_NUM_BATCHED_TOKENS=131072
- Startup flags include: gpu-memory-utilization=0.70, max-num-seqs=8, max-num-batched-tokens=131072, enable-prefix-caching, reasoning-parser=nemotron_v3, tool-call-parser=qwen3_coder, kv-cache-dtype=fp8
- Reported startup details: GPU KV cache size 2,972,816 tokens; maximum concurrency for 131,072 tokens per request 99.80x; available KV cache memory 42.53 GiB.
