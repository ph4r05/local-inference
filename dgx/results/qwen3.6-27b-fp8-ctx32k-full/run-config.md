# Run Config

- Model: `Qwen/Qwen3.6-27B-FP8`
- Startup script: `scripts/start-qwen3.6-27b-fp8.sh`
- Container image: `nvcr.io/nvidia/vllm:26.04-py3`
- Port: `8000`
- Tensor parallel size: `1`
- MAX_MODEL_LEN / served context window: `32768`
- Suite cases: `1:1,1:2,1:4,2:4,4:8,8:16,16:32,32:64`
- Prompt mode: `unique`
- Prompt words: `256`
- Max output tokens: `128`
- Warmup requests per case: `1`
- Resource interval: `1.0`

## Notes

Official Qwen FP8 baseline.

`MAX_MODEL_LEN` is the serving context window. It is separate from benchmark prompt/output lengths and can affect memory pressure, scheduling, and throughput.
