# Run Config

- Model: `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`
- Startup script: `scripts/start-nemotron3-nano-omni-30b-nvfp4.sh`
- Container image: `vllm/vllm-openai:v0.20.0`
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

Nano script overrides image to vLLM 0.20 and uses KV cache fp8 plus Nano Omni multimedia/model-card flags.

`MAX_MODEL_LEN` is the serving context window. It is separate from benchmark prompt/output lengths and can affect memory pressure, scheduling, and throughput.
