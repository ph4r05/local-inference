# Run Config

- Model: `RedHatAI/Qwen3.6-35B-A3B-NVFP4`
- Startup script: `scripts/start-redhatai-qwen3.6-35b-a3b-nvfp4.sh`
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

Third-party NVFP4 quantization. NVIDIA 26.04 image failed on tokenizer; vLLM 0.20 image worked.

`MAX_MODEL_LEN` is the serving context window. It is separate from benchmark prompt/output lengths and can affect memory pressure, scheduling, and throughput.
