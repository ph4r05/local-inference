# Run Config

- Model: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`
- Startup script: `/home/ph4r05/vllm-nemotron.sh`
- Container image: `nvcr.io/nvidia/vllm:26.04-py3`
- Port: `8000`
- Tensor parallel size: `1`
- MAX_MODEL_LEN / served context window: `32768`
- Suite cases: `1:1,1:2,1:4,2:4,4:8,8:16,32:64 or earlier comparable sweep; see summary/results JSON`
- Prompt mode: `unique`
- Prompt words: `256`
- Max output tokens: `128`
- Warmup requests per case: `1`
- Resource interval: `1.0`

## Notes

Backfilled from earlier run using original vllm-nemotron.sh style startup.

`MAX_MODEL_LEN` is the serving context window. It is separate from benchmark prompt/output lengths and can affect memory pressure, scheduling, and throughput.
