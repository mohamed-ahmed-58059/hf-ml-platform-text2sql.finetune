#!/usr/bin/env bash
# Start Llama-3.1-8B BASE on vLLM for the base-model eval baseline (completion mode).
# Run manually; Ctrl-C to stop. evaluate.py talks to it on localhost:8000.
# Gated model -> needs HF_TOKEN (read from .env below).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="meta-llama/Llama-3.1-8B"

# Same CUDA 13.0 toolkit pin as serve_hermes.sh (matches torch +cu130 for flashinfer JIT).
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"

# Gated repo: export the HF token so vLLM can pull the weights.
export HF_TOKEN="$(grep -m1 '^HF_TOKEN=' "$ROOT/.env" | cut -d= -f2-)"

# Base model: no chat template / tool calling -> plain completion serving.
uv run vllm serve "$MODEL" \
  --quantization bitsandbytes --max-model-len 8192 --gpu-memory-utilization 0.85
