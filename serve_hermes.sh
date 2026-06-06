#!/usr/bin/env bash
# Start Hermes-3-8B on vLLM for table naming. Run manually; Ctrl-C to stop.
# (prepare_wikisql.py talks to it on localhost:8000.)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use the system CUDA 13.0 toolkit (matches torch's +cu130) so flashinfer's JIT
# compiles cleanly. The venv's bundled nvcc was 13.3 — too new for flashinfer's
# headers, which caused the compiler/headers incompatibility.
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"

VLLM_SERVER_DEV_MODE=1 uv run vllm serve NousResearch/Hermes-3-Llama-3.1-8B \
  --quantization bitsandbytes --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice --tool-call-parser hermes --enable-sleep-mode
