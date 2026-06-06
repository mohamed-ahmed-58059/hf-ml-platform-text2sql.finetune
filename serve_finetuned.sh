#!/usr/bin/env bash
# Serve Llama-3.1-8B base + the fine-tuned LoRA adapter on vLLM (for the "finetuned" eval).
# Usage: ./serve_finetuned.sh <adapter_path_or_hub_repo>
#   e.g. ./serve_finetuned.sh out/text2sql-20260604-120000/checkpoint-6785
# Run manually; Ctrl-C to stop. evaluate.py (BASELINE="finetuned") talks to it on :8000.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE="meta-llama/Llama-3.1-8B"
ADAPTER="${1:?pass the adapter dir or hub repo, e.g. out/<run>/checkpoint-6785}"

# Same CUDA 13.0 toolkit pin as the other serve scripts (torch +cu130 / flashinfer JIT).
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export HF_TOKEN="$(grep -m1 '^HF_TOKEN=' "$ROOT/.env" | cut -d= -f2-)"

# Base served 4-bit; the adapter is loaded on top as a LoRA module named "text2sql".
# --max-lora-rank must be >= LORA_R (16 in train.py).
uv run vllm serve "$BASE" \
  --quantization bitsandbytes --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --enable-lora --lora-modules text2sql="$ADAPTER" --max-lora-rank 16
