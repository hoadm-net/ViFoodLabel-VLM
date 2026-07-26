#!/usr/bin/env bash
# Run this ON THE GPU MACHINE (Linux + NVIDIA CUDA) — vLLM does not run on macOS,
# so this is a separate environment from the rest of this repo, not a `uv` script.
#
# Serves Vintern-3B-beta behind an OpenAI-compatible endpoint on port 8000,
# matching configs/models.yaml's `vintern-3b.base_url` (http://localhost:8000/v1).
#
# If the GPU machine is remote (not the same host you run the benchmark scripts
# from), either SSH-tunnel port 8000 back to your dev machine:
#   ssh -L 8000:localhost:8000 user@gpu-host
# or edit `base_url` in configs/models.yaml to the GPU host's reachable address.
#
# Usage: bash scripts/serve_vintern.sh

set -euo pipefail

python3 -m venv .venv-vllm
source .venv-vllm/bin/activate
pip install -U vllm

vllm serve 5CD-AI/Vintern-3B-beta \
    --trust-remote-code \
    --max-model-len 8192 \
    --port 8000
