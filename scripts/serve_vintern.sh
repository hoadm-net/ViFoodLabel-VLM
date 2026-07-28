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
#
# Troubleshooting -- "CUDA initialization: driver too old" / cuda.is_available()
# False after `pip install -U vllm`: on an older-driver datacenter GPU (e.g. a
# T4 with a driver whose max supported CUDA is 12.6), the latest vLLM can pull
# a torch build targeting a newer CUDA major than the driver supports. Minor
# CUDA-version compatibility only bridges same-major gaps; a wheel from a newer
# major needs either a matching `cuda-compat-<major>-0` forward-compat package
# (not always sufficient) or an older vLLM pinned to a torch build with a wheel
# for your driver's CUDA major, e.g.:
#   pip install vllm==0.9.2 --extra-index-url https://download.pytorch.org/whl/cu126
# If that also fails at import with `ValueError: 'aimv2' is already used by a
# Transformers config`, pin a transformers release contemporary with that vLLM
# version instead of whatever latest `pip install -U vllm` pulled in, e.g.:
#   pip install "transformers==4.52.4"

set -euo pipefail

python3 -m venv .venv-vllm
source .venv-vllm/bin/activate
pip install -U vllm

# --max-model-len: must exceed DEFAULT_MAX_OUTPUT_TOKENS (16000, client.py) plus
# the prompt's input tokens (schema + classification rules + image tiles), or
# every real call 400s with "maximum context length exceeded" -- 8192 was too
# small and made every call fail. 24576 leaves ~8.5k for input, well above the
# ~1.5-4k observed for one image (up to 13 InternVL tiles); the Qwen2.5-3B
# backbone natively supports up to 32768.
#
# --limit-mm-per-prompt video=0: vLLM's InternVL processor's dummy multimodal
# profiling data generation for the video modality crashes on startup with
# `TypeError: Cannot handle this data type: (1, 1, 3), <i8` on this vLLM/
# transformers combination (a vLLM bug in the dummy-video-data path, unrelated
# to Vintern itself, which is never sent video input by this project). Setting
# the video limit to 0 skips that dummy-data generation entirely.
vllm serve 5CD-AI/Vintern-3B-beta \
    --trust-remote-code \
    --max-model-len 24576 \
    --limit-mm-per-prompt '{"image": 4, "video": 0}' \
    --port 8000
