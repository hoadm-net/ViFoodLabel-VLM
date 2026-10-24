# Models

Seven models across three groups, structured around a specific question:
does model scale / being closed-source matter more than domain
specialization for this task? The registry (slugs, pricing, endpoints) lives
in [`configs/models.yaml`](../configs/models.yaml), loaded via
`src/vifoodlabel/config.py::ModelSpec`. Availability and pricing were
verified live against OpenRouter's model listing at the time this roster was
put together — re-verify before relying on it, since this space moves fast.

## Closed frontier (via OpenRouter)

| Model | Slug | Input $/M | Output $/M |
|---|---|---|---|
| GPT-5.4 Standard | `openai/gpt-5.4` | 2.50 | 15.00 |
| Claude Sonnet 5 | `anthropic/claude-sonnet-5` | 2.00 | 10.00 |
| Grok 4.20 | `x-ai/grok-4.20` | 1.25 | 2.50 |

Two swaps happened here already, both caught by re-verifying live rather
than trusting an earlier pick — see git history of this file for the
concrete examples:

- `google/gemini-3-pro-preview` (the original pick) was discontinued by
  Google on 2026-03-09 — OpenRouter started returning `404 No endpoints
  found` for it. Caught by `json_validity_rate` showing 0% for this model
  on the first real test run. Replaced with `google/gemini-3.1-pro-preview`.
- Every current Gemini Pro/Flash-tier model tried afterward
  (`gemini-3.1-pro-preview`, `gemini-2.5-pro`, `gemini-3.5-flash`) rejects
  `reasoning: {enabled: false}` with `400 Reasoning is mandatory for this
  endpoint` — see [experimental-design.md](experimental-design.md#inference-parameters-every-call-every-tier)
  for why every call disables reasoning. Grok 4.20 accepts it cleanly
  (verified live) and is cheaper besides, so it replaced Gemini rather than
  carrying a per-model exception for one model in the roster.

## Open-source large VLMs (via OpenRouter)

| Model | Slug | Input $/M | Output $/M |
|---|---|---|---|
| Qwen3-VL-235B-A22B-Instruct | `qwen/qwen3-vl-235b-a22b-instruct` | 0.20 | 0.88 |

(Prices as listed for the original submission run. Re-verified 2026-10-24
for the revision runs: Qwen3-VL-235B is now 0.21/1.90 and MiMo-V2.5
0.119/0.238 — `configs/models.yaml` holds the current values.)
| GLM-4.6V | `z-ai/glm-4.6v` | 0.30 | 0.90 |
| MiMo-V2.5 | `xiaomi/mimo-v2.5` | 0.105 | 0.28 |

## Vietnamese document-specialized (self-hosted)

**Vintern-3B-beta** (`5CD-AI/Vintern-3B-beta`) — a Vietnamese-native
vision-language model (InternViT-300M + Qwen2.5-3B-Instruct backbone,
InternVL2.5 architecture) fine-tuned specifically for Vietnamese OCR,
document extraction, and VQA.

This group was originally scoped around narrow OCR-specialist models
(dots.ocr, DeepSeek-OCR, GLM-OCR) but those turned out to be a poor fit:
none are available via OpenRouter (self-hosting only), and — more
importantly — they operate through **fixed prompt modes** (e.g. dots.ocr's
`prompt_layout_all_en`/`prompt_web_parsing`) rather than free-form
instruction-following, so they can't take our custom 9-field JSON-schema
prompt the same way the other 6 models do; using them would require an
entirely different two-stage (OCR → separate structuring LLM) pipeline that
confounds the comparison. Vintern doesn't have that problem — it's
instruction-tuned and takes the exact same prompt as every other model in
this benchmark — while still being a genuine "small, Vietnamese-specialized"
counterpoint to the large general VLMs above.

### Self-hosting Vintern

vLLM doesn't run on macOS, so this needs a separate Linux + NVIDIA GPU
machine (a ≤24GB card such as a 3090/4090/L4 is enough for this 3B model) —
not the same `uv` environment as the rest of this repo:

```bash
bash scripts/serve_vintern.sh   # run on the GPU machine
```

This starts a fresh venv, installs vLLM, and serves an OpenAI-compatible
endpoint on `http://localhost:8000/v1`, matching the `vintern-3b` entry in
`configs/models.yaml`. If the GPU machine isn't the same host you run the
benchmark scripts from, either SSH-tunnel port 8000 back
(`ssh -L 8000:localhost:8000 user@gpu-host`) or edit that entry's `base_url`
to the GPU host's reachable address.

`--max-model-len` is set to 24576, comfortably above
`DEFAULT_MAX_OUTPUT_TOKENS` (16000 — see
[experimental-design.md](experimental-design.md#inference-parameters-every-call-every-tier))
plus prompt/image tokens; every real request 400s if this doesn't leave
enough headroom. Vintern is also the only model in the roster with a known
generation-loop failure mode (repeats a short substring to the full token
ceiling instead of stopping), which is why it gets two Vintern-only
mitigations (`repetition_penalty`, a streaming loop-cutoff) that no other
model in the roster needs — see experimental-design.md's inference
parameters section for the full explanation; it's the one place inference
isn't identical across all 7 models.

## Added in revision: general ~3B controls and support models

Reviewer request: separate Vietnamese specialization from scale directly.
Two general VLMs at Vintern's scale, self-hosted exactly like it (same
`repetition_penalty` + loop cut-off, since the call path is shared by
every local model):

| Model | Slug | Why |
|---|---|---|
| InternVL2.5-4B | `OpenGVLab/InternVL2_5-4B` | InternViT-300M + Qwen2.5-3B-Instruct — Vintern's architecture family and exact LLM backbone, without the Vietnamese fine-tuning |
| Qwen2.5-VL-3B-Instruct | `Qwen/Qwen2.5-VL-3B-Instruct` | a strong general 3B VLM from another family |

Support models carry a `role` in `configs/models.yaml` and are never part
of a default selection (they must be named with `--models`):

- `qwen2.5-3b-text` (`Qwen/Qwen2.5-3B-Instruct`, self-hosted, role
  `ocr_llm`) — text-only structuring LLM for the OCR -> LLM baseline;
  Vintern's own language backbone.
- `judge-gemini-3.1-pro` (`google/gemini-3.1-pro-preview`, role `judge`) —
  taxonomy-validation labeller, deliberately from a family outside the
  evaluated roster.

### Self-hosting on an older datacenter GPU (V100, used for the revision)

The revision runs served all four local models on one Tesla V100-32GB
(compute capability 7.0). Recent PyTorch CUDA 12.8+ wheels no longer ship
sm_70 kernels, so pin the stack used by `scripts/serve_vintern.sh`'s
troubleshooting note: `vllm==0.9.2`, `transformers==4.52.4`, torch 2.7.0
(cu126). V100 has no bf16 — pass `--dtype float16`. Ports:
Vintern 8000, InternVL 8001, Qwen2.5-VL 8002, Qwen2.5-3B text 8003
(`configs/models.yaml`). Practicalities that cost time:
- Start servers one at a time — concurrent starts on one GPU break vLLM's
  memory profiling. Two fit side by side at `--gpu-memory-utilization 0.45`
  (Qwen2.5-VL needs ~0.6: its image-profiling pass is larger).
- Don't name your own env var `VLLM_PORT`: vLLM uses it internally.
- Text-only models: omit `--limit-mm-per-prompt`.
