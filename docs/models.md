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
