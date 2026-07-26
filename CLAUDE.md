# CLAUDE.md

Project instructions for Claude Code in this repository.

## What this is

Code and experimental harness for **ViFoodLabel-VLM**, a benchmark paper
(target: *Discover Artificial Intelligence*, Springer Nature) evaluating
zero-shot structured extraction — Vietnamese food label photo → 9-field
JSON — across 7 VLMs. See `README.md` for the overview and `docs/` for full
methodology (dataset schema, metric definitions, experimental design, model
roster) before touching scoring logic or experiment design.

## Conventions

- Package layout: `src/vifoodlabel/`, managed with `uv` (not pip/poetry/conda).
  Install with `uv sync`; run scripts with `uv run scripts/<name>.py`.
- One script per experimental tier in `scripts/` (`01`–`05`, numbered by
  tier, plus `aggregate_report.py`). Each is self-contained and documented
  in its own module docstring.
- The model registry (slugs, pricing, endpoints, group) lives in
  `configs/models.yaml`, loaded via `src/vifoodlabel/config.py::ModelSpec`.
  Never hardcode a model slug or price anywhere else.
- Every API call is cached to disk (`results/raw/`, gitignored) keyed by
  `(tier, model, image_id, condition)`, written *before* scoring — this is
  what makes every script resumable and idempotent. Preserve this pattern in
  any new script; don't call the API without going through
  `src/vifoodlabel/runner.py`.

## Hard rules

- **Never commit `data/` or `results/`.** Both are gitignored on purpose:
  `data/` is unpublished ground truth and real product photography for a
  paper under review; `results/` is regenerable and may contain extracted
  content from that data. If you ever see either about to be staged
  (`git add -A` etc.), stop and check `.gitignore` rather than committing.
- **Never add an AI co-author trailer to commits in this repo** (no
  `Co-Authored-By: Claude ...` or similar). The target journal is sensitive
  to AI-assisted-code disclosure — commit only under the repository owner's
  local git identity, with an otherwise-normal commit message.
- **Never guess a model's API slug, availability, or pricing from training
  data.** This space moves fast and stale assumptions here cost real money
  or silently break a run. Verify live (web search/fetch against OpenRouter,
  or the provider's own docs) before adding or changing an entry in
  `configs/models.yaml`.
- **Prefer `--dry-run` before any real run against paid models**, especially
  before changing prompt text (longer prompts = more input tokens across
  every model) or widening `--images`/removing `--limit`.
