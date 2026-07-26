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
  Install with `uv sync`.
- **Single CLI entry point: `main.py` at repo root**, via subcommands
  (`benchmark`, `prompt-sensitivity`, `perturbation`, `error-sample`,
  `score-taxonomy`, `report` — one per experimental tier). Run with
  `uv run main.py <subcommand> --help`.
- Each subcommand's argument parsing + orchestration lives in its own module
  under `src/vifoodlabel/cli/` (`benchmark.py`, `prompt_sensitivity.py`,
  etc.) — `main.py` only dispatches. Keep it that way: a subcommand module
  should only ever import from `vifoodlabel.cli.common` and the core
  `vifoodlabel.*` library modules, never from another `cli/` module. This is
  what lets you fix/extend one subcommand without having to re-test the
  others — they don't share any mutable state or code path beyond the
  already-tested core library.
- The self-hosted Vintern-3B-beta model is the one exception: it's set up
  via `scripts/serve_vintern.sh` on a separate GPU machine, not through
  `main.py`. Once served, it's picked up automatically by every subcommand
  through `configs/models.yaml` like any other model.
- The model registry (slugs, pricing, endpoints, group) lives in
  `configs/models.yaml`, loaded via `src/vifoodlabel/config.py::ModelSpec`.
  Never hardcode a model slug or price anywhere else.
- Every API call is cached to disk (`results/raw/`, gitignored) keyed by
  `(tier, model, image_id, condition)`, written *before* scoring — this is
  what makes every subcommand resumable and idempotent. Preserve this
  pattern in any new subcommand; don't call the API without going through
  `src/vifoodlabel/runner.py`.
- **Test suite: `uv run pytest`** (`tests/`, one file per `src/vifoodlabel/`
  module). No network calls -- `client.py`/`runner.py` tests mock the
  OpenAI SDK call and the disk cache (via `tmp_path`/`monkeypatch`), never
  hit OpenRouter or write into the real `results/`. Several tests exist
  specifically because a real scoring bug was found by hand (search the
  test docstrings/git log for "regression") -- when you fix a bug in
  `matching.py`/`normalize.py`/`schema.py`/etc., add the failing case as a
  test before fixing it, the same way. Run the suite before considering any
  change to scoring logic done.

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
  before widening `--images`/`--start-id`/`--end-id`/removing `--limit` —
  it prints the scope (images x models x conditions = call count) without
  calling anything. It does NOT estimate a dollar cost (removed; it needed
  periodic recalibration to stay accurate and wasn't relied on) -- actual
  spend is tracked from real per-call token usage in
  `results/cost_ledger.csv`, cross-checked against the OpenRouter
  dashboard. `--resume` (prints cached vs. remaining count) is cheap to run
  first too.
