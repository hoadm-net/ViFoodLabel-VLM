# Experimental design

Four tiers, run in order via `main.py`'s subcommands (each subcommand's
logic lives in its own module under `src/vifoodlabel/cli/`, documented at
the top of the file). Every tier shares one on-disk response cache keyed by
`(tier, model, image_id, condition)` (`src/vifoodlabel/cache.py`), so re-runs
are free and, where conditions coincide across tiers, later tiers reuse
earlier tiers' API calls instead of re-purchasing them.

## Inference parameters (every call, every tier)

`temperature=0`, `max_tokens=16000` (`src/vifoodlabel/client.py`). Extended
reasoning/"thinking" is explicitly disabled (`reasoning: {enabled: false}`
via OpenRouter's unified reasoning parameter) uniformly across all 6
OpenRouter-routed models — not applied to the self-hosted Vintern, whose
vLLM endpoint doesn't recognize the field.

This was a deliberate call, not a default left untouched: MiMo-V2.5 was
observed defaulting to thinking-enabled and burning its entire output-token
budget on hidden reasoning without ever emitting an answer (a known
OpenRouter/provider behavior for that model, not a prompt issue — see
`git log` for `client.py` around 2026-07). Rather than leaving each model at
whatever its provider defaults to (uncontrolled and, per the above,
occasionally pathological) or special-casing just the broken model,
reasoning is switched off the same way for every model, so all 7 are
compared on their direct zero-shot answer under identical inference
parameters.

**One disclosed exception, local to Vintern**: at `temperature=0` with no
repetition penalty, Vintern-3B-beta loops (repeats a short substring) to the
full `max_tokens` ceiling instead of emitting an end-of-sequence token — a
failure mode never observed on the 6 OpenRouter-routed models. Two
mitigations apply only to the self-hosted call path (`extra_body`, local
models only): `repetition_penalty=1.15` (grid-searched on real images —
cuts the loop rate but doesn't eliminate it), and a streaming watch that
detects a short substring repeating many times back-to-back and closes the
connection immediately, salvaging whatever content came before the loop the
same way truncation already does (tagged `generation_loop_detected`, a
structural issue distinct from `output_truncated` — see
[metrics.md](metrics.md#json-validity--structural-issues)). This is the one
place inference isn't byte-for-byte identical across all 7 models; it exists
because Vintern has no equivalent lever to the OpenRouter reasoning
parameter for fixing the same class of runaway-generation problem.

## Tier 1 — main benchmark

`uv run main.py benchmark` — the full dataset × all 7 models, canonical
condition (`vi_zero`: Vietnamese instructions, zero-shot, clean image).
Produces the headline numbers: field-level F1 and pairing accuracy per
model, with bootstrap CI and pairwise McNemar significance (see
[metrics.md](metrics.md)).

## Tier 2 — prompt sensitivity

`uv run main.py prompt-sensitivity` — two one-factor-at-a-time comparisons
against the `vi_zero` baseline, not a full 2×2 factorial: `vi_zero` vs
`vi_one` isolates the shot-count effect (does a one-shot example help?),
`vi_zero` vs `en_zero` isolates the instruction-language effect (does the
language the instructions are written in matter?). The fourth cell of the
factorial, `en_one`, is deliberately not run — it would only measure the
language×shot interaction, which isn't a question this benchmark asks, at
the cost of a fourth condition's worth of calls on every model (see
`prompts.py`'s `ALL_PROMPT_CONDITIONS`).

Runs on the same subset mechanism as Tier 3 (default 120 images, ~20% of
600). This subset is **pinned**, not recomputed per run:
`configs/subset_120.json`, generated once by `scripts/select_subset.py` (a
random draw, seed 42 — see `cli/common.py`'s
`add_subset_args`/`DEFAULT_SUBSET_SIZE`/`load_pinned_subset`), so the same
120 images are used across every future run and every tier that needs this
subset, immune to any later change in the sampling code or the underlying
image pool. Passing a non-default `--subset-size`/`--seed` still draws a
fresh ad hoc subset instead of using the pinned one. Like Tier 3, this is a
paired per-image comparison across conditions rather than the main
benchmark's headline per-model claim, so a subset gives adequate power
without paying for 600 images × 3 conditions. Uses the *same* cache
namespace as Tier 1 (`tier="benchmark"`), so the `vi_zero` leg is never
re-run — only `vi_one` and `en_zero` cost anything.

The one-shot exemplar is a **synthetic, text-only illustrative example** — a
fictional product with made-up values, described in the prompt text, not
paired with a real photo. This was a deliberate methodology choice: pairing
a real one-shot exemplar image+JSON would either burn one of the 600 images
(shrinking the scored set and creating a leakage asymmetry between zero- and
one-shot conditions) or require sourcing a 601st dedicated image. A
synthetic textual example demonstrates the JSON format and the bilingual
nutrition name/value pairing convention without touching the evaluation set
at all — every one of the 600 images stays scorable under every condition.
The exact wording lives in [`prompts/`](../prompts/) (plain `.txt` files,
assembled by `src/vifoodlabel/prompts.py`) — review or cite that directory
directly rather than the code.

## Tier 3 — perturbation robustness

`uv run main.py perturbation` — synthetic corruptions applied to the same
pinned subset as Tier 2 (`configs/subset_120.json`, default 120 images) of
otherwise-clean photos, in place of collecting real hard-to-read photos:

| Corruption | Levels | Implementation |
|---|---|---|
| Blur | Gaussian σ ∈ {1.5, 3.5, 6.0} | `PIL.ImageFilter.GaussianBlur` |
| Glare | intensity/radius ∈ {(.35,.18), (.55,.22), (.75,.28)} | radial highlight added on top of the pixels (not a true screen blend), with a squared falloff from center for a concentrated hotspot rather than a flat disc; center position is random, offset per image id so it doesn't land in the same spot on every photo (see `_per_image_seed` — deterministic/reproducible given the same seed, still varies image to image) |
| Rotation | 5° / 15° / 30° | rotated, then cropped to the largest axis-aligned inscribed rectangle and resized back to the original canvas — **no blank corners**, so the model can't infer "this was rotated" from a padding artifact |

See `src/vifoodlabel/perturbation.py`. The subset size trades off cost
(GPT-5.4 in particular is ~10-50x pricier per token than the open models)
against having enough images per condition for a meaningful degradation
curve.

The "clean" (severity-0) baseline for the degradation curve is **not**
re-run under this tier — `report` pulls Tier 1's already-cached `vi_zero`
results for the same image ids, so Tier 3 only ever pays for the 9 corrupted
conditions (3 kinds × 3 severities).

## Tier 4 — error taxonomy

`uv run main.py error-sample` samples incorrect (image, model, field)
instances from a prior run into two identical CSVs
(`error_sample_coder_{a,b}.csv`) with blank `error_category`/`notes` columns
for two independent human coders. Suggested category vocabulary: diacritics,
wrong unit, pairing error (cross-row swap), missing additive, wrong language
(right content, wrong-language portion of a multi-language label — see
[annotation-guidelines.md](annotation-guidelines.md)), hallucination, wrong
value, missing field, output truncated (hit `max_tokens`, not a real
extraction failure — see `client.py`'s `finish_reason` tracking), generation
loop (Vintern-only, auto-detected and cut short — see [inference
parameters](#inference-parameters-every-call-every-tier) above), malformed
JSON.

`uv run main.py score-taxonomy` computes Cohen's κ between the two completed
sheets, writes a per-row agreement table (agreed rows get a final category;
disagreements are flagged `NEEDS_ADJUDICATION` for manual resolution), and a
preliminary category-distribution table over the agreed rows.

## Combining results

`uv run main.py report` reads whatever tiers have been run so far
(gracefully skipping missing ones) and produces the summary tables —
per-model/condition F1 and pairing accuracy, bootstrap CIs, McNemar tables,
and the Tier-3 degradation-curve figure.
