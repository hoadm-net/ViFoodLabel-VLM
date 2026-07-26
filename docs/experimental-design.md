# Experimental design

Four tiers, run in order (each script is documented at the top of its file
in `scripts/`). Every tier shares one on-disk response cache keyed by
`(tier, model, image_id, condition)` (`src/vifoodlabel/cache.py`), so re-runs
are free and, where conditions coincide across tiers, later tiers reuse
earlier tiers' API calls instead of re-purchasing them.

## Tier 1 — main benchmark

`scripts/01_run_benchmark.py` — the full dataset × all 7 models, canonical
condition (`vi_zero`: Vietnamese instructions, zero-shot, clean image).
Produces the headline numbers: field-level F1 and pairing accuracy per
model, with bootstrap CI and pairwise McNemar significance (see
[metrics.md](metrics.md)).

## Tier 2 — prompt sensitivity

`scripts/02_run_prompt_sensitivity.py` — 2×2 ablation: instruction language
(`vi`/`en`) × shot count (`zero`/`one`). Uses the *same* cache namespace as
Tier 1 (`tier="benchmark"`), so the `vi_zero` leg is never re-run — only the
other 3 conditions cost anything.

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

`scripts/03_run_perturbation.py` — synthetic corruptions applied to a
stratified random subset (default 120 images, seeded) of otherwise-clean
photos, in place of collecting real hard-to-read photos:

| Corruption | Levels | Implementation |
|---|---|---|
| Blur | Gaussian σ ∈ {1.5, 3.5, 6.0} | `PIL.ImageFilter.GaussianBlur` |
| Glare | intensity/radius ∈ {(.35,.18), (.55,.22), (.75,.28)} | additive radial highlight, screen-blended |
| Rotation | 5° / 15° / 30° | rotated, then cropped to the largest axis-aligned inscribed rectangle and resized back to the original canvas — **no blank corners**, so the model can't infer "this was rotated" from a padding artifact |

See `src/vifoodlabel/perturbation.py`. The subset size trades off cost
(GPT-5.4 in particular is ~10-50x pricier per token than the open models)
against having enough images per condition for a meaningful degradation
curve.

The "clean" (severity-0) baseline for the degradation curve is **not**
re-run under this tier — `aggregate_report.py` pulls Tier 1's already-cached
`vi_zero` results for the same image ids, so Tier 3 only ever pays for the 9
corrupted conditions (3 kinds × 3 severities).

## Tier 4 — error taxonomy

`scripts/04_export_error_sample.py` samples incorrect (image, model, field)
instances from a prior run into two identical CSVs
(`error_sample_coder_{a,b}.csv`) with blank `error_category`/`notes` columns
for two independent human coders. Suggested category vocabulary: diacritics,
wrong unit, pairing error (cross-row swap), missing additive, wrong language
(right content, wrong-language portion of a multi-language label — see
[annotation-guidelines.md](annotation-guidelines.md)), hallucination, wrong
value, missing field, malformed JSON.

`scripts/05_score_error_taxonomy.py` computes Cohen's κ between the two
completed sheets, writes a per-row agreement table (agreed rows get a final
category; disagreements are flagged `NEEDS_ADJUDICATION` for manual
resolution), and a preliminary category-distribution table over the agreed
rows.

## Combining results

`scripts/aggregate_report.py` reads whatever tiers have been run so far
(gracefully skipping missing ones) and produces the summary tables —
per-model/condition F1 and pairing accuracy, bootstrap CIs, McNemar tables,
and the Tier-3 degradation-curve figure.
