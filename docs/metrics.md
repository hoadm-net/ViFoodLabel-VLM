# Metrics

Two headline metrics, both computed per image and aggregated across the
dataset: **field-level F1** (is each of the 9 fields' content right?) and
**nutrition pairing accuracy** (are name/value pairs in the `nutrition` table
correctly matched — the RQ specifically about bilingual, multi-row tables).

Implementation: [`src/vifoodlabel/matching.py`](../src/vifoodlabel/matching.py)
(per-field matchers) and [`src/vifoodlabel/metrics.py`](../src/vifoodlabel/metrics.py)
(per-image scoring + aggregation).

> **If you open `results/scored/tier*_*.csv` yourself** (Excel, a fresh
> `pd.read_csv()`, R, ...): pass `dtype={"image_id": str}` (pandas) or
> equivalent. Left unspecified, `image_id` values like `"0001"` get
> silently read back as the integer `1`, losing the zero-padding — this
> repo's own code always does this (see `upsert_scored_csv` in
> `metrics.py`), but a naive read outside it won't.

## Normalization

Before any comparison, text is Unicode-NFC normalized, whitespace-collapsed,
and lowercased (`normalize_text`). Numbers are parsed handling both
Vietnamese (`34,242` = decimal comma) and mixed thousand-separator formats
(`normalize_number`); units are aliased to a canonical form (`normalize_unit`,
e.g. `gram`/`gr`/`g` → `g`); dates are best-effort normalized to ISO
`yyyy-mm-dd` where they parse as literal dates, else compared as normalized
text (many `mfg_date`/`expiry_date` values are instructions, not dates).

A separate `strip_diacritics` normalization exists *only* to flag "content is
right, Vietnamese diacritics are wrong" as its own error-taxonomy category —
it never decides whether a field counts as correct.

## Scalar fields (`product_name`, `origin`, `net_weight`, `mfg_date`, `expiry_date`)

Two match levels are computed and both reported:
- **strict**: exact match after normalization.
- **lenient**: strict match, or `rapidfuzz` token-sort similarity ≥ 90 (an
  ANLS-style tolerance, as used in DocVQA-style benchmarks). This is the
  primary "is it correct" signal used in aggregate scores.

`net_weight` additionally gets a numeric+unit-aware comparison (number within
a small relative/absolute tolerance, unit aliased) before falling back to
text matching, since weights are usually `<number> <unit>` strings.

## List fields (`ingredient`, `additive`, `warning`)

Scored as a set-matching problem: predicted and ground-truth items are
optimally paired via the Hungarian algorithm
(`scipy.optimize.linear_sum_assignment`) on pairwise `rapidfuzz` similarity,
then a pair only counts as matched if similarity ≥ 80. From matched-pair
count `TP`:

```
precision = TP / len(pred_items)   (1.0 if pred_items is empty)
recall    = TP / len(gt_items)     (1.0 if gt_items is empty)
f1        = harmonic mean(precision, recall)
```

This one formula also correctly handles the empty-list edge cases (both
empty → 1.0; only one side empty → 0.0 on the non-vacuous side) without
special-casing.

**Merge-aware bonus**: after standard bipartite matching, if the *leftover
unmatched* items on one side, concatenated together, are a high-similarity
match for a single leftover item on the other side (e.g. two ground-truth
warning sentences the model combined into one array entry), both sides are
credited fully for that portion instead of scoring a hard miss. This is
deliberately narrow — only "several items on one side == one item on the
other," not arbitrary many-to-many regrouping — and doesn't give free
credit to a genuinely hallucinated extra item sitting alongside a real
match.

## Nutrition pairing (`nutrition[{name, value}]`)

This is the metric the RQ cares about most, and it's deliberately split into
two independent questions:

1. **Name matching** — same bipartite-matching approach as list fields, but
   similarity first checks a bilingual alias table (`canonical_nutrient_name`,
   e.g. *Năng lượng* ≡ *Energy*, *Chất đạm* ≡ *Protein*, *Đường tổng số* ≡
   *Total Sugar*) before falling back to fuzzy string similarity — this is
   what lets a model's English-column answer match a Vietnamese-column ground
   truth row, or vice versa. Produces `name_precision/recall/f1`, same
   formula as list fields.

2. **Value correctness vs. pairing correctness**, for each matched
   name-pair: the number/unit is compared (`_value_matches`). If it's wrong,
   we additionally check whether that same predicted value is numerically
   correct for a *different* row in the ground-truth table. If so, it's
   counted as a **pairing error** (a cross-row swap — the model read the
   right number off the label but attached it to the wrong nutrient name),
   distinct from simply extracting a wrong/hallucinated number.

```
value_accuracy    = value_correct / n_matched_pairs
pairing_accuracy  = 1 - (n_pairing_errors / len(gt_entries))
```

`pairing_accuracy` is reported as its own headline, per-image metric in
`[0, 1]`, directly comparable/aggregatable (bootstrap CI, etc.) alongside
`macro_field_f1`.

## Aggregation

- **`macro_field_f1`** (per image): mean of the lenient-match indicator for
  the 5 scalar fields, the F1 for the 3 list fields, and the nutrition
  name-F1 — i.e. one F1-equivalent number per field, averaged over all 9.
  `macro_field_f1_strict` is the same using strict scalar matching.
- **Bootstrap CI**: percentile-method bootstrap (image-level resampling,
  `B = 10,000`) for every headline metric, per (model, condition) —
  `stats.bootstrap_ci` / `bootstrap_ci_table`.
- **McNemar's test**: pairwise significance between models on a paired
  binary per-image outcome (exact-match, i.e. `f1 ≥ 0.999`), per field, with
  Holm-Bonferroni correction across all `C(n_models, 2)` pairwise comparisons
  — `stats.mcnemar_pairwise`.

## JSON validity / structural issues

Every prediction also carries a `json_valid` flag and a list of
`structural_issues` (JSON parse failure, JSON-repair was needed, a field
missing from the model's output, wrong type coerced, etc. — see
`schema.coerce_prediction`), plus two issues detected upstream in the client/
runner rather than at parse time: `output_truncated` (the response hit
`max_tokens`, tracked via `finish_reason`) and `generation_loop_detected`
(self-hosted Vintern only — an auto-detected repeated-substring loop cut
short before it could burn the full token budget; see
[experimental-design.md](experimental-design.md#inference-parameters-every-call-every-tier)).
A model that can't produce parseable JSON scores zero on the affected fields
rather than crashing the pipeline, and the issue list feeds directly into
the [error taxonomy](experimental-design.md#tier-4--error-taxonomy).
