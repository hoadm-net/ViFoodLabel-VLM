"""Tier 4a -- sample incorrect fields from a prior run and automatically
classify each into the error-category taxonomy (see error_taxonomy.py for
the rules and which categories are heuristic/low-confidence).

Reads from the on-disk cache only (no API calls) -- run `benchmark` first.
Produces one CSV (error_taxonomy.csv) with `error_category` already filled
in and a `low_confidence` flag marking rows worth spot-checking by hand
first. Classification is deterministic -- there's no human-coding step or
inter-coder Cohen's kappa here; if spot-checking finds the error rate too
high, the rules in error_taxonomy.py need revisiting, not more coders.

Category vocabulary (see error_taxonomy.py for exact assignment rules):
  diacritics       - content right, Vietnamese diacritics wrong
  wrong_unit        - number right, unit wrong/missing
  pairing_error      - nutrition value correct but attached to the wrong name
  missing_additive     - an additive present on the label was dropped
  hallucination          - content not present on the label at all
                          (low-confidence heuristic -- see error_taxonomy.py;
                          also covers what would be "wrong_language", since
                          ground truth has no record of non-Vietnamese label
                          text to confirm that distinction automatically)
  wrong_value              - any other incorrect value/content
  missing_field               - field left empty when the label had content
  output_truncated              - response hit the max_tokens ceiling;
                                  missing/cut-off fields aren't the model's
                                  choice (see structural_issues column)
  generation_loop                - self-hosted-only: cut short after an
                                    auto-detected repetition loop
  malformed_json                  - see json_valid/api_error columns
"""

from __future__ import annotations

import argparse
import random

from vifoodlabel.cli.common import add_dataset_args, resolve_dataset, resolve_selected_models
from vifoodlabel.config import SCORED_RESULTS_DIR
from vifoodlabel.error_taxonomy import LOW_CONFIDENCE_CATEGORIES, classify_error
from vifoodlabel.io_utils import labeled_only
from vifoodlabel.prompts import CANONICAL_CONDITION
from vifoodlabel.runner import load_cached_records, score_records
from vifoodlabel.schema import LIST_FIELDS, NUTRITION_FIELD, SCALAR_FIELDS, field_value_str


def _iter_incorrect_fields(scores):
    for s in scores:
        for f in SCALAR_FIELDS:
            r = s.scalar_results[f]
            if not r.lenient_match:
                yield s, f, "scalar", r.similarity / 100.0
        for f in LIST_FIELDS:
            r = s.list_results[f]
            if r.f1 < 1.0:
                yield s, f, "list", r.f1
        n = s.nutrition_result
        if n is not None and (n.name_f1 < 1.0 or n.pairing_accuracy < 1.0 or n.value_accuracy < 1.0):
            yield s, NUTRITION_FIELD, "nutrition", n.name_f1


def add_arguments(parser: argparse.ArgumentParser) -> None:
    add_dataset_args(parser)
    parser.add_argument("--tier", default="benchmark")
    parser.add_argument("--condition", default=CANONICAL_CONDITION)
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)


def run(args: argparse.Namespace) -> None:
    models = resolve_selected_models(args)
    items = labeled_only(resolve_dataset(args))
    if not items:
        print("No ground-truth-labeled images found in the selection.")
        return

    records = load_cached_records(args.tier, models, items, args.condition)
    if not records:
        print(f"No cached responses found for tier={args.tier!r} condition={args.condition!r}. Run 'main.py benchmark' first.")
        return

    scores = score_records(records)
    rows = []
    for s, field, field_type, score in _iter_incorrect_fields(scores):
        category = classify_error(s, field, field_type)
        rows.append({
            "image_id": s.image_id,
            "model_key": s.model_key,
            "condition": s.condition,
            "field": field,
            "field_type": field_type,
            "score": round(score, 3),
            "pred_value": field_value_str(s.pred, field) if s.pred else "",
            "gt_value": field_value_str(s.gt, field) if s.gt else "",
            "json_valid": s.json_valid,
            "api_error": s.api_error or "",
            "structural_issues": "; ".join(s.structural_issues),
            "error_category": category,
            "low_confidence": category in LOW_CONFIDENCE_CATEGORIES,
            "notes": "",
        })

    if not rows:
        print("No incorrect fields found — nothing to sample.")
        return

    rng = random.Random(args.seed)
    sample = rows if len(rows) <= args.sample_size else rng.sample(rows, args.sample_size)
    sample.sort(key=lambda r: (r["model_key"], r["image_id"], r["field"]))

    import pandas as pd

    df = pd.DataFrame(sample)
    SCORED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SCORED_RESULTS_DIR / "error_taxonomy.csv"
    df.to_csv(out_path, index=False)

    print(f"\n{len(rows)} total incorrect (image, model, field) instances found; sampled {len(sample)}.")
    print(f"Wrote {len(df)} auto-classified rows to {out_path}")
    print(f"{int(df['low_confidence'].sum())} rows flagged low_confidence -- spot-check these first (see error_taxonomy.py).")

    # Per model, not pooled -- error mix differs a lot by model (e.g. only
    # Vintern ever hits generation_loop), so a single pooled distribution
    # would mostly just reflect whichever model produced the most errors.
    counts = pd.crosstab(df["model_key"], df["error_category"])
    shares = pd.crosstab(df["model_key"], df["error_category"], normalize="index")
    print("\nError category counts per model:")
    print(counts.to_string())
    print("\nError category share within each model's own errors:")
    print(shares.round(3).to_string())
