"""OCR -> LLM baseline (revision, Reviewer 2.4): a conventional two-stage
pipeline -- OCR transcript of the label, then a text-only LLM fills the same
9-field schema -- scored exactly like Tier 1.

Decouples text recognition from structured emission: for a model that can
take either input (e.g. MiMo-V2.5), "OCR text -> model" vs Tier 1's
"image -> model" isolates the recognition step; "OCR text ->
Qwen2.5-3B-Instruct" (Vintern-3B-beta's own language backbone) vs
"image -> Vintern" asks the same question at 3B scale.

The OCR stage runs separately (scripts/run_ocr.py, own environment) and must
be done first; images without a cached transcript are skipped with a count.
The prompt is the canonical Vietnamese zero-shot prompt with the OCR text in
place of the image (prompts/task_instruction_ocr_vi.txt) -- same field
descriptions and classification rules, nothing else changed.
"""

from __future__ import annotations

import argparse

from vifoodlabel.cli.common import (
    add_dataset_args,
    add_execution_args,
    print_dry_run_scope,
    print_resume_status,
    resolve_dataset,
)
from vifoodlabel.config import SCORED_RESULTS_DIR, resolve_models
from vifoodlabel.io_utils import labeled_only
from vifoodlabel.metrics import (
    FIELD_SCORE_KEY,
    IMAGE_SCORE_KEY,
    image_level_summary,
    model_summary,
    scores_to_dataframe,
    upsert_scored_csv,
)
from vifoodlabel.ocr import DEFAULT_OCR_ENGINE, load_ocr_text, ocr_condition_name
from vifoodlabel.prompts import build_ocr_instruction
from vifoodlabel.runner import RunItem, run_and_score

TIER = "ocr_pipeline"

# Vintern's own LLM backbone (free, self-hosted) and the cheapest of the
# leading three Tier 1 models -- each has an image-input counterpart already
# in Tier 1 to compare against. Override with --models.
DEFAULT_MODELS = ["qwen2.5-3b-text", "mimo-v2.5"]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    add_dataset_args(parser)
    add_execution_args(parser)
    parser.add_argument("--engine", default=DEFAULT_OCR_ENGINE, help=f"OCR engine whose cached transcripts to use (default: {DEFAULT_OCR_ENGINE})")


def run(args: argparse.Namespace) -> None:
    models = resolve_models(args.models or DEFAULT_MODELS)
    items = resolve_dataset(args)
    condition = ocr_condition_name(args.engine)

    transcripts = {item.image_id: load_ocr_text(args.engine, item.image_id) for item in items}
    missing = [i for i, t in transcripts.items() if t is None]
    items = [item for item in items if transcripts[item.image_id] is not None]

    print(f"OCR -> LLM baseline: {len(items)} images x {len(models)} models, engine={args.engine}, condition={condition}")
    if missing:
        print(f"  {len(missing)} selected images have no OCR transcript yet (run scripts/run_ocr.py) -- skipped")
    print(f"  {len(labeled_only(items))}/{len(items)} images currently have ground truth (only those will be scored)")
    if args.resume:
        print_resume_status(TIER, models, items, [condition])

    if args.dry_run:
        print_dry_run_scope(models, items, [condition])
        return

    run_items = [
        RunItem(item=item, model=model, condition=condition,
                instruction=build_ocr_instruction(transcripts[item.image_id]), image_path=None)
        for model in models
        for item in items
    ]
    scores = run_and_score(TIER, run_items, concurrency_per_model=args.concurrency, force=args.force)

    if not scores:
        print("No scores produced.")
        return

    upsert_scored_csv(scores_to_dataframe(scores), SCORED_RESULTS_DIR / "ocr_pipeline_field_scores.csv", FIELD_SCORE_KEY)
    merged = upsert_scored_csv(image_level_summary(scores), SCORED_RESULTS_DIR / "ocr_pipeline_image_scores.csv", IMAGE_SCORE_KEY)
    print("\nPer-model summary (all scored so far):")
    print(model_summary(merged).to_string(index=False))
