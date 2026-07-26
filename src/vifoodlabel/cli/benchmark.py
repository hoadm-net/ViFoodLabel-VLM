"""Tier 1 -- main benchmark: full dataset x all models, canonical zero-shot VI prompt, clean images."""

from __future__ import annotations

import argparse

from vifoodlabel.cli.common import (
    add_dataset_args,
    add_execution_args,
    print_dry_run_scope,
    print_resume_status,
    resolve_dataset,
    resolve_selected_models,
)
from vifoodlabel.config import SCORED_RESULTS_DIR
from vifoodlabel.io_utils import labeled_only
from vifoodlabel.metrics import (
    FIELD_SCORE_KEY,
    IMAGE_SCORE_KEY,
    image_level_summary,
    model_summary,
    scores_to_dataframe,
    upsert_scored_csv,
)
from vifoodlabel.prompts import CANONICAL_CONDITION, CANONICAL_LANGUAGE, CANONICAL_SHOT, build_instruction
from vifoodlabel.runner import RunItem, run_and_score

TIER = "benchmark"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    add_dataset_args(parser)
    add_execution_args(parser)


def run(args: argparse.Namespace) -> None:
    models = resolve_selected_models(args)
    items = resolve_dataset(args)
    instruction = build_instruction(CANONICAL_LANGUAGE, CANONICAL_SHOT)

    print(f"Tier 1 benchmark: {len(items)} images x {len(models)} models, condition={CANONICAL_CONDITION}")
    n_labeled = len(labeled_only(items))
    print(f"  {n_labeled}/{len(items)} images currently have ground truth (only those will be scored)")
    if args.resume:
        print_resume_status(TIER, models, items, [CANONICAL_CONDITION])

    if args.dry_run:
        print_dry_run_scope(models, items, [CANONICAL_CONDITION])
        return

    run_items = [
        RunItem(item=item, model=model, condition=CANONICAL_CONDITION, instruction=instruction, image_path=item.image_path)
        for model in models
        for item in items
    ]
    scores = run_and_score(TIER, run_items, concurrency_per_model=args.concurrency, force=args.force)

    if not scores:
        print("No scores produced (no ground truth available yet for the selected images).")
        return

    field_df = scores_to_dataframe(scores)
    image_df = image_level_summary(scores)
    upsert_scored_csv(field_df, SCORED_RESULTS_DIR / "tier1_field_scores.csv", FIELD_SCORE_KEY)
    merged_image_df = upsert_scored_csv(image_df, SCORED_RESULTS_DIR / "tier1_image_scores.csv", IMAGE_SCORE_KEY)

    print("\nPer-model summary (all models scored so far, not just this run):")
    print(model_summary(merged_image_df).to_string(index=False))
