"""Tier 2 -- prompt sensitivity ablation: {vi,en} x {zero,one}-shot.

Uses the same cache namespace ("benchmark") as Tier 1, keyed by condition
(vi_zero/vi_one/en_zero/en_one) -- so the vi_zero leg transparently reuses
Tier 1's results instead of re-purchasing them from the API.
"""

from __future__ import annotations

import argparse

from vifoodlabel.cli.common import (
    add_dataset_args,
    add_execution_args,
    print_resume_status,
    resolve_dataset,
    resolve_selected_models,
)
from vifoodlabel.config import SCORED_RESULTS_DIR
from vifoodlabel.cost import estimate_run_cost
from vifoodlabel.io_utils import labeled_only
from vifoodlabel.metrics import (
    FIELD_SCORE_KEY,
    IMAGE_SCORE_KEY,
    image_level_summary,
    model_summary,
    scores_to_dataframe,
    upsert_scored_csv,
)
from vifoodlabel.prompts import ALL_PROMPT_CONDITIONS, build_instruction, condition_name
from vifoodlabel.runner import RunItem, run_and_score

TIER = "benchmark"  # shared cache namespace with Tier 1, see module docstring


def add_arguments(parser: argparse.ArgumentParser) -> None:
    add_dataset_args(parser)
    add_execution_args(parser)


def run(args: argparse.Namespace) -> None:
    models = resolve_selected_models(args)
    items = resolve_dataset(args)
    n_labeled = len(labeled_only(items))
    all_conditions = [condition_name(lang, shot) for lang, shot in ALL_PROMPT_CONDITIONS]

    print(f"Tier 2 prompt sensitivity: {len(items)} images x {len(models)} models x {len(ALL_PROMPT_CONDITIONS)} conditions")
    print(f"  {n_labeled}/{len(items)} images currently have ground truth (only those will be scored)")
    if args.resume:
        print_resume_status(TIER, models, items, all_conditions)

    if args.dry_run:
        for language, shot in ALL_PROMPT_CONDITIONS:
            instruction = build_instruction(language, shot)
            est = estimate_run_cost(models, instruction, n_images=len(items))
            print(f"\nCondition {condition_name(language, shot)}:")
            print(est.to_string(index=False))
        return

    run_items = [
        RunItem(
            item=item, model=model, condition=condition_name(language, shot),
            instruction=build_instruction(language, shot), image_path=item.image_path,
        )
        for language, shot in ALL_PROMPT_CONDITIONS
        for model in models
        for item in items
    ]
    scores = run_and_score(TIER, run_items, concurrency_per_model=args.concurrency, force=args.force)

    if not scores:
        print("No scores produced (no ground truth available yet for the selected images).")
        return

    field_df = scores_to_dataframe(scores)
    image_df = image_level_summary(scores)
    upsert_scored_csv(field_df, SCORED_RESULTS_DIR / "tier2_field_scores.csv", FIELD_SCORE_KEY)
    merged_image_df = upsert_scored_csv(image_df, SCORED_RESULTS_DIR / "tier2_image_scores.csv", IMAGE_SCORE_KEY)

    print("\nPer-model x condition summary (all scored so far, not just this run):")
    print(model_summary(merged_image_df).sort_values(["model_key", "condition"]).to_string(index=False))
