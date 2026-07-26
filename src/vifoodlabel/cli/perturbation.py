"""Tier 3 -- perturbation robustness: blur/glare/rotation x 3 severities, on a
stratified subset of the dataset (default 120 images) using the canonical
zero-shot VI prompt. The "clean" baseline for the degradation curve is NOT
re-run here -- it's Tier 1's already-cached result for the same image ids
(see `report`), so this only ever pays for the 9 corrupted conditions.
"""

from __future__ import annotations

import argparse
import random

from vifoodlabel.cli.common import (
    add_dataset_args,
    add_execution_args,
    print_resume_status,
    resolve_image_ids,
    resolve_selected_models,
)
from vifoodlabel.config import SCORED_RESULTS_DIR
from vifoodlabel.cost import estimate_run_cost
from vifoodlabel.io_utils import dataset_index, labeled_only, list_image_ids
from vifoodlabel.metrics import (
    FIELD_SCORE_KEY,
    IMAGE_SCORE_KEY,
    image_level_summary,
    model_summary,
    scores_to_dataframe,
    upsert_scored_csv,
)
from vifoodlabel.perturbation import all_perturbation_conditions, materialize, perturbation_condition_name
from vifoodlabel.prompts import CANONICAL_LANGUAGE, CANONICAL_SHOT, build_instruction
from vifoodlabel.runner import RunItem, run_and_score

TIER = "perturbation"
DEFAULT_SUBSET_SIZE = 120
DEFAULT_SEED = 42


def select_subset(subset_size: int, seed: int) -> list[str]:
    all_ids = list_image_ids()
    if subset_size >= len(all_ids):
        return all_ids
    return sorted(random.Random(seed).sample(all_ids, subset_size))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    add_dataset_args(parser)
    add_execution_args(parser)
    parser.add_argument("--subset-size", type=int, default=DEFAULT_SUBSET_SIZE, help="Stratified subset size (ignored if --images/--start-id/--end-id given)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Subset selection seed")


def run(args: argparse.Namespace) -> None:
    models = resolve_selected_models(args)
    image_ids = resolve_image_ids(args)
    if image_ids is None:
        image_ids = select_subset(args.subset_size, args.seed)
    if args.limit is not None:
        image_ids = image_ids[: args.limit]
    items = dataset_index(image_ids)
    conditions = all_perturbation_conditions()
    condition_names = [perturbation_condition_name(kind, severity) for kind, severity in conditions]

    print(f"Tier 3 perturbation: {len(items)} images x {len(models)} models x {len(conditions)} conditions")
    n_labeled = len(labeled_only(items))
    print(f"  {n_labeled}/{len(items)} images currently have ground truth (only those will be scored)")
    if args.resume:
        print_resume_status(TIER, models, items, condition_names)

    instruction = build_instruction(CANONICAL_LANGUAGE, CANONICAL_SHOT)

    if args.dry_run:
        est = estimate_run_cost(models, instruction, n_images=len(items), n_conditions=len(conditions))
        print(est.to_string(index=False))
        return

    run_items = []
    for kind, severity in conditions:
        condition = perturbation_condition_name(kind, severity)
        for item in items:
            perturbed_path = materialize(item.image_path, item.image_id, kind, severity, seed=args.seed)
            for model in models:
                run_items.append(RunItem(item=item, model=model, condition=condition, instruction=instruction, image_path=perturbed_path))

    scores = run_and_score(TIER, run_items, concurrency_per_model=args.concurrency, force=args.force)

    if not scores:
        print("No scores produced (no ground truth available yet for the selected images).")
        return

    field_df = scores_to_dataframe(scores)
    image_df = image_level_summary(scores)
    upsert_scored_csv(field_df, SCORED_RESULTS_DIR / "tier3_field_scores.csv", FIELD_SCORE_KEY)
    merged_image_df = upsert_scored_csv(image_df, SCORED_RESULTS_DIR / "tier3_image_scores.csv", IMAGE_SCORE_KEY)

    print("\nPer-model x condition summary (all scored so far, not just this run):")
    print(model_summary(merged_image_df).sort_values(["model_key", "condition"]).to_string(index=False))
