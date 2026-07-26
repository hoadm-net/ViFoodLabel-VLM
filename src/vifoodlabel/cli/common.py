"""Shared CLI plumbing for main.py's subcommands.

Split into two argument groups on purpose: `add_dataset_args` (model/image
selection) applies to every subcommand that reads a dataset slice;
`add_execution_args` (concurrency/force/resume/dry-run) only applies to the
three subcommands that actually call a model API (benchmark,
prompt-sensitivity, perturbation) — error-sample/score-taxonomy/report never
touch the network, so they don't get flags that wouldn't do anything.
"""

from __future__ import annotations

import argparse

from vifoodlabel.cache import load_cached
from vifoodlabel.config import ModelSpec, resolve_models
from vifoodlabel.io_utils import DatasetItem, dataset_index, list_image_ids
from vifoodlabel.runner import DEFAULT_CONCURRENCY_PER_MODEL

ID_WIDTH = 4  # matches data/images/NNNN.jpeg


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models", nargs="*", default=None, help="Model keys or slugs (default: all)")
    parser.add_argument("--images", nargs="*", default=None, help="Specific image ids, e.g. 0001 0002")
    parser.add_argument("--start-id", type=int, default=None, help="First image id, inclusive (e.g. 1). Ignored if --images given.")
    parser.add_argument("--end-id", type=int, default=None, help="Last image id, inclusive (e.g. 600). Ignored if --images given.")
    parser.add_argument("--limit", type=int, default=None, help="Only use the first N selected images")


def add_execution_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY_PER_MODEL, help="Concurrent requests per model")
    parser.add_argument("--force", action="store_true", help="Bypass the response cache and re-call the API")
    parser.add_argument("--resume", action="store_true", help="Print how many (model, image, condition) triples are already cached before running")
    parser.add_argument("--dry-run", action="store_true", help="Estimate cost only, no API calls")


def _ids_from_range(start: int, end: int) -> list[str]:
    if start > end:
        raise ValueError(f"--start-id ({start}) must be <= --end-id ({end})")
    return [f"{i:0{ID_WIDTH}d}" for i in range(start, end + 1)]


def resolve_image_ids(args: argparse.Namespace) -> list[str] | None:
    """None means "every image in data/images/" (dataset_index's default)."""
    if args.images:
        return list(args.images)
    if args.start_id is not None or args.end_id is not None:
        start = args.start_id if args.start_id is not None else 1
        end = args.end_id if args.end_id is not None else len(list_image_ids())
        return _ids_from_range(start, end)
    return None


def resolve_dataset(args: argparse.Namespace) -> list[DatasetItem]:
    items = dataset_index(resolve_image_ids(args))
    if args.limit is not None:
        items = items[: args.limit]
    return items


def resolve_selected_models(args: argparse.Namespace) -> list[ModelSpec]:
    return resolve_models(args.models)


def print_resume_status(
    tier: str, models: list[ModelSpec], items: list[DatasetItem], conditions: list[str]
) -> None:
    total = len(models) * len(items) * len(conditions)
    cached = sum(
        1
        for model in models
        for item in items
        for condition in conditions
        if load_cached(tier, model, item.image_id, condition) is not None
    )
    print(f"  resume: {cached}/{total} (model, image, condition) triples already cached, {total - cached} remaining")
