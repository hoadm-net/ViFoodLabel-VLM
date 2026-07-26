"""Shared CLI plumbing for the 01/02/03 experiment-runner scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vifoodlabel.config import ModelSpec, resolve_models  # noqa: E402
from vifoodlabel.io_utils import DatasetItem, dataset_index  # noqa: E402
from vifoodlabel.runner import DEFAULT_CONCURRENCY_PER_MODEL  # noqa: E402


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models", nargs="*", default=None, help="Model keys or slugs (default: all 4)")
    parser.add_argument("--images", nargs="*", default=None, help="Specific image ids, e.g. 0001 0002")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N images")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY_PER_MODEL, help="Concurrent requests per model")
    parser.add_argument("--force", action="store_true", help="Bypass the response cache and re-call the API")
    parser.add_argument("--dry-run", action="store_true", help="Estimate cost only, no API calls")


def resolve_dataset(args: argparse.Namespace) -> list[DatasetItem]:
    items = dataset_index(args.images)
    if args.limit is not None:
        items = items[: args.limit]
    return items


def resolve_selected_models(args: argparse.Namespace) -> list[ModelSpec]:
    return resolve_models(args.models)
