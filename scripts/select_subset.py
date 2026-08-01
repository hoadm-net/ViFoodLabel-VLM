#!/usr/bin/env python3
"""One-off utility: materialize the default subset used by Tier 2/3 (and any
future tier that wants the same images) into configs/subset_120.json, so it
stays a fixed, checked-in list -- not just "reproducible given the same seed
and code." Run once from the repo root:

    uv run scripts/select_subset.py

Re-running requires --force. The whole point of the pinned file is that the
subset doesn't move out from under already-cached Tier 2/3 API results.
"""

from __future__ import annotations

import argparse
import json

from vifoodlabel.cli.common import DEFAULT_SUBSET_SEED, DEFAULT_SUBSET_SIZE, SUBSET_FILE, select_subset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing subset_120.json")
    args = parser.parse_args()

    if SUBSET_FILE.exists() and not args.force:
        raise SystemExit(
            f"{SUBSET_FILE} already exists -- refusing to overwrite without --force "
            "(this would change the subset out from under any cached Tier 2/3 "
            "results that assume the current one)."
        )

    image_ids = select_subset(DEFAULT_SUBSET_SIZE, DEFAULT_SUBSET_SEED)
    SUBSET_FILE.write_text(
        json.dumps(
            {"size": DEFAULT_SUBSET_SIZE, "seed": DEFAULT_SUBSET_SEED, "image_ids": image_ids},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(image_ids)} image ids to {SUBSET_FILE}")


if __name__ == "__main__":
    main()
