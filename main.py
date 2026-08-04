#!/usr/bin/env python3
"""Single CLI entry point for the ViFoodLabel-VLM experimental harness.

Each subcommand maps 1:1 to a tier in docs/experimental-design.md. Every
subcommand's logic lives in its own module under src/vifoodlabel/cli/ and
only touches the shared, already-tested library code in src/vifoodlabel/
(client, runner, matching, metrics, stats, cache, cost, ...) -- changing one
subcommand never risks another's behavior, and there's nothing left to
retest beyond the subcommand you actually touched.

Vintern-3B-beta (self-hosted) is NOT run through this CLI -- it has its own
setup script (scripts/serve_vintern.sh) since it runs on a separate Linux +
GPU machine. Once served, it's just another model in configs/models.yaml and
every subcommand below picks it up automatically via --models vintern-3b.

Usage:
    uv run main.py benchmark --dry-run
    uv run main.py benchmark --start-id 1 --end-id 3 --models mimo-v2.5
    uv run main.py prompt-sensitivity --start-id 1 --end-id 3
    uv run main.py perturbation --start-id 1 --end-id 3 --resume
    uv run main.py error-sample --sample-size 300
    uv run main.py label-agreement
    uv run main.py report
"""

from __future__ import annotations

import argparse

from vifoodlabel.cli import benchmark, error_sample, label_agreement, perturbation, prompt_sensitivity, report

SUBCOMMANDS = {
    "benchmark": benchmark,
    "prompt-sensitivity": prompt_sensitivity,
    "perturbation": perturbation,
    "error-sample": error_sample,
    "label-agreement": label_agreement,
    "report": report,
}


def _first_doc_line(module) -> str:
    doc = module.__doc__ or ""
    return doc.strip().splitlines()[0] if doc.strip() else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, module in SUBCOMMANDS.items():
        subparser = subparsers.add_parser(name, help=_first_doc_line(module), description=module.__doc__)
        module.add_arguments(subparser)

    args = parser.parse_args()
    SUBCOMMANDS[args.command].run(args)


if __name__ == "__main__":
    main()
