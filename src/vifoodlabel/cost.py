"""Running spend ledger, built from actual per-call token usage.

(There used to be a pre-run cost estimator here too, based on assumed
token counts. Dropped: it needed periodic recalibration to stay accurate
and wasn't relied on -- spend is tracked here from real usage instead, and
checked against the OpenRouter dashboard directly.)
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from vifoodlabel.config import RESULTS_DIR, ModelSpec

LEDGER_PATH = RESULTS_DIR / "cost_ledger.csv"


def append_ledger(
    tier: str, model: ModelSpec, image_id: str, condition: str,
    input_tokens: int, output_tokens: int, cost_usd: float,
    ledger_path: Path | None = None,
) -> None:
    # Resolved at call time, not as a default argument -- a default would
    # bind the path at import and ignore monkeypatching (see tests/test_cost.py).
    ledger_path = ledger_path or LEDGER_PATH
    is_new = not ledger_path.exists()
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["tier", "model_key", "image_id", "condition", "input_tokens", "output_tokens", "cost_usd"])
        writer.writerow([tier, model.key, image_id, condition, input_tokens, output_tokens, f"{cost_usd:.6f}"])


def read_ledger_total(ledger_path: Path | None = None) -> float:
    ledger_path = ledger_path or LEDGER_PATH
    if not ledger_path.exists():
        return 0.0
    df = pd.read_csv(ledger_path)
    return float(df["cost_usd"].sum())
