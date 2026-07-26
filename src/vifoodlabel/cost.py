"""Cost estimation (dry-run) and a running spend ledger.

The dry-run estimate is a ballpark only: image tokenization varies by provider
and resolution, and we don't call the API to find out. It uses a fixed
per-image token assumption plus a chars/4 heuristic for prompt text — good
enough to sanity-check "is this run $5 or $500" before spending real money.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from vifoodlabel.config import RESULTS_DIR, ModelSpec

LEDGER_PATH = RESULTS_DIR / "cost_ledger.csv"

# Ballpark assumption for a photographed label image at typical upload resolution.
ASSUMED_IMAGE_TOKENS = 1100
ASSUMED_OUTPUT_TOKENS = 600
CHARS_PER_TOKEN = 4.0


def estimate_call_cost(model: ModelSpec, instruction_text: str) -> dict:
    text_tokens = int(len(instruction_text) / CHARS_PER_TOKEN)
    input_tokens = text_tokens + ASSUMED_IMAGE_TOKENS
    output_tokens = ASSUMED_OUTPUT_TOKENS
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": model.cost_usd(input_tokens, output_tokens),
    }


def estimate_run_cost(
    models: list[ModelSpec], instruction_text: str, n_images: int, n_conditions: int = 1
) -> pd.DataFrame:
    rows = []
    for model in models:
        per_call = estimate_call_cost(model, instruction_text)
        n_calls = n_images * n_conditions
        rows.append({
            "model_key": model.key,
            "n_calls": n_calls,
            "est_cost_per_call_usd": per_call["cost_usd"],
            "est_total_cost_usd": per_call["cost_usd"] * n_calls,
        })
    df = pd.DataFrame(rows)
    total = pd.DataFrame([{
        "model_key": "TOTAL", "n_calls": df["n_calls"].sum(),
        "est_cost_per_call_usd": float("nan"), "est_total_cost_usd": df["est_total_cost_usd"].sum(),
    }])
    return pd.concat([df, total], ignore_index=True)


def append_ledger(
    tier: str, model: ModelSpec, image_id: str, condition: str,
    input_tokens: int, output_tokens: int, cost_usd: float,
    ledger_path: Path = LEDGER_PATH,
) -> None:
    is_new = not ledger_path.exists()
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["tier", "model_key", "image_id", "condition", "input_tokens", "output_tokens", "cost_usd"])
        writer.writerow([tier, model.key, image_id, condition, input_tokens, output_tokens, f"{cost_usd:.6f}"])


def read_ledger_total(ledger_path: Path = LEDGER_PATH) -> float:
    if not ledger_path.exists():
        return 0.0
    df = pd.read_csv(ledger_path)
    return float(df["cost_usd"].sum())
