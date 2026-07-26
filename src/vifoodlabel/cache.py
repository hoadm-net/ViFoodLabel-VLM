"""Disk cache for raw API responses, keyed by (tier, model, image_id, condition).

This is what makes every script resumable/idempotent, and lets later tiers
reuse earlier tiers' calls for free (e.g. Tier 2's vi/zero-shot leg and Tier
3's "clean" baseline are the same cache key as Tier 1's main run).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from vifoodlabel.config import RAW_RESULTS_DIR, ModelSpec
from vifoodlabel.io_utils import load_json, save_json


def cache_path(tier: str, model: ModelSpec, image_id: str, condition: str) -> Path:
    return RAW_RESULTS_DIR / tier / model.slug_sanitized / f"{image_id}__{condition}.json"


def load_cached(tier: str, model: ModelSpec, image_id: str, condition: str) -> dict | None:
    path = cache_path(tier, model, image_id, condition)
    if not path.exists():
        return None
    return load_json(path)


def save_cached(
    tier: str,
    model: ModelSpec,
    image_id: str,
    condition: str,
    *,
    content: str | None,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    latency_s: float,
    error: str | None,
    finish_reason: str | None = None,
) -> Path:
    record = {
        "tier": tier,
        "model_key": model.key,
        "model_slug": model.slug,
        "image_id": image_id,
        "condition": condition,
        "content": content,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "latency_s": latency_s,
        "error": error,
        "finish_reason": finish_reason,
        "cached_at": datetime.now(timezone.utc).isoformat(),
    }
    path = cache_path(tier, model, image_id, condition)
    save_json(path, record)
    return path
