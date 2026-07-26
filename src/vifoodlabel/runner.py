"""Generic orchestration: (image, model, condition) -> cache-or-call -> parse -> score.

Every script (Tier 1/2/3) just builds a list of `RunItem`s and hands them to
`run_all`; scoring against ground truth (where available) happens uniformly
in `score_records`. This is what makes every tier resumable and automatically
scale up as more ground-truth labels arrive.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from vifoodlabel.cache import load_cached, save_cached
from vifoodlabel.client import TRUNCATION_FINISH_REASON, VLMClient, build_client
from vifoodlabel.config import ModelSpec
from vifoodlabel.cost import append_ledger
from vifoodlabel.io_utils import DatasetItem, load_ground_truth
from vifoodlabel.metrics import ImageScore, score_prediction
from vifoodlabel.parsing import parse_model_json
from vifoodlabel.schema import coerce_prediction

DEFAULT_CONCURRENCY_PER_MODEL = 5


@dataclass
class RunItem:
    item: DatasetItem
    model: ModelSpec
    condition: str
    instruction: str
    image_path: Path  # overridable, e.g. a perturbed image for Tier 3


async def _run_one(client: VLMClient, tier: str, semaphore: asyncio.Semaphore, run_item: RunItem, force: bool) -> dict:
    if not force:
        cached = load_cached(tier, run_item.model, run_item.item.image_id, run_item.condition)
        if cached is not None:
            return cached

    async with semaphore:
        response = await client.extract(run_item.model, run_item.image_path, run_item.instruction)

    save_cached(
        tier, run_item.model, run_item.item.image_id, run_item.condition,
        content=response.content, input_tokens=response.input_tokens, output_tokens=response.output_tokens,
        cost_usd=response.cost_usd, latency_s=response.latency_s, error=response.error,
        finish_reason=response.finish_reason,
    )
    if response.input_tokens or response.output_tokens:
        append_ledger(
            tier, run_item.model, run_item.item.image_id, run_item.condition,
            response.input_tokens, response.output_tokens, response.cost_usd,
        )
    return {"content": response.content, "error": response.error}


async def run_all(
    tier: str,
    run_items: list[RunItem],
    concurrency_per_model: int = DEFAULT_CONCURRENCY_PER_MODEL,
    force: bool = False,
) -> list[tuple[RunItem, dict]]:
    # One client per distinct endpoint (e.g. all OpenRouter models share one;
    # a self-hosted model like Vintern gets its own pointed at localhost).
    clients: dict[str, VLMClient] = {}

    def get_client(model: ModelSpec) -> VLMClient:
        url = model.resolved_base_url
        if url not in clients:
            clients[url] = build_client(model)
        return clients[url]

    semaphores: dict[str, asyncio.Semaphore] = {}
    results: list[tuple[RunItem, dict]] = []
    try:
        async def process(ri: RunItem) -> tuple[RunItem, dict]:
            sem = semaphores.setdefault(ri.model.key, asyncio.Semaphore(concurrency_per_model))
            record = await _run_one(get_client(ri.model), tier, sem, ri, force)
            return ri, record

        tasks = [asyncio.ensure_future(process(ri)) for ri in run_items]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=tier):
            results.append(await coro)
    finally:
        for client in clients.values():
            await client.aclose()
    return results


def score_records(records: list[tuple[RunItem, dict]]) -> list[ImageScore]:
    scores = []
    for ri, record in records:
        if not ri.item.has_ground_truth:
            continue
        gt = load_ground_truth(ri.item)
        content = record.get("content")
        api_error = record.get("error")
        extra_issues: list[str] = []
        raw_data = None

        if record.get("finish_reason") == TRUNCATION_FINISH_REASON:
            # Hit max_tokens -- content (if any) may be a cut-off prefix.
            # json_repair can still turn that into "valid" JSON, which would
            # otherwise silently look like the model just omitted fields
            # rather than never finishing. Flag it either way so it's
            # auditable instead of hidden inside a recall miss.
            extra_issues.append("output_truncated")

        if content is None:
            extra_issues.append("api_error")
            json_valid = False
        else:
            parse_result = parse_model_json(content)
            if parse_result.parse_failed:
                extra_issues.append("json_parse_failed")
                json_valid = False
            else:
                raw_data = parse_result.data
                json_valid = True
                if parse_result.used_repair:
                    extra_issues.append("json_repair_used")

        pred, structural_issues = coerce_prediction(raw_data)
        scores.append(
            score_prediction(
                image_id=ri.item.image_id,
                model_key=ri.model.key,
                condition=ri.condition,
                gt=gt,
                pred=pred,
                json_valid=json_valid,
                structural_issues=extra_issues + structural_issues,
                api_error=api_error,
            )
        )
    return scores


def load_cached_records(
    tier: str, models: list[ModelSpec], items: list[DatasetItem], condition: str
) -> list[tuple[RunItem, dict]]:
    """Rebuild (RunItem, record) pairs from disk cache only — no API calls.

    Used by downstream analysis scripts (error-taxonomy export, report
    aggregation) to re-derive full ImageScore objects (with pred/gt text)
    from a prior run without paying for the API again.
    """
    records: list[tuple[RunItem, dict]] = []
    for model in models:
        for item in items:
            cached = load_cached(tier, model, item.image_id, condition)
            if cached is None:
                continue
            ri = RunItem(item=item, model=model, condition=condition, instruction="", image_path=item.image_path)
            records.append((ri, cached))
    return records


def run_and_score(
    tier: str,
    run_items: list[RunItem],
    concurrency_per_model: int = DEFAULT_CONCURRENCY_PER_MODEL,
    force: bool = False,
) -> list[ImageScore]:
    records = asyncio.run(run_all(tier, run_items, concurrency_per_model=concurrency_per_model, force=force))
    return score_records(records)
