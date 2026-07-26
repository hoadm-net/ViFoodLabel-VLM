"""runner.py -- includes a regression test for a real bug: failed API calls
(insufficient credits, transient outages, ...) used to be cached the same as
successful ones, making the failure permanent until an explicit --force
(which would also re-pay for every already-succeeded call in the run)."""

from __future__ import annotations

import asyncio

from vifoodlabel.client import RawResponse
from vifoodlabel.config import ModelSpec
from vifoodlabel.io_utils import DatasetItem
from vifoodlabel.runner import RunItem, _run_one, score_records


def _model() -> ModelSpec:
    return ModelSpec(
        key="test-model", slug="test/model", display_name="Test", group="test",
        open_weight=True, price_input_per_m=0.0, price_output_per_m=0.0,
    )


def _run_item(tmp_path) -> RunItem:
    item = DatasetItem(image_id="0001", image_path=tmp_path / "0001.jpeg", label_path=tmp_path / "0001.json")
    return RunItem(item=item, model=_model(), condition="vi_zero", instruction="extract", image_path=item.image_path)


class FakeClient:
    """Duck-types VLMClient: just needs an async .extract()."""

    def __init__(self, response: RawResponse):
        self._response = response

    async def extract(self, model, image_path, instruction):
        return self._response


def _patch_cache_and_ledger(monkeypatch, tmp_path):
    monkeypatch.setattr("vifoodlabel.cache.RAW_RESULTS_DIR", tmp_path / "raw")
    monkeypatch.setattr("vifoodlabel.cost.LEDGER_PATH", tmp_path / "cost_ledger.csv")


async def test_failed_call_is_not_cached(tmp_path, monkeypatch):
    _patch_cache_and_ledger(monkeypatch, tmp_path)
    from vifoodlabel.cache import load_cached

    ri = _run_item(tmp_path)
    failed = RawResponse(model_slug=ri.model.slug, content=None, input_tokens=0, output_tokens=0,
                          cost_usd=0.0, latency_s=0.1, error="APIStatusError: 402 insufficient credits")
    record = await _run_one(FakeClient(failed), "benchmark", asyncio.Semaphore(1), ri, force=False)

    assert record["content"] is None
    assert record["error"] == failed.error
    assert load_cached("benchmark", ri.model, ri.item.image_id, ri.condition) is None


async def test_successful_call_is_cached(tmp_path, monkeypatch):
    _patch_cache_and_ledger(monkeypatch, tmp_path)
    from vifoodlabel.cache import load_cached

    ri = _run_item(tmp_path)
    ok = RawResponse(model_slug=ri.model.slug, content='{"product_name": "x"}', input_tokens=10,
                      output_tokens=5, cost_usd=0.001, latency_s=0.1)
    record = await _run_one(FakeClient(ok), "benchmark", asyncio.Semaphore(1), ri, force=False)

    assert record["content"] == '{"product_name": "x"}'
    cached = load_cached("benchmark", ri.model, ri.item.image_id, ri.condition)
    assert cached is not None
    assert cached["content"] == '{"product_name": "x"}'


async def test_rerun_after_failure_retries_without_force(tmp_path, monkeypatch):
    # The actual regression scenario: call fails (out of credits), then the
    # exact same command is rerun later (credits topped up) -- it must hit
    # the API again, not silently return the stale cached failure.
    _patch_cache_and_ledger(monkeypatch, tmp_path)
    ri = _run_item(tmp_path)
    failed = RawResponse(model_slug=ri.model.slug, content=None, input_tokens=0, output_tokens=0,
                          cost_usd=0.0, latency_s=0.1, error="402 insufficient credits")
    await _run_one(FakeClient(failed), "benchmark", asyncio.Semaphore(1), ri, force=False)

    ok = RawResponse(model_slug=ri.model.slug, content='{"product_name": "x"}', input_tokens=10,
                      output_tokens=5, cost_usd=0.001, latency_s=0.1)
    second_client = FakeClient(ok)
    record = await _run_one(second_client, "benchmark", asyncio.Semaphore(1), ri, force=False)

    assert record["content"] == '{"product_name": "x"}'  # actually retried, not stuck on the cached error


async def test_cached_success_is_not_recalled(tmp_path, monkeypatch):
    _patch_cache_and_ledger(monkeypatch, tmp_path)
    ri = _run_item(tmp_path)
    ok = RawResponse(model_slug=ri.model.slug, content='{"product_name": "x"}', input_tokens=10,
                      output_tokens=5, cost_usd=0.001, latency_s=0.1)
    await _run_one(FakeClient(ok), "benchmark", asyncio.Semaphore(1), ri, force=False)

    class ExplodingClient:
        async def extract(self, model, image_path, instruction):
            raise AssertionError("should not be called -- result was already cached")

    record = await _run_one(ExplodingClient(), "benchmark", asyncio.Semaphore(1), ri, force=False)
    assert record["content"] == '{"product_name": "x"}'


class TestScoreRecordsTruncationFlag:
    def test_truncated_finish_reason_is_flagged(self, tmp_path):
        item = DatasetItem(image_id="0001", image_path=tmp_path / "0001.jpeg", label_path=tmp_path / "0001.json")
        gt = {
            "product_name": "x", "ingredient": [], "additive": [], "warning": [], "nutrition": [],
            "origin": "x", "net_weight": "x", "mfg_date": "x", "expiry_date": "x",
        }
        import json
        item.label_path.write_text(json.dumps(gt), encoding="utf-8")

        ri = RunItem(item=item, model=_model(), condition="vi_zero", instruction="x", image_path=item.image_path)
        record = {"content": '{"product_name": "x"', "error": None, "finish_reason": "length"}
        scores = score_records([(ri, record)])

        assert len(scores) == 1
        assert "output_truncated" in scores[0].structural_issues

    def test_normal_stop_is_not_flagged(self, tmp_path):
        item = DatasetItem(image_id="0001", image_path=tmp_path / "0001.jpeg", label_path=tmp_path / "0001.json")
        gt = {
            "product_name": "x", "ingredient": [], "additive": [], "warning": [], "nutrition": [],
            "origin": "x", "net_weight": "x", "mfg_date": "x", "expiry_date": "x",
        }
        import json
        item.label_path.write_text(json.dumps(gt), encoding="utf-8")

        ri = RunItem(item=item, model=_model(), condition="vi_zero", instruction="x", image_path=item.image_path)
        record = {"content": json.dumps(gt), "error": None, "finish_reason": "stop"}
        scores = score_records([(ri, record)])

        assert "output_truncated" not in scores[0].structural_issues

    def test_unlabeled_image_is_skipped(self, tmp_path):
        item = DatasetItem(image_id="0001", image_path=tmp_path / "0001.jpeg", label_path=tmp_path / "does_not_exist.json")
        ri = RunItem(item=item, model=_model(), condition="vi_zero", instruction="x", image_path=item.image_path)
        scores = score_records([(ri, {"content": "{}", "error": None})])
        assert scores == []
