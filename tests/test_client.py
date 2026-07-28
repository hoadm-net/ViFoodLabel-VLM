"""client.py -- includes a regression test for a real bug: Gemini 3.1 Pro
rejects reasoning={enabled: false} with a 400 "Reasoning is mandatory for
this endpoint" error, which used to propagate and fail every single call
for that model instead of falling back to its default reasoning behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from openai import BadRequestError
from PIL import Image

from vifoodlabel.client import LOCAL_EXTRA_BODY, VLMClient, _CallOutcome, _has_repetition_loop
from vifoodlabel.config import ModelSpec


def _model(is_local: bool = False) -> ModelSpec:
    return ModelSpec(
        key="test-model", slug="test/model", display_name="Test", group="test",
        open_weight=True, price_input_per_m=1.0, price_output_per_m=2.0,
        base_url="http://localhost:8000/v1" if is_local else None,
    )


def _fake_outcome(content: str | None = "{}", finish_reason: str = "stop",
                    prompt_tokens: int = 10, completion_tokens: int = 5) -> _CallOutcome:
    return _CallOutcome(content=content, finish_reason=finish_reason,
                         input_tokens=prompt_tokens, output_tokens=completion_tokens)


def _bad_request(message: str) -> BadRequestError:
    # openai's exception classes want a real httpx Response-shaped body;
    # a minimal stand-in that survives str(exc) is all this needs.
    body = {"error": {"message": message}}
    response = SimpleNamespace(request=SimpleNamespace(), status_code=400, headers={})
    return BadRequestError(message=message, response=response, body=body)


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "0001.jpeg"
    Image.new("RGB", (10, 10), color="white").save(path, format="JPEG")
    return path


class TestExtract:
    async def test_successful_call(self, image_path):
        client = VLMClient(base_url="http://example.invalid/v1", api_key="test")
        client._create = _record_and_return(_fake_outcome(content='{"x": 1}', finish_reason="stop"))

        r = await client.extract(_model(), image_path, "extract this")

        assert r.ok is True
        assert r.content == '{"x": 1}'
        assert r.input_tokens == 10
        assert r.output_tokens == 5
        assert r.finish_reason == "stop"
        assert r.truncated is False

    async def test_truncated_finish_reason(self, image_path):
        client = VLMClient(base_url="http://example.invalid/v1", api_key="test")
        client._create = _record_and_return(_fake_outcome(content='{"x": 1', finish_reason="length"))

        r = await client.extract(_model(), image_path, "extract this")

        assert r.truncated is True
        assert r.finish_reason == "length"

    async def test_empty_content_is_an_error(self, image_path):
        client = VLMClient(base_url="http://example.invalid/v1", api_key="test")
        client._create = _record_and_return(_fake_outcome(content=None))

        r = await client.extract(_model(), image_path, "extract this")

        assert r.ok is False
        assert "empty_response" in r.error

    async def test_unexpected_exception_never_propagates(self, image_path):
        # extract() must degrade any failure to a RawResponse, never raise --
        # a single bad call in a large concurrent batch must not take down
        # every other in-flight call.
        client = VLMClient(base_url="http://example.invalid/v1", api_key="test")

        async def boom(*a, **k):
            raise RuntimeError("something unexpected")

        client._call = boom

        r = await client.extract(_model(), image_path, "extract this")
        assert r.ok is False
        assert "RuntimeError" in r.error


class TestReasoningFallback:
    async def test_mandatory_reasoning_error_falls_back_to_default(self, image_path):
        # The exact real case: Gemini 3.1 Pro rejects reasoning={enabled:
        # False} outright. Must retry once with no reasoning param, not fail
        # the whole call.
        client = VLMClient(base_url="http://example.invalid/v1", api_key="test")
        calls = []

        async def fake_create(model, instruction, image_data_url, extra_body):
            calls.append(extra_body)
            if extra_body:
                raise _bad_request("Reasoning is mandatory for this endpoint and cannot be disabled.")
            return _fake_outcome(content='{"x": 1}')

        client._create = fake_create

        outcome = await client._call(_model(), "extract this", "data:image/jpeg;base64,x")

        assert outcome.content == '{"x": 1}'
        assert calls == [{"reasoning": {"enabled": False}}, {}]

    async def test_second_call_for_same_model_skips_straight_to_fallback(self, image_path):
        # Once a model's slug is known to reject the param, subsequent calls
        # shouldn't waste a round trip re-discovering that.
        client = VLMClient(base_url="http://example.invalid/v1", api_key="test")
        model = _model()
        calls = []

        async def fake_create(model_, instruction, image_data_url, extra_body):
            calls.append(extra_body)
            if extra_body:
                raise _bad_request("Reasoning is mandatory for this endpoint.")
            return _fake_outcome()

        client._create = fake_create

        await client._call(model, "x", "data:image/jpeg;base64,x")
        await client._call(model, "x", "data:image/jpeg;base64,x")

        # First call: tries enabled=False, falls back. Second call: goes
        # straight to {} -- only 3 total attempts, not 4.
        assert calls == [{"reasoning": {"enabled": False}}, {}, {}]

    async def test_unrelated_bad_request_error_is_not_retried(self, image_path):
        client = VLMClient(base_url="http://example.invalid/v1", api_key="test")
        calls = []

        async def fake_create(model, instruction, image_data_url, extra_body):
            calls.append(extra_body)
            raise _bad_request("Invalid image format.")

        client._create = fake_create

        with pytest.raises(BadRequestError):
            await client._call(_model(), "x", "data:image/jpeg;base64,x")
        assert len(calls) == 1  # no fallback attempted for an unrelated error

    def test_local_model_never_gets_reasoning_param(self):
        client = VLMClient(base_url="http://localhost:8000/v1", api_key="EMPTY")
        extra_body = client._extra_body_for(_model(is_local=True))
        assert "reasoning" not in extra_body

    def test_local_model_gets_repetition_penalty(self):
        # Vintern-3B-beta reliably loops to the max_tokens ceiling at
        # temperature=0 with no repetition penalty (see LOCAL_EXTRA_BODY in
        # client.py) -- every self-hosted/local model call must carry it.
        client = VLMClient(base_url="http://localhost:8000/v1", api_key="EMPTY")
        assert client._extra_body_for(_model(is_local=True)) == LOCAL_EXTRA_BODY


class TestLoopCutoffDispatch:
    async def test_local_model_routes_through_streaming(self):
        client = VLMClient(base_url="http://localhost:8000/v1", api_key="EMPTY")
        calls = {"streaming": 0, "nonstreaming": 0}

        async def fake_streaming(model, instruction, image_data_url, extra_body):
            calls["streaming"] += 1
            return _fake_outcome()

        async def fake_nonstreaming(model, instruction, image_data_url, extra_body):
            calls["nonstreaming"] += 1
            return _fake_outcome()

        client._create_streaming = fake_streaming
        client._create = fake_nonstreaming

        await client._call(_model(is_local=True), "x", "data:image/jpeg;base64,x")
        await client._call(_model(is_local=False), "x", "data:image/jpeg;base64,x")

        assert calls == {"streaming": 1, "nonstreaming": 1}

    async def test_loop_cutoff_finish_reason_surfaces_on_raw_response(self, image_path):
        client = VLMClient(base_url="http://localhost:8000/v1", api_key="EMPTY")
        client._create_streaming = _record_and_return(
            _fake_outcome(content="{'ingredient': ['a', 'a', 'a'", finish_reason="loop_cutoff")
        )

        r = await client.extract(_model(is_local=True), image_path, "extract this")

        assert r.ok is True
        assert r.loop_cutoff is True
        assert r.truncated is False
        assert r.finish_reason == "loop_cutoff"


class TestRepetitionLoopDetection:
    def test_normal_json_is_not_flagged(self):
        text = '{"product_name": "Bánh xốp", "ingredient": ["Bột mì", "Đường", "Muối"]}'
        assert _has_repetition_loop(text) is False

    def test_not_enough_text_yet_is_not_flagged(self):
        assert _has_repetition_loop("short prefix") is False

    def test_short_single_character_run_is_not_flagged(self):
        # 20 chars isn't enough to confirm min_repeats(5) periods of
        # min_unit(8)+ chars each.
        assert _has_repetition_loop("a" * 20) is False

    def test_long_single_character_run_is_flagged(self):
        assert _has_repetition_loop("a" * 50) is True

    def test_multi_word_phrase_cycle_is_flagged(self):
        # the real failure mode observed on a live Vintern-3B-beta call
        # (image 0003, rp=1.15): a several-word phrase repeating verbatim.
        unit = "hạt bí đỏ, hạt bí trắng, hạt bí xanh, hạt bí đen, "
        text = "some real content first " + unit * 6
        assert _has_repetition_loop(text) is True

    def test_repeated_json_field_is_flagged(self):
        # the other observed failure mode (image 0001, rp=1.0): a whole
        # repeated {name: value} object.
        unit = "{'name': 'Chất béo thực vật', 'value': '0 g'}, "
        text = "prefix " + unit * 6
        assert _has_repetition_loop(text) is True

    def test_scattered_short_repeats_are_not_flagged(self):
        # distinct nutrition rows legitimately sharing a unit ("g") must not
        # false-positive just because "g" recurs.
        text = "Chất đạm: 5 g, Chất béo: 3 g, Carbohydrat: 10 g, Natri: 20 mg"
        assert _has_repetition_loop(text) is False


class _FakeChunk:
    def __init__(self, delta_content=None, finish_reason=None, usage=None):
        delta = SimpleNamespace(content=delta_content)
        choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
        self.choices = [choice]
        self.usage = usage


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks
        self.closed = False

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for chunk in self._chunks:
            yield chunk

    async def close(self):
        self.closed = True


class TestConsumeStreamWithLoopGuard:
    async def test_cuts_off_early_and_closes_on_detected_loop(self):
        unit = "{'name': 'Chất béo thực vật', 'value': '0 g'}, "
        chunks = [_FakeChunk(delta_content="prefix ")]
        chunks += [_FakeChunk(delta_content=unit) for _ in range(10)]
        stream = _FakeStream(chunks)
        client = VLMClient(base_url="http://localhost:8000/v1", api_key="EMPTY")

        outcome = await client._consume_stream_with_loop_guard(stream)

        assert outcome.finish_reason == "loop_cutoff"
        assert stream.closed is True
        assert outcome.content.startswith("prefix ")
        assert outcome.content.count(unit) < 10  # cut short, not all 10 consumed
        assert 0 < outcome.output_tokens <= 11

    async def test_normal_stream_completes_without_cutoff(self):
        chunks = [
            _FakeChunk(delta_content='{"x": 1}'),
            _FakeChunk(delta_content=None, finish_reason="stop"),
        ]
        stream = _FakeStream(chunks)
        client = VLMClient(base_url="http://localhost:8000/v1", api_key="EMPTY")

        outcome = await client._consume_stream_with_loop_guard(stream)

        assert outcome.finish_reason == "stop"
        assert stream.closed is False
        assert outcome.content == '{"x": 1}'

    async def test_usage_chunk_overrides_chunk_count_estimate(self):
        chunks = [
            _FakeChunk(delta_content='{"x": 1}'),
            _FakeChunk(usage=SimpleNamespace(prompt_tokens=42, completion_tokens=7)),
        ]
        stream = _FakeStream(chunks)
        client = VLMClient(base_url="http://localhost:8000/v1", api_key="EMPTY")

        outcome = await client._consume_stream_with_loop_guard(stream)

        assert outcome.input_tokens == 42
        assert outcome.output_tokens == 7


def _record_and_return(response):
    async def fake_create(model, instruction, image_data_url, extra_body):
        return response

    return fake_create
