"""client.py -- includes a regression test for a real bug: Gemini 3.1 Pro
rejects reasoning={enabled: false} with a 400 "Reasoning is mandatory for
this endpoint" error, which used to propagate and fail every single call
for that model instead of falling back to its default reasoning behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from openai import BadRequestError
from PIL import Image

from vifoodlabel.client import LOCAL_EXTRA_BODY, VLMClient
from vifoodlabel.config import ModelSpec


def _model(is_local: bool = False) -> ModelSpec:
    return ModelSpec(
        key="test-model", slug="test/model", display_name="Test", group="test",
        open_weight=True, price_input_per_m=1.0, price_output_per_m=2.0,
        base_url="http://localhost:8000/v1" if is_local else None,
    )


def _fake_response(content: str | None = "{}", finish_reason: str = "stop",
                    prompt_tokens: int = 10, completion_tokens: int = 5):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


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
        client._create = _record_and_return(_fake_response(content='{"x": 1}', finish_reason="stop"))

        r = await client.extract(_model(), image_path, "extract this")

        assert r.ok is True
        assert r.content == '{"x": 1}'
        assert r.input_tokens == 10
        assert r.output_tokens == 5
        assert r.finish_reason == "stop"
        assert r.truncated is False

    async def test_truncated_finish_reason(self, image_path):
        client = VLMClient(base_url="http://example.invalid/v1", api_key="test")
        client._create = _record_and_return(_fake_response(content='{"x": 1', finish_reason="length"))

        r = await client.extract(_model(), image_path, "extract this")

        assert r.truncated is True
        assert r.finish_reason == "length"

    async def test_empty_content_is_an_error(self, image_path):
        client = VLMClient(base_url="http://example.invalid/v1", api_key="test")
        client._create = _record_and_return(_fake_response(content=None))

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
            return _fake_response(content='{"x": 1}')

        client._create = fake_create

        response = await client._call(_model(), "extract this", "data:image/jpeg;base64,x")

        assert response.choices[0].message.content == '{"x": 1}'
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
            return _fake_response()

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


def _record_and_return(response):
    async def fake_create(model, instruction, image_data_url, extra_body):
        return response

    return fake_create
