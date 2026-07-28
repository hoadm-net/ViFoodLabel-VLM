"""Thin async wrapper around any OpenAI-compatible chat-completions endpoint.

Used both for OpenRouter (the 6 hosted models) and a local vLLM server (the
self-hosted Vintern-3B-beta) — `ModelSpec.base_url`/`api_key_env` decide which.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from vifoodlabel.config import ModelSpec
from vifoodlabel.io_utils import image_to_data_url

load_dotenv()

# json.JSONDecodeError: seen in practice -- OpenRouter occasionally returns a
# truncated/malformed HTTP response body (HTTP 200, but the SDK's response.json()
# fails), most likely a transient proxy/provider hiccup. Worth a retry like any
# other transient failure rather than surfacing as a hard crash.
RETRYABLE_EXCEPTIONS = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError, json.JSONDecodeError)

# Generous headroom, not a real budget: some labels have long ingredient/
# additive/nutrition lists (dense imported products especially), and the
# 3000-token default silently truncated several real responses mid-JSON
# (json_repair then patched the cut-off into "valid" JSON, hiding it -- see
# RawResponse.truncated / finish_reason below). Kept finite rather than
# unbounded purely as a runaway-generation cost guard for unattended batch
# runs, not because labels are expected to need this much.
DEFAULT_MAX_OUTPUT_TOKENS = 16000
DEFAULT_TEMPERATURE = 0.0
TRUNCATION_FINISH_REASON = "length"

# vLLM's own convention for a local, no-auth-required server (see vLLM docs).
LOCAL_API_KEY_PLACEHOLDER = "EMPTY"

# Vintern-3B-beta (the only self-hosted/local model) reliably falls into a
# degenerate repetition loop at temperature=0 with no repetition penalty --
# observed hitting the full 16000-token max_tokens ceiling on every call
# without emitting an EOS, at ~470s/call. Empirically grid-searched on this
# model+prompt: repetition_penalty=1.0 (off) and 1.3 both loop to the ceiling
# (1.3 into a *different* degenerate loop, e.g. repeating a single rare
# token) -- 1.15 is the value that reliably stops cleanly at a real EOS in a
# few hundred tokens. vLLM's OpenAI-compatible server accepts this as a
# non-standard sampling param via extra_body.
LOCAL_EXTRA_BODY = {"repetition_penalty": 1.15}

# repetition_penalty=1.15 only cuts the loop rate, it doesn't eliminate it
# (still observed on ~1/3 real images) -- and Tier 3's perturbed (blurred/
# glared/rotated) images are expected to confuse this small model more often,
# not less. A full loop burns the entire 16000-token budget at ~30 tok/s on a
# single GPU (~8 minutes) for zero usable content past the point it started
# cycling, which doesn't scale to hundreds of perturbed-image calls. Detected
# by watching the streamed output for a short substring ("unit") repeated
# many times back-to-back in a row -- catches both a single-token spam loop
# and a multi-word phrase cycle (both observed in practice) -- and cutting
# the connection the moment that's confirmed, instead of waiting for the
# server's own max_tokens ceiling. Local/self-hosted only (`extract()`):
# never observed on the 6 OpenRouter models, and streaming would change an
# already-validated call path for the paid models for no observed benefit.
LOOP_WINDOW_CHARS = 800
LOOP_MIN_UNIT_CHARS = 8
LOOP_MAX_UNIT_CHARS = 200
LOOP_MIN_REPEATS = 5
LOOP_CUTOFF_FINISH_REASON = "loop_cutoff"


def _has_repetition_loop(
    text: str,
    window: int = LOOP_WINDOW_CHARS,
    min_unit: int = LOOP_MIN_UNIT_CHARS,
    max_unit: int = LOOP_MAX_UNIT_CHARS,
    min_repeats: int = LOOP_MIN_REPEATS,
) -> bool:
    """True if the tail of `text` is some short substring repeated
    `min_repeats`+ times back-to-back. `min_unit` excludes trivial repeats
    (a run of commas/spaces from normal JSON formatting); `min_repeats`
    excludes coincidental short repeats (e.g. several distinct nutrition
    rows that happen to share a unit like "g")."""
    tail = text[-window:]
    n = len(tail)
    for period in range(min_unit, min(max_unit, n // min_repeats) + 1):
        span = period * min_repeats
        segment = tail[-span:]
        if segment[:period] * min_repeats == segment:
            return True
    return False


@dataclass
class _CallOutcome:
    content: str | None
    finish_reason: str | None
    input_tokens: int
    output_tokens: int


@dataclass
class RawResponse:
    model_slug: str
    content: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    error: str | None = None
    finish_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.content is not None

    @property
    def truncated(self) -> bool:
        """Hit the max_tokens ceiling -- content (if any) may be a cut-off
        prefix. json_repair can often still parse it, which would otherwise
        silently hide that the last field(s) are missing, not "not extracted"."""
        return self.finish_reason == TRUNCATION_FINISH_REASON

    @property
    def loop_cutoff(self) -> bool:
        """We detected a degenerate repetition loop mid-generation and closed
        the connection early (see LOOP_CUTOFF_FINISH_REASON) -- distinct from
        `truncated`, which is the *server's own* max_tokens ceiling."""
        return self.finish_reason == LOOP_CUTOFF_FINISH_REASON


def resolve_api_key(model: ModelSpec) -> str:
    env_name = model.api_key_env or "OPENROUTER_API_KEY"
    key = os.environ.get(env_name)
    if key:
        return key
    if model.is_local:
        return LOCAL_API_KEY_PLACEHOLDER
    raise RuntimeError(
        f"{env_name} is not set. Copy .env.example to .env and fill in your key."
    )


_REASONING_MANDATORY_MARKER = "reasoning is mandatory"


class VLMClient:
    def __init__(self, base_url: str, api_key: str):
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        # Models (by slug) whose endpoint has told us they reject
        # reasoning={enabled: false} -- remembered so we stop retrying the
        # doomed request and go straight to their default (reasoning-on)
        # behavior on every subsequent call.
        self._reasoning_disable_unsupported: set[str] = set()

    def _extra_body_for(self, model: ModelSpec) -> dict:
        # Disabled by default across every OpenRouter-routed model -- observed
        # MiMo-V2.5 defaulting to thinking-enabled and burning the entire
        # max_tokens budget on hidden reasoning without ever emitting an
        # answer (a known OpenRouter/provider behavior, not specific to our
        # prompt). Applying the same reasoning=off condition to all models
        # controls for that confound uniformly rather than special-casing one
        # model -- except where the provider flatly rejects it (some models,
        # e.g. Gemini 3.1 Pro, error with "reasoning is mandatory for this
        # endpoint"), in which case we fall back to that model's default.
        # Skipped entirely for self-hosted models (Vintern/vLLM): vLLM's
        # OpenAI-compatible server may reject this OpenRouter-specific field.
        if model.is_local or model.slug in self._reasoning_disable_unsupported:
            return dict(LOCAL_EXTRA_BODY) if model.is_local else {}
        return {"reasoning": {"enabled": False}}

    def _messages_for(self, instruction: str, image_data_url: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ]

    async def _create(self, model: ModelSpec, instruction: str, image_data_url: str, extra_body: dict) -> _CallOutcome:
        response = await self._client.chat.completions.create(
            model=model.slug,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            extra_body=extra_body,
            messages=self._messages_for(instruction, image_data_url),
        )
        usage = response.usage
        choice = response.choices[0] if response.choices else None
        return _CallOutcome(
            content=choice.message.content if choice else None,
            finish_reason=choice.finish_reason if choice else None,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

    async def _consume_stream_with_loop_guard(self, stream) -> _CallOutcome:
        """Drain a chat-completion stream, watching for a degenerate
        repetition loop (see LOOP_* constants) and closing the connection
        early if one is confirmed, instead of waiting for max_tokens.

        `stream` just needs to be an async iterable of chunks shaped like
        the OpenAI SDK's, plus an async `close()` -- kept duck-typed so the
        loop-detection logic itself is testable without the real SDK.
        """
        parts: list[str] = []
        finish_reason: str | None = None
        input_tokens = 0
        output_tokens = 0
        async for chunk in stream:
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                input_tokens = chunk_usage.prompt_tokens
                output_tokens = chunk_usage.completion_tokens
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta.content if choice.delta else None
            if delta:
                parts.append(delta)
                if _has_repetition_loop("".join(parts)):
                    finish_reason = LOOP_CUTOFF_FINISH_REASON
                    await stream.close()
                    break
            if choice.finish_reason:
                finish_reason = choice.finish_reason
        return _CallOutcome(
            content="".join(parts) or None,
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            # Cut short before the server's own final usage chunk -- fall
            # back to one token per streamed delta (vLLM streams ~1
            # token/chunk), close enough for the cost ledger given this is
            # the free self-hosted model (price 0/0, so cost_usd is
            # unaffected either way).
            output_tokens=output_tokens or len(parts),
        )

    async def _create_streaming(self, model: ModelSpec, instruction: str, image_data_url: str, extra_body: dict) -> _CallOutcome:
        stream = await self._client.chat.completions.create(
            model=model.slug,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            extra_body=extra_body,
            stream=True,
            stream_options={"include_usage": True},
            messages=self._messages_for(instruction, image_data_url),
        )
        return await self._consume_stream_with_loop_guard(stream)

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=1, max=60),
        reraise=True,
    )
    async def _call(self, model: ModelSpec, instruction: str, image_data_url: str) -> _CallOutcome:
        extra_body = self._extra_body_for(model)
        creator = self._create_streaming if model.is_local else self._create
        try:
            return await creator(model, instruction, image_data_url, extra_body)
        except BadRequestError as exc:
            if extra_body and _REASONING_MANDATORY_MARKER in str(exc).lower():
                self._reasoning_disable_unsupported.add(model.slug)
                return await creator(model, instruction, image_data_url, {})
            raise

    async def extract(self, model: ModelSpec, image_path: Path, instruction: str) -> RawResponse:
        image_data_url = image_to_data_url(image_path)
        start = time.monotonic()
        try:
            outcome = await self._call(model, instruction, image_data_url)
        except Exception as exc:
            # Deliberately broad: this call is one of many running concurrently
            # in a batch (600 images x 7 models x N conditions) -- a single
            # unexpected failure (retryable-exhausted, APIStatusError, or
            # anything not explicitly anticipated) must degrade to a recorded
            # per-item error, never take down the whole in-flight batch.
            return RawResponse(
                model_slug=model.slug,
                content=None,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                latency_s=time.monotonic() - start,
                error=f"{type(exc).__name__}: {exc}",
            )

        latency_s = time.monotonic() - start
        if outcome.content is None:
            return RawResponse(
                model_slug=model.slug,
                content=None,
                input_tokens=outcome.input_tokens,
                output_tokens=outcome.output_tokens,
                cost_usd=model.cost_usd(outcome.input_tokens, outcome.output_tokens),
                latency_s=latency_s,
                error="empty_response: no content in first choice",
                finish_reason=outcome.finish_reason,
            )

        return RawResponse(
            model_slug=model.slug,
            content=outcome.content,
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
            cost_usd=model.cost_usd(outcome.input_tokens, outcome.output_tokens),
            latency_s=latency_s,
            finish_reason=outcome.finish_reason,
        )

    async def aclose(self) -> None:
        await self._client.close()


def build_client(model: ModelSpec) -> VLMClient:
    return VLMClient(base_url=model.resolved_base_url, api_key=resolve_api_key(model))
