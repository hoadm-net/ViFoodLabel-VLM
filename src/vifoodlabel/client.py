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
            return {}
        return {"reasoning": {"enabled": False}}

    async def _create(self, model: ModelSpec, instruction: str, image_data_url: str, extra_body: dict):
        return await self._client.chat.completions.create(
            model=model.slug,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            extra_body=extra_body,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
        )

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=1, max=60),
        reraise=True,
    )
    async def _call(self, model: ModelSpec, instruction: str, image_data_url: str):
        extra_body = self._extra_body_for(model)
        try:
            return await self._create(model, instruction, image_data_url, extra_body)
        except BadRequestError as exc:
            if extra_body and _REASONING_MANDATORY_MARKER in str(exc).lower():
                self._reasoning_disable_unsupported.add(model.slug)
                return await self._create(model, instruction, image_data_url, {})
            raise

    async def extract(self, model: ModelSpec, image_path: Path, instruction: str) -> RawResponse:
        image_data_url = image_to_data_url(image_path)
        start = time.monotonic()
        try:
            response = await self._call(model, instruction, image_data_url)
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
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        finish_reason = response.choices[0].finish_reason if response.choices else None
        content = response.choices[0].message.content if response.choices else None
        if content is None:
            return RawResponse(
                model_slug=model.slug,
                content=None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=model.cost_usd(input_tokens, output_tokens),
                latency_s=latency_s,
                error="empty_response: no content in first choice",
                finish_reason=finish_reason,
            )

        return RawResponse(
            model_slug=model.slug,
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=model.cost_usd(input_tokens, output_tokens),
            latency_s=latency_s,
            finish_reason=finish_reason,
        )

    async def aclose(self) -> None:
        await self._client.close()


def build_client(model: ModelSpec) -> VLMClient:
    return VLMClient(base_url=model.resolved_base_url, api_key=resolve_api_key(model))
