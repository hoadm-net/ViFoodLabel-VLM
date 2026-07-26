"""Thin async wrapper around any OpenAI-compatible chat-completions endpoint.

Used both for OpenRouter (the 6 hosted models) and a local vLLM server (the
self-hosted Vintern-3B-beta) — `ModelSpec.base_url`/`api_key_env` decide which.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from vifoodlabel.config import ModelSpec
from vifoodlabel.io_utils import image_to_data_url

load_dotenv()

RETRYABLE_EXCEPTIONS = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)

DEFAULT_MAX_OUTPUT_TOKENS = 3000
DEFAULT_TEMPERATURE = 0.0

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

    @property
    def ok(self) -> bool:
        return self.error is None and self.content is not None


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


class VLMClient:
    def __init__(self, base_url: str, api_key: str):
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=1, max=60),
        reraise=True,
    )
    async def _call(self, model: ModelSpec, instruction: str, image_data_url: str):
        return await self._client.chat.completions.create(
            model=model.slug,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
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

    async def extract(self, model: ModelSpec, image_path: Path, instruction: str) -> RawResponse:
        image_data_url = image_to_data_url(image_path)
        start = time.monotonic()
        try:
            response = await self._call(model, instruction, image_data_url)
        except (*RETRYABLE_EXCEPTIONS, APIStatusError) as exc:
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
            )

        return RawResponse(
            model_slug=model.slug,
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=model.cost_usd(input_tokens, output_tokens),
            latency_s=latency_s,
        )

    async def aclose(self) -> None:
        await self._client.close()


def build_client(model: ModelSpec) -> VLMClient:
    return VLMClient(base_url=model.resolved_base_url, api_key=resolve_api_key(model))
