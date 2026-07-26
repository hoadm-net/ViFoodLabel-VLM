"""Robust extraction of a JSON object from raw VLM text output.

Models frequently wrap JSON in markdown fences, add a stray sentence before/after,
or emit slightly malformed JSON (trailing commas, smart quotes). This tries a
sequence of increasingly forgiving strategies and records which one worked (or
that all of them failed) — a `json_repair_used` / `parse_failed` flag feeds
directly into the error taxonomy (malformed JSON is a scored failure mode, not
a crash).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from json_repair import repair_json

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


@dataclass
class ParseResult:
    data: dict | None
    parse_failed: bool
    used_repair: bool
    raw_text: str


def _strip_fence(text: str) -> str:
    match = _FENCE_RE.search(text)
    return match.group(1) if match else text


def _largest_brace_span(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def parse_model_json(raw_text: str) -> ParseResult:
    text = raw_text.strip()

    for candidate in (text, _strip_fence(text)):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return ParseResult(data=data, parse_failed=False, used_repair=False, raw_text=raw_text)
        except (json.JSONDecodeError, ValueError):
            pass

    fenced = _strip_fence(text)
    span = _largest_brace_span(fenced)
    if span is None:
        # No closing brace anywhere -- e.g. truncated before any nested
        # object completed. repair_json can often still infer and close
        # the missing structure on its own; worth trying on whatever's
        # left from the first '{' onward rather than giving up here.
        start = fenced.find("{")
        span = fenced[start:] if start != -1 else None

    if span is not None:
        try:
            data = json.loads(span)
            if isinstance(data, dict):
                return ParseResult(data=data, parse_failed=False, used_repair=False, raw_text=raw_text)
        except (json.JSONDecodeError, ValueError):
            pass

        try:
            repaired = repair_json(span, return_objects=True)
            if isinstance(repaired, dict):
                return ParseResult(data=repaired, parse_failed=False, used_repair=True, raw_text=raw_text)
        except Exception:
            pass

    return ParseResult(data=None, parse_failed=True, used_repair=False, raw_text=raw_text)
