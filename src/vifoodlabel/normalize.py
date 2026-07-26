"""Vietnamese text, number/unit, and date normalization used by the matcher.

Two text-normalization levels are exposed on purpose:
- `normalize_text` (NFC, trimmed, lowercased) is used for all real matching.
- `strip_diacritics` is used *only* to detect "content is right, diacritics are
  wrong" as its own error-taxonomy category — it is never used to decide
  whether a field counts as correct.
"""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"[-+]?\d[\d.,]*")

_UNIT_ALIASES = {
    "g": "g", "gr": "g", "gram": "g", "grams": "g",
    "mg": "mg", "milligram": "mg",
    "kg": "kg",
    "kcal": "kcal", "calo": "kcal", "calorie": "kcal", "calories": "kcal",
    "ml": "ml", "milliliter": "ml",
    "l": "l", "lit": "l", "liter": "l", "lít": "l",
    "%": "%", "pct": "%", "percent": "%",
    "mcg": "mcg", "microgram": "mcg", "µg": "mcg",
}

_DATE_PATTERNS = [
    (re.compile(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})$"), lambda m: f"{m[3]}-{int(m[2]):02d}-{int(m[1]):02d}"),
    (re.compile(r"^(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})$"), lambda m: f"{m[1]}-{int(m[2]):02d}-{int(m[3]):02d}"),
]


def normalize_text(s: str) -> str:
    """NFC-normalize, collapse whitespace, lowercase. The baseline for all matching."""
    s = unicodedata.normalize("NFC", s or "")
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s.lower()


def strip_diacritics(s: str) -> str:
    """Remove Vietnamese diacritics (for error-taxonomy diagnosis only)."""
    s = s.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", s)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", stripped)


def normalize_number(raw: str) -> float | None:
    """Parse a Vietnamese- or English-formatted number: '34,242' / '0.25' / '1.234,5'."""
    match = _NUMBER_RE.search(raw or "")
    if not match:
        return None
    token = match.group(0)
    has_dot = "." in token
    has_comma = "," in token
    if has_dot and has_comma:
        token = token.replace(".", "").replace(",", ".")
    elif has_comma:
        token = token.replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return None


def normalize_unit(raw: str) -> str | None:
    token = normalize_text(raw).replace(" ", "")
    if not token:
        return None
    return _UNIT_ALIASES.get(token, token)


def extract_number_unit(raw: str) -> tuple[float | None, str | None]:
    """Best-effort split of a value string like '70 kcal' -> (70.0, 'kcal')."""
    text = raw or ""
    num_match = _NUMBER_RE.search(text)
    if not num_match:
        return None, None
    number = normalize_number(num_match.group(0))
    remainder = text[num_match.end():].strip()
    unit_match = re.match(r"[^\d]*", remainder)
    unit_raw = unit_match.group(0).strip() if unit_match else ""
    unit = normalize_unit(unit_raw) if unit_raw else None
    return number, unit


def normalize_date(raw: str) -> str:
    """Best-effort date normalization to ISO yyyy-mm-dd; falls back to normalized text
    (many mfg_date/expiry_date values are instructions, not literal dates)."""
    text = normalize_text(raw)
    compact = text.replace(" ", "")
    for pattern, formatter in _DATE_PATTERNS:
        m = pattern.match(compact)
        if m:
            return formatter(m)
    return text
