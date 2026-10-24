"""Read-side of the OCR -> LLM baseline: cached OCR transcripts.

The OCR itself runs outside this package (scripts/run_ocr.py, in its own
environment -- see that script's docstring for why), and writes one
`results/ocr/<engine>/<image_id>.txt` per image. This module only locates
and loads those files; it never runs OCR, so the harness stays free of the
OCR stack's dependencies.
"""

from __future__ import annotations

from pathlib import Path

from vifoodlabel.config import OCR_RESULTS_DIR

DEFAULT_OCR_ENGINE = "paddle-vietocr"


def ocr_text_path(engine: str, image_id: str) -> Path:
    return OCR_RESULTS_DIR / engine / f"{image_id}.txt"


def load_ocr_text(engine: str, image_id: str) -> str | None:
    """The image's OCR transcript, or None if scripts/run_ocr.py hasn't
    produced it yet. An empty transcript (no text detected) is returned as
    "" -- a legitimate OCR outcome, still worth sending to the LLM."""
    path = ocr_text_path(engine, image_id)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def ocr_condition_name(engine: str, base_condition: str = "vi_zero") -> str:
    """Cache/scoring condition for "this prompt condition, fed OCR text from
    `engine` instead of the image" -- kept distinct from every image-input
    condition so the two can never collide in the cache."""
    return f"{base_condition}__ocr-{engine}"
