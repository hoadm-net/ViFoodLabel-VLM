"""ocr.py -- read-side of the OCR -> LLM baseline's transcript cache."""

from __future__ import annotations

import vifoodlabel.ocr as ocr


def test_missing_transcript_is_none(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "OCR_RESULTS_DIR", tmp_path)
    assert ocr.load_ocr_text("eng", "0001") is None


def test_transcript_loaded_and_stripped(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "OCR_RESULTS_DIR", tmp_path)
    (tmp_path / "eng").mkdir()
    (tmp_path / "eng" / "0001.txt").write_text("Bánh xốp\nĐường\n", encoding="utf-8")
    assert ocr.load_ocr_text("eng", "0001") == "Bánh xốp\nĐường"


def test_empty_transcript_is_empty_string_not_none(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "OCR_RESULTS_DIR", tmp_path)
    (tmp_path / "eng").mkdir()
    (tmp_path / "eng" / "0002.txt").write_text("\n", encoding="utf-8")
    assert ocr.load_ocr_text("eng", "0002") == ""


def test_condition_name_never_collides_with_image_conditions():
    assert ocr.ocr_condition_name("paddle-vietocr") == "vi_zero__ocr-paddle-vietocr"
