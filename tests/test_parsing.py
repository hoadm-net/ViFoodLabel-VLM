"""parsing.py -- robust JSON extraction from raw VLM text output."""

from __future__ import annotations

import json

from vifoodlabel.parsing import parse_model_json


class TestParseModelJson:
    def test_clean_json(self):
        r = parse_model_json('{"product_name": "x"}')
        assert not r.parse_failed
        assert not r.used_repair
        assert r.data == {"product_name": "x"}

    def test_markdown_fenced_json(self):
        r = parse_model_json('```json\n{"product_name": "x"}\n```')
        assert not r.parse_failed
        assert r.data == {"product_name": "x"}

    def test_fenced_without_language_tag(self):
        r = parse_model_json('```\n{"product_name": "x"}\n```')
        assert not r.parse_failed
        assert r.data == {"product_name": "x"}

    def test_stray_prose_before_and_after(self):
        r = parse_model_json('Here is the JSON:\n{"product_name": "x"}\nHope that helps!')
        assert not r.parse_failed
        assert r.data == {"product_name": "x"}

    def test_trailing_comma_gets_repaired(self):
        r = parse_model_json('{"product_name": "x", "origin": "y",}')
        assert not r.parse_failed
        assert r.used_repair is True
        assert r.data["product_name"] == "x"

    def test_truncated_mid_json_gets_repaired_when_possible(self):
        # The real truncation case: response cut off mid-array. json_repair
        # closes the structure; result is "valid" but incomplete -- callers
        # must check finish_reason separately to detect this (see runner.py).
        r = parse_model_json('{"product_name": "x", "ingredient": ["a", "b"')
        assert not r.parse_failed
        assert r.used_repair is True

    def test_completely_broken_text_fails_gracefully(self):
        r = parse_model_json("I cannot process this image.")
        assert r.parse_failed is True
        assert r.data is None

    def test_empty_string(self):
        r = parse_model_json("")
        assert r.parse_failed is True

    def test_non_object_json_is_treated_as_failure(self):
        # A bare JSON array or scalar isn't a valid prediction shape.
        r = parse_model_json("[1, 2, 3]")
        assert r.parse_failed is True

    def test_real_vietnamese_content_roundtrips(self):
        original = {"product_name": "Bánh xốp ống Deka Jumbo cà phê trắng White Coffee",
                    "ingredient": ["Bột mì (34,242%)", "Đường"]}
        raw_text = json.dumps(original, ensure_ascii=False)
        r = parse_model_json(raw_text)
        assert not r.parse_failed
        assert r.data == original
