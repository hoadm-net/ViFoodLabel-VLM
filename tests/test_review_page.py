"""review_page.py -- the offline taxonomy-review page."""

from __future__ import annotations

import json

from vifoodlabel.prompts import _load
from vifoodlabel.review_page import REVIEW_COLUMNS, build_review_page, category_definitions


def _rows():
    return [{"image_id": "0001", "model_key": "m", "field": "origin", "gt_value": "Việt Nam",
             "pred_value": "</script><b>x</b>", "flags": "none", "judge_category": "wrong_value", "judge_rationale": "r"}]


def test_no_placeholders_left():
    page = build_review_page(_rows(), ["wrong_value"], "defs")
    for placeholder in ("__ROWS__", "__CATEGORIES__", "__DEFS__", "__COLUMNS__"):
        assert placeholder not in page


def test_model_output_cannot_close_the_script_block():
    page = build_review_page(_rows(), ["wrong_value"], "defs")
    script = page.split("<script>")[1]
    assert "</script>" not in script.split("</script>\n</body>")[0]


def test_rows_round_trip_and_rule_category_absent():
    page = build_review_page(_rows(), ["wrong_value"], "defs")
    blob = page.split("const ROWS = ")[1].split(";\n")[0].replace("<\\/", "</")
    assert json.loads(blob) == _rows()
    assert "rule_category" not in page


def test_export_columns_match_import_expectation():
    assert REVIEW_COLUMNS[:3] == ["image_id", "model_key", "field"]
    assert {"human_category", "human_confirmed", "human_note"} <= set(REVIEW_COLUMNS)


def test_definitions_are_the_judges_category_list():
    defs = category_definitions(_load("taxonomy_judge_en.txt"))
    assert defs.startswith("Error categories")
    assert "hallucination" in defs and "Return ONLY" not in defs
