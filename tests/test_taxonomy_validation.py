"""taxonomy_validation.py -- stratified sampling, judge-reply parsing and
rule-vs-reference agreement for the Tier 4 validation."""

from __future__ import annotations

import math

import pytest

from vifoodlabel.taxonomy_validation import (
    category_agreement,
    describe_flags,
    parse_judge_response,
    stratified_sample,
)


class TestStratifiedSample:
    def _rows(self):
        return [{"rule_category": "wrong_value", "i": i} for i in range(100)] + [
            {"rule_category": "diacritics", "i": i} for i in range(3)
        ]

    def test_caps_large_categories_and_keeps_all_of_rare_ones(self):
        sample = stratified_sample(self._rows(), per_category=10, seed=1)
        cats = [r["rule_category"] for r in sample]
        assert cats.count("wrong_value") == 10
        assert cats.count("diacritics") == 3

    def test_deterministic_given_seed(self):
        a = stratified_sample(self._rows(), per_category=10, seed=7)
        b = stratified_sample(self._rows(), per_category=10, seed=7)
        assert a == b


class TestParseJudgeResponse:
    def test_valid_reply(self):
        v = parse_judge_response('{"category": "Hallucination", "content_on_label": false, "rationale": "Not on label."}')
        assert v == {"category": "hallucination", "content_on_label": False, "rationale": "Not on label."}

    def test_fenced_reply_still_parses(self):
        v = parse_judge_response('```json\n{"category": "wrong_value", "content_on_label": true, "rationale": "x"}\n```')
        assert v["category"] == "wrong_value"

    def test_out_of_vocabulary_category_is_none(self):
        # Must not silently count as (dis)agreement with some real category.
        assert parse_judge_response('{"category": "wrong_language"}')["category"] is None

    def test_empty_or_garbage_reply(self):
        assert parse_judge_response(None)["category"] is None
        assert parse_judge_response("I think it's a hallucination")["category"] is None


class TestCategoryAgreement:
    def test_perfect_agreement(self):
        overall, per_cat, _ = category_agreement(["a", "b", "a"], ["a", "b", "a"])
        assert overall["accuracy"] == 1.0
        assert overall["cohen_kappa"] == pytest.approx(1.0)
        assert (per_cat["rule_precision"] == 1.0).all()

    def test_rule_precision_and_recall(self):
        rule = ["hallucination", "hallucination", "hallucination", "wrong_value"]
        ref = ["hallucination", "wrong_value", "wrong_value", "wrong_value"]
        overall, per_cat, confusion = category_agreement(rule, ref)
        h = per_cat.set_index("category").loc["hallucination"]
        assert h["rule_precision"] == pytest.approx(1 / 3)
        assert h["rule_recall"] == pytest.approx(1.0)
        w = per_cat.set_index("category").loc["wrong_value"]
        assert w["rule_recall"] == pytest.approx(1 / 3)
        assert overall["accuracy"] == pytest.approx(0.5)
        assert confusion.loc["hallucination", "wrong_value"] == 2

    def test_category_never_assigned_by_rule_has_nan_precision(self):
        _, per_cat, _ = category_agreement(["a", "a"], ["a", "b"])
        assert math.isnan(per_cat.set_index("category").loc["b", "rule_precision"])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            category_agreement(["a"], [])


class TestDescribeFlags:
    def test_structural_flags_described(self):
        text = describe_flags(["output_truncated", "json_repair_used"])
        assert "cut off at the output-token limit" in text
        assert "repair" not in text  # non-deciding issues are not passed on

    def test_no_flags(self):
        assert describe_flags([]).startswith("none")
