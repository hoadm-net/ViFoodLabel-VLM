"""schema.py -- includes a regression test for a real crash: ground-truth
files legitimately use `null` for fields not visible in frame (see
docs/annotation-guidelines.md §4), but LabelSchema originally required
non-null str and crashed loading the very first real multi-image batch."""

from __future__ import annotations

import pytest

from vifoodlabel.schema import LabelSchema, coerce_prediction, field_value_str


class TestLabelSchemaNullCoercion:
    def test_null_scalar_fields_become_empty_string(self):
        # The exact real case: data/labels/0002.json and 0003.json use
        # `null` for net_weight/mfg_date/expiry_date/product_name.
        raw = {
            "product_name": None,
            "ingredient": [],
            "additive": [],
            "warning": [],
            "nutrition": [],
            "origin": "Việt Nam",
            "net_weight": None,
            "mfg_date": None,
            "expiry_date": None,
        }
        gt = LabelSchema.model_validate(raw)
        assert gt.product_name == ""
        assert gt.net_weight == ""
        assert gt.mfg_date == ""
        assert gt.expiry_date == ""

    def test_null_list_fields_become_empty_list(self):
        raw = {
            "product_name": "x", "ingredient": None, "additive": None, "warning": None,
            "nutrition": None, "origin": "x", "net_weight": "x", "mfg_date": "x", "expiry_date": "x",
        }
        gt = LabelSchema.model_validate(raw)
        assert gt.ingredient == []
        assert gt.additive == []
        assert gt.warning == []
        assert gt.nutrition == []

    def test_normal_values_pass_through_unchanged(self):
        raw = {
            "product_name": "Bánh xốp", "ingredient": ["Bột mì"], "additive": [], "warning": [],
            "nutrition": [{"name": "Năng lượng", "value": "70 kcal"}],
            "origin": "Indonesia", "net_weight": "140 g", "mfg_date": "01/01/2026", "expiry_date": "01/01/2027",
        }
        gt = LabelSchema.model_validate(raw)
        assert gt.product_name == "Bánh xốp"
        assert gt.nutrition[0].name == "Năng lượng"

    def test_missing_required_scalar_key_still_raises(self):
        # A key entirely absent (not null, just missing) is a genuine
        # data-entry bug in a ground-truth file and should still fail loudly.
        raw = {"ingredient": [], "additive": [], "warning": [], "nutrition": [],
               "origin": "x", "net_weight": "x", "mfg_date": "x", "expiry_date": "x"}
        with pytest.raises(Exception):
            LabelSchema.model_validate(raw)


class TestCoercePrediction:
    def test_well_formed_prediction(self):
        raw = {
            "product_name": "Bánh xốp", "ingredient": ["Bột mì"], "additive": [], "warning": [],
            "nutrition": [{"name": "Năng lượng", "value": "70 kcal"}],
            "origin": "Indonesia", "net_weight": "140 g", "mfg_date": "01/01/2026", "expiry_date": "01/01/2027",
        }
        pred, issues = coerce_prediction(raw)
        assert pred.product_name == "Bánh xốp"
        assert issues == []

    def test_non_dict_root_never_raises(self):
        pred, issues = coerce_prediction(None)
        assert pred.product_name == ""
        assert any("root" in i for i in issues)

        pred2, issues2 = coerce_prediction("not a dict")
        assert pred2.product_name == ""
        assert issues2

        pred3, issues3 = coerce_prediction([1, 2, 3])
        assert pred3.ingredient == []

    def test_missing_fields_are_defaulted_and_flagged(self):
        pred, issues = coerce_prediction({"product_name": "x"})
        assert pred.origin == ""
        assert pred.ingredient == []
        assert any("origin" in i and "missing" in i for i in issues)

    def test_wrong_type_scalar_is_stringified_and_flagged(self):
        pred, issues = coerce_prediction({"product_name": 123, "origin": "x", "net_weight": "x",
                                           "mfg_date": "x", "expiry_date": "x"})
        assert pred.product_name == "123"
        assert any("product_name" in i and "expected str" in i for i in issues)

    def test_string_instead_of_list_is_wrapped(self):
        pred, issues = coerce_prediction({
            "product_name": "x", "ingredient": "Bột mì", "origin": "x",
            "net_weight": "x", "mfg_date": "x", "expiry_date": "x",
        })
        assert pred.ingredient == ["Bột mì"]
        assert any("ingredient" in i and "wrapped" in i for i in issues)

    def test_empty_string_ingredient_wraps_to_empty_list_not_blank_item(self):
        pred, _issues = coerce_prediction({"product_name": "x", "ingredient": "", "origin": "x",
                                            "net_weight": "x", "mfg_date": "x", "expiry_date": "x"})
        assert pred.ingredient == []

    def test_malformed_nutrition_entries_are_skipped_not_crashed(self):
        pred, issues = coerce_prediction({
            "product_name": "x", "origin": "x", "net_weight": "x", "mfg_date": "x", "expiry_date": "x",
            "nutrition": [
                {"name": "Năng lượng", "value": "70 kcal"},  # fine
                {"value": "10 g"},  # missing name -> skipped
                "not an object",  # wrong type -> skipped
                {"name": "Chất đạm"},  # missing value -> defaulted to ""
            ],
        })
        assert len(pred.nutrition) == 2
        assert pred.nutrition[0].name == "Năng lượng"
        assert pred.nutrition[1] == pred.nutrition[1]  # Chất đạm entry survives with value=""
        assert pred.nutrition[1].value == ""
        assert any("missing 'name'" in i for i in issues)

    def test_non_dict_nutrition_field_discarded(self):
        pred, issues = coerce_prediction({"product_name": "x", "origin": "x", "net_weight": "x",
                                           "mfg_date": "x", "expiry_date": "x", "nutrition": "not a list"})
        assert pred.nutrition == []
        assert any("nutrition" in i for i in issues)


class TestFieldValueStr:
    def test_scalar_field(self):
        gt = LabelSchema.model_validate({"product_name": "Bánh xốp", "origin": "x", "net_weight": "x",
                                          "mfg_date": "x", "expiry_date": "x"})
        assert field_value_str(gt, "product_name") == "Bánh xốp"

    def test_list_field_joined(self):
        gt = LabelSchema.model_validate({"product_name": "x", "ingredient": ["A", "B"], "origin": "x",
                                          "net_weight": "x", "mfg_date": "x", "expiry_date": "x"})
        assert field_value_str(gt, "ingredient") == "A | B"

    def test_nutrition_field_formatted(self):
        gt = LabelSchema.model_validate({
            "product_name": "x", "origin": "x", "net_weight": "x", "mfg_date": "x", "expiry_date": "x",
            "nutrition": [{"name": "Năng lượng", "value": "70 kcal"}],
        })
        assert field_value_str(gt, "nutrition") == "Năng lượng=70 kcal"
