"""error_taxonomy.py -- rule-based automatic classification of incorrect
(image, model, field) instances into the Tier 4 error-category vocabulary."""

from __future__ import annotations

from vifoodlabel.error_taxonomy import LOW_CONFIDENCE_CATEGORIES, classify_error
from vifoodlabel.matching import match_list_field, match_nutrition, match_scalar
from vifoodlabel.metrics import ImageScore
from vifoodlabel.schema import NutritionEntry


def _score(**overrides) -> ImageScore:
    base = dict(
        image_id="0001", model_key="test-model", condition="vi_zero",
        json_valid=True, structural_issues=[], api_error=None,
        scalar_results={}, list_results={}, nutrition_result=None, gt=None, pred=None,
    )
    base.update(overrides)
    return ImageScore(**base)


class _Obj:
    """Minimal stand-in for LabelSchema, just needs attribute access."""
    def __init__(self, **fields):
        self.__dict__.update(fields)


class TestStructuralIssuesTakePriority:
    def test_invalid_json_is_malformed_json(self):
        s = _score(json_valid=False, scalar_results={"origin": match_scalar("origin", "x", "y")})
        assert classify_error(s, "origin", "scalar") == "malformed_json"

    def test_json_parse_failed_flag_is_malformed_json(self):
        s = _score(structural_issues=["json_parse_failed"], scalar_results={"origin": match_scalar("origin", "x", "y")})
        assert classify_error(s, "origin", "scalar") == "malformed_json"

    def test_output_truncated_flag_wins_over_content_mismatch(self):
        s = _score(structural_issues=["output_truncated"], scalar_results={"origin": match_scalar("origin", "", "Vietnam")})
        assert classify_error(s, "origin", "scalar") == "output_truncated"

    def test_generation_loop_flag_wins_over_content_mismatch(self):
        s = _score(structural_issues=["generation_loop_detected"], scalar_results={"origin": match_scalar("origin", "", "Vietnam")})
        assert classify_error(s, "origin", "scalar") == "generation_loop"


class TestClassifyScalar:
    def test_diacritics_only(self):
        r = match_scalar("origin", "Viet Nam", "Việt Nam")
        s = _score(scalar_results={"origin": r}, gt=_Obj(origin="Việt Nam"), pred=_Obj(origin="Viet Nam"))
        assert classify_error(s, "origin", "scalar") == "diacritics"

    def test_missing_field(self):
        r = match_scalar("origin", "", "Việt Nam")
        s = _score(scalar_results={"origin": r}, gt=_Obj(origin="Việt Nam"), pred=_Obj(origin=""))
        assert classify_error(s, "origin", "scalar") == "missing_field"

    def test_hallucination_when_gt_is_empty(self):
        r = match_scalar("origin", "Thailand", "")
        s = _score(scalar_results={"origin": r}, gt=_Obj(origin=""), pred=_Obj(origin="Thailand"))
        assert classify_error(s, "origin", "scalar") == "hallucination"

    def test_hallucination_when_totally_unrelated(self):
        r = match_scalar("product_name", "Bánh quy socola", "Sữa tươi tiệt trùng")
        s = _score(scalar_results={"product_name": r},
                    gt=_Obj(product_name="Sữa tươi tiệt trùng"), pred=_Obj(product_name="Bánh quy socola"))
        assert classify_error(s, "product_name", "scalar") == "hallucination"

    def test_wrong_unit_net_weight(self):
        # Same number, different unit token -- match_scalar's own numeric+
        # unit check only accepts equal units, so this is a genuine mismatch
        # that classify_error should attribute to the unit, not the number.
        r = match_scalar("net_weight", "140 ml", "140 g")
        s = _score(scalar_results={"net_weight": r}, gt=_Obj(net_weight="140 g"), pred=_Obj(net_weight="140 ml"))
        assert classify_error(s, "net_weight", "scalar") == "wrong_unit"

    def test_generic_wrong_value(self):
        r = match_scalar("net_weight", "120 g", "140 g")
        s = _score(scalar_results={"net_weight": r}, gt=_Obj(net_weight="140 g"), pred=_Obj(net_weight="120 g"))
        assert classify_error(s, "net_weight", "scalar") == "wrong_value"

    def test_hallucination_is_low_confidence(self):
        assert "hallucination" in LOW_CONFIDENCE_CATEGORIES
        assert "diacritics" not in LOW_CONFIDENCE_CATEGORIES


class TestClassifyList:
    def test_missing_field_when_pred_empty(self):
        r = match_list_field([], ["Sữa", "Đường"])
        s = _score(list_results={"ingredient": r})
        assert classify_error(s, "ingredient", "list") == "missing_field"

    def test_hallucination_when_gt_empty(self):
        r = match_list_field(["Sữa", "Đường"], [])
        s = _score(list_results={"ingredient": r})
        assert classify_error(s, "ingredient", "list") == "hallucination"

    def test_missing_additive_specific_to_additive_field(self):
        r = match_list_field(["Lecithin"], ["Lecithin", "INS 330"])
        s = _score(list_results={"additive": r})
        assert classify_error(s, "additive", "list") == "missing_additive"

    def test_missing_item_in_non_additive_field_is_not_missing_additive(self):
        r = match_list_field(["Sữa"], ["Sữa", "Đường"])
        s = _score(list_results={"ingredient": r})
        assert classify_error(s, "ingredient", "list") != "missing_additive"

    def test_hallucination_when_more_spurious_than_missing(self):
        # 3 predicted, only 1 matches -> precision way below recall
        r = match_list_field(["Sữa", "Xoài", "Chuối"], ["Sữa"])
        s = _score(list_results={"warning": r})
        assert classify_error(s, "warning", "list") == "hallucination"


class TestClassifyNutrition:
    def test_missing_field_when_pred_empty(self):
        n = match_nutrition([], [NutritionEntry(name="Năng lượng", value="100 kcal")])
        s = _score(nutrition_result=n)
        assert classify_error(s, "nutrition", "nutrition") == "missing_field"

    def test_hallucination_when_gt_empty(self):
        n = match_nutrition([NutritionEntry(name="Năng lượng", value="100 kcal")], [])
        s = _score(nutrition_result=n)
        assert classify_error(s, "nutrition", "nutrition") == "hallucination"

    def test_pairing_error(self):
        gt = [NutritionEntry(name="Năng lượng", value="100 kcal"), NutritionEntry(name="Chất đạm", value="5 g")]
        pred = [NutritionEntry(name="Năng lượng", value="5 g"), NutritionEntry(name="Chất đạm", value="100 kcal")]
        n = match_nutrition(pred, gt)
        s = _score(nutrition_result=n)
        assert classify_error(s, "nutrition", "nutrition") == "pairing_error"

    def test_wrong_value(self):
        gt = [NutritionEntry(name="Năng lượng", value="100 kcal")]
        pred = [NutritionEntry(name="Năng lượng", value="80 kcal")]
        n = match_nutrition(pred, gt)
        s = _score(nutrition_result=n)
        assert classify_error(s, "nutrition", "nutrition") == "wrong_value"


class TestUnknownFieldType:
    def test_raises(self):
        import pytest
        s = _score()
        with pytest.raises(ValueError):
            classify_error(s, "origin", "not_a_type")
