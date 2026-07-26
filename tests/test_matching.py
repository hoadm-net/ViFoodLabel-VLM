"""matching.py -- the two headline metrics live here. Includes regression
tests for two real scoring bugs found by manually inspecting error-sample
output (see git log around 2026-07): quote/punctuation normalization
deflating scalar-field scores, and list-field scoring 0.0 for a
content-correct-but-differently-segmented answer."""

from __future__ import annotations

import pytest

from vifoodlabel.matching import (
    canonical_nutrient_name,
    match_list_field,
    match_nutrition,
    match_scalar,
)
from vifoodlabel.schema import NutritionEntry


class TestMatchScalar:
    def test_exact_match(self):
        r = match_scalar("product_name", "Bánh xốp", "Bánh xốp")
        assert r.strict_match and r.lenient_match
        assert r.similarity == 100.0

    def test_completely_different_is_wrong(self):
        r = match_scalar("product_name", "Bánh xốp", "Nước ép lựu")
        assert not r.strict_match
        assert not r.lenient_match

    def test_quote_and_trailing_period_regression(self):
        # The exact real case: two identical-content answers scored 0.597
        # similarity (below the 90 lenient threshold) before normalize_text
        # folded quote styles and stripped trailing punctuation.
        pred = 'Xem "PRODUCTION CODE" trên bao bì'
        gt = "Xem “PRODUCTION CODE” trên bao bì."
        r = match_scalar("mfg_date", pred, gt)
        assert r.strict_match is True
        assert r.similarity == 100.0

    def test_diacritic_only_mismatch_flagged_but_not_correct(self):
        r = match_scalar("origin", "Viet Nam", "Việt Nam")
        assert not r.strict_match
        assert r.diacritic_only_mismatch is True

    def test_net_weight_numeric_tolerance(self):
        r = match_scalar("net_weight", "140g", "140 g (14 g x 10 cây).")
        assert r.lenient_match is True

    def test_net_weight_wrong_unit_not_matched_numerically(self):
        r = match_scalar("net_weight", "140 mg", "140 g")
        # Falls through to text comparison since units disagree; "140 mg"
        # vs "140 g" should not fuzzy-match as the same weight.
        assert r.strict_match is False

    def test_date_relative_instruction_exact_match(self):
        r = match_scalar("mfg_date", "06 tháng trước HSD.", "06 tháng trước HSD")
        assert r.lenient_match is True

    def test_empty_vs_empty_is_a_match(self):
        r = match_scalar("origin", "", "")
        assert r.strict_match is True

    def test_none_inputs_are_safe(self):
        r = match_scalar("origin", None, None)
        assert r.strict_match is True


class TestMatchListField:
    def test_perfect_match(self):
        items = ["Bột mì", "Đường", "Muối"]
        r = match_list_field(items, items)
        assert r.precision == 1.0
        assert r.recall == 1.0
        assert r.f1 == 1.0

    def test_both_empty_is_perfect(self):
        r = match_list_field([], [])
        assert r.precision == 1.0 and r.recall == 1.0 and r.f1 == 1.0

    def test_pred_empty_gt_nonempty_is_zero_recall(self):
        r = match_list_field([], ["Bột mì"])
        assert r.recall == 0.0
        assert r.precision == 1.0  # vacuously: no false positives

    def test_gt_empty_pred_nonempty_is_zero_precision(self):
        r = match_list_field(["Bột mì"], [])
        assert r.precision == 0.0
        assert r.recall == 1.0  # vacuously: nothing was missed

    def test_merge_regression_two_gt_sentences_combined_by_pred(self):
        # The exact real case: model combined two GT warning sentences into
        # one list entry. Content is 100% correct, just re-segmented -- this
        # used to score 0.0 (bipartite matching alone can't see it).
        pred = ["Có chứa lúa mì, đậu nành và sữa. Có thể chứa trứng và đậu phộng."]
        gt = ["Có chứa lúa mì, đậu nành và sữa.", "Có thể chứa trứng và đậu phộng."]
        r = match_list_field(pred, gt)
        assert r.precision == 1.0
        assert r.recall == 1.0
        assert r.f1 == 1.0

    def test_merge_regression_symmetric_pred_split_gt_combined(self):
        pred = ["Có chứa lúa mì, đậu nành và sữa.", "Có thể chứa trứng và đậu phộng."]
        gt = ["Có chứa lúa mì, đậu nành và sữa. Có thể chứa trứng và đậu phộng."]
        r = match_list_field(pred, gt)
        assert r.precision == 1.0
        assert r.recall == 1.0

    def test_merge_bonus_does_not_credit_hallucination(self):
        # A leftover pred item that ISN'T just a re-segmentation of GT
        # content must still be penalized -- merge-aware matching should
        # only fire on genuine content-level correspondence.
        pred = ["Có chứa lúa mì, đậu nành và sữa.", "Sản phẩm này chứa hạt điều."]
        gt = ["Có chứa lúa mì, đậu nành và sữa."]
        r = match_list_field(pred, gt)
        assert r.recall == 1.0
        assert r.precision == 0.5  # the hallucinated second item still costs precision

    def test_partial_overlap(self):
        pred = ["Bột mì", "Đường", "Muối bịa"]
        gt = ["Bột mì", "Đường", "Tiêu"]
        r = match_list_field(pred, gt)
        assert r.precision == pytest.approx(2 / 3)
        assert r.recall == pytest.approx(2 / 3)


class TestNutrientAliasing:
    def test_vi_en_alias_equivalence(self):
        assert canonical_nutrient_name("Năng lượng") == canonical_nutrient_name("Energy")
        assert canonical_nutrient_name("Chất đạm") == canonical_nutrient_name("Protein")
        assert canonical_nutrient_name("Natri") == canonical_nutrient_name("Sodium")


class TestMatchNutrition:
    def _entry(self, name: str, value: str) -> NutritionEntry:
        return NutritionEntry(name=name, value=value)

    def test_perfect_match_same_language(self):
        entries = [self._entry("Năng lượng", "70 kcal"), self._entry("Chất đạm", "0 g")]
        r = match_nutrition(entries, entries)
        assert r.name_f1 == 1.0
        assert r.value_accuracy == 1.0
        assert r.pairing_accuracy == 1.0

    def test_bilingual_name_still_matches(self):
        pred = [self._entry("Energy", "70 kcal")]
        gt = [self._entry("Năng lượng", "70 kcal")]
        r = match_nutrition(pred, gt)
        assert r.name_f1 == 1.0
        assert r.value_accuracy == 1.0

    def test_cross_row_pairing_error_detected(self):
        # Value is numerically correct, but attached to the wrong nutrient
        # name -- this is exactly the "pairing" failure mode from the RQ,
        # distinct from a plain wrong/missing value.
        pred = [
            self._entry("Chất đạm", "10 g"),  # should be Carbohydrat's value
            self._entry("Carbohydrat", "0 g"),  # should be Chất đạm's value
        ]
        gt = [
            self._entry("Chất đạm", "0 g"),
            self._entry("Carbohydrat", "10 g"),
        ]
        r = match_nutrition(pred, gt)
        assert r.name_f1 == 1.0  # both names matched correctly
        assert r.value_accuracy == 0.0  # neither value is right for its own row
        assert r.pairing_accuracy == 0.0  # both are swaps, not plain misses
        assert r.n_pairing_errors == 2

    def test_genuinely_wrong_value_is_not_a_pairing_error(self):
        # Wrong value that doesn't correspond to any other GT row either --
        # a plain extraction error, not a cross-row swap.
        pred = [self._entry("Chất đạm", "999 g")]
        gt = [self._entry("Chất đạm", "0 g")]
        r = match_nutrition(pred, gt)
        assert r.value_accuracy == 0.0
        assert r.n_pairing_errors == 0
        assert r.pairing_accuracy == 1.0  # not a pairing error -> doesn't penalize this metric

    def test_empty_both_is_perfect(self):
        r = match_nutrition([], [])
        assert r.name_f1 == 1.0
        assert r.pairing_accuracy == 1.0
