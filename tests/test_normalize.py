"""normalize.py -- includes a regression test for a real scoring bug: curly
quotes + trailing punctuation pushed two identical-content strings below the
lenient-match threshold (see matching.py's ScalarMatchResult and git log)."""

from __future__ import annotations

from vifoodlabel.normalize import (
    extract_number_unit,
    normalize_date,
    normalize_number,
    normalize_text,
    normalize_unit,
    strip_diacritics,
)


class TestNormalizeText:
    def test_lowercases_and_collapses_whitespace(self):
        assert normalize_text("  Bột   Mì  ") == "bột mì"

    def test_none_and_empty_are_safe(self):
        assert normalize_text(None) == ""
        assert normalize_text("") == ""

    def test_curly_and_straight_quotes_are_equivalent(self):
        # The exact real-world pair that motivated this fix.
        straight = 'Xem "BEST BEFORE" trên bao bì'
        curly = "Xem “BEST BEFORE” trên bao bì."
        assert normalize_text(straight) == normalize_text(curly)

    def test_single_curly_quotes_fold_too(self):
        assert normalize_text("‘abc’") == normalize_text("'abc'")

    def test_trailing_sentence_punctuation_is_stripped(self):
        assert normalize_text("Có chứa sữa.") == normalize_text("Có chứa sữa")
        assert normalize_text("Thật vậy?!") == normalize_text("Thật vậy")

    def test_internal_punctuation_is_preserved(self):
        # Only *trailing* punctuation is stripped -- a period mid-sentence
        # (e.g. an abbreviation or a second sentence) must survive.
        result = normalize_text("Có chứa sữa. Có thể chứa trứng.")
        assert "sữa." in result or "sữa. " in result.strip() + " "
        assert result.endswith("trứng")

    def test_nfc_normalization(self):
        # Combining-character vs precomposed Vietnamese diacritic should
        # compare equal after NFC normalization.
        precomposed = "Việt Nam"
        decomposed = "Việt Nam"
        assert normalize_text(precomposed) == normalize_text(decomposed)


class TestStripDiacritics:
    def test_strips_common_vietnamese_diacritics(self):
        assert strip_diacritics("Việt Nam") == "Viet Nam"

    def test_dinh_stroke_handled_specially(self):
        assert strip_diacritics("Đường") == "Duong"
        assert strip_diacritics("đậu nành") == "dau nanh"


class TestNormalizeNumber:
    def test_plain_integer(self):
        assert normalize_number("70 kcal") == 70.0

    def test_vietnamese_decimal_comma(self):
        assert normalize_number("34,242%") == 34.242

    def test_english_decimal_point(self):
        assert normalize_number("0.25%") == 0.25

    def test_thousands_dot_plus_decimal_comma(self):
        assert normalize_number("1.234,5") == 1234.5

    def test_no_number_returns_none(self):
        assert normalize_number("Xem trên bao bì") is None

    def test_none_input(self):
        assert normalize_number(None) is None


class TestNormalizeUnit:
    def test_known_aliases_collapse(self):
        assert normalize_unit("gram") == "g"
        assert normalize_unit("Gr") == "g"
        assert normalize_unit("Calories") == "kcal"

    def test_unknown_unit_passed_through(self):
        assert normalize_unit("IU") == "iu"

    def test_empty_returns_none(self):
        assert normalize_unit("") is None


class TestExtractNumberUnit:
    def test_number_and_unit(self):
        assert extract_number_unit("70 kcal") == (70.0, "kcal")

    def test_number_only(self):
        number, unit = extract_number_unit("3")
        assert number == 3.0
        assert unit is None

    def test_no_number(self):
        assert extract_number_unit("Xem trên hộp") == (None, None)

    def test_percentage(self):
        number, unit = extract_number_unit("34,242%")
        assert number == 34.242
        assert unit == "%"


class TestNormalizeDate:
    def test_dmy_to_iso(self):
        assert normalize_date("01/01/2026") == "2026-01-01"

    def test_ymd_to_iso(self):
        assert normalize_date("2026-01-01") == "2026-01-01"

    def test_dash_separator(self):
        assert normalize_date("01-01-2026") == "2026-01-01"

    def test_non_date_falls_back_to_normalized_text(self):
        # Relative/instructional dates are common in this dataset and must
        # not crash or be mangled -- see docs/annotation-guidelines.md.
        text = 'Xem "PRODUCTION CODE" trên bao bì.'
        assert normalize_date(text) == normalize_text(text)

    def test_relative_date_text_preserved(self):
        assert normalize_date("06 tháng trước HSD") == normalize_text("06 tháng trước HSD")
