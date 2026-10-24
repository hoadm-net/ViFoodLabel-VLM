"""prompts.py -- includes a regression test for a real bug: the one-shot
synthetic example used to render with doubled curly braces ({{...}} instead
of {...}), producing invalid JSON in the exemplar shown to the model."""

from __future__ import annotations

import json

from vifoodlabel.prompts import ALL_PROMPT_CONDITIONS, CANONICAL_CONDITION, build_instruction, condition_name


class TestBuildInstruction:
    def test_all_four_combinations_render_without_leftover_placeholders(self):
        # Covers all 4 (language, shot) combinations that build_instruction
        # supports, independent of which ones ALL_PROMPT_CONDITIONS actually
        # runs by default (en_one is a valid combination even though Tier 2
        # doesn't include it in its default ablation -- see prompts.py).
        for language in ("vi", "en"):
            for shot in ("zero", "one"):
                text = build_instruction(language, shot)
                assert "{fields}" not in text
                assert "{example}" not in text
                assert "{classification_rules}" not in text
                assert len(text) > 0

    def test_zero_shot_has_no_example_section(self):
        text = build_instruction("vi", "zero")
        assert "Ví dụ minh họa" not in text

    def test_one_shot_vi_example_is_valid_json(self):
        # The exact real bug: this used to contain {{"product_name": ...}}
        # (doubled braces) instead of valid JSON.
        text = build_instruction("vi", "one")
        idx = text.find("=>")
        assert idx != -1
        snippet = text[idx + 2:]
        end = snippet.find("\n\n")
        json_text = snippet[:end].strip()
        parsed = json.loads(json_text)  # raises if this regresses
        assert "product_name" in parsed

    def test_one_shot_en_example_is_valid_json(self):
        text = build_instruction("en", "one")
        idx = text.find("=>")
        snippet = text[idx + 2:]
        end = snippet.find("\n\n")
        json_text = snippet[:end].strip()
        parsed = json.loads(json_text)
        assert "product_name" in parsed

    def test_one_shot_example_is_not_from_the_real_dataset(self):
        # The exemplar must be synthetic (see docs/experimental-design.md) --
        # it must not be word-for-word the real 0001.json product name, or
        # every one-shot condition would leak an evaluation image.
        text = build_instruction("vi", "one")
        assert "Deka Jumbo" not in text  # the real 0001.json product name

    def test_condition_naming(self):
        assert condition_name("vi", "zero") == "vi_zero"
        assert condition_name("en", "one") == "en_one"

    def test_canonical_condition_is_vi_zero(self):
        assert CANONICAL_CONDITION == "vi_zero"

    def test_all_prompt_conditions_has_three_entries(self):
        # One-factor-at-a-time against vi_zero, not a full 2x2 -- en_one is
        # deliberately excluded, see prompts.py.
        assert len(ALL_PROMPT_CONDITIONS) == 3
        assert set(ALL_PROMPT_CONDITIONS) == {
            ("vi", "zero"), ("vi", "one"), ("en", "zero"),
        }


class TestOcrInstruction:
    def test_embeds_transcript_and_shared_rules(self):
        from vifoodlabel.prompts import _load, build_ocr_instruction

        text = build_ocr_instruction("Thành phần: Bột mì, Đường")
        assert "Thành phần: Bột mì, Đường" in text
        # Same field descriptions and classification rules as the image prompt.
        assert _load("field_descriptions_vi.txt") in text
        assert _load("classification_rules_vi.txt") in text
        for placeholder in ("{fields}", "{classification_rules}", "{example}", "{ocr_text}"):
            assert placeholder not in text

    def test_transcript_with_braces_is_not_reformatted(self):
        from vifoodlabel.prompts import build_ocr_instruction

        assert "{weird} {{text}}" in build_ocr_instruction("{weird} {{text}}")

    def test_only_vietnamese_supported(self):
        import pytest

        from vifoodlabel.prompts import build_ocr_instruction

        with pytest.raises(ValueError):
            build_ocr_instruction("x", language="en")


class TestTaxonomyJudgeInstruction:
    def test_fills_values_and_never_mentions_rule_output(self):
        from vifoodlabel.prompts import build_taxonomy_judge_instruction

        text = build_taxonomy_judge_instruction("origin", "Indonesia", "", "none")
        assert "Field: origin" in text
        assert "Indonesia" in text
        assert "(empty)" in text  # empty prediction made explicit
        assert "rule" not in text.lower().split("annotation conventions")[0]
        assert '{"category"' in text  # literal JSON braces survived .format
