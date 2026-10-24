"""Prompt construction for the {vi,en} x {zero,one}-shot conditions.

The actual prompt wording lives in `prompts/*.txt` (repo root), not here —
this module only assembles those pieces. Keeping the wording in plain text
files makes it easy to review/tweak without touching code, diff cleanly in
git, and reuse verbatim as supplementary material in the paper.

The one-shot condition demonstrates the expected JSON *format* with a
synthetic, entirely made-up product (fake brand, fake numbers) described in
text only — no real benchmark image is ever used as the exemplar, so all 600
images stay eligible for scoring under every condition (no data leakage).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from vifoodlabel.config import PROMPTS_DIR

Language = Literal["vi", "en"]
Shot = Literal["zero", "one"]


@lru_cache(maxsize=None)
def _load(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def condition_name(language: Language, shot: Shot) -> str:
    return f"{language}_{shot}"


def build_instruction(language: Language, shot: Shot) -> str:
    example = ""
    if shot == "one":
        example = _load(f"one_shot_example_{language}.txt")
    fields = _load(f"field_descriptions_{language}.txt")
    classification_rules = _load(f"classification_rules_{language}.txt")
    template = _load(f"task_instruction_{language}.txt")
    return template.format(fields=fields, classification_rules=classification_rules, example=example)


def build_ocr_instruction(ocr_text: str, language: Language = "vi", shot: Shot = "zero") -> str:
    """The OCR -> LLM baseline's prompt: identical field descriptions and
    classification rules to `build_instruction`, with the label's OCR
    transcript in place of the image. Only the Vietnamese template exists --
    the baseline is compared against Tier 1's canonical vi_zero condition."""
    if language != "vi":
        raise ValueError("The OCR -> LLM baseline only has a Vietnamese template (task_instruction_ocr_vi.txt)")
    example = _load(f"one_shot_example_{language}.txt") if shot == "one" else ""
    template = _load(f"task_instruction_ocr_{language}.txt")
    return template.format(
        fields=_load(f"field_descriptions_{language}.txt"),
        classification_rules=_load(f"classification_rules_{language}.txt"),
        example=example,
        ocr_text=ocr_text,
    )


# The canonical Tier-1 benchmark condition.
CANONICAL_LANGUAGE: Language = "vi"
CANONICAL_SHOT: Shot = "zero"
CANONICAL_CONDITION = condition_name(CANONICAL_LANGUAGE, CANONICAL_SHOT)

# One-factor-at-a-time against the vi_zero baseline, not a full 2x2 factorial
# -- vi_zero vs vi_one isolates the shot-count effect, vi_zero vs en_zero
# isolates the instruction-language effect. en_one (both factors changed at
# once) is deliberately omitted: it would only add the language x shot
# interaction term, which isn't a question this benchmark asks, at the cost
# of a fourth condition's worth of calls on every model.
ALL_PROMPT_CONDITIONS: list[tuple[Language, Shot]] = [
    ("vi", "zero"),
    ("vi", "one"),
    ("en", "zero"),
]


# One-line field semantics for the taxonomy judge (paper Table 1).
FIELD_SEMANTICS = {
    "product_name": "full product name as printed on the label",
    "ingredient": "list of ingredient items, printed percentages retained",
    "additive": "list of food additives, identified by function rather than position",
    "warning": "list of safety warnings and advisories",
    "nutrition": "list of {name, value} rows of the nutrition facts table",
    "origin": "country or place of manufacture or origin",
    "net_weight": "net weight or volume",
    "mfg_date": "manufacturing date, recorded verbatim (may be an instruction where to find it)",
    "expiry_date": "expiry date, recorded verbatim (may be an instruction where to find it)",
}


def build_taxonomy_judge_instruction(field: str, gt_value: str, pred_value: str, flags: str) -> str:
    """Prompt for the taxonomy-validation judge (revision, Reviewer 2.2).
    Deliberately never shows the rule-based category -- the judge labels
    blind, so agreement with the rule is a real check, not an anchor."""
    return _load("taxonomy_judge_en.txt").format(
        field=field,
        field_semantics=FIELD_SEMANTICS[field],
        gt_value=gt_value if gt_value.strip() else "(empty)",
        pred_value=pred_value if pred_value.strip() else "(empty)",
        flags=flags,
    )
