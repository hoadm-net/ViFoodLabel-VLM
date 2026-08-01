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
