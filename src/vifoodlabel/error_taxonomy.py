"""Automatic error-category classification for Tier 4 (error taxonomy).

Deterministic and rule-based -- not human-coded. Built entirely on the match
results already computed for scoring (matching.py), so a category is only
ever assigned from a signal the matching engine can actually observe.

Two categories from the original design have no reliable automatic signal:

- `wrong_language`: ground truth only records the correct (Vietnamese)
  answer, never what non-Vietnamese text was also printed on the label, so
  there's no basis to confirm a mismatch is specifically "read from the
  wrong language" rather than any other kind of wrong value. Not assigned
  here at all -- those cases fall through to `wrong_value` or
  `hallucination` like any other mismatch.
- `hallucination`: DOES get a heuristic below (near-zero text similarity to
  the ground truth, or content where the label had none at all), because
  that's the best signal available -- but it's the least reliable rule in
  this module and should be the first thing spot-checked by a human.

`LOW_CONFIDENCE_CATEGORIES` marks which categories come from a heuristic
rather than a near-certain structural/numeric signal, so downstream tooling
can flag them for priority review.
"""

from __future__ import annotations

from vifoodlabel.metrics import ImageScore
from vifoodlabel.normalize import extract_number_unit
from vifoodlabel.schema import field_value_str

# Below this rapidfuzz token_sort_ratio, a predicted scalar value is treated
# as unrelated to the ground truth rather than merely wrong -- chosen
# conservatively; revisit after human spot-checking (see module docstring).
HALLUCINATION_SIMILARITY_THRESHOLD = 20.0

LOW_CONFIDENCE_CATEGORIES = frozenset({"hallucination"})


def classify_error(score: ImageScore, field: str, field_type: str) -> str:
    """Best-effort automatic error category for one incorrect (image, model,
    field) instance. Always returns a category -- the generic fallback is
    "wrong_value"."""
    if not score.json_valid or "json_parse_failed" in score.structural_issues:
        return "malformed_json"
    if "output_truncated" in score.structural_issues:
        return "output_truncated"
    if "generation_loop_detected" in score.structural_issues:
        return "generation_loop"

    if field_type == "scalar":
        return _classify_scalar(score, field)
    if field_type == "list":
        return _classify_list(score, field)
    if field_type == "nutrition":
        return _classify_nutrition(score)
    raise ValueError(f"Unknown field_type: {field_type!r}")


def _classify_scalar(score: ImageScore, field: str) -> str:
    r = score.scalar_results[field]
    pred = field_value_str(score.pred, field) if score.pred else ""
    gt = field_value_str(score.gt, field) if score.gt else ""

    if r.diacritic_only_mismatch:
        return "diacritics"
    if not pred.strip() and gt.strip():
        return "missing_field"
    if not gt.strip() and pred.strip():
        return "hallucination"  # label had nothing here; model invented content

    if field == "net_weight":
        p_num, p_unit = extract_number_unit(pred)
        g_num, g_unit = extract_number_unit(gt)
        if p_num is not None and g_num is not None and p_unit != g_unit:
            numeric_close = abs(p_num - g_num) <= max(0.05 * abs(g_num), 0.05)
            if numeric_close:
                return "wrong_unit"

    if gt.strip() and r.similarity < HALLUCINATION_SIMILARITY_THRESHOLD:
        return "hallucination"
    return "wrong_value"


def _classify_list(score: ImageScore, field: str) -> str:
    r = score.list_results[field]
    if r.n_pred == 0 and r.n_gt > 0:
        return "missing_field"
    if r.n_gt == 0 and r.n_pred > 0:
        return "hallucination"
    if field == "additive" and r.recall < 1.0:
        return "missing_additive"
    if r.precision < r.recall:
        # relatively more spurious/unmatched predicted items than missed
        # ground-truth items -- more invented than dropped.
        return "hallucination"
    return "wrong_value"


def _classify_nutrition(score: ImageScore) -> str:
    n = score.nutrition_result
    if n.n_pred == 0 and n.n_gt > 0:
        return "missing_field"
    if n.n_gt == 0 and n.n_pred > 0:
        return "hallucination"
    if n.pairing_accuracy < 1.0:
        return "pairing_error"
    if n.value_accuracy < 1.0:
        return "wrong_value"
    if n.name_precision < n.name_recall:
        return "hallucination"
    if n.name_recall < 1.0:
        return "missing_field"
    return "wrong_value"
