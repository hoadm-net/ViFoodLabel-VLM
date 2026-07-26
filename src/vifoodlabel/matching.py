"""Field-matching logic behind the two headline metrics (field-level F1, pairing accuracy).

Three matchers:
- `match_scalar`   — the 5 scalar fields (strict + lenient/fuzzy, w/ diacritic-only flag).
- `match_list_field` — the 3 list fields (ingredient/additive/warning), via optimal
  bipartite (Hungarian) assignment on string similarity, then precision/recall/F1.
- `match_nutrition`  — the nutrition[{name,value}] field: bipartite name matching
  (with a bilingual VI/EN alias table) followed by value comparison, explicitly
  separating plain value-extraction errors from *pairing* errors (a value that is
  numerically correct for a *different* GT row than the one it's attached to).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from vifoodlabel.normalize import (
    extract_number_unit,
    normalize_date,
    normalize_text,
    strip_diacritics,
)
from vifoodlabel.schema import NutritionEntry

DEFAULT_LENIENT_THRESHOLD = 90.0
DEFAULT_LIST_MATCH_THRESHOLD = 80.0
DEFAULT_NUTRITION_NAME_THRESHOLD = 75.0
DEFAULT_VALUE_ABS_TOL = 0.05
DEFAULT_VALUE_REL_TOL = 0.05

# Canonical bilingual nutrient-name aliases. Extend as new labels surface variants.
_NUTRIENT_ALIASES: dict[str, str] = {
    "năng lượng": "energy", "energy": "energy", "calo": "energy",
    "chất đạm": "protein", "đạm": "protein", "protein": "protein",
    "chất béo": "fat", "béo": "fat", "fat": "fat", "lipid": "fat", "total fat": "fat",
    "chất béo bão hòa": "saturated fat", "saturated fat": "saturated fat", "chất béo bão hoà": "saturated fat",
    "carbohydrat": "carbohydrate", "carbohydrate": "carbohydrate",
    "chất bột đường": "carbohydrate", "tinh bột": "carbohydrate", "carbohydrat tổng số": "carbohydrate",
    "đường tổng số": "total sugar", "total sugar": "total sugar", "đường": "sugar", "sugar": "sugar",
    "natri": "sodium", "sodium": "sodium",
    "chất xơ": "fiber", "fiber": "fiber", "fibre": "fiber", "chất xơ tổng số": "fiber",
    "cholesterol": "cholesterol",
    "canxi": "calcium", "calcium": "calcium",
    "protein tổng số": "protein",
}


def canonical_nutrient_name(name: str) -> str:
    return _NUTRIENT_ALIASES.get(normalize_text(name), normalize_text(name))


def _name_similarity(a: str, b: str) -> float:
    if canonical_nutrient_name(a) == canonical_nutrient_name(b):
        return 100.0
    return fuzz.token_sort_ratio(normalize_text(a), normalize_text(b))


def _text_similarity(a: str, b: str) -> float:
    return fuzz.token_sort_ratio(normalize_text(a), normalize_text(b))


def _bipartite_match(
    pred_items: list[str], gt_items: list[str], sim_fn, threshold: float
) -> list[tuple[int, int, float]]:
    """Optimal assignment maximizing total similarity; returns pairs at/above threshold."""
    if not pred_items or not gt_items:
        return []
    sim = [[sim_fn(p, g) for g in gt_items] for p in pred_items]
    cost = [[100.0 - s for s in row] for row in sim]
    pred_idx, gt_idx = linear_sum_assignment(cost)
    matches = []
    for pi, gi in zip(pred_idx, gt_idx):
        s = sim[pi][gi]
        if s >= threshold:
            matches.append((int(pi), int(gi), float(s)))
    return matches


# ---------------------------------------------------------------------------
# Scalar fields
# ---------------------------------------------------------------------------


@dataclass
class ScalarMatchResult:
    strict_match: bool
    lenient_match: bool
    similarity: float
    diacritic_only_mismatch: bool


def match_scalar(
    field_name: str, pred: str, gt: str, threshold: float = DEFAULT_LENIENT_THRESHOLD
) -> ScalarMatchResult:
    pred, gt = pred or "", gt or ""

    if field_name == "net_weight":
        p_num, p_unit = extract_number_unit(pred)
        g_num, g_unit = extract_number_unit(gt)
        if g_num is not None and p_num is not None:
            numeric_ok = abs(p_num - g_num) <= max(DEFAULT_VALUE_REL_TOL * abs(g_num), DEFAULT_VALUE_ABS_TOL)
            unit_ok = (g_unit is None) or (p_unit is None) or (g_unit == p_unit)
            if numeric_ok and unit_ok:
                return ScalarMatchResult(strict_match=True, lenient_match=True, similarity=100.0, diacritic_only_mismatch=False)

    if field_name in ("mfg_date", "expiry_date"):
        pred_norm, gt_norm = normalize_date(pred), normalize_date(gt)
        if pred_norm == gt_norm and gt_norm:
            return ScalarMatchResult(strict_match=True, lenient_match=True, similarity=100.0, diacritic_only_mismatch=False)

    pred_norm, gt_norm = normalize_text(pred), normalize_text(gt)
    strict = pred_norm == gt_norm
    similarity = 100.0 if strict else float(fuzz.token_sort_ratio(pred_norm, gt_norm))
    lenient = strict or similarity >= threshold
    diacritic_only = (not strict) and bool(gt_norm) and strip_diacritics(pred_norm) == strip_diacritics(gt_norm)
    return ScalarMatchResult(strict_match=strict, lenient_match=lenient, similarity=similarity, diacritic_only_mismatch=diacritic_only)


# ---------------------------------------------------------------------------
# List fields (ingredient / additive / warning)
# ---------------------------------------------------------------------------


@dataclass
class SetMatchResult:
    precision: float
    recall: float
    f1: float
    matched_pairs: list[tuple[int, int, float]] = field(default_factory=list)
    n_pred: int = 0
    n_gt: int = 0


def _merge_aware_bonus(
    pred_items: list[str], gt_items: list[str], threshold: float, matches: list[tuple[int, int, float]]
) -> tuple[int, int]:
    """Credit leftover (unmatched) items when one side's leftovers, joined
    together, are a high-similarity match for a single leftover item on the
    other side -- e.g. two ground-truth warning sentences the model merged
    into one list entry (or vice versa). Content-equivalent, just
    re-segmented differently, so it shouldn't score as a total miss.

    Deliberately narrow: only "several items on one side == one item on the
    other" is credited, not arbitrary many-to-many regrouping, since that's
    the actual failure mode observed (adjacent sentences split/merged), not
    a license to paper over hallucinated or dropped items.

    Returns (extra_pred_credit, extra_gt_credit) to add to the numerators
    of precision and recall respectively.
    """
    matched_pred = {p for p, _g, _s in matches}
    matched_gt = {g for _p, g, _s in matches}
    leftover_pred = [i for i in range(len(pred_items)) if i not in matched_pred]
    leftover_gt = [i for i in range(len(gt_items)) if i not in matched_gt]
    if not leftover_pred or not leftover_gt:
        return 0, 0

    if len(leftover_gt) > 1:
        merged_gt_text = " ".join(gt_items[i] for i in leftover_gt)
        if any(_text_similarity(pred_items[i], merged_gt_text) >= threshold for i in leftover_pred):
            return 1, len(leftover_gt)

    if len(leftover_pred) > 1:
        merged_pred_text = " ".join(pred_items[i] for i in leftover_pred)
        if any(_text_similarity(merged_pred_text, gt_items[i]) >= threshold for i in leftover_gt):
            return len(leftover_pred), 1

    return 0, 0


def match_list_field(
    pred_items: list[str], gt_items: list[str], threshold: float = DEFAULT_LIST_MATCH_THRESHOLD
) -> SetMatchResult:
    matches = _bipartite_match(pred_items, gt_items, _text_similarity, threshold)
    extra_pred, extra_gt = _merge_aware_bonus(pred_items, gt_items, threshold, matches)
    tp = len(matches)
    precision = (tp + extra_pred) / len(pred_items) if pred_items else 1.0
    recall = (tp + extra_gt) / len(gt_items) if gt_items else 1.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return SetMatchResult(precision=precision, recall=recall, f1=f1, matched_pairs=matches, n_pred=len(pred_items), n_gt=len(gt_items))


# ---------------------------------------------------------------------------
# Nutrition pairing
# ---------------------------------------------------------------------------


@dataclass
class NutritionMatchResult:
    name_precision: float
    name_recall: float
    name_f1: float
    value_accuracy: float
    pairing_accuracy: float
    n_gt: int
    n_pred: int
    n_matched: int
    n_pairing_errors: int


def _value_matches(pred_value: str, gt_value: str) -> bool:
    p_num, p_unit = extract_number_unit(pred_value)
    g_num, g_unit = extract_number_unit(gt_value)
    if g_num is None:
        return normalize_text(pred_value) == normalize_text(gt_value)
    if p_num is None:
        return False
    numeric_ok = abs(p_num - g_num) <= max(DEFAULT_VALUE_REL_TOL * abs(g_num), DEFAULT_VALUE_ABS_TOL)
    unit_ok = (g_unit is None) or (p_unit is None) or (g_unit == p_unit)
    return numeric_ok and unit_ok


def match_nutrition(
    pred_entries: list[NutritionEntry],
    gt_entries: list[NutritionEntry],
    name_threshold: float = DEFAULT_NUTRITION_NAME_THRESHOLD,
) -> NutritionMatchResult:
    pred_names = [e.name for e in pred_entries]
    gt_names = [e.name for e in gt_entries]
    matches = _bipartite_match(pred_names, gt_names, _name_similarity, name_threshold)

    tp = len(matches)
    name_precision = tp / len(pred_entries) if pred_entries else 1.0
    name_recall = tp / len(gt_entries) if gt_entries else 1.0
    name_f1 = 0.0 if (name_precision + name_recall) == 0 else 2 * name_precision * name_recall / (name_precision + name_recall)

    value_correct = 0
    pairing_errors = 0
    for pred_idx, gt_idx, _sim in matches:
        pred_entry, gt_entry = pred_entries[pred_idx], gt_entries[gt_idx]
        if _value_matches(pred_entry.value, gt_entry.value):
            value_correct += 1
        else:
            # Is this predicted value actually correct for a *different* GT row?
            # That's the cross-row "pairing" failure mode called out in the RQ,
            # distinct from a plain wrong/missing value.
            if any(_value_matches(pred_entry.value, other.value) for j, other in enumerate(gt_entries) if j != gt_idx):
                pairing_errors += 1

    value_accuracy = value_correct / tp if tp else (1.0 if not gt_entries else 0.0)
    pairing_accuracy = 1.0 - (pairing_errors / len(gt_entries)) if gt_entries else 1.0

    return NutritionMatchResult(
        name_precision=name_precision,
        name_recall=name_recall,
        name_f1=name_f1,
        value_accuracy=value_accuracy,
        pairing_accuracy=pairing_accuracy,
        n_gt=len(gt_entries),
        n_pred=len(pred_entries),
        n_matched=tp,
        n_pairing_errors=pairing_errors,
    )
