"""Validation of the rule-based error taxonomy (revision, Reviewer 2.2).

A stratified sample of incorrect (image, model, field) instances is labelled
blind by an LLM judge (from a model family outside the evaluated roster),
then confirmed or corrected by a human. The rule-based category
(error_taxonomy.py) is scored against that reference: Cohen's kappa overall,
plus per-category precision/recall of the rule -- precision being the
question "when the rule says X, is it X?" that the 29% low-confidence
(hallucination) share raises.

Pure functions only; the API call and file I/O live in
cli/taxonomy_validation.py.
"""

from __future__ import annotations

import random
from collections import defaultdict

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from vifoodlabel.error_taxonomy import ALL_CATEGORIES
from vifoodlabel.parsing import parse_model_json

_FLAG_TEXT = {
    "json_parse_failed": "the system's whole response could not be parsed as JSON",
    "api_error": "the system returned no response at all",
    "output_truncated": "the response was cut off at the output-token limit",
    "generation_loop_detected": "the response degenerated into a repetition loop and was cut off",
}


def describe_flags(structural_issues: list[str]) -> str:
    """Human-readable response-level flags for the judge prompt. Only the
    structural signals behind the three structural categories are passed --
    the other issues (coerced types, repaired JSON) don't decide a category."""
    texts = [_FLAG_TEXT[i] for i in structural_issues if i in _FLAG_TEXT]
    return "; ".join(texts) if texts else "none (the response was complete, valid JSON)"


def stratified_sample(rows: list[dict], per_category: int, seed: int, category_key: str = "rule_category") -> list[dict]:
    """Up to `per_category` rows from every category present (all of a
    category's rows when it has fewer), so rare categories are represented
    instead of being swamped by wrong_value/hallucination."""
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_category[row[category_key]].append(row)
    rng = random.Random(seed)
    sample = []
    for category in sorted(by_category):
        pool = by_category[category]
        sample.extend(pool if len(pool) <= per_category else rng.sample(pool, per_category))
    return sample


def parse_judge_response(content: str | None) -> dict:
    """{"category", "content_on_label", "rationale"} from the judge's reply;
    category is None when the reply is unusable or names a category outside
    the vocabulary (so it can't silently count as agreement)."""
    empty = {"category": None, "content_on_label": None, "rationale": ""}
    if not content:
        return empty
    result = parse_model_json(content)
    data = result.data if not result.parse_failed and isinstance(result.data, dict) else None
    if data is None:
        return empty
    category = str(data.get("category", "")).strip().lower()
    return {
        "category": category if category in ALL_CATEGORIES else None,
        "content_on_label": data.get("content_on_label"),
        "rationale": str(data.get("rationale", "")).strip(),
    }


def category_agreement(rule: list[str], reference: list[str]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Agreement of the rule's categories with a reference labelling.

    Returns (overall, per_category, confusion): overall has n, raw accuracy
    and Cohen's kappa; per_category has, for each category, how often the
    rule assigned it, its precision (rule says X -> reference says X) and
    recall (reference says X -> rule says X); confusion is rule x reference.
    """
    if len(rule) != len(reference):
        raise ValueError("rule and reference must be the same length")
    n = len(rule)
    if n == 0:
        return {"n": 0, "accuracy": float("nan"), "cohen_kappa": float("nan")}, pd.DataFrame(), pd.DataFrame()
    accuracy = sum(a == b for a, b in zip(rule, reference)) / n
    kappa = cohen_kappa_score(rule, reference, labels=sorted(set(rule) | set(reference)))
    overall = {"n": n, "accuracy": accuracy, "cohen_kappa": float(kappa)}

    rows = []
    for category in sorted(set(rule) | set(reference), key=lambda c: ALL_CATEGORIES.index(c) if c in ALL_CATEGORIES else 99):
        n_rule = sum(r == category for r in rule)
        n_ref = sum(r == category for r in reference)
        tp = sum(a == category and b == category for a, b in zip(rule, reference))
        rows.append({
            "category": category, "n_rule": n_rule, "n_reference": n_ref, "n_agree": tp,
            "rule_precision": tp / n_rule if n_rule else float("nan"),
            "rule_recall": tp / n_ref if n_ref else float("nan"),
        })
    confusion = pd.crosstab(pd.Series(rule, name="rule"), pd.Series(reference, name="reference"))
    return overall, pd.DataFrame(rows), confusion
