"""Per-image scoring and aggregation into flat tables ready for stats/reporting."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from vifoodlabel.matching import (
    NutritionMatchResult,
    ScalarMatchResult,
    SetMatchResult,
    match_list_field,
    match_nutrition,
    match_scalar,
)
from vifoodlabel.schema import LIST_FIELDS, SCALAR_FIELDS, LabelSchema


@dataclass
class ImageScore:
    image_id: str
    model_key: str
    condition: str
    json_valid: bool
    structural_issues: list[str] = field(default_factory=list)
    api_error: str | None = None
    scalar_results: dict[str, ScalarMatchResult] = field(default_factory=dict)
    list_results: dict[str, SetMatchResult] = field(default_factory=dict)
    nutrition_result: NutritionMatchResult | None = None
    gt: LabelSchema | None = None
    pred: LabelSchema | None = None

    @property
    def macro_field_f1(self) -> float:
        """Mean F1 across all 9 fields (lenient match for scalars)."""
        values = [float(r.lenient_match) for r in self.scalar_results.values()]
        values += [r.f1 for r in self.list_results.values()]
        if self.nutrition_result is not None:
            values.append(self.nutrition_result.name_f1)
        return sum(values) / len(values) if values else 0.0

    @property
    def macro_field_f1_strict(self) -> float:
        values = [float(r.strict_match) for r in self.scalar_results.values()]
        values += [r.f1 for r in self.list_results.values()]
        if self.nutrition_result is not None:
            values.append(self.nutrition_result.name_f1)
        return sum(values) / len(values) if values else 0.0

    @property
    def pairing_accuracy(self) -> float:
        return self.nutrition_result.pairing_accuracy if self.nutrition_result else 1.0


def score_prediction(
    image_id: str,
    model_key: str,
    condition: str,
    gt: LabelSchema,
    pred: LabelSchema,
    json_valid: bool,
    structural_issues: list[str] | None = None,
    api_error: str | None = None,
) -> ImageScore:
    scalar_results = {
        f: match_scalar(f, getattr(pred, f), getattr(gt, f)) for f in SCALAR_FIELDS
    }
    list_results = {
        f: match_list_field(getattr(pred, f), getattr(gt, f)) for f in LIST_FIELDS
    }
    nutrition_result = match_nutrition(pred.nutrition, gt.nutrition)
    return ImageScore(
        image_id=image_id,
        model_key=model_key,
        condition=condition,
        json_valid=json_valid,
        structural_issues=structural_issues or [],
        api_error=api_error,
        scalar_results=scalar_results,
        list_results=list_results,
        nutrition_result=nutrition_result,
        gt=gt,
        pred=pred,
    )


def scores_to_dataframe(scores: list[ImageScore]) -> pd.DataFrame:
    """One row per (image, model, condition, field)."""
    rows = []
    for s in scores:
        for f, r in s.scalar_results.items():
            rows.append({
                "image_id": s.image_id, "model_key": s.model_key, "condition": s.condition,
                "field": f, "field_type": "scalar",
                "precision": float(r.lenient_match), "recall": float(r.lenient_match), "f1": float(r.lenient_match),
                "strict_f1": float(r.strict_match), "similarity": r.similarity,
                "diacritic_only_mismatch": r.diacritic_only_mismatch,
                "pairing_accuracy": None, "value_accuracy": None,
                "json_valid": s.json_valid, "api_error": s.api_error,
            })
        for f, r in s.list_results.items():
            rows.append({
                "image_id": s.image_id, "model_key": s.model_key, "condition": s.condition,
                "field": f, "field_type": "list",
                "precision": r.precision, "recall": r.recall, "f1": r.f1,
                "strict_f1": None, "similarity": None, "diacritic_only_mismatch": None,
                "pairing_accuracy": None, "value_accuracy": None,
                "json_valid": s.json_valid, "api_error": s.api_error,
            })
        if s.nutrition_result is not None:
            n = s.nutrition_result
            rows.append({
                "image_id": s.image_id, "model_key": s.model_key, "condition": s.condition,
                "field": "nutrition", "field_type": "nutrition",
                "precision": n.name_precision, "recall": n.name_recall, "f1": n.name_f1,
                "strict_f1": None, "similarity": None, "diacritic_only_mismatch": None,
                "pairing_accuracy": n.pairing_accuracy, "value_accuracy": n.value_accuracy,
                "json_valid": s.json_valid, "api_error": s.api_error,
            })
    return pd.DataFrame(rows)


def image_level_summary(scores: list[ImageScore]) -> pd.DataFrame:
    """One row per (image, model, condition) with the headline macro metrics."""
    rows = [{
        "image_id": s.image_id, "model_key": s.model_key, "condition": s.condition,
        "macro_field_f1": s.macro_field_f1, "macro_field_f1_strict": s.macro_field_f1_strict,
        "pairing_accuracy": s.pairing_accuracy, "json_valid": s.json_valid,
        "api_error": s.api_error, "n_structural_issues": len(s.structural_issues),
    } for s in scores]
    return pd.DataFrame(rows)


def model_field_summary(field_df: pd.DataFrame) -> pd.DataFrame:
    """Mean precision/recall/F1 per (model, condition, field), plus support count."""
    return (
        field_df.groupby(["model_key", "condition", "field"])
        .agg(
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            n_images=("image_id", "nunique"),
        )
        .reset_index()
    )


def model_summary(image_df: pd.DataFrame) -> pd.DataFrame:
    """Mean headline metrics per (model, condition), plus JSON-validity rate."""
    return (
        image_df.groupby(["model_key", "condition"])
        .agg(
            mean_macro_field_f1=("macro_field_f1", "mean"),
            mean_macro_field_f1_strict=("macro_field_f1_strict", "mean"),
            mean_pairing_accuracy=("pairing_accuracy", "mean"),
            json_validity_rate=("json_valid", "mean"),
            n_images=("image_id", "nunique"),
        )
        .reset_index()
    )
