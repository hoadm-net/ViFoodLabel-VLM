"""The 9-field ground-truth schema, and a lenient coercer for raw VLM output.

Ground truth files are expected to conform exactly (`LabelSchema.model_validate`
fails loudly on malformed GT — that's a data-entry bug worth catching). Model
predictions are messy by construction: a field may be missing, the wrong type,
or nested oddly. `coerce_prediction` never raises; it does its best and returns
a list of `structural_issues` describing every place it had to guess, which
feeds the error-taxonomy tier (a model that constantly needs coercion is a
model producing malformed JSON, which is itself a scored failure mode).
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

SCALAR_FIELDS = ["product_name", "origin", "net_weight", "mfg_date", "expiry_date"]
LIST_FIELDS = ["ingredient", "additive", "warning"]
NUTRITION_FIELD = "nutrition"
ALL_FIELDS = SCALAR_FIELDS + LIST_FIELDS + [NUTRITION_FIELD]


class NutritionEntry(BaseModel):
    name: str
    value: str


class LabelSchema(BaseModel):
    """Strict schema. Ground-truth files must validate against this as-is.

    Annotators may legitimately write `null` for any field that isn't
    visible in the photographed frame (docs/annotation-guidelines.md §4) --
    `null` and "empty" ("" / []) mean the same thing here, so both are
    accepted and `null` is normalized to the field's empty value below.
    """

    product_name: str
    ingredient: list[str] = Field(default_factory=list)
    additive: list[str] = Field(default_factory=list)
    warning: list[str] = Field(default_factory=list)
    nutrition: list[NutritionEntry] = Field(default_factory=list)
    origin: str
    net_weight: str
    mfg_date: str
    expiry_date: str

    @field_validator(*SCALAR_FIELDS, mode="before")
    @classmethod
    def _null_scalar_to_empty_string(cls, value: object) -> object:
        return "" if value is None else value

    @field_validator(*LIST_FIELDS, NUTRITION_FIELD, mode="before")
    @classmethod
    def _null_list_to_empty_list(cls, value: object) -> object:
        return [] if value is None else value


def _coerce_str(value: object, path: str, issues: list[str]) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    issues.append(f"{path}: expected str, got {type(value).__name__}; stringified")
    return str(value)


def _coerce_str_list(value: object, path: str, issues: list[str]) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        issues.append(f"{path}: expected list, got str; wrapped in single-item list")
        return [value] if value else []
    if not isinstance(value, list):
        issues.append(f"{path}: expected list, got {type(value).__name__}; discarded")
        return []
    out = []
    for i, item in enumerate(value):
        out.append(_coerce_str(item, f"{path}[{i}]", issues))
    return out


def _coerce_nutrition(value: object, issues: list[str]) -> list[NutritionEntry]:
    if value is None:
        return []
    if not isinstance(value, list):
        issues.append(f"nutrition: expected list, got {type(value).__name__}; discarded")
        return []
    out = []
    for i, item in enumerate(value):
        path = f"nutrition[{i}]"
        if not isinstance(item, dict):
            issues.append(f"{path}: expected object, got {type(item).__name__}; skipped")
            continue
        name = item.get("name")
        val = item.get("value")
        if name is None:
            issues.append(f"{path}: missing 'name'; skipped")
            continue
        if val is None:
            issues.append(f"{path}: missing 'value'; defaulted to ''")
        out.append(
            NutritionEntry(
                name=_coerce_str(name, f"{path}.name", issues),
                value=_coerce_str(val, f"{path}.value", issues) if val is not None else "",
            )
        )
    return out


def field_value_str(obj: LabelSchema, field_name: str) -> str:
    """Human-readable flattening of a field's value, for error-sample exports."""
    value = getattr(obj, field_name)
    if field_name == NUTRITION_FIELD:
        return "; ".join(f"{e.name}={e.value}" for e in value)
    if isinstance(value, list):
        return " | ".join(value)
    return value


def coerce_prediction(raw: object) -> tuple[LabelSchema, list[str]]:
    """Best-effort coercion of a raw parsed-JSON prediction into LabelSchema.

    Returns (schema, structural_issues). Never raises.
    """
    issues: list[str] = []
    if not isinstance(raw, dict):
        issues.append(f"root: expected object, got {type(raw).__name__}; using empty record")
        raw = {}

    kwargs = {
        "product_name": _coerce_str(raw.get("product_name"), "product_name", issues),
        "ingredient": _coerce_str_list(raw.get("ingredient"), "ingredient", issues),
        "additive": _coerce_str_list(raw.get("additive"), "additive", issues),
        "warning": _coerce_str_list(raw.get("warning"), "warning", issues),
        "nutrition": _coerce_nutrition(raw.get("nutrition"), issues),
        "origin": _coerce_str(raw.get("origin"), "origin", issues),
        "net_weight": _coerce_str(raw.get("net_weight"), "net_weight", issues),
        "mfg_date": _coerce_str(raw.get("mfg_date"), "mfg_date", issues),
        "expiry_date": _coerce_str(raw.get("expiry_date"), "expiry_date", issues),
    }
    for field in ALL_FIELDS:
        if field not in raw:
            issues.append(f"{field}: missing from output; defaulted")

    return LabelSchema(**kwargs), issues
