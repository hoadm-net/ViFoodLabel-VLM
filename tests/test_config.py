"""config.py -- model registry role filtering: support models (the OCR
pipeline's text LLM, the taxonomy judge) must never join a default run,
or e.g. `benchmark` with no --models would send label images to a
text-only model and to the judge."""

from __future__ import annotations

import pytest

from vifoodlabel.config import load_model_registry, resolve_models


def test_default_selection_is_evaluated_models_only():
    assert all(m.role == "evaluated" for m in resolve_models(None))


def test_support_models_exist_but_are_excluded_by_default():
    registry = load_model_registry()
    support = {k for k, m in registry.items() if m.role != "evaluated"}
    assert support, "registry should define at least one support model"
    assert support.isdisjoint({m.key for m in resolve_models(None)})


def test_explicit_selector_reaches_any_role():
    registry = load_model_registry()
    judge = next(k for k, m in registry.items() if m.role == "judge")
    assert [m.key for m in resolve_models([judge])] == [judge]


def test_default_role_selects_that_role():
    assert all(m.role == "ocr_llm" for m in resolve_models(None, default_role="ocr_llm"))


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        resolve_models(["no-such-model"])
