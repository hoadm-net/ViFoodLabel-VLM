"""cost.py -- includes a regression test for a real bug: append_ledger bound
LEDGER_PATH as a default argument at import time, so monkeypatching
cost.LEDGER_PATH (as test_runner.py does) had no effect and the runner
tests appended fake "test-model" rows to the real results/cost_ledger.csv."""

from __future__ import annotations

import vifoodlabel.cost as cost
from vifoodlabel.config import ModelSpec


def _model() -> ModelSpec:
    return ModelSpec(key="m", slug="m/m", display_name="M", group="g", open_weight=True,
                     price_input_per_m=0.0, price_output_per_m=0.0)


def test_monkeypatched_ledger_path_is_honored(tmp_path, monkeypatch):
    target = tmp_path / "ledger.csv"
    monkeypatch.setattr(cost, "LEDGER_PATH", target)
    cost.append_ledger("benchmark", _model(), "0001", "vi_zero", 10, 5, 0.001)
    assert target.exists()
    assert cost.read_ledger_total() == 0.001


def test_explicit_path_still_works(tmp_path):
    target = tmp_path / "explicit.csv"
    cost.append_ledger("benchmark", _model(), "0001", "vi_zero", 10, 5, 0.002, ledger_path=target)
    assert cost.read_ledger_total(target) == 0.002
