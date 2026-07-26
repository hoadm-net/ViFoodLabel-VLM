"""cli/common.py -- id-range resolution and the shared random-subset
selection used by both ablation tiers (Tier 2/3)."""

from __future__ import annotations

from argparse import Namespace

from vifoodlabel.cli.common import DEFAULT_SUBSET_SIZE, resolve_image_ids, select_subset


def _args(**overrides) -> Namespace:
    base = dict(images=None, start_id=None, end_id=None)
    base.update(overrides)
    return Namespace(**base)


class TestResolveImageIds:
    def test_explicit_images_wins(self):
        ids = resolve_image_ids(_args(images=["0001", "0005"], start_id=1, end_id=600))
        assert ids == ["0001", "0005"]

    def test_no_selection_returns_none(self):
        assert resolve_image_ids(_args()) is None

    def test_start_end_range_zero_padded(self):
        ids = resolve_image_ids(_args(start_id=1, end_id=3))
        assert ids == ["0001", "0002", "0003"]

    def test_start_only_uses_end_of_dataset(self, monkeypatch):
        monkeypatch.setattr("vifoodlabel.cli.common.list_image_ids", lambda: [f"{i:04d}" for i in range(1, 601)])
        ids = resolve_image_ids(_args(start_id=598))
        assert ids == ["0598", "0599", "0600"]

    def test_end_only_starts_at_1(self):
        ids = resolve_image_ids(_args(end_id=2))
        assert ids == ["0001", "0002"]

    def test_start_after_end_raises(self):
        import pytest
        with pytest.raises(ValueError):
            resolve_image_ids(_args(start_id=5, end_id=2))


class TestSelectSubset:
    def test_deterministic_given_same_seed(self, monkeypatch):
        monkeypatch.setattr("vifoodlabel.cli.common.list_image_ids", lambda: [f"{i:04d}" for i in range(1, 601)])
        a = select_subset(120, seed=42)
        b = select_subset(120, seed=42)
        assert a == b

    def test_different_seed_different_subset(self, monkeypatch):
        monkeypatch.setattr("vifoodlabel.cli.common.list_image_ids", lambda: [f"{i:04d}" for i in range(1, 601)])
        a = select_subset(120, seed=1)
        b = select_subset(120, seed=2)
        assert a != b

    def test_respects_requested_size(self, monkeypatch):
        monkeypatch.setattr("vifoodlabel.cli.common.list_image_ids", lambda: [f"{i:04d}" for i in range(1, 601)])
        assert len(select_subset(120, seed=42)) == 120

    def test_size_larger_than_dataset_returns_everything(self, monkeypatch):
        monkeypatch.setattr("vifoodlabel.cli.common.list_image_ids", lambda: [f"{i:04d}" for i in range(1, 11)])
        assert len(select_subset(120, seed=42)) == 10

    def test_default_subset_size_is_120(self):
        assert DEFAULT_SUBSET_SIZE == 120
