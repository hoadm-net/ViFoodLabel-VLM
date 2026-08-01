"""cli/common.py -- id-range resolution and the shared random-subset
selection used by both ablation tiers (Tier 2/3)."""

from __future__ import annotations

import json
from argparse import Namespace

from vifoodlabel.cli.common import (
    DEFAULT_SUBSET_SEED,
    DEFAULT_SUBSET_SIZE,
    load_pinned_subset,
    resolve_image_ids,
    resolve_subset_image_ids,
    select_subset,
)


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


class TestLoadPinnedSubset:
    def test_returns_none_when_file_absent(self, monkeypatch, tmp_path):
        monkeypatch.setattr("vifoodlabel.cli.common.SUBSET_FILE", tmp_path / "missing.json")
        assert load_pinned_subset() is None

    def test_returns_image_ids_from_file(self, monkeypatch, tmp_path):
        path = tmp_path / "subset_120.json"
        path.write_text(json.dumps({"size": 3, "seed": 42, "image_ids": ["0001", "0002", "0003"]}))
        monkeypatch.setattr("vifoodlabel.cli.common.SUBSET_FILE", path)
        assert load_pinned_subset() == ["0001", "0002", "0003"]


class TestResolveSubsetImageIds:
    def test_explicit_images_wins_over_subset(self):
        ids = resolve_subset_image_ids(_args(images=["0001"], subset_size=DEFAULT_SUBSET_SIZE, seed=DEFAULT_SUBSET_SEED))
        assert ids == ["0001"]

    def test_uses_pinned_subset_at_default_size_and_seed(self, monkeypatch, tmp_path):
        path = tmp_path / "subset_120.json"
        path.write_text(json.dumps({"size": 120, "seed": 42, "image_ids": ["0009", "0010"]}))
        monkeypatch.setattr("vifoodlabel.cli.common.SUBSET_FILE", path)
        ids = resolve_subset_image_ids(_args(subset_size=DEFAULT_SUBSET_SIZE, seed=DEFAULT_SUBSET_SEED))
        assert ids == ["0009", "0010"]

    def test_falls_back_to_live_selection_when_no_pinned_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr("vifoodlabel.cli.common.SUBSET_FILE", tmp_path / "missing.json")
        monkeypatch.setattr("vifoodlabel.cli.common.list_image_ids", lambda: [f"{i:04d}" for i in range(1, 601)])
        ids = resolve_subset_image_ids(_args(subset_size=DEFAULT_SUBSET_SIZE, seed=DEFAULT_SUBSET_SEED))
        assert len(ids) == DEFAULT_SUBSET_SIZE

    def test_custom_size_or_seed_ignores_pinned_file(self, monkeypatch, tmp_path):
        path = tmp_path / "subset_120.json"
        path.write_text(json.dumps({"size": 120, "seed": 42, "image_ids": ["0009", "0010"]}))
        monkeypatch.setattr("vifoodlabel.cli.common.SUBSET_FILE", path)
        monkeypatch.setattr("vifoodlabel.cli.common.list_image_ids", lambda: [f"{i:04d}" for i in range(1, 601)])
        ids = resolve_subset_image_ids(_args(subset_size=5, seed=99))
        assert ids != ["0009", "0010"]
        assert len(ids) == 5
