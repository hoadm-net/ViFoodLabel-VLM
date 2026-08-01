"""perturbation.py -- synthetic image corruptions for Tier 3."""

from __future__ import annotations

import numpy as np
from PIL import Image

from vifoodlabel.perturbation import (
    PERTURBATION_TYPES,
    SEVERITY_LEVELS,
    all_perturbation_conditions,
    apply_perturbation,
    perturbation_condition_name,
)


def _sample_image(w=400, h=300) -> Image.Image:
    # A synthetic image with actual content (not solid color) so blur/glare
    # have something to visibly act on.
    arr = np.random.default_rng(0).integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


class TestAllPerturbationConditions:
    def test_3x3_combinations(self):
        conditions = all_perturbation_conditions()
        assert len(conditions) == 9
        assert set(k for k, _s in conditions) == set(PERTURBATION_TYPES)
        assert set(s for _k, s in conditions) == set(SEVERITY_LEVELS)

    def test_condition_naming(self):
        assert perturbation_condition_name("blur", 2) == "blur_s2"


class TestApplyPerturbation:
    def test_blur_preserves_dimensions(self):
        img = _sample_image()
        out = apply_perturbation(img, "blur", 2)
        assert out.size == img.size

    def test_rotation_preserves_dimensions(self):
        img = _sample_image()
        out = apply_perturbation(img, "rotation", 2)
        assert out.size == img.size

    def test_glare_preserves_dimensions(self):
        img = _sample_image()
        out = apply_perturbation(img, "glare", 2, seed=1)
        assert out.size == img.size

    def test_rotation_has_no_blank_corners(self):
        # The whole point of cropping to the largest inscribed rectangle:
        # no padding/blank-corner artifact that would tip the model off
        # that the image was rotated.
        img = Image.new("RGB", (400, 300), color=(120, 80, 40))
        out = apply_perturbation(img, "rotation", 3)
        corners = [out.getpixel((0, 0)), out.getpixel((out.width - 1, 0)),
                   out.getpixel((0, out.height - 1)), out.getpixel((out.width - 1, out.height - 1))]
        # A blank-corner bug would show pure white (255,255,255) padding.
        assert all(c != (255, 255, 255) for c in corners)

    def test_higher_blur_severity_is_blurrier(self):
        img = _sample_image()
        mild = np.asarray(apply_perturbation(img, "blur", 1))
        severe = np.asarray(apply_perturbation(img, "blur", 3))
        # Variance of pixel values drops as blur smooths out noise.
        assert severe.astype(float).var() < mild.astype(float).var()

    def test_glare_brightens_the_image(self):
        img = Image.new("RGB", (200, 200), color=(50, 50, 50))
        out = np.asarray(apply_perturbation(img, "glare", 3, seed=1))
        assert out.astype(float).mean() > 50

    def test_unknown_kind_raises(self):
        import pytest
        with pytest.raises(ValueError):
            apply_perturbation(_sample_image(), "not_a_kind", 1)

    def test_glare_is_deterministic_given_seed(self):
        img = _sample_image()
        a = np.asarray(apply_perturbation(img, "glare", 2, seed=42))
        b = np.asarray(apply_perturbation(img, "glare", 2, seed=42))
        assert np.array_equal(a, b)


class TestMaterialize(object):
    def test_caches_to_disk_and_is_idempotent(self, tmp_path, monkeypatch):
        from vifoodlabel import perturbation as perturbation_module

        monkeypatch.setattr(perturbation_module, "PERTURBED_IMAGES_DIR", tmp_path)
        src = tmp_path / "source.jpeg"
        _sample_image().save(src, format="JPEG")

        out1 = perturbation_module.materialize(src, "0001", "blur", 1, seed=0)
        assert out1.exists()
        mtime1 = out1.stat().st_mtime

        out2 = perturbation_module.materialize(src, "0001", "blur", 1, seed=0)
        assert out2 == out1
        assert out2.stat().st_mtime == mtime1  # not regenerated

    def test_glare_position_differs_across_image_ids_given_same_seed(self, tmp_path, monkeypatch):
        # Regression test: materialize() used to pass the same base seed
        # straight through to _apply_glare for every image, and a fresh
        # np.random.default_rng(seed) always draws the same first value for
        # the same seed -- so every image got glare in the identical
        # relative spot instead of a genuinely varying "random position".
        from vifoodlabel import perturbation as perturbation_module

        monkeypatch.setattr(perturbation_module, "PERTURBED_IMAGES_DIR", tmp_path)
        src = tmp_path / "source.jpeg"
        Image.new("RGB", (200, 200), color=(30, 30, 30)).save(src, format="JPEG")

        out1 = perturbation_module.materialize(src, "0001", "glare", 3, seed=42)
        out2 = perturbation_module.materialize(src, "0002", "glare", 3, seed=42)
        arr1 = np.asarray(Image.open(out1))
        arr2 = np.asarray(Image.open(out2))
        assert not np.array_equal(arr1, arr2)


class TestPerImageSeed:
    def test_differs_across_image_ids(self):
        from vifoodlabel.perturbation import _per_image_seed

        assert _per_image_seed(42, "0001") != _per_image_seed(42, "0002")

    def test_deterministic_given_same_inputs(self):
        from vifoodlabel.perturbation import _per_image_seed

        assert _per_image_seed(42, "0007") == _per_image_seed(42, "0007")
