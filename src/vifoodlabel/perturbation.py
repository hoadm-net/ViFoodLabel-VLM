"""Synthetic image corruptions for the Tier-3 robustness curve: blur, glare, rotation.

Each corruption has 3 severity levels. Rotation is cropped back down to the
original canvas via the largest-inscribed-rectangle formula so there are no
blank/white corners that would tip the model off that the image was rotated.
Perturbed images are materialized to disk (not regenerated per model call) so
they're cached, reusable across all 4 models, and inspectable by hand.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from vifoodlabel.config import PERTURBED_IMAGES_DIR

PERTURBATION_TYPES = ["blur", "glare", "rotation"]
SEVERITY_LEVELS = [1, 2, 3]

_BLUR_SIGMA = {1: 1.5, 2: 3.5, 3: 6.0}
_ROTATION_DEGREES = {1: 5, 2: 15, 3: 30}
# (additive intensity in [0,1], patch radius as a fraction of min(h, w))
_GLARE_PARAMS = {1: (0.35, 0.18), 2: (0.55, 0.22), 3: (0.75, 0.28)}


def perturbation_condition_name(kind: str, severity: int) -> str:
    return f"{kind}_s{severity}"


def all_perturbation_conditions() -> list[tuple[str, int]]:
    return [(k, s) for k in PERTURBATION_TYPES for s in SEVERITY_LEVELS]


def _apply_blur(image: Image.Image, severity: int) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=_BLUR_SIGMA[severity]))


def _largest_inscribed_rect(w: int, h: int, angle_deg: float) -> tuple[float, float]:
    """Largest axis-aligned rectangle that fits inside a w x h image after
    rotating it by angle_deg, with no blank corners. Standard closed-form."""
    angle = math.radians(angle_deg)
    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr = (w * cos_a - h * sin_a) / cos_2a
        hr = (h * cos_a - w * sin_a) / cos_2a
    return wr, hr


def _apply_rotation(image: Image.Image, severity: int) -> Image.Image:
    angle = _ROTATION_DEGREES[severity]
    w, h = image.size
    rotated = image.rotate(angle, resample=Image.BICUBIC, expand=True)
    rw, rh = _largest_inscribed_rect(w, h, angle)
    rw, rh = min(rw, rotated.width), min(rh, rotated.height)
    left = (rotated.width - rw) / 2
    top = (rotated.height - rh) / 2
    cropped = rotated.crop((left, top, left + rw, top + rh))
    return cropped.resize((w, h), Image.LANCZOS)


def _apply_glare(image: Image.Image, severity: int, seed: int) -> Image.Image:
    intensity, radius_frac = _GLARE_PARAMS[severity]
    rng = np.random.default_rng(seed)
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    radius = radius_frac * min(h, w)
    cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    falloff = np.clip(1.0 - dist / radius, 0.0, 1.0) ** 2
    glared = arr + falloff[..., None] * intensity * 255.0
    return Image.fromarray(np.clip(glared, 0, 255).astype(np.uint8))


def apply_perturbation(image: Image.Image, kind: str, severity: int, seed: int = 0) -> Image.Image:
    if kind == "blur":
        return _apply_blur(image, severity)
    if kind == "rotation":
        return _apply_rotation(image, severity)
    if kind == "glare":
        return _apply_glare(image, severity, seed)
    raise ValueError(f"Unknown perturbation kind: {kind!r}")


def perturbed_image_path(image_id: str, kind: str, severity: int) -> Path:
    condition = perturbation_condition_name(kind, severity)
    return PERTURBED_IMAGES_DIR / condition / f"{image_id}.jpeg"


def _per_image_seed(seed: int, image_id: str) -> int:
    """Offsets the base seed by the image id so glare's random highlight
    position (the only perturbation with any randomness) varies per image
    instead of landing in the exact same relative spot on every photo --
    while staying fully deterministic/reproducible for a given (seed,
    image_id) pair. A single shared seed would otherwise re-seed the RNG
    identically on every call, since `np.random.default_rng(seed)` always
    produces the same first draw for the same seed."""
    return seed + int(image_id)


def materialize(image_path: Path, image_id: str, kind: str, severity: int, seed: int = 0, force: bool = False) -> Path:
    """Generate (if not already cached) the perturbed image and return its path."""
    out_path = perturbed_image_path(image_id, kind, severity)
    if out_path.exists() and not force:
        return out_path
    image = Image.open(image_path).convert("RGB")
    perturbed = apply_perturbation(image, kind, severity, seed=_per_image_seed(seed, image_id))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    perturbed.save(out_path, format="JPEG", quality=92)
    return out_path
