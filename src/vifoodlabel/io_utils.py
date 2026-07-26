"""Dataset indexing, image encoding, and small JSON I/O helpers."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

from vifoodlabel.config import IMAGES_DIR, LABELS_DIR
from vifoodlabel.schema import LabelSchema

_MIME_BY_SUFFIX = {
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


@dataclass(frozen=True)
class DatasetItem:
    image_id: str
    image_path: Path
    label_path: Path

    @property
    def has_ground_truth(self) -> bool:
        return self.label_path.exists()


def list_image_ids() -> list[str]:
    """All image ids (filename stems) in data/images, sorted."""
    return sorted(p.stem for p in IMAGES_DIR.glob("*.jpeg"))


def dataset_index(image_ids: list[str] | None = None) -> list[DatasetItem]:
    """Build the (image, label) index. If image_ids is given, restrict to those."""
    ids = image_ids if image_ids is not None else list_image_ids()
    items = []
    for image_id in ids:
        image_path = IMAGES_DIR / f"{image_id}.jpeg"
        if not image_path.exists():
            raise FileNotFoundError(f"No image found for id '{image_id}' at {image_path}")
        items.append(
            DatasetItem(
                image_id=image_id,
                image_path=image_path,
                label_path=LABELS_DIR / f"{image_id}.json",
            )
        )
    return items


def labeled_only(items: list[DatasetItem]) -> list[DatasetItem]:
    """Filter to items that currently have a ground-truth file on disk."""
    return [item for item in items if item.has_ground_truth]


def load_ground_truth(item: DatasetItem) -> LabelSchema:
    raw = load_json(item.label_path)
    return LabelSchema.model_validate(raw)


def image_to_data_url(path: Path) -> str:
    mime = _MIME_BY_SUFFIX.get(path.suffix.lower(), "application/octet-stream")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
