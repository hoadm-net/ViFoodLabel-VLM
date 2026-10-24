#!/usr/bin/env python3
"""OCR stage of the OCR -> LLM baseline (Reviewer 2.4). Run this in its own
environment, NOT the project's `uv` env -- like scripts/serve_vintern.sh,
its heavy dependencies (PaddlePaddle, PyTorch, VietOCR) would otherwise
bloat every install of the harness for one baseline:

    python3 -m venv .venv-ocr && . .venv-ocr/bin/activate
    pip install paddlepaddle==3.3.1 paddleocr==3.7.0 "setuptools<81" vietocr==0.3.13 \\
        torch==2.7.0 torchvision==0.22.0 --extra-index-url https://download.pytorch.org/whl/cu126
    python scripts/run_ocr.py                    # all of data/images
    python scripts/run_ocr.py --images 0001 0002

Engine "paddle-vietocr": PaddleOCR's PP-OCRv6 text-line detector + VietOCR
(vgg_transformer) recognizer. PaddleOCR's own `lang="vi"` recognizer was
tried first and rejected: it drops most stacked-diacritic characters
("Bánh xp őng" for "Bánh xốp ống", "Cht béo" for "Chất béo"), which would
make the baseline weak for a reason unrelated to the question it's meant
to answer. VietOCR is the de facto Vietnamese text-line recognizer.

Output, one file per image (the harness's `ocr.py` only ever reads these):
    results/ocr/<engine>/<image_id>.txt   one recognized line per text line,
                                          in reading order
    results/ocr/<engine>/<image_id>.json  boxes + per-line text, for audit
Existing outputs are skipped, so the script is resumable like every tier.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = REPO_ROOT / "data" / "images"
OCR_DIR = REPO_ROOT / "results" / "ocr"
ENGINE = "paddle-vietocr"


def reading_order(boxes: list[tuple[float, float, float, float]]) -> list[int]:
    """Group boxes into text lines (vertical-center overlap with the line's
    first box), lines top to bottom, boxes within a line left to right."""
    idx = sorted(range(len(boxes)), key=lambda i: (boxes[i][1] + boxes[i][3]) / 2)
    lines: list[list[int]] = []
    for i in idx:
        x0, y0, x1, y1 = boxes[i]
        center = (y0 + y1) / 2
        if lines:
            ref = boxes[lines[-1][0]]
            if ref[1] <= center <= ref[3]:
                lines[-1].append(i)
                continue
        lines.append([i])
    return [i for line in lines for i in sorted(line, key=lambda j: boxes[j][0])]


def build_engine():
    from paddleocr import TextDetection
    from vietocr.tool.config import Cfg
    from vietocr.tool.predictor import Predictor

    # enable_mkldnn=False: PaddlePaddle 3.3's oneDNN CPU path crashes on the
    # PP-OCRv6 detector ("ConvertPirAttribute2RuntimeAttribute not support").
    detector = TextDetection(model_name="PP-OCRv6_medium_det", device="cpu", enable_mkldnn=False)
    cfg = Cfg.load_config_from_name("vgg_transformer")
    cfg["device"] = os.environ.get("OCR_DEVICE") or ("cuda:0" if _cuda_available() else "cpu")
    cfg["predictor"]["beamsearch"] = False
    return detector, Predictor(cfg)


def _cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def ocr_image(detector, recognizer, path: Path) -> dict:
    import numpy as np
    from PIL import Image

    image = Image.open(path).convert("RGB")
    polys = detector.predict(str(path))[0]["dt_polys"]
    boxes = []
    for poly in polys:
        arr = np.asarray(poly, dtype=float)
        (x0, y0), (x1, y1) = arr.min(0), arr.max(0)
        boxes.append((float(x0), float(y0), float(x1), float(y1)))
    boxes = [b for b in boxes if b[2] - b[0] >= 4 and b[3] - b[1] >= 4]
    order = reading_order(boxes)
    crops = [image.crop(tuple(int(v) for v in boxes[i])) for i in order]
    texts = recognizer.predict_batch(crops) if crops else []
    return {"lines": [{"box": boxes[i], "text": t} for i, t in zip(order, texts)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", nargs="*", default=None, help="Image ids (default: all of data/images)")
    parser.add_argument("--shard", type=int, default=0, help="This worker's shard index")
    parser.add_argument("--num-shards", type=int, default=1, help="Total parallel workers")
    args = parser.parse_args()

    ids = args.images or sorted(p.stem for p in IMAGES_DIR.glob("*.jpeg"))
    ids = [i for k, i in enumerate(ids) if k % args.num_shards == args.shard]
    out_dir = OCR_DIR / ENGINE
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in ids if not (out_dir / f"{i}.txt").exists()]
    print(f"[shard {args.shard}] {len(ids)} images, {len(todo)} to do", flush=True)
    if not todo:
        return

    detector, recognizer = build_engine()
    for n, image_id in enumerate(todo, 1):
        start = time.monotonic()
        result = ocr_image(detector, recognizer, IMAGES_DIR / f"{image_id}.jpeg")
        result.update({"engine": ENGINE, "image_id": image_id})
        (out_dir / f"{image_id}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        # .txt last: its presence is the "done" marker for resumption.
        (out_dir / f"{image_id}.txt").write_text("\n".join(l["text"] for l in result["lines"]) + "\n", encoding="utf-8")
        print(f"[shard {args.shard}] {n}/{len(todo)} {image_id} {len(result['lines'])} lines {time.monotonic() - start:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
