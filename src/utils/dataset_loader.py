"""
Unified dataset helpers for YOLO detection, ViT classification, and SAM segmentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml
from PIL import Image
from torch.utils.data import Dataset

from .paths import DATA_DIR, DATA_YAML


def load_class_names(yaml_path: Path | None = None) -> list[str]:
    """Load class names from the YOLO data.yaml file."""
    yaml_path = yaml_path or DATA_YAML
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names", {})
    if isinstance(names, dict):
        return [names[i] for i in sorted(names, key=lambda k: int(k))]
    return list(names)


def yolo_label_path(image_path: Path) -> Path:
    """Map an image path to its YOLO label file."""
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def read_yolo_class_id(label_path: Path) -> int | None:
    """Return the dominant class id in a YOLO label file."""
    if not label_path.exists():
        return None

    counts: dict[int, int] = {}
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            counts[cls_id] = counts.get(cls_id, 0) + 1

    if not counts:
        return None
    return max(counts, key=counts.get)


def iter_split_images(split: str) -> Iterable[Path]:
    """Yield image paths for train/valid/test splits."""
    image_dir = DATA_DIR / split / "images"
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        yield from sorted(image_dir.glob(pattern))


class YOLOClassificationDataset(Dataset):
    """
    ViT classification dataset built from YOLO annotations.
    Each image inherits the most frequent class id in its label file.
    """

    def __init__(self, split: str, transform=None):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for image_path in iter_split_images(split):
            label_path = yolo_label_path(image_path)
            cls_id = read_yolo_class_id(label_path)
            if cls_id is not None:
                self.samples.append((image_path, cls_id))

        if not self.samples:
            raise RuntimeError(f"No labeled images found for split '{split}'")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def yolo_to_xyxy(bbox_yolo: list[float], img_w: int, img_h: int) -> list[float]:
    """Convert YOLO cx,cy,w,h (normalized) to pixel x1,y1,x2,y2."""
    cx, cy, bw, bh = bbox_yolo
    x1 = (cx - bw / 2) * img_w
    y1 = (cy - bh / 2) * img_h
    x2 = (cx + bw / 2) * img_w
    y2 = (cy + bh / 2) * img_h
    return [x1, y1, x2, y2]
