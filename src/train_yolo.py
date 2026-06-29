"""
Fine-tune YOLOv8 on the fruit detection dataset.
Run only when you want to retrain; otherwise use export_yolo_artifacts.py.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ultralytics import YOLO

from utils.paths import (
    CURVE_YOLO,
    CURVES_DIR,
    DATA_YAML,
    METRICS_DIR,
    METRICS_YOLO,
    MODELS_DIR,
    PRETRAINED_YOLO,
    YOLO_WEIGHTS,
)


def train_yolo() -> None:
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

    model = YOLO(str(PRETRAINED_YOLO))
    start = time.time()

    model.train(
        data=str(DATA_YAML),
        epochs=15,
        imgsz=416,
        batch=4,
        project=str(MODELS_DIR),
        name="yolo_finetuned",
        pretrained=True,
        exist_ok=True,
    )

    results = model.val()
    train_time = time.time() - start

    precision = float(results.box.mp)
    recall = float(results.box.mr)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    best_src = MODELS_DIR / "yolo_finetuned" / "weights" / "best.pt"
    YOLO_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    if best_src.exists():
        YOLO_WEIGHTS.write_bytes(best_src.read_bytes())

    metrics = {
        "model": "YOLOv8n",
        "task": "detection",
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mAP50": round(float(results.box.map50), 4),
        "mAP50_95": round(float(results.box.map), 4),
        "train_time_sec": round(train_time, 1),
        "model_size_mb": round(YOLO_WEIGHTS.stat().st_size / (1024 * 1024), 2),
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_YOLO, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    results_png = MODELS_DIR / "yolo_finetuned" / "results.png"
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    if results_png.exists():
        CURVE_YOLO.write_bytes(results_png.read_bytes())

    print("YOLO training complete.")
    print(metrics)


if __name__ == "__main__":
    train_yolo()
