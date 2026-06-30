from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils.paths import (
    CURVE_YOLO,
    CURVES_DIR,
    METRICS_DIR,
    METRICS_YOLO,
    MODELS_DIR,
    YOLO_RESULTS_CSV,
    YOLO_RESULTS_PNG,
    YOLO_RUN_WEIGHTS,
    YOLO_WEIGHTS,
)


def copy_weights() -> None:
    if not YOLO_RUN_WEIGHTS.exists():
        raise FileNotFoundError(f"Fine-tuned YOLO weights not found: {YOLO_RUN_WEIGHTS}")

    YOLO_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(YOLO_RUN_WEIGHTS, YOLO_WEIGHTS)
    print(f"[YOLO] Weights copied -> {YOLO_WEIGHTS}")


def _read_results_csv() -> list[dict]:
    rows = []
    with open(YOLO_RESULTS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def build_metrics() -> dict:
    rows = _read_results_csv()
    if not rows:
        raise RuntimeError(f"No rows found in {YOLO_RESULTS_CSV}")

    last = rows[-1]
    precision = float(last["metrics/precision(B)"])
    recall = float(last["metrics/recall(B)"])
    map50 = float(last["metrics/mAP50(B)"])
    map5095 = float(last["metrics/mAP50-95(B)"])
    train_time_sec = float(last["time"])

    size_mb = round(YOLO_WEIGHTS.stat().st_size / (1024 * 1024), 2)

    latency_ms = None
    try:
        from ultralytics import YOLO

        model = YOLO(str(YOLO_WEIGHTS))
        dummy = np.zeros((416, 416, 3), dtype=np.uint8)
        start = time.perf_counter()
        model.predict(source=dummy, verbose=False)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
    except Exception as exc:
        print(f"[YOLO] Latency benchmark skipped: {exc}")

    metrics = {
        "model": "YOLOv8n",
        "task": "detection",
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(_f1(precision, recall), 4),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map5095, 4),
        "latency_ms": latency_ms,
        "model_size_mb": size_mb,
        "train_time_sec": round(train_time_sec, 1),
        "epochs_trained": len(rows),
        "history": {
            "train_box_loss": [float(r["train/box_loss"]) for r in rows],
            "train_cls_loss": [float(r["train/cls_loss"]) for r in rows],
            "train_dfl_loss": [float(r["train/dfl_loss"]) for r in rows],
            "val_box_loss": [float(r["val/box_loss"]) for r in rows],
            "val_cls_loss": [float(r["val/cls_loss"]) for r in rows],
            "val_dfl_loss": [float(r["val/dfl_loss"]) for r in rows],
            "precision": [float(r["metrics/precision(B)"]) for r in rows],
            "recall": [float(r["metrics/recall(B)"]) for r in rows],
            "mAP50": [float(r["metrics/mAP50(B)"]) for r in rows],
        },
    }
    return metrics


def save_metrics(metrics: dict) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_YOLO, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[YOLO] Metrics saved -> {METRICS_YOLO}")


def save_curves(metrics: dict) -> None:
    CURVES_DIR.mkdir(parents=True, exist_ok=True)

    if YOLO_RESULTS_PNG.exists():
        shutil.copy2(YOLO_RESULTS_PNG, CURVE_YOLO)
        print(f"[YOLO] Curves copied -> {CURVE_YOLO}")
        return

    history = metrics["history"]
    epochs = range(1, len(history["train_box_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("YOLOv8 Fine-tuning Curves", fontsize=14, fontweight="bold")

    axes[0].plot(epochs, history["train_box_loss"], label="Train box loss", color="#22c55e")
    axes[0].plot(epochs, history["val_box_loss"], label="Val box loss", color="#3b82f6")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Box Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["mAP50"], label="mAP@50", color="#f59e0b")
    axes[1].plot(epochs, history["precision"], label="Precision", color="#22c55e")
    axes[1].plot(epochs, history["recall"], label="Recall", color="#3b82f6")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Metrics")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CURVE_YOLO, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[YOLO] Curves generated -> {CURVE_YOLO}")


def export_yolo_artifacts() -> None:
    copy_weights()
    metrics = build_metrics()
    save_metrics(metrics)
    save_curves(metrics)


if __name__ == "__main__":
    export_yolo_artifacts()
