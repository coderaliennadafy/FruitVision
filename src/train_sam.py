"""
Fine-tune SAM mask decoder for fruit segmentation.
Uses YOLO bounding boxes as prompts and pseudo ground-truth masks.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import SamModel, SamProcessor

from utils.dataset_loader import yolo_to_xyxy
from utils.paths import (
    CURVE_SAM,
    CURVES_DIR,
    DATA_DIR,
    METRICS_DIR,
    METRICS_SAM,
    MODELS_DIR,
    SAM_WEIGHTS,
)

MODEL_NAME = "facebook/sam-vit-base"

EPOCHS = 1
BATCH_SIZE = 1

MAX_TRAIN_SAMPLES = 20
MAX_VAL_SAMPLES = 5

LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAM_OUTPUT_DIR = MODELS_DIR / "sam_finetuned"


class FruitSAMDataset(Dataset):
    """Pairs YOLO annotations with images for SAM decoder fine-tuning."""

    def __init__(self, split: str):
        self.items: list[tuple[Path, list[float]]] = []
        split_dir = DATA_DIR / split
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        for img_path in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
                parts = line.split()
                if len(parts) == 5:
                    self.items.append((img_path, list(map(float, parts[1:]))))

        if not self.items:
            raise RuntimeError(f"No SAM training samples found in split '{split}'")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        img_path, bbox_yolo = self.items[index]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        bbox = yolo_to_xyxy(bbox_yolo, w, h)

        mask = torch.zeros((h, w), dtype=torch.float32)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        mask[y1:y2, x1:x2] = 1.0
        return img, bbox, mask


def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred).view(-1)
    target = target.view(-1)
    inter = (pred * target).sum()
    return 1 - (2 * inter + eps) / (pred.sum() + target.sum() + eps)


def seg_loss(pred_masks, gt_masks):
    bce = F.binary_cross_entropy_with_logits(pred_masks, gt_masks)
    return bce + dice_loss(pred_masks, gt_masks)


def compute_iou(pred_mask, gt_mask, threshold=0.5):
    pred_bin = (torch.sigmoid(pred_mask) > threshold).float()
    inter = (pred_bin * gt_mask).sum()
    union = pred_bin.sum() + gt_mask.sum() - inter
    return (inter / (union + 1e-6)).item()


def train_sam() -> None:
    print(f"[SAM] Device: {DEVICE}")
    SAM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    processor = SamProcessor.from_pretrained(MODEL_NAME)
    model = SamModel.from_pretrained(MODEL_NAME)

    for name, param in model.named_parameters():
        param.requires_grad = "mask_decoder" in name
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    train_ds = FruitSAMDataset("train")
    val_ds = FruitSAMDataset("valid")
    test_ds = FruitSAMDataset("test")
    if len(train_ds) > MAX_TRAIN_SAMPLES:
        train_ds.items = train_ds.items[:MAX_TRAIN_SAMPLES]
    if len(val_ds) > MAX_VAL_SAMPLES:
        val_ds.items = val_ds.items[:MAX_VAL_SAMPLES]
    print(f"[SAM] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}", flush=True)

    history = {"train_loss": [], "val_loss": [], "val_iou": []}
    best_iou = 0.0
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_ds), BATCH_SIZE):
            
            if i % 20 == 0:
                print(f"Training batch {i}/{len(train_ds)}")
                
            batch = [train_ds[j] for j in range(i, min(i + BATCH_SIZE, len(train_ds)))]
            imgs, bboxes, gt_masks = zip(*batch)

            inputs = processor(
                images=list(imgs),
                input_boxes=[[b] for b in bboxes],
                return_tensors="pt",
            ).to(DEVICE)

            gt_tensor = torch.stack([
                F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(256, 256), mode="nearest").squeeze()
                for m in gt_masks
            ]).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(**inputs, multimask_output=False)
            pred_masks = outputs.pred_masks.squeeze(2)
            gt_tensor = gt_tensor.unsqueeze(1)
            loss = seg_loss(pred_masks, gt_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            print(f"Batch {i}/{len(train_ds)} done")

            ep_loss += loss.item()
            n_batches += 1

        train_loss = ep_loss / max(n_batches, 1)

        model.eval()
        val_loss_sum, iou_sum, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for i in range(0, min(len(val_ds), 120), BATCH_SIZE):
                batch = [val_ds[j] for j in range(i, min(i + BATCH_SIZE, len(val_ds)))]
                imgs, bboxes, gt_masks = zip(*batch)
                inputs = processor(
                    images=list(imgs),
                    input_boxes=[[b] for b in bboxes],
                    return_tensors="pt",
                ).to(DEVICE)
                gt_tensor = torch.stack([
                    F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(256, 256), mode="nearest").squeeze()
                    for m in gt_masks
                ]).to(DEVICE)
                outputs = model(**inputs, multimask_output=False)
                pred_masks = outputs.pred_masks.squeeze(2)
                gt_tensor = gt_tensor.unsqueeze(1)
                val_loss_sum += seg_loss(pred_masks, gt_tensor).item()
                iou_sum += compute_iou(pred_masks, gt_tensor)
                n_val += 1

        val_loss = val_loss_sum / max(n_val, 1)
        val_iou = iou_sum / max(n_val, 1)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)

        print(
            f"Epoch {epoch:02d}/{EPOCHS}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_iou={val_iou:.4f}"
        )

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), SAM_WEIGHTS)
            print(f"  Best decoder saved (IoU={best_iou:.4f})")

    train_time = time.time() - t_start

    model.eval()
    iou_list, inf_times = [], []
    with torch.no_grad():
        for i in range(0, min(len(test_ds), 120), BATCH_SIZE):
            batch = [test_ds[j] for j in range(i, min(i + BATCH_SIZE, len(test_ds)))]
            imgs, bboxes, gt_masks = zip(*batch)
            inputs = processor(
                images=list(imgs),
                input_boxes=[[b] for b in bboxes],
                return_tensors="pt",
            ).to(DEVICE)
            gt_tensor = torch.stack([
                F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(256, 256), mode="nearest").squeeze()
                for m in gt_masks
            ]).to(DEVICE)
            t0 = time.perf_counter()
            outputs = model(**inputs, multimask_output=False)
            inf_times.append((time.perf_counter() - t0) / len(batch))
            pred_masks = outputs.pred_masks.squeeze(2)
            gt_tensor = gt_tensor.unsqueeze(1)
            iou_list.append(compute_iou(pred_masks, gt_tensor))

    mean_iou = float(np.mean(iou_list)) if iou_list else best_iou
    latency = float(np.mean(inf_times) * 1000) if inf_times else 0.0
    model_size_mb = round(SAM_WEIGHTS.stat().st_size / (1024 * 1024), 2) if SAM_WEIGHTS.exists() else 0.0

    metrics = {
        "model": "SAM-vit-base (decoder fine-tuned)",
        "task": "segmentation",
        "epochs_trained": EPOCHS,
        "train_time_sec": round(train_time, 1),
        "mean_iou": round(mean_iou, 4),
        "latency_ms": round(latency, 2),
        "model_size_mb": model_size_mb,
        "history": {k: [round(v, 4) for v in vals] for k, vals in history.items()},
    }

    with open(METRICS_SAM, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("SAM Fine-tuning Curves (Decoder Only)", fontsize=14, fontweight="bold")
    axes[0].plot(epochs_range, history["train_loss"], label="Train Loss", color="#22c55e")
    axes[0].plot(epochs_range, history["val_loss"], label="Val Loss", color="#3b82f6")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs_range, history["val_iou"], label="Val IoU", color="#f59e0b")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CURVE_SAM, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[SAM] Metrics -> {METRICS_SAM}")
    print(f"[SAM] Curves  -> {CURVE_SAM}")


if __name__ == "__main__":
    train_sam()
