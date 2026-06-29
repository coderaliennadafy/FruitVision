"""
Fine-tune Vision Transformer (ViT) for fruit classification.
Uses YOLO labels to derive image-level class targets.
"""

from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor

from utils.dataset_loader import YOLOClassificationDataset, load_class_names
from utils.paths import (
    CURVE_VIT,
    CURVES_DIR,
    METRICS_DIR,
    METRICS_VIT,
    VIT_MODEL_DIR,
)

MODEL_NAME = "google/vit-base-patch16-224"
EPOCHS = 8
BATCH_SIZE = 32
LR = 2e-5
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def train_vit() -> None:
    print(f"[ViT] Device: {DEVICE}")
    VIT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names()
    train_ds = YOLOClassificationDataset("train", transform=get_transforms(train=True))
    val_ds = YOLOClassificationDataset("valid", transform=get_transforms(train=False))
    test_ds = YOLOClassificationDataset("test", transform=get_transforms(train=False))

    num_classes = len(class_names)
    print(
        f"[ViT] Classes: {num_classes} | "
        f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}"
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    id2label = {i: name for i, name in enumerate(class_names)}
    label2id = {name: i for i, name in id2label.items()}

    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(DEVICE)

    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(pixel_values=imgs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_ds)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(pixel_values=imgs)
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_ds)
        val_acc = correct / total
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(VIT_MODEL_DIR)
            print(f"  Best model saved (val_acc={best_val_acc:.4f})")

    train_time = time.time() - t_start

    model.eval()
    all_preds, all_labels = [], []
    inf_times = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            t0 = time.perf_counter()
            out = model(pixel_values=imgs)
            inf_times.append((time.perf_counter() - t0) / imgs.size(0))
            all_preds.extend(out.logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    latency = float(np.mean(inf_times) * 1000)

    model_size_mb = sum(
        f.stat().st_size for f in VIT_MODEL_DIR.rglob("*") if f.is_file()
    ) / (1024 * 1024)

    metrics = {
        "model": "ViT-base-patch16-224",
        "task": "classification",
        "num_classes": num_classes,
        "epochs_trained": EPOCHS,
        "train_time_sec": round(train_time, 1),
        "best_val_acc": round(best_val_acc, 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "latency_ms": round(latency, 2),
        "model_size_mb": round(model_size_mb, 2),
        "history": history,
    }

    with open(METRICS_VIT, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    epochs_range = range(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("ViT Fine-tuning Curves", fontsize=14, fontweight="bold")
    axes[0].plot(epochs_range, history["train_loss"], label="Train Loss", color="#22c55e")
    axes[0].plot(epochs_range, history["val_loss"], label="Val Loss", color="#3b82f6")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs_range, history["val_acc"], label="Val Accuracy", color="#f59e0b")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CURVE_VIT, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[ViT] Metrics -> {METRICS_VIT}")
    print(f"[ViT] Curves  -> {CURVE_VIT}")


if __name__ == "__main__":
    train_vit()
