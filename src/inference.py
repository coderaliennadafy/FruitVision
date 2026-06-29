"""
CLI inference for the three fine-tuned models.

Usage:
    python inference.py --model yolo --image path/to/img.jpg
    python inference.py --model all --image path/to/img.jpg
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.paths import SAM_WEIGHTS, VIT_MODEL_DIR, YOLO_WEIGHTS


def infer_yolo(image_path: str, conf: float = 0.25):
    from ultralytics import YOLO

    model = YOLO(str(YOLO_WEIGHTS))
    image = Image.open(image_path).convert("RGB")
    t0 = time.perf_counter()
    results = model.predict(source=np.array(image), conf=conf, verbose=False)
    elapsed = (time.perf_counter() - t0) * 1000
    boxes = results[0].boxes
    print(f"[YOLO] {elapsed:.1f} ms | {len(boxes)} detections")
    for i, (cls_id, conf_val) in enumerate(zip(boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy())):
        print(f"  [{i + 1}] class={int(cls_id)} conf={conf_val:.3f}")


def infer_vit(image_path: str):
    import torch
    from transformers import ViTForImageClassification, ViTImageProcessor

    model = ViTForImageClassification.from_pretrained(str(VIT_MODEL_DIR))
    processor = ViTImageProcessor.from_pretrained(str(VIT_MODEL_DIR))
    model.eval()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(**inputs).logits
    elapsed = (time.perf_counter() - t0) * 1000
    probs = torch.softmax(logits, dim=-1)[0]
    top = probs.topk(5)
    print(f"[ViT] {elapsed:.1f} ms")
    for score, idx in zip(top.values, top.indices):
        print(f"  {model.config.id2label[int(idx)]}: {score.item() * 100:.2f}%")


def infer_sam(image_path: str):
    import torch
    from transformers import SamModel, SamProcessor

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    state = torch.load(SAM_WEIGHTS, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    bbox = [[w * 0.05, h * 0.05, w * 0.95, h * 0.95]]
    inputs = processor(images=image, input_boxes=[bbox], return_tensors="pt")
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    elapsed = (time.perf_counter() - t0) * 1000
    mask = outputs.pred_masks[0, 0, 0].numpy() > 0
    print(f"[SAM] {elapsed:.1f} ms | coverage={mask.mean() * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["yolo", "vit", "sam", "all"], default="yolo")
    parser.add_argument("--image", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    if args.model in ("yolo", "all"):
        infer_yolo(args.image, args.conf)
    if args.model in ("vit", "all"):
        infer_vit(args.image)
    if args.model in ("sam", "all"):
        infer_sam(args.image)


if __name__ == "__main__":
    main()
