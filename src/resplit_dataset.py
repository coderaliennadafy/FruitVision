"""
Resplit fruit dataset to 60/20/20 (train/valid/test).
Run once from project root: python src/resplit_dataset.py
"""
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "fruit_dataset"
SPLITS = ("train", "valid", "test")
RATIOS = (0.60, 0.20, 0.20)
SEED = 42

def collect_pairs():
    pairs = []
    for split in SPLITS:
        img_dir = DATA_DIR / split / "images"
        lbl_dir = DATA_DIR / split / "labels"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                pairs.append((img_path, lbl_path))
    return pairs

def main():
    pairs = collect_pairs()
    print(f"Total image+label pairs: {len(pairs)}")

    random.seed(SEED)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * RATIOS[0])
    n_valid = int(n * RATIOS[1])
    # test = le reste (pour arriver exactement à n)

    buckets = {
        "train": pairs[:n_train],
        "valid": pairs[n_train:n_train + n_valid],
        "test": pairs[n_train + n_valid:],
    }

    for split, items in buckets.items():
        print(f"{split}: {len(items)} ({100 * len(items) / n:.1f}%)")

    staging = DATA_DIR / "_resplit_staging"
    if staging.exists():
        shutil.rmtree(staging)

    for split, items in buckets.items():
        img_out = staging / split / "images"
        lbl_out = staging / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img_path, lbl_path in items:
            shutil.copy2(img_path, img_out / img_path.name)
            shutil.copy2(lbl_path, lbl_out / lbl_path.name)

    # Remplacer les anciens splits
    for split in SPLITS:
        old = DATA_DIR / split
        if old.exists():
            shutil.rmtree(old)

    for split in SPLITS:
        shutil.move(str(staging / split), str(DATA_DIR / split))

    shutil.rmtree(staging)
    print("Done. New split saved in data/fruit_dataset/")

if __name__ == "__main__":
    main()