"""
modes_masks_to_yolo.py — convert the MoDES stray-animals dataset
(images + segmentation masks) into YOLO detection format.

MoDES ships RGB frames in images/ and a foreground mask per frame in masks/.
There are no bounding boxes, so we derive a box from each mask's foreground
region (the animal) and label it class 0 = "animal". The result is a standard
YOLO detection dataset you can train with train.py --data.

Usage:
    python tools/modes_masks_to_yolo.py \
        --src datasets/stray_animals/src0/out2 \
        --out datasets/stray_animals/yolo \
        --limit 20000
"""
from __future__ import annotations

import argparse
import glob
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def find_mask(masks_dir: Path, stem: str):
    import re
    # images are 'fgbg000000', masks are 'mask000000' (same number)
    candidates = [stem, stem.replace("fgbg", "mask")]
    m = re.search(r"(\d+)", stem)
    if m:
        candidates.append(f"mask{m.group(1)}")
    for cand in candidates:
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            p = masks_dir / f"{cand}{ext}"
            if p.exists():
                return p
    return None


def mask_to_boxes(mask_path: Path, min_area_frac: float = 0.001):
    """Return list of normalised (cx, cy, w, h) boxes from a foreground mask."""
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return []
    h, w = m.shape[:2]
    # background is a clean 0; the animal is the bright foreground. Threshold low
    # (20) to capture the FULL silhouette while ignoring minor JPG noise (~1-14).
    _, binm = cv2.threshold(m, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_frac * w * h
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh < min_area:
            continue
        boxes.append(((x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with images/ and masks/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=20000, help="0 = use all images")
    ap.add_argument("--val_frac", type=float, default=0.2)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    images_dir, masks_dir = src / "images", src / "masks"

    imgs = sorted(glob.glob(str(images_dir / "*")))
    print("total images found:", len(imgs))
    if args.limit:
        random.seed(0)
        random.shuffle(imgs)
        imgs = imgs[: args.limit]
        print("processing subset:", len(imgs))

    # quick diagnostic: are masks binary or multi-value (encoding types)?
    for s in imgs[:3]:
        mp = find_mask(masks_dir, Path(s).stem)
        if mp is not None:
            vals = np.unique(cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE))
            print(f"  sample mask {mp.name}: unique values = {vals[:10]}")

    for split in ["train", "valid"]:
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    random.seed(1)
    kept = nobox = nomask = 0
    for img in imgs:
        stem = Path(img).stem
        mp = find_mask(masks_dir, stem)
        if mp is None:
            nomask += 1
            continue
        boxes = mask_to_boxes(mp)
        if not boxes:
            nobox += 1
            continue
        split = "valid" if random.random() < args.val_frac else "train"
        shutil.copy(img, out / split / "images" / Path(img).name)
        (out / split / "labels" / f"{stem}.txt").write_text(
            "\n".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in boxes)
        )
        kept += 1

    import yaml
    yaml.safe_dump(
        {"path": str(out.resolve()), "train": "train/images", "val": "valid/images",
         "names": ["animal"], "nc": 1},
        open(out / "data.yaml", "w"),
    )
    print(f"DONE: kept {kept} | no-mask {nomask} | empty-mask {nobox}")
    print("data.yaml ->", out / "data.yaml")


if __name__ == "__main__":
    main()
