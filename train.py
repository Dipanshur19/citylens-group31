"""
train.py — one command to train any category on the H200.

Examples
--------
    python train.py --category fire_smoke
    python train.py --category streetlights --epochs 150 --batch 96
    python train.py --category accidents
    python train.py --all                      # train every category in turn

Outputs
-------
    runs/<category>/train/...        full training run (plots, results.csv)
    weights/<category>_best.pt       the trained model (kept safe here)
    runs/<category>/<category>_predictions.csv   bbox coords (detection only)
    runs/<category>/pred/            annotated frames with boxes drawn
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from configs import CONFIGS, OUT_ROOT, WEIGHTS_DIR, get_config
from prepare_data import prepare


def _save_weights(run_dir: Path, category: str) -> Path:
    src = run_dir / "weights" / "best.pt"
    Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
    dst = Path(WEIGHTS_DIR) / f"{category}_best.pt"
    if src.exists():
        shutil.copy(src, dst)
        print(f"[weights] saved -> {dst}")
    return dst


def _export_predictions(weights: Path, source: str, out_dir: Path, names):
    """Run inference, save annotated frames + a bbox CSV (required deliverable)."""
    import cv2
    from ultralytics import YOLO

    det = YOLO(str(weights))
    (out_dir / "pred").mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{out_dir.name}_predictions.csv"
    rows = []
    for r in det.predict(source=source, conf=0.25, stream=True, verbose=False):
        frame = Path(r.path).name
        cv2.imwrite(str(out_dir / "pred" / frame), r.plot())
        h, w = r.orig_shape
        for b in r.boxes:
            x, y, bw, bh = b.xywh[0].tolist()
            rows.append({
                "frame": frame, "class": det.names[int(b.cls)],
                "confidence": round(float(b.conf), 4),
                "x": round(x - bw / 2, 1), "y": round(y - bh / 2, 1),
                "width": round(bw, 1), "height": round(bh, 1),
                "x_norm": round((x - bw / 2) / w, 4), "y_norm": round((y - bh / 2) / h, 4),
                "w_norm": round(bw / w, 4), "h_norm": round(bh / h, 4),
            })
    with open(csv_path, "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["frame"])
        wri.writeheader(); wri.writerows(rows)
    print(f"[predict] {len(rows)} detections -> {csv_path}")


def train_category(category: str, data_override: str = None, **overrides):
    from ultralytics import YOLO

    cfg = get_config(category, **overrides)
    print(f"\n=== TRAINING: {category} ({cfg['task']}) on {cfg['model']} ===")
    data = data_override or prepare(category)
    if data_override:
        print(f"[data] using local dataset override -> {data}")

    model = YOLO(cfg["model"])
    common = dict(
        data=str(data), epochs=cfg["epochs"], imgsz=cfg["imgsz"], batch=cfg["batch"],
        patience=cfg.get("patience", 25), cache=cfg.get("cache", False),
        device=0, workers=cfg.get("workers", 16), amp=True, cos_lr=True,
        project=f"{OUT_ROOT}/{category}", name="train", exist_ok=True, plots=True,
    )
    if cfg["task"] == "detect":
        common.update(close_mosaic=15, mixup=0.1, copy_paste=0.1)
        if "fraction" in cfg:
            common["fraction"] = cfg["fraction"]
    else:  # classify: add regularization + augmentation to fight overfitting
        common.update(dropout=0.2, erasing=0.5, degrees=10.0, fliplr=0.5)

    results = model.train(**common)
    run_dir = Path(results.save_dir)

    # validate + print metrics
    metrics = model.val(data=str(data))
    if cfg["task"] == "detect":
        print(f"\n[{category}] mAP@50={float(metrics.box.map50):.4f} "
              f"mAP@50-95={float(metrics.box.map):.4f} "
              f"P={float(metrics.box.mp):.4f} R={float(metrics.box.mr):.4f}")
    else:
        print(f"\n[{category}] top1 accuracy = {float(metrics.top1):.4f}")

    weights = _save_weights(run_dir, category)

    # detection: produce annotated frames + bbox CSV from the val/test split
    if cfg["task"] == "detect":
        import yaml
        ycfg = yaml.safe_load(open(data))
        base = Path(ycfg["path"])
        src = base / ycfg.get("test", ycfg.get("val", "valid/images"))
        _export_predictions(weights, str(src), Path(OUT_ROOT) / category, ycfg["names"])

    print(f"=== DONE: {category} ===\n")
    return weights


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", choices=list(CONFIGS))
    ap.add_argument("--all", action="store_true", help="train every category")
    ap.add_argument("--model"); ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int); ap.add_argument("--imgsz", type=int)
    ap.add_argument("--data", help="path to a local data.yaml (skips auto-download)")
    args = ap.parse_args()

    overrides = dict(model=args.model, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)
    if args.all:
        for cat in CONFIGS:
            train_category(cat, **overrides)
    elif args.category:
        train_category(args.category, data_override=args.data, **overrides)
    else:
        ap.error("pass --category <name> or --all")
