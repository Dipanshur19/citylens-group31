"""
common.py — shared helpers used by every category script.

Keep this small and dependency-light so it can be pasted into a Kaggle cell
or imported when the repo is added as a Kaggle dataset/utility script.
"""
from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def get_output_dir(category: str) -> Path:
    """Return (and create) the output folder for a given category."""
    base = Path(os.environ.get("CITYLENS_OUT", "outputs")) / category
    (base / "weights").mkdir(parents=True, exist_ok=True)
    (base / "pred_frames").mkdir(parents=True, exist_ok=True)
    (base / "csv").mkdir(parents=True, exist_ok=True)
    return base


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
def load_yolo(weights: str = "yolo11s.pt"):
    """
    Load an Ultralytics YOLO model. Falls back to YOLOv8 if YOLO11 weights
    are unavailable in the environment.
    """
    from ultralytics import YOLO

    try:
        return YOLO(weights)
    except Exception:
        return YOLO("yolov8s.pt")


# ---------------------------------------------------------------------------
# Training wrapper (consistent settings across all categories)
# ---------------------------------------------------------------------------
def train_yolo(
    data_yaml: str,
    category: str,
    base_weights: str = "yolo11s.pt",
    epochs: int = 60,
    imgsz: int = 640,
    batch: int = 16,
    patience: int = 15,
):
    """
    Fine-tune a YOLO model and return the trained model + results.

    Tuned for FAST training on a single Kaggle GPU (T4/P100) while still
    aiming for high mAP. Early stopping (patience) avoids wasted epochs.
    """
    model = load_yolo(base_weights)
    out = get_output_dir(category)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        project=str(out),
        name="train",
        exist_ok=True,
        # accuracy-boosting / speed-friendly settings
        cos_lr=True,        # cosine LR schedule -> smoother convergence
        close_mosaic=10,    # disable mosaic for last 10 epochs -> cleaner boxes
        amp=True,           # mixed precision -> faster on GPU
        cache=True,         # cache images in RAM -> faster epochs
        plots=True,
    )
    return model, results


def validate_and_report(model, data_yaml: str) -> dict:
    """Run validation and print the key metrics judges want to see."""
    metrics = model.val(data=data_yaml)
    report = {
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50-95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
    }
    print("\n===== VALIDATION REPORT =====")
    for k, v in report.items():
        print(f"{k:>10}: {v}")
    print("=============================\n")
    return report
