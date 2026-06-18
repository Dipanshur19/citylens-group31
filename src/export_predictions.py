"""
export_predictions.py — REQUIRED DELIVERABLE

Runs a trained YOLO model over a folder of images (or a video) and writes:
  1. Annotated output frames with bounding boxes drawn  (deliverable #4)
  2. A CSV listing, per detection: frame, class, confidence,
     x, y, width, height (pixels) and normalised coords      (deliverable #5)

Usage (Kaggle cell or CLI):
    from src.export_predictions import export
    export("outputs/fire_smoke/weights/best.pt",
           source="/kaggle/input/your-test-images",
           category="fire_smoke")
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2

from .common import get_output_dir, load_yolo


def export(weights: str, source: str, category: str, conf: float = 0.25):
    model = load_yolo(weights)
    out = get_output_dir(category)
    csv_path = out / "csv" / f"{category}_predictions.csv"

    rows = []
    # stream=True keeps memory low for large folders / videos
    results = model.predict(source=source, conf=conf, stream=True, verbose=False)

    for i, r in enumerate(results):
        frame_name = Path(r.path).name if r.path else f"frame_{i:06d}.jpg"

        # Save annotated frame with boxes drawn
        annotated = r.plot()  # numpy BGR image with boxes
        cv2.imwrite(str(out / "pred_frames" / frame_name), annotated)

        h, w = r.orig_shape
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bw, bh = x2 - x1, y2 - y1
            rows.append(
                {
                    "frame": frame_name,
                    "class": cls_name,
                    "confidence": round(confidence, 4),
                    "x": round(x1, 1),
                    "y": round(y1, 1),
                    "width": round(bw, 1),
                    "height": round(bh, 1),
                    "x_norm": round(x1 / w, 4),
                    "y_norm": round(y1 / h, 4),
                    "w_norm": round(bw / w, 4),
                    "h_norm": round(bh / h, 4),
                }
            )

    fields = [
        "frame", "class", "confidence",
        "x", "y", "width", "height",
        "x_norm", "y_norm", "w_norm", "h_norm",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[{category}] {len(rows)} detections written to {csv_path}")
    print(f"[{category}] annotated frames in {out / 'pred_frames'}")
    return csv_path
