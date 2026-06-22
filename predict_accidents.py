"""
predict_accidents.py — produce the accident model's deliverables.

Accidents is a CLASSIFICATION model (no bounding boxes), so the equivalents of
the detection deliverables are:
  #4 output frames -> each frame saved with the predicted label overlaid
  #5 per-frame results -> CSV of frame, predicted_class, confidence
  #8 dark spots (bonus) -> accident frames aggregated by source clip/location

Outputs land where make_submission.py expects them:
  runs/accidents/pred/*.jpg
  runs/accidents/accidents_predictions.csv
  runs/accidents/dark_spots.csv

Usage:  python predict_accidents.py
"""
from __future__ import annotations

import csv
import glob
import re
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights" / "accidents_best.pt"
SRC = ROOT / "datasets" / "accidents" / "frames" / "val"
OUT = ROOT / "runs" / "accidents"
(OUT / "pred").mkdir(parents=True, exist_ok=True)

model = YOLO(str(WEIGHTS))
imgs = (glob.glob(str(SRC / "**" / "*.jpg"), recursive=True)
        + glob.glob(str(SRC / "**" / "*.png"), recursive=True))
print(f"classifying {len(imgs)} frames ...")

rows = []
for r in model.predict(source=imgs, imgsz=384, stream=True, verbose=False):
    name = Path(r.path).name
    top = int(r.probs.top1)
    cls = model.names[top]
    conf = float(r.probs.top1conf)

    img = cv2.imread(r.path)
    if img is not None:
        color = (0, 0, 255) if cls == "accident" else (0, 170, 0)
        cv2.putText(img, f"{cls} {conf:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.imwrite(str(OUT / "pred" / name), img)
    rows.append({"frame": name, "predicted_class": cls, "confidence": round(conf, 4)})

# #5 per-frame predictions CSV
with open(OUT / "accidents_predictions.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["frame", "predicted_class", "confidence"])
    w.writeheader()
    w.writerows(rows)

# #8 dark spots: group accident frames by source clip (filename prefix before digits).
# In deployment 'location' would be the camera/GPS id; here we use the clip name.
def clip_of(name: str) -> str:
    m = re.match(r"([A-Za-z_\-]+)", name)
    return m.group(1).strip("_-") if m else "clip"

acc = [clip_of(r["frame"]) for r in rows if r["predicted_class"] == "accident"]
counts = Counter(acc)
with open(OUT / "dark_spots.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["location_or_clip", "accident_frames", "risk_level"])
    for loc, n in counts.most_common():
        risk = "High" if n >= 10 else ("Medium" if n >= 3 else "Low")
        w.writerow([loc, n, risk])

n_acc = sum(1 for r in rows if r["predicted_class"] == "accident")
print(f"wrote {len(rows)} predictions ({n_acc} accident) -> {OUT/'accidents_predictions.csv'}")
print(f"annotated frames -> {OUT/'pred'}")
print(f"dark spots -> {OUT/'dark_spots.csv'}")
