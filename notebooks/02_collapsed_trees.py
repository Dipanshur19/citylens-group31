# =============================================================================
# CATEGORY 2 (20%) — COLLAPSED TREES / STRUCTURES  | CityLens Group 31
# =============================================================================
# Detects fallen trees and road obstructions (poles, barricades, debris).
# Bonus analytic: classification into obstacle varieties.
# Paste each "# %% CELL" into its own Kaggle cell. GPU ON, Internet ON.
# =============================================================================

# %% CELL 1 — install + imports -----------------------------------------------
# !pip install -q ultralytics roboflow
from pathlib import Path
from ultralytics import YOLO

WORK = Path("/kaggle/working")
DATA = WORK / "trees_data"
DATA.mkdir(parents=True, exist_ok=True)

# %% CELL 2 — download dataset -------------------------------------------------
from roboflow import Roboflow
rf = Roboflow(api_key="PASTE_YOUR_ROBOFLOW_API_KEY")

def rf_download(workspace, project, location, fmt="yolov8", prefer=1):
    """Download a Roboflow dataset, auto-finding a valid version number."""
    proj = rf.workspace(workspace).project(project)
    order = [prefer] + [v for v in range(1, 12) if v != prefer]
    tried = []
    for v in order:
        try:
            return proj.version(v).download(fmt, location=location)
        except Exception:
            tried.append(v)
    raise RuntimeError(f"Could not download {workspace}/{project}; tried {tried}")

# Fallen trees (with palms) — 8.7k images
ds = rf_download("overflow-thaap", "fallen-trees-with-palms", str(DATA / "fallen_trees"))
data_yaml = Path(ds.location) / "data.yaml"
print("data.yaml:", data_yaml)

# %% CELL 3 — (optional) inspect classes --------------------------------------
import yaml
print(yaml.safe_load(open(data_yaml))["names"])
# The dataset's own classes already cover obstacle varieties. If you want to add
# generic road obstacles (pole/barricade), source an extra Roboflow set and merge
# using the remap pattern from notebooks/01_fire_smoke.py (CELL 3).

# %% CELL 4 — train ------------------------------------------------------------
model = YOLO("yolo11s.pt")
results = model.train(
    data=str(data_yaml),
    epochs=60, imgsz=640, batch=16, patience=15,
    cos_lr=True, close_mosaic=10, amp=True, cache=True,
    project=str(WORK / "runs"), name="collapsed_trees", exist_ok=True,
)

# %% CELL 5 — validate ---------------------------------------------------------
m = model.val(data=str(data_yaml))
print("mAP@50:", round(float(m.box.map50), 4), "| mAP@50-95:", round(float(m.box.map), 4),
      "| P:", round(float(m.box.mp), 4), "| R:", round(float(m.box.mr), 4))

# %% CELL 6 — predict + draw boxes + export obstacle-type CSV ------------------
import pandas as pd
best = WORK / "runs" / "collapsed_trees" / "weights" / "best.pt"
det = YOLO(str(best))
rows = []
for r in det.predict(source=str(Path(ds.location) / "test" / "images"),
                     conf=0.25, save=True, stream=True):
    h, w = r.orig_shape
    for b in r.boxes:
        x, y, bw, bh = b.xywh[0].tolist()
        rows.append({
            "frame": Path(r.path).name,
            "obstacle_type": det.names[int(b.cls)],   # <- bonus classification
            "confidence": round(float(b.conf), 4),
            "x": round(x - bw/2, 1), "y": round(y - bh/2, 1),
            "width": round(bw, 1), "height": round(bh, 1),
        })
pd.DataFrame(rows).to_csv(WORK / "collapsed_trees_predictions.csv", index=False)
print("Saved ->", WORK / "collapsed_trees_predictions.csv", "| detections:", len(rows))
