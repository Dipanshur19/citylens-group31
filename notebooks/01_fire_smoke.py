# =============================================================================
# CATEGORY 1 (40%) — FIRE & SMOKE DETECTION  | CityLens Group 31
# =============================================================================
# Trains YOLO11 DIRECTLY on the ECO fire/smoke dataset (no risky remapping).
# Paste each "# %% CELL" block into its own Kaggle notebook cell.
# Settings -> Accelerator: GPU T4. Internet: ON.
# =============================================================================

# %% CELL 1 — install + imports -----------------------------------------------
!pip install -q ultralytics roboflow
import os, glob, yaml
from pathlib import Path
from ultralytics import YOLO

WORK = Path("/kaggle/working")
DATA = WORK / "fire_data"
DATA.mkdir(parents=True, exist_ok=True)

# %% CELL 2 — download the ECO fire & smoke dataset (~10k images) --------------
# Free key: roboflow.com -> Settings -> Roboflow API.
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

ds_eco = rf_download("eco-group", "fire-smoke-yvnrc", str(DATA / "eco"))
ROOT = Path(ds_eco.location)
print("Downloaded to:", ROOT)

# %% CELL 3 — inspect labels + build a clean data.yaml (NO remapping) ----------
# This both verifies labels exist AND fixes the dataset paths so Ultralytics
# finds images + labels. If "train label files" > 0 and a sample shows numbers,
# training will work correctly.
cfg = yaml.safe_load(open(ROOT / "data.yaml"))
print("Classes in dataset:", cfg["names"])

lbls = glob.glob(str(ROOT / "train" / "labels" / "*.txt"))
non_empty = [p for p in lbls if os.path.getsize(p) > 0]
print(f"train label files: {len(lbls)}  | non-empty: {len(non_empty)}")
if non_empty:
    print("sample label (class x y w h):\n", open(non_empty[0]).read().strip()[:200])
else:
    print("WARNING: no non-empty labels found — check the dataset export format!")

# point train/val/test at the real folders with absolute base path
cfg["path"] = str(ROOT)
cfg["train"] = "train/images"
cfg["val"] = "valid/images" if (ROOT / "valid" / "images").exists() else "val/images"
if (ROOT / "test" / "images").exists():
    cfg["test"] = "test/images"
data_yaml = ROOT / "data_fixed.yaml"
yaml.safe_dump(cfg, open(data_yaml, "w"))
print("Using data.yaml ->", data_yaml)

# %% CELL 4 — train ------------------------------------------------------------
# yolo11s = fast + accurate. Switch to yolo11m.pt for more mAP if you have time.
model = YOLO("yolo11s.pt")
results = model.train(
    data=str(data_yaml),
    epochs=60, imgsz=640, batch=16, patience=15,
    cos_lr=True, close_mosaic=10, amp=True, cache=True,
    project=str(WORK / "runs"), name="fire_smoke", exist_ok=True,
)

# %% CELL 5 — validate + read metrics for README ------------------------------
metrics = model.val(data=str(data_yaml))
print("mAP@50    :", round(float(metrics.box.map50), 4))
print("mAP@50-95 :", round(float(metrics.box.map), 4))
print("precision :", round(float(metrics.box.mp), 4))
print("recall    :", round(float(metrics.box.mr), 4))
# If mAP@50 < 0.90: train more epochs or switch to yolo11m.pt.

# %% CELL 6 — predict on test images + draw boxes + export bbox CSV ------------
import pandas as pd
best = WORK / "runs" / "fire_smoke" / "weights" / "best.pt"
det = YOLO(str(best))
test_src = str(ROOT / (("test/images") if (ROOT / "test" / "images").exists() else "valid/images"))

rows = []
for r in det.predict(source=test_src, conf=0.25, save=True, stream=True):
    h, w = r.orig_shape
    for b in r.boxes:
        x, y, bw, bh = b.xywh[0].tolist()
        rows.append({
            "frame": Path(r.path).name,
            "class": det.names[int(b.cls)],
            "confidence": round(float(b.conf), 4),
            "x": round(x - bw / 2, 1), "y": round(y - bh / 2, 1),
            "width": round(bw, 1), "height": round(bh, 1),
        })
pd.DataFrame(rows).to_csv(WORK / "fire_smoke_predictions.csv", index=False)
print("Annotated frames in runs/fire_smoke/ ; bbox CSV saved. detections:", len(rows))

# %% CELL 7 — BONUS analytics: severity / vulnerability / burning area ---------
SEV_MILD, SEV_MOD = 0.05, 0.20      # fraction of frame covered by fire/smoke
VUL_MILD, VUL_MOD = 2, 8            # nearby people exposed
MPP = 0.02                          # metres per pixel — CALIBRATE if camera known

def level(v, a, b):
    return "Mild" if v <= a else ("Moderate" if v <= b else "Severe")

coco = YOLO("yolo11s.pt")  # COCO model for person/context detection
arows = []
for r in det.predict(source=test_src, conf=0.25, stream=True, verbose=False):
    h, w = r.orig_shape
    fire_area = sum(float(b.xywh[0][2]) * float(b.xywh[0][3]) for b in r.boxes)
    cr = coco.predict(source=r.path, conf=0.3, verbose=False)[0]
    exposed = sum(1 for b in cr.boxes if coco.names[int(b.cls)] == "person")
    coverage = fire_area / (w * h)
    arows.append({
        "frame": Path(r.path).name,
        "severity": level(coverage, SEV_MILD, SEV_MOD),
        "vulnerability": level(exposed, VUL_MILD, VUL_MOD),
        "burning_area_sqm": round(fire_area * MPP**2, 2),
        "fire_coverage_pct": round(coverage * 100, 2),
        "people_exposed": exposed,
    })
adf = pd.DataFrame(arows)
adf.to_csv(WORK / "fire_smoke_analytics.csv", index=False)
print(adf.head(20))
print("Saved analytics ->", WORK / "fire_smoke_analytics.csv")
