# =============================================================================
# CATEGORY 1 (40%) — FIRE & SMOKE DETECTION  | CityLens Group 31
# =============================================================================
# Paste each "# %% CELL" block into its own Kaggle notebook cell.
# Settings -> Accelerator: GPU T4 x2 (or P100). Internet: ON.
# =============================================================================

# %% CELL 1 — install + imports -----------------------------------------------
# !pip install -q ultralytics roboflow
import os, shutil, yaml
from pathlib import Path
from ultralytics import YOLO

WORK = Path("/kaggle/working")
DATA = WORK / "fire_data"
DATA.mkdir(parents=True, exist_ok=True)

# %% CELL 2 — download datasets from Roboflow ---------------------------------
# Get a free API key: roboflow.com -> Settings -> Roboflow API.
from roboflow import Roboflow
rf = Roboflow(api_key="PASTE_YOUR_ROBOFLOW_API_KEY")

def rf_download(workspace, project, location, fmt="yolov8", prefer=1):
    """Download a Roboflow dataset, auto-finding a valid version number so the
    notebook doesn't break if version 1 was renumbered."""
    proj = rf.workspace(workspace).project(project)
    order = [prefer] + [v for v in range(1, 12) if v != prefer]
    tried = []
    for v in order:
        try:
            return proj.version(v).download(fmt, location=location)
        except Exception:
            tried.append(v)
    raise RuntimeError(f"Could not download {workspace}/{project}; tried {tried}")

# ECO Group fire & smoke (~10k images) — the main dataset
ds_eco = rf_download("eco-group", "fire-smoke-yvnrc", str(DATA / "eco"))
# Wildfire smoke (737 images) — adds outdoor/wildfire variety
ds_wild = rf_download("public", "wildfire-smoke", str(DATA / "wildfire"))

# %% CELL 3 — merge datasets into ONE with unified classes {0:fire, 1:smoke} ---
# Different datasets number their classes differently, so we remap every label
# file to a single shared class order before training.
UNIFIED = ["fire", "smoke"]
NAME_MAP = {           # map raw dataset class-names -> unified names
    "fire": "fire", "flame": "fire", "flames": "fire", "burning": "fire",
    "smoke": "smoke", "smoking": "smoke",
}
MERGED = DATA / "merged"

def remap_dataset(src_root: Path):
    """Copy images/labels from a Roboflow dataset into MERGED, remapping class ids."""
    with open(src_root / "data.yaml") as f:
        cfg = yaml.safe_load(f)
    src_names = cfg["names"]
    for split in ["train", "valid", "test"]:
        img_dir = src_root / split / "images"
        lbl_dir = src_root / split / "labels"
        if not img_dir.exists():
            continue
        out_img = MERGED / split / "images"; out_img.mkdir(parents=True, exist_ok=True)
        out_lbl = MERGED / split / "labels"; out_lbl.mkdir(parents=True, exist_ok=True)
        for img in img_dir.iterdir():
            shutil.copy(img, out_img / f"{src_root.name}_{img.name}")
            lbl = lbl_dir / f"{img.stem}.txt"
            new_lines = []
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    raw = src_names[int(parts[0])].lower()
                    mapped = NAME_MAP.get(raw)
                    if mapped is None:
                        continue  # skip classes we don't use
                    parts[0] = str(UNIFIED.index(mapped))
                    new_lines.append(" ".join(parts))
            (out_lbl / f"{src_root.name}_{img.stem}.txt").write_text("\n".join(new_lines))

remap_dataset(DATA / "eco")
remap_dataset(DATA / "wildfire")

# write the unified data.yaml
data_yaml = MERGED / "data.yaml"
with open(data_yaml, "w") as f:
    yaml.safe_dump({
        "path": str(MERGED),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {i: n for i, n in enumerate(UNIFIED)},
    }, f)
print("Merged dataset ready at", MERGED)

# %% CELL 4 — train ------------------------------------------------------------
# yolo11s = fast + accurate. Use yolo11m.pt if you have time and want more mAP.
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
# If mAP@50 < 0.90: train more epochs, switch to yolo11m, or add the domestic
# fire dataset (github.com/datacluster-labs/Domestic-Fire-and-Smoke-Dataset).

# %% CELL 6 — predict on test images + draw boxes + export CSV -----------------
best = WORK / "runs" / "fire_smoke" / "weights" / "best.pt"
det = YOLO(str(best))
pred = det.predict(source=str(MERGED / "test" / "images"),
                   conf=0.25, save=True, stream=True)

# %% CELL 7 — BONUS analytics: severity / vulnerability / burning area ---------
# Detect people+buildings with a COCO model, fire/smoke with our model, then
# combine for the analytics report.
import pandas as pd
from dataclasses import asdict
import sys; sys.path.append("/kaggle/working/citylens-group31")  # if repo added
# Inline copy of analytics so the notebook is self-contained:
SEV_MILD, SEV_MOD = 0.05, 0.20
VUL_MILD, VUL_MOD = 2, 8
MPP = 0.02  # metres per pixel — CALIBRATE per camera if known

def level(v, a, b): return "Mild" if v <= a else ("Moderate" if v <= b else "Severe")

coco = YOLO("yolo11s.pt")  # for person/building context
rows = []
for r in det.predict(source=str(MERGED / "test" / "images"), conf=0.25, stream=True):
    h, w = r.orig_shape
    fire_area = sum(float(b.xywh[0][2]) * float(b.xywh[0][3]) for b in r.boxes)
    # context objects
    cr = coco.predict(source=r.path, conf=0.3, verbose=False)[0]
    exposed = sum(1 for b in cr.boxes if coco.names[int(b.cls)] in ("person",))
    coverage = fire_area / (w * h)
    rows.append({
        "frame": Path(r.path).name,
        "severity": level(coverage, SEV_MILD, SEV_MOD),
        "vulnerability": level(exposed, VUL_MILD, VUL_MOD),
        "burning_area_sqm": round(fire_area * MPP**2, 2),
        "fire_coverage_pct": round(coverage * 100, 2),
        "people_exposed": exposed,
    })
df = pd.DataFrame(rows)
df.to_csv(WORK / "fire_smoke_analytics.csv", index=False)
print(df.head(20))
print("Saved analytics ->", WORK / "fire_smoke_analytics.csv")
