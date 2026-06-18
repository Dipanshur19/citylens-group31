# =============================================================================
# CATEGORY 3 (20%) — DAMAGED STREET LIGHTS  | CityLens Group 31
# =============================================================================
# Detects street lights; bonus analytics = OFF-state + flickering detection.
# Paste each "# %% CELL" into its own Kaggle cell. GPU ON, Internet ON.
# =============================================================================

# %% CELL 1 — install + imports -----------------------------------------------
# !pip install -q ultralytics roboflow
import shutil, yaml
from pathlib import Path
from ultralytics import YOLO

WORK = Path("/kaggle/working")
DATA = WORK / "light_data"
MERGED = DATA / "merged"
DATA.mkdir(parents=True, exist_ok=True)

# %% CELL 2 — download datasets ------------------------------------------------
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

ds_dmg = rf_download("godspeed-yqpeo", "damaged-lights", str(DATA / "damaged"))
ds_sod = rf_download("streetlight-detection", "sodioum-only-jkq3f", str(DATA / "sodium"))
# Street-Light-Dataset2 (github.com/Team16Project/Street-Light-Dataset) can be
# added with the kaggle/github download + the same remap helper if needed.

# %% CELL 3 — merge into unified classes --------------------------------------
# Goal classes: streetlight (and, if labels exist, damaged / off variants).
UNIFIED = ["streetlight", "damaged_light"]
NAME_MAP = {
    "light": "streetlight", "streetlight": "streetlight", "street-light": "streetlight",
    "sodium": "streetlight", "lamp": "streetlight", "on": "streetlight",
    "damaged": "damaged_light", "damaged-light": "damaged_light",
    "broken": "damaged_light", "off": "damaged_light", "fault": "damaged_light",
}

def remap(src_root: Path):
    cfg = yaml.safe_load(open(Path(src_root) / "data.yaml"))
    names = cfg["names"]
    for split in ["train", "valid", "test"]:
        idir = Path(src_root) / split / "images"
        ldir = Path(src_root) / split / "labels"
        if not idir.exists():
            continue
        oi = MERGED / split / "images"; oi.mkdir(parents=True, exist_ok=True)
        ol = MERGED / split / "labels"; ol.mkdir(parents=True, exist_ok=True)
        for img in idir.iterdir():
            shutil.copy(img, oi / f"{Path(src_root).name}_{img.name}")
            lbl = ldir / f"{img.stem}.txt"; lines = []
            if lbl.exists():
                for ln in lbl.read_text().splitlines():
                    p = ln.split()
                    if not p:
                        continue
                    raw = names[int(p[0])].lower()
                    mapped = NAME_MAP.get(raw)
                    if mapped is None:
                        continue
                    p[0] = str(UNIFIED.index(mapped)); lines.append(" ".join(p))
            (ol / f"{Path(src_root).name}_{img.stem}.txt").write_text("\n".join(lines))

remap(ds_dmg.location); remap(ds_sod.location)
data_yaml = MERGED / "data.yaml"
yaml.safe_dump({"path": str(MERGED), "train": "train/images", "val": "valid/images",
                "test": "test/images", "names": {i: n for i, n in enumerate(UNIFIED)}},
               open(data_yaml, "w"))
print("Merged ->", MERGED)

# %% CELL 4 — train ------------------------------------------------------------
model = YOLO("yolo11s.pt")
model.train(data=str(data_yaml), epochs=60, imgsz=640, batch=16, patience=15,
            cos_lr=True, close_mosaic=10, amp=True, cache=True,
            project=str(WORK / "runs"), name="streetlights", exist_ok=True)

# %% CELL 5 — validate ---------------------------------------------------------
m = model.val(data=str(data_yaml))
print("mAP@50:", round(float(m.box.map50), 4), "| mAP@50-95:", round(float(m.box.map), 4))

# %% CELL 6 — BONUS: OFF-state + flickering on a video -------------------------
# Point VIDEO at any street CCTV clip. We detect lights per frame, decide ON/OFF
# by lamp brightness, and flag flickering via ON<->OFF transitions over time.
import cv2, numpy as np, pandas as pd
from collections import deque

BRIGHT_ON = 130; FLICK_MIN = 3; WIN = 12; MATCH = 40
best = WORK / "runs" / "streetlights" / "weights" / "best.pt"
det = YOLO(str(best))
VIDEO = "/kaggle/input/your-street-video/clip.mp4"   # <- set this

def is_on(frame, xyxy):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    crop = frame[max(y1,0):y2, max(x1,0):x2]
    if crop.size == 0: return False
    g = crop.mean(axis=2); top = np.sort(g.ravel())[int(g.size*0.9):]
    return float(top.mean()) >= BRIGHT_ON

tracks = {}; next_id = 0; rows = []
cap = cv2.VideoCapture(VIDEO); fidx = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    res = det.predict(frame, conf=0.25, verbose=False)[0]
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist(); cx, cy = (x1+x2)/2, (y1+y2)/2
        on = is_on(frame, (x1, y1, x2, y2))
        tid = None
        for k, t in tracks.items():
            if (t["pos"][0]-cx)**2 + (t["pos"][1]-cy)**2 <= MATCH**2: tid = k; break
        if tid is None:
            tid = next_id; next_id += 1; tracks[tid] = {"pos": (cx, cy), "h": deque(maxlen=WIN)}
        tracks[tid]["pos"] = (cx, cy); tracks[tid]["h"].append(1 if on else 0)
        h = list(tracks[tid]["h"]); trans = sum(1 for a, b2 in zip(h, h[1:]) if a != b2)
        rows.append({"frame": fidx, "light_id": tid, "state": "ON" if on else "OFF",
                     "flickering": trans >= FLICK_MIN, "transitions": trans})
    fidx += 1
cap.release()
df = pd.DataFrame(rows)
df.to_csv(WORK / "streetlight_states.csv", index=False)
print(df.groupby("light_id").agg(off=("state", lambda s: (s=="OFF").sum()),
                                 flicker=("flickering", "max")))
print("Saved ->", WORK / "streetlight_states.csv")
