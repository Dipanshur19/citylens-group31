# =============================================================================
# CATEGORY 5 (10%) — DEAD / STRAY ANIMALS ON ROAD  | CityLens Group 31
# =============================================================================
# Detects animals + bonus analytics: type, Dead/Alive, count & dwell time.
# Dataset: MoDES (bsridevi/modes-dataset-of-stray-animals) -> Add as Kaggle input.
# GPU ON.
# =============================================================================

# %% CELL 1 — install + imports -----------------------------------------------
# !pip install -q ultralytics
import yaml
from pathlib import Path
from ultralytics import YOLO

WORK = Path("/kaggle/working")
MODES = Path("/kaggle/input/modes-dataset-of-stray-animals")  # adjust if different

# %% CELL 2 — locate / build data.yaml ----------------------------------------
# MoDES ships YOLO-format labels. If a data.yaml already exists, use it; otherwise
# build one. Inspect the folder first:
print(list(MODES.glob("*"))[:20])

# If the dataset already has a yaml:
existing = list(MODES.rglob("data.yaml"))
if existing:
    data_yaml = existing[0]
else:
    # Build one — EDIT 'names' to match the dataset's animal classes
    data_yaml = WORK / "animals.yaml"
    yaml.safe_dump({
        "path": str(MODES),
        "train": "train/images",
        "val": "valid/images",
        "names": {0: "cow", 1: "dog", 2: "goat", 3: "buffalo", 4: "cat"},
    }, open(data_yaml, "w"))
print("Using data.yaml:", data_yaml)

# %% CELL 3 — train ------------------------------------------------------------
model = YOLO("yolo11s.pt")
model.train(data=str(data_yaml), epochs=50, imgsz=640, batch=16, patience=12,
            cos_lr=True, close_mosaic=10, amp=True, cache=True,
            project=str(WORK / "runs"), name="stray_animals", exist_ok=True)

# %% CELL 4 — validate ---------------------------------------------------------
m = model.val(data=str(data_yaml))
print("mAP@50:", round(float(m.box.map50), 4), "| mAP@50-95:", round(float(m.box.map), 4))

# %% CELL 5 — BONUS: count + dwell time + Dead/Alive on a video ----------------
# Uses YOLO tracking (ByteTrack) so each animal keeps a stable ID across frames.
import pandas as pd
best = WORK / "runs" / "stray_animals" / "weights" / "best.pt"
det = YOLO(str(best))

VIDEO = "/kaggle/input/your-road-video/clip.mp4"   # <- set this
FPS = 25.0
DEAD_MIN_DWELL_SEC, DEAD_MAX_MOVE_PX = 4.0, 25.0

tracks = {}   # id -> dict(type, first, last, pos, move)
fidx = 0
for r in det.track(source=VIDEO, persist=True, conf=0.3,
                   tracker="bytetrack.yaml", stream=True, verbose=False):
    for b in r.boxes:
        if b.id is None:
            continue
        tid = int(b.id)
        x, y, w, h = b.xywh[0].tolist(); cx, cy = x, y
        name = det.names[int(b.cls)]
        if tid not in tracks:
            tracks[tid] = {"type": name, "first": fidx, "last": fidx,
                           "pos": (cx, cy), "move": 0.0}
        else:
            t = tracks[tid]
            t["move"] += ((cx - t["pos"][0])**2 + (cy - t["pos"][1])**2) ** 0.5
            t["pos"] = (cx, cy); t["last"] = fidx
    fidx += 1

rows = []
for tid, t in tracks.items():
    dwell = (t["last"] - t["first"] + 1) / FPS
    dead = dwell >= DEAD_MIN_DWELL_SEC and t["move"] <= DEAD_MAX_MOVE_PX
    rows.append({"track_id": tid, "animal_type": t["type"],
                 "dwell_time_sec": round(dwell, 2),
                 "status": "Dead" if dead else "Alive",
                 "total_movement_px": round(t["move"], 1)})
df = pd.DataFrame(rows)
df.to_csv(WORK / "animal_analytics.csv", index=False)
print("Total distinct animals:", len(df))
print(df)
print("Saved ->", WORK / "animal_analytics.csv")
