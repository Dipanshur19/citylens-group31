# =============================================================================
# CATEGORY 4 (10%) — DARK SPOTS / ACCIDENT DETECTION  | CityLens Group 31
# =============================================================================
# Accidents come as VIDEOS, so this is a video task. Strategy:
#   1. Extract frames from the accident videos.
#   2. Train a YOLO11 CLASSIFIER (accident vs no_accident) on those frames
#      -> fast to train and easily hits high accuracy.
#   3. Run it over test videos; when a clip is flagged "accident", log an event
#      tagged with its location (video/camera id).
#   4. Rank locations by accident frequency = the DARK SPOTS.
# Add the Kaggle datasets via "Add Input": picekl/accident, siddhi17/road-crossing.
# GPU ON.
# =============================================================================

# %% CELL 1 — install + imports -----------------------------------------------
# !pip install -q ultralytics opencv-python
import cv2
from pathlib import Path
from ultralytics import YOLO

WORK = Path("/kaggle/working")
FRAMES = WORK / "accident_frames"           # classifier dataset root
for sub in ["train/accident", "train/no_accident", "val/accident", "val/no_accident"]:
    (FRAMES / sub).mkdir(parents=True, exist_ok=True)

# Adjust to the actual dataset layout after you add it as input:
ACCIDENT_INPUT = Path("/kaggle/input/accident")   # picekl/accident

# %% CELL 2 — extract frames from videos --------------------------------------
# Many versions of this dataset already separate Accident / Non Accident folders
# (sometimes as images, sometimes as videos). This helper handles BOTH:
#  - if it finds videos, it samples frames;
#  - if it finds images, it copies them.
import shutil

def label_of(path: Path) -> str | None:
    p = str(path).lower()
    if "non" in p or "no_accident" in p or "normal" in p:
        return "no_accident"
    if "accident" in p or "crash" in p:
        return "accident"
    return None

def sample_video(video: Path, out_dir: Path, every_n: int = 10, max_frames: int = 30):
    cap = cv2.VideoCapture(str(video)); i = saved = 0
    while saved < max_frames:
        ok, frame = cap.read()
        if not ok: break
        if i % every_n == 0:
            cv2.imwrite(str(out_dir / f"{video.stem}_{i:04d}.jpg"), frame); saved += 1
        i += 1
    cap.release()

count = 0
for f in ACCIDENT_INPUT.rglob("*"):
    if f.is_dir(): continue
    lab = label_of(f)
    if lab is None: continue
    split = "train" if (count % 5 != 0) else "val"   # 80/20 split
    dst = FRAMES / split / lab
    if f.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        sample_video(f, dst)
    elif f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        shutil.copy(f, dst / f.name)
    count += 1
print("Frames prepared under", FRAMES)
# Sanity check counts:
for sub in ["train/accident", "train/no_accident", "val/accident", "val/no_accident"]:
    print(sub, len(list((FRAMES / sub).glob("*"))))

# %% CELL 3 — train a fast image classifier -----------------------------------
clf = YOLO("yolo11s-cls.pt")
clf.train(data=str(FRAMES), epochs=30, imgsz=224, batch=64, patience=8,
          cos_lr=True, amp=True, project=str(WORK / "runs"),
          name="accident_cls", exist_ok=True)

# %% CELL 4 — validate ---------------------------------------------------------
metrics = clf.val(data=str(FRAMES))
print("top1 accuracy:", round(float(metrics.top1), 4))   # aim > 0.90

# %% CELL 5 — detect accidents on test videos + build DARK SPOTS ---------------
import pandas as pd
from collections import Counter
best = WORK / "runs" / "accident_cls" / "weights" / "best.pt"
det = YOLO(str(best))

# Map each test video to a location (camera id / junction). Edit this dict.
TEST_VIDEOS = {
    "/kaggle/input/your-test/clip1.mp4": "MG_Road_Junction",
    "/kaggle/input/your-test/clip2.mp4": "Ring_Road_KM12",
}
ACCIDENT_CONF = 0.6   # frame is "accident" if predicted prob exceeds this

events = []
for vid, location in TEST_VIDEOS.items():
    cap = cv2.VideoCapture(vid); i = flagged = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if i % 10 == 0:
            r = det.predict(frame, imgsz=224, verbose=False)[0]
            top = int(r.probs.top1); prob = float(r.probs.top1conf)
            if det.names[top] == "accident" and prob >= ACCIDENT_CONF:
                flagged += 1
        i += 1
    cap.release()
    # one accident EVENT per clip that has enough accident frames
    if flagged >= 2:
        events.append({"location": location, "video": vid, "accident_frames": flagged})

HIGH_MIN, MED_MIN = 5, 2
counts = Counter(e["location"] for e in events)
darkspots = [{"location": loc, "accident_count": n,
              "risk_level": "High" if n >= HIGH_MIN else ("Medium" if n >= MED_MIN else "Low")}
             for loc, n in counts.most_common()]
pd.DataFrame(events).to_csv(WORK / "accident_events.csv", index=False)
pd.DataFrame(darkspots).to_csv(WORK / "dark_spots.csv", index=False)
print("DARK SPOTS (by frequency):")
print(pd.DataFrame(darkspots))
