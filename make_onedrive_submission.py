"""
make_onedrive_submission.py — arrange deliverables into the OneDrive folder
structure the organizers created (one folder per primary category):

    Fire&Smoke/  CollapsedTrees/  DamagedLights/  DarkSpots/  Dead&StrayAnimals/
    Photos&Videos/

Each category folder gets: the trained model, its prediction CSV, output frames
with boxes drawn, and a model card (architecture + metrics + dataset + references).
Photos&Videos/ holds the input images used. Upload each local subfolder's
contents into the matching OneDrive folder.

Run:  python make_onedrive_submission.py
Output: ./onedrive_submission/<category folders>
"""
from __future__ import annotations

import glob
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "onedrive_submission"
GITHUB = "https://github.com/Dipanshur19/citylens-group31"
MAX_OUTPUT_FRAMES = 40
MAX_INPUT_SAMPLES = 30

# category -> OneDrive folder name
FOLDER = {
    "fire_smoke": "Fire&Smoke",
    "collapsed_trees": "CollapsedTrees",
    "streetlights": "DamagedLights",
    "accidents": "DarkSpots",
    "stray_animals": "Dead&StrayAnimals",
}

# category -> (model, metric name, score, one-line description + analytics note)
INFO = {
    "fire_smoke": ("YOLO11x (single-class)", "mAP@50", "0.871",
                   "Detects fire & smoke. Bonus analytics: severity / vulnerability / "
                   "burning-area (src/analytics/fire_severity.py)."),
    "collapsed_trees": ("YOLO11x", "mAP@50", "0.817",
                        "Detects fallen trees / road obstructions (bounding boxes)."),
    "streetlights": ("YOLO11l", "mAP@50", "0.730",
                     "Detects street lights. OFF-state + flickering detection via the "
                     "brightness analytic (src/analytics/streetlight_state.py)."),
    "accidents": ("YOLO11l-cls", "top-1 accuracy", "0.889",
                  "Classifies accident vs no-accident frames. dark_spots.csv ranks "
                  "accident frequency by location (src/analytics/accident_darkspots.py)."),
    "stray_animals": ("YOLO11x", "mAP@50", "0.977",
                      "Detects animals on road. Count + dwell-time + dead/alive via "
                      "tracking (src/analytics/animal_dwell.py)."),
}

INPUT_DIRS = {
    "fire_smoke": ["datasets/fire_smoke"],
    "collapsed_trees": ["datasets/trees_full", "datasets/collapsed_trees"],
    "streetlights": ["datasets/streetlights/merged"],
    "accidents": ["datasets/accidents/frames/val"],
    "stray_animals": ["datasets/stray_animals/yolo"],
}

DATASET_LINK = {
    "fire_smoke": "https://universe.roboflow.com/eco-group/fire-smoke-yvnrc",
    "collapsed_trees": "https://universe.roboflow.com/overflow-thaap/fallen-trees-with-palms",
    "streetlights": "https://universe.roboflow.com/godspeed-yqpeo/damaged-lights ; "
                    "https://universe.roboflow.com/streetlight-detection/sodioum-only-jkq3f",
    "accidents": "https://www.kaggle.com/datasets/ckay16/accident-detection-from-cctv-footage",
    "stray_animals": "https://www.kaggle.com/datasets/bsridevi/modes-dataset-of-stray-animals",
}


def copy_n(srcs, dst: Path, n: int) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    c = 0
    for s in srcs:
        if c >= n:
            break
        try:
            shutil.copy(s, dst / Path(s).name); c += 1
        except Exception:
            pass
    return c


def model_card(cat: str) -> str:
    model, metric, score, desc = INFO[cat]
    return f"""# {FOLDER[cat]} — Model Card (Group 31, KPI Group 4)

{desc}

## Model
- **Architecture:** Ultralytics {model}, COCO-pretrained, fine-tuned on a single NVIDIA H200.
- **Performance:** {metric} = **{score}**
- **Weights file:** `{cat}_best.pt` (in this folder)

## Files in this folder
- `{cat}_best.pt` — trained model weights
- `{cat}_predictions.csv` — per-frame predictions (bounding-box coords + dimensions,
  or class + confidence for the accident classifier)
- `output_frames/` — output images with detections drawn / labels overlaid
- `analytics/` — bonus analytics CSVs (if generated)

## Input images
Sample training/testing frames are in the shared `Photos&Videos/{FOLDER[cat]}/` folder.

## Dataset
{DATASET_LINK[cat]}

## Code & full documentation
GitHub repository (all code, comments, README, references): {GITHUB}

## Tools / references
Ultralytics YOLO11, PyTorch, OpenCV, Roboflow/Kaggle datasets. An AI assistant
(Kiro) was used to help build and debug the pipeline. See REFERENCES.md in the repo.
"""


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    pv = OUT / "Photos&Videos"
    pv.mkdir()

    report = []
    for cat, folder in FOLDER.items():
        cdir = OUT / folder
        cdir.mkdir(parents=True, exist_ok=True)

        # weights
        w = ROOT / "weights" / f"{cat}_best.pt"
        has_w = w.exists()
        if has_w:
            shutil.copy(w, cdir / f"{cat}_best.pt")

        # predictions CSV
        csvs = glob.glob(str(ROOT / "runs" / "**" / f"{cat}_predictions.csv"), recursive=True)
        if csvs:
            shutil.copy(csvs[0], cdir / f"{cat}_predictions.csv")

        # output frames
        frames = []
        for pd in glob.glob(str(ROOT / "runs" / "**" / "pred"), recursive=True):
            if cat in pd:
                frames += glob.glob(pd + "/*")
        n_out = copy_n(frames, cdir / "output_frames", MAX_OUTPUT_FRAMES)

        # analytics CSVs relevant to this category
        an = []
        for pat in [f"*{cat}*analytics*.csv", "dark_spots.csv", "*states*.csv", "*dwell*.csv"]:
            an += glob.glob(str(ROOT / "runs" / "**" / pat), recursive=True)
            an += glob.glob(str(ROOT / pat))
        an = [a for a in set(an) if cat in a or (cat == "accidents" and "dark_spots" in a)]
        if an:
            copy_n(an, cdir / "analytics", 20)

        # model card
        (cdir / "README.md").write_text(model_card(cat))

        # input samples -> Photos&Videos/<folder>/
        imgs = []
        for d in INPUT_DIRS.get(cat, []):
            imgs += glob.glob(str(ROOT / d / "**" / "images" / "*"), recursive=True)
            imgs += glob.glob(str(ROOT / d / "**" / "*.jpg"), recursive=True)
        n_in = copy_n(sorted(set(imgs)), pv / folder, MAX_INPUT_SAMPLES)

        report.append(f"  {folder:18s} weights={'Y' if has_w else 'N'} "
                      f"csv={'Y' if csvs else 'N'} out_frames={n_out} input(P&V)={n_in}")

    # also drop the global docs into the root for convenience
    for f in ["README.md", "REFERENCES.md"]:
        if (ROOT / f).exists():
            shutil.copy(ROOT / f, OUT / f)

    print("=== ONEDRIVE SUBMISSION BUILT -> onedrive_submission/ ===")
    print("\n".join(report))
    print("\nUpload each local folder's CONTENTS into the matching OneDrive folder:")
    for folder in FOLDER.values():
        print(f"   onedrive_submission/{folder}/  ->  OneDrive Group31/{folder}/")
    print("   onedrive_submission/Photos&Videos/ -> OneDrive Group31/Photos&Videos/")
    print(f"\nAlso ensure the GitHub repo is public/shared: {GITHUB}")


if __name__ == "__main__":
    main()
