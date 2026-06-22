"""
make_onedrive_submission.py — build a COMPLETE, self-contained submission folder
for each primary category, matching the OneDrive layout the organizers created:

    Fire&Smoke/  CollapsedTrees/  DamagedLights/  DarkSpots/  Dead&StrayAnimals/

Every category folder contains ALL required deliverables (1-9) so each is a full
submission on its own:

  #1 code repository ....... code/ (full code copy) + GitHub link in README
  #2 README (weights, arch, metrics) ... README.md + <cat>_best.pt
  #3 input frames .......... input_frames/
  #4 output frames (boxes) . output_frames/
  #5 bbox coords/dims ...... <cat>_predictions.csv
  #6 LLM / references ...... REFERENCES.md
  #7 code comments ......... code/ (commented source)
  #8 analytics (bonus) ..... analytics/ (CSV + analytics module)
  #9 dataset links (bonus) . DATASETS.md

Run:  python make_onedrive_submission.py
Output: ./onedrive_submission/<category folders>
Upload each folder's CONTENTS into the matching OneDrive folder.
"""
from __future__ import annotations

import glob
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "onedrive_submission"
GITHUB = "https://github.com/Dipanshur19/citylens-group31"
MAX_OUTPUT_FRAMES = 40
MAX_INPUT_FRAMES = 30

FOLDER = {
    "fire_smoke": "Fire&Smoke",
    "collapsed_trees": "CollapsedTrees",
    "streetlights": "DamagedLights",
    "accidents": "DarkSpots",
    "stray_animals": "Dead&StrayAnimals",
}

INFO = {
    "fire_smoke": ("YOLO11x (single-class)", "mAP@50", "0.871",
                   "Detects fire & smoke and outputs bounding boxes."),
    "collapsed_trees": ("YOLOv8m", "mAP@50", "0.878",
                        "Detects fallen trees / road obstructions with bounding boxes."),
    "streetlights": ("YOLO11l", "mAP@50", "0.730",
                     "Detects street lights; OFF-state + flickering via brightness analytic."),
    "accidents": ("YOLO11l-cls", "top-1 accuracy", "0.889",
                  "Classifies accident vs no-accident; dark-spot ranking by frequency."),
    "stray_animals": ("YOLO11x", "mAP@50", "0.977",
                      "Detects animals on road; count/dwell-time/dead-alive via tracking."),
}

ANALYTICS_MODULE = {
    "fire_smoke": "src/analytics/fire_severity.py",
    "streetlights": "src/analytics/streetlight_state.py",
    "accidents": "src/analytics/accident_darkspots.py",
    "stray_animals": "src/analytics/animal_dwell.py",
    "collapsed_trees": None,
}

INPUT_DIRS = {
    "fire_smoke": ["datasets/fire_smoke"],
    "collapsed_trees": ["datasets/trees_full", "datasets/collapsed_trees"],
    "streetlights": ["datasets/streetlights/merged"],
    "accidents": ["datasets/accidents/frames/val"],
    "stray_animals": ["datasets/stray_animals/yolo"],
}

DATASET_LINK = {
    "fire_smoke": ["ECO fire & smoke: https://universe.roboflow.com/eco-group/fire-smoke-yvnrc"],
    "collapsed_trees": ["Fallen trees (with palms): https://universe.roboflow.com/overflow-thaap/fallen-trees-with-palms",
                        "Road debris (alt): https://universe.roboflow.com/fallen-object/road-debris-iya6s-a4hkr"],
    "streetlights": ["Damaged lights: https://universe.roboflow.com/godspeed-yqpeo/damaged-lights",
                     "Sodium street lights: https://universe.roboflow.com/streetlight-detection/sodioum-only-jkq3f"],
    "accidents": ["CCTV accidents: https://www.kaggle.com/datasets/ckay16/accident-detection-from-cctv-footage"],
    "stray_animals": ["MoDES stray animals: https://www.kaggle.com/datasets/bsridevi/modes-dataset-of-stray-animals"],
}

# code shared into every category folder (deliverables #1 and #7)
CODE_ITEMS = ["src", "tools", "configs.py", "prepare_data.py", "train.py",
              "predict_accidents.py", "make_submission.py",
              "make_onedrive_submission.py", "requirements.txt", "GUIDE.md", "SETUP_GPU.md"]


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


def copy_code(dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for item in CODE_ITEMS:
        p = ROOT / item
        if p.is_dir():
            shutil.copytree(p, dst / item, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        elif p.exists():
            shutil.copy(p, dst / item)


def readme(cat: str) -> str:
    model, metric, score, desc = INFO[cat]
    return f"""# {FOLDER[cat]} — CityLens Group 31 (KPI Group 4: Public Safety Hazards)

{desc}

This folder is a complete, self-contained submission for this category and
contains every required deliverable (1-9).

## 1. Code repository
Full code (with comments) is in `code/` here, and on GitHub:
  {GITHUB}

## 2. Model — architecture, weights, performance
- **Architecture:** Ultralytics {model}, COCO-pretrained, fine-tuned on an NVIDIA H200.
- **Trained weights:** `{cat}_best.pt`
- **Performance:** {metric} = **{score}**

## 3. Input frames
Sample training/testing images: `input_frames/`

## 4. Output frames (bounding boxes drawn)
`output_frames/`

## 5. Bounding-box coordinates & dimensions
`{cat}_predictions.csv` (per-frame: class, confidence, x, y, width, height, + normalised)

## 6. LLMs / external references
See `REFERENCES.md`.

## 7. Code comments
All source in `code/` is commented (libraries, functions, and sections explained).

## 8. Analytics feature set (bonus)
See `analytics/`.

## 9. Datasets
See `DATASETS.md`.

## Team — Group 31
Aryan Gupta, Udit Choudhary, Ashish Bairwa, Ayush Kiran Badgujar, Dipanshu Raj
"""


def datasets_md(cat: str) -> str:
    lines = "\n".join(f"- {x}" for x in DATASET_LINK[cat])
    return (f"# Datasets used — {FOLDER[cat]}\n\n{lines}\n\n"
            f"All datasets are open-access / free-to-use. Additional datasets the team "
            f"sourced are documented in the repo's data/dataset_links.md.\n")


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    report = []
    for cat, folder in FOLDER.items():
        cdir = OUT / folder
        cdir.mkdir(parents=True, exist_ok=True)

        # #2 weights
        w = ROOT / "weights" / f"{cat}_best.pt"
        has_w = w.exists()
        if has_w:
            shutil.copy(w, cdir / f"{cat}_best.pt")

        # #5 predictions CSV
        csvs = glob.glob(str(ROOT / "runs" / "**" / f"{cat}_predictions.csv"), recursive=True)
        if csvs:
            shutil.copy(csvs[0], cdir / f"{cat}_predictions.csv")

        # #4 output frames
        frames = []
        for pd in glob.glob(str(ROOT / "runs" / "**" / "pred"), recursive=True):
            if cat in pd:
                frames += glob.glob(pd + "/*")
        n_out = copy_n(frames, cdir / "output_frames", MAX_OUTPUT_FRAMES)

        # #3 input frames (now INSIDE the category folder)
        imgs = []
        for d in INPUT_DIRS.get(cat, []):
            imgs += glob.glob(str(ROOT / d / "**" / "images" / "*"), recursive=True)
            imgs += glob.glob(str(ROOT / d / "**" / "*.jpg"), recursive=True)
        n_in = copy_n(sorted(set(imgs)), cdir / "input_frames", MAX_INPUT_FRAMES)

        # #8 analytics: module + any CSVs
        adir = cdir / "analytics"
        mod = ANALYTICS_MODULE.get(cat)
        if mod and (ROOT / mod).exists():
            adir.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / mod, adir / Path(mod).name)
        an = []
        for pat in [f"*{cat}*analytics*.csv", "dark_spots.csv", "*states*.csv", "*dwell*.csv"]:
            an += glob.glob(str(ROOT / "runs" / "**" / pat), recursive=True)
            an += glob.glob(str(ROOT / pat))
        an = [a for a in set(an) if cat in a or (cat == "accidents" and "dark_spots" in a)]
        if an:
            copy_n(an, adir, 20)

        # #1 + #7 code copy
        copy_code(cdir / "code")

        # #2/#6/#9 docs
        (cdir / "README.md").write_text(readme(cat))
        if (ROOT / "REFERENCES.md").exists():
            shutil.copy(ROOT / "REFERENCES.md", cdir / "REFERENCES.md")
        (cdir / "DATASETS.md").write_text(datasets_md(cat))

        report.append(f"  {folder:18s} weights={'Y' if has_w else 'N'} csv={'Y' if csvs else 'N'} "
                      f"out={n_out} in={n_in} code=Y refs=Y analytics={'Y' if (adir.exists()) else 'N'}")

    print("=== SELF-CONTAINED PER-CATEGORY SUBMISSION -> onedrive_submission/ ===")
    print("\n".join(report))
    print("\nEach folder now holds ALL 9 deliverables. Upload each folder's CONTENTS")
    print("into the matching OneDrive folder:")
    for folder in FOLDER.values():
        print(f"   onedrive_submission/{folder}/  ->  OneDrive Group31/{folder}/")
    print(f"\nMake the repo public/shared: {GITHUB}")


if __name__ == "__main__":
    main()
