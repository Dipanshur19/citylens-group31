"""
make_submission.py — assemble ALL hackathon deliverables into ./submission/
and zip it. Run from the repo root on the H200:

    python make_submission.py

Produces: citylens_group31_submission.zip  (ready to upload)

Maps to the rubric:
  #1 code repo .......... submission/code/ (+ the GitHub link in README)
  #2 README + weights ... submission/README.md, REFERENCES.md, submission/weights/
  #3 input images ....... submission/input_samples/<cat>/
  #4 output frames ...... submission/output_frames/<cat>/   (boxes drawn)
  #5 bbox coords ........ submission/bbox_csv/<cat>_predictions.csv
  #6 LLM/refs ........... submission/REFERENCES.md
  #7 code comments ...... throughout submission/code/
  #8 analytics (bonus) .. submission/analytics/  (+ src/analytics modules in code)
  #9 dataset links ...... submission/code/data/dataset_links.md
"""
from __future__ import annotations

import glob
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUB = ROOT / "submission"
CATS = ["fire_smoke", "collapsed_trees", "streetlights", "accidents", "stray_animals"]
MAX_OUTPUT_FRAMES = 40
MAX_INPUT_SAMPLES = 25

# where each category's training images live (best-effort)
INPUT_DIRS = {
    "fire_smoke": ["datasets/fire_smoke"],
    "collapsed_trees": ["datasets/trees_full", "datasets/collapsed_trees"],
    "streetlights": ["datasets/streetlights/merged"],
    "accidents": ["datasets/accidents/frames/train"],
    "stray_animals": ["datasets/stray_animals/yolo"],
}


def _copy_n(srcs, dst: Path, n: int) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for s in srcs:
        if count >= n:
            break
        try:
            shutil.copy(s, dst / Path(s).name)
            count += 1
        except Exception:
            pass
    return count


def main():
    if SUB.exists():
        shutil.rmtree(SUB)
    SUB.mkdir(parents=True)

    # ---- #1/#7: code + key docs -------------------------------------------
    for f in ["README.md", "REFERENCES.md", "GUIDE.md", "SETUP_GPU.md", "requirements.txt"]:
        if (ROOT / f).exists():
            shutil.copy(ROOT / f, SUB / f)
    (SUB / "code").mkdir(exist_ok=True)
    for item in ["src", "tools", "notebooks", "data", "configs.py",
                 "prepare_data.py", "train.py"]:
        p = ROOT / item
        if p.is_dir():
            shutil.copytree(p, SUB / "code" / item, dirs_exist_ok=True)
        elif p.exists():
            shutil.copy(p, SUB / "code" / item)

    # ---- #2: trained weights ----------------------------------------------
    n_w = _copy_n(sorted(glob.glob(str(ROOT / "weights" / "*.pt"))), SUB / "weights", 99)

    # ---- per-category artifacts -------------------------------------------
    report = []
    for cat in CATS:
        # #5 bbox CSV  (runs/<cat>/<cat>_predictions.csv)
        csvs = glob.glob(str(ROOT / "runs" / "**" / f"{cat}_predictions.csv"), recursive=True)
        if csvs:
            (SUB / "bbox_csv").mkdir(parents=True, exist_ok=True)
            shutil.copy(csvs[0], SUB / "bbox_csv" / f"{cat}_predictions.csv")

        # #4 output frames with boxes  (runs/<cat>/pred/*)
        frames = []
        for pd in glob.glob(str(ROOT / "runs" / "**" / "pred"), recursive=True):
            if cat in pd:
                frames += glob.glob(pd + "/*")
        n_out = _copy_n(frames, SUB / "output_frames" / cat, MAX_OUTPUT_FRAMES)

        # #3 input samples
        imgs = []
        for d in INPUT_DIRS.get(cat, []):
            imgs += glob.glob(str(ROOT / d / "**" / "images" / "*"), recursive=True)
            imgs += glob.glob(str(ROOT / d / "**" / "*.jpg"), recursive=True)
        n_in = _copy_n(sorted(set(imgs)), SUB / "input_samples" / cat, MAX_INPUT_SAMPLES)

        # training curve plot, if present
        plots = [p for p in glob.glob(str(ROOT / "runs" / "**" / "results.png"), recursive=True)
                 if cat in p]
        if plots:
            (SUB / "training_plots").mkdir(exist_ok=True)
            shutil.copy(plots[0], SUB / "training_plots" / f"{cat}_results.png")

        report.append(f"  {cat:16s} weights={'Y' if (SUB/'weights'/(cat+'_best.pt')).exists() else 'N'}"
                      f"  bbox_csv={'Y' if csvs else 'N'}  out_frames={n_out}  inputs={n_in}")

    # ---- #8: any analytics CSVs already generated -------------------------
    an = []
    for pat in ["*analytics*.csv", "dark_spots*.csv", "*_states.csv", "*dwell*.csv"]:
        an += glob.glob(str(ROOT / "**" / pat), recursive=True)
    an = [a for a in an if "submission/" not in a]
    if an:
        (SUB / "analytics").mkdir(exist_ok=True)
        for a in set(an):
            try:
                shutil.copy(a, SUB / "analytics" / Path(a).name)
            except Exception:
                pass

    # ---- zip ---------------------------------------------------------------
    zip_path = ROOT / "citylens_group31_submission.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in SUB.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(ROOT))

    print("=== SUBMISSION ASSEMBLED ===")
    print(f"weights copied: {n_w}")
    print("\n".join(report))
    print(f"\nZIP -> {zip_path}  ({zip_path.stat().st_size / 1e6:.1f} MB)")
    print("\nReview ./submission/ then upload the zip. Don't forget to make the "
          "GitHub repo accessible to judges (deliverable #1).")


if __name__ == "__main__":
    main()
