"""
prepare_data.py — download + prepare datasets for each category.

Detection categories -> produces a clean YOLO `data.yaml` (paths fixed,
optional single-class collapse, optional multi-dataset merge).
Classification (accidents) -> extracts video frames into train/val folders.

Run standalone:   python prepare_data.py --category fire_smoke
Or it is called automatically by train.py.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import yaml

from configs import DATA_ROOT, ROBOFLOW_API_KEY, get_config


# ---------------------------------------------------------------------------
# Downloaders
# ---------------------------------------------------------------------------
def download_roboflow(workspace: str, project: str, dest: str, fmt="yolov8", prefer=1):
    """Download a Roboflow dataset, auto-finding a valid version number."""
    from roboflow import Roboflow

    if not ROBOFLOW_API_KEY or "PASTE" in ROBOFLOW_API_KEY:
        raise RuntimeError("Set ROBOFLOW_API_KEY env var (see SETUP_GPU.md).")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    proj = rf.workspace(workspace).project(project)
    order = [prefer] + [v for v in range(1, 12) if v != prefer]
    tried = []
    last_err = None
    for v in order:
        try:
            ds = proj.version(v).download(fmt, location=dest)
            print(f"[roboflow] {workspace}/{project} v{v} -> {dest}")
            return ds.location
        except Exception as e:
            tried.append(v)
            last_err = e
    raise RuntimeError(
        f"Could not download {workspace}/{project}; tried versions {tried}. "
        f"Last error: {type(last_err).__name__}: {last_err}"
    )


def download_kaggle(slug: str, dest: str):
    """Download + unzip a Kaggle dataset (needs ~/.kaggle/kaggle.json)."""
    Path(dest).mkdir(parents=True, exist_ok=True)
    zip_guess = Path(dest) / (slug.split("/")[-1] + ".zip")
    if not any(Path(dest).iterdir()):
        print(f"[kaggle] downloading {slug} ...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", dest],
            check=True,
        )
        for z in glob.glob(str(Path(dest) / "*.zip")):
            print(f"[kaggle] unzipping {z}")
            with zipfile.ZipFile(z) as f:
                f.extractall(dest)
    else:
        print(f"[kaggle] {dest} already populated, skipping download")
    return dest


# ---------------------------------------------------------------------------
# Detection prep
# ---------------------------------------------------------------------------
def _find_split_dir(root: Path, split: str):
    for name in ([split] + (["valid", "val"] if split == "val" else [])):
        if (root / name / "images").exists():
            return root / name
    return None


def _relabel_single_class(root: Path):
    """Force every box in every label file to class 0."""
    for txt in glob.glob(str(root / "**" / "labels" / "*.txt"), recursive=True):
        lines = []
        for ln in open(txt).read().splitlines():
            p = ln.split()
            if len(p) >= 5:
                p[0] = "0"
                lines.append(" ".join(p))
        open(txt, "w").write("\n".join(lines))
    for c in glob.glob(str(root / "**" / "*.cache"), recursive=True):
        os.remove(c)


def _write_yaml(root: Path, names) -> Path:
    cfg = {"path": str(root.resolve()), "train": "train/images"}
    val = _find_split_dir(root, "val")
    cfg["val"] = (val.name + "/images") if val else "train/images"
    if (root / "test" / "images").exists():
        cfg["test"] = "test/images"
    cfg["names"] = list(names) if not isinstance(names, dict) else names
    cfg["nc"] = len(cfg["names"])
    out = root / "data_fixed.yaml"
    yaml.safe_dump(cfg, open(out, "w"))
    print(f"[data] {out} | classes: {cfg['names']}")
    return out


def _merge_detect(roots: list[Path], dest: Path, merge_names: dict, unified: list):
    """Merge several YOLO datasets into one, remapping class names -> unified ids."""
    for split in ["train", "valid", "val", "test"]:
        for root in roots:
            sdir = root / split
            if not (sdir / "images").exists():
                continue
            names = yaml.safe_load(open(root / "data.yaml"))["names"]
            oi = dest / ("valid" if split == "val" else split) / "images"
            ol = dest / ("valid" if split == "val" else split) / "labels"
            oi.mkdir(parents=True, exist_ok=True)
            ol.mkdir(parents=True, exist_ok=True)
            for img in (sdir / "images").iterdir():
                tag = root.name
                shutil.copy(img, oi / f"{tag}_{img.name}")
                lbl = sdir / "labels" / f"{img.stem}.txt"
                out_lines = []
                if lbl.exists():
                    for ln in lbl.read_text().splitlines():
                        p = ln.split()
                        if not p:
                            continue
                        raw = names[int(p[0])].lower()
                        mapped = merge_names.get(raw)
                        if mapped is None:
                            continue
                        p[0] = str(unified.index(mapped))
                        out_lines.append(" ".join(p))
                (ol / f"{tag}_{img.stem}.txt").write_text("\n".join(out_lines))
    return _write_yaml(dest, unified)


def prepare_detect(cfg: dict) -> Path:
    cat = cfg["category"]
    base = Path(DATA_ROOT) / cat
    base.mkdir(parents=True, exist_ok=True)

    roots = []
    for i, ds in enumerate(cfg["datasets"]):
        dest = base / f"src{i}"
        if ds["type"] == "roboflow":
            loc = download_roboflow(ds["workspace"], ds["project"], str(dest))
        elif ds["type"] == "kaggle":
            loc = download_kaggle(ds["slug"], str(dest))
        else:
            raise ValueError(ds["type"])
        roots.append(Path(loc))

    # multi-dataset merge (e.g. streetlights)
    if cfg.get("unified"):
        merged = base / "merged"
        data_yaml = _merge_detect(roots, merged, cfg["merge_names"], cfg["unified"])
        root = merged
    else:
        root = roots[0]
        # diagnostics
        lbls = glob.glob(str(root / "train" / "labels" / "*.txt"))
        non_empty = [p for p in lbls if os.path.getsize(p) > 0]
        print(f"[data] train labels: {len(lbls)} | non-empty: {len(non_empty)}")
        names = yaml.safe_load(open(root / "data.yaml")).get("names")
        if cfg.get("single_class"):
            _relabel_single_class(root)
            names = [cfg["single_class"]]
        elif isinstance(names, list) and len(names) == 2 and any(
            not str(n).isalpha() for n in names
        ):
            names = ["fire", "smoke"]  # clean obvious junk names
        data_yaml = _write_yaml(root, names)
    return data_yaml


# ---------------------------------------------------------------------------
# Classification prep (accidents)
# ---------------------------------------------------------------------------
def prepare_classify(cfg: dict) -> Path:
    import cv2

    cat = cfg["category"]
    base = Path(DATA_ROOT) / cat
    raw = base / "raw"
    frames = base / "frames"
    for sub in ["train/accident", "train/no_accident", "val/accident", "val/no_accident"]:
        (frames / sub).mkdir(parents=True, exist_ok=True)

    download_kaggle(cfg["datasets"][0]["slug"], str(raw))

    def label_of(p: Path):
        s = str(p).lower()
        if "non" in s or "no_accident" in s or "normal" in s:
            return "no_accident"
        if "accident" in s or "crash" in s:
            return "accident"
        return None

    stride = cfg.get("frame_stride", 10)
    maxf = cfg.get("max_frames_per_video", 30)
    count = 0
    for f in raw.rglob("*"):
        if f.is_dir():
            continue
        lab = label_of(f)
        if lab is None:
            continue
        split = "train" if count % 5 != 0 else "val"
        dst = frames / split / lab
        if f.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            cap = cv2.VideoCapture(str(f)); i = saved = 0
            while saved < maxf:
                ok, fr = cap.read()
                if not ok:
                    break
                if i % stride == 0:
                    cv2.imwrite(str(dst / f"{f.stem}_{i:04d}.jpg"), fr); saved += 1
                i += 1
            cap.release()
        elif f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            shutil.copy(f, dst / f.name)
        count += 1

    for sub in ["train/accident", "train/no_accident", "val/accident", "val/no_accident"]:
        print(sub, len(list((frames / sub).glob("*"))))
    return frames


# ---------------------------------------------------------------------------
def prepare(category: str) -> Path:
    cfg = get_config(category)
    if cfg["task"] == "classify":
        return prepare_classify(cfg)
    return prepare_detect(cfg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    args = ap.parse_args()
    path = prepare(args.category)
    print("PREPARED ->", path)
