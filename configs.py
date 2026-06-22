"""
configs.py — central training configuration for all 5 KPI Group 4 categories.

Tuned for a single NVIDIA H200 (143 GB VRAM): largest YOLO11 models, big
batches, and larger image sizes for maximum accuracy. Override any value from
the command line, e.g.  python train.py --category fire_smoke --batch 96

API keys: set them as environment variables (recommended) before running:
    export ROBOFLOW_API_KEY="xxxx"
    # Kaggle: place kaggle.json in ~/.kaggle/ (see SETUP_GPU.md)
"""
import os

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "PASTE_YOUR_ROBOFLOW_API_KEY")

# Where prepared datasets and outputs go
DATA_ROOT = os.environ.get("CITYLENS_DATA", "datasets")
OUT_ROOT = os.environ.get("CITYLENS_OUT", "runs")
WEIGHTS_DIR = "weights"

CONFIGS = {
    # ----------------------------------------------------------------- 40% ---
    "fire_smoke": {
        "task": "detect",
        "datasets": [
            {"type": "roboflow", "workspace": "eco-group", "project": "fire-smoke-yvnrc"},
        ],
        # Collapse the messy 2-class labels into one hazard class -> higher mAP.
        # Set to None to keep the dataset's own classes.
        "single_class": "fire_smoke",
        "model": "yolo11x.pt",
        "epochs": 120, "imgsz": 768, "batch": 64, "patience": 30, "cache": True,
    },
    # ----------------------------------------------------------------- 20% ---
    "collapsed_trees": {
        "task": "detect",
        # Try each candidate in order; use the FIRST that downloads successfully.
        "try_alternatives": True,
        "datasets": [
            {"type": "roboflow", "workspace": "fallen-object", "project": "road-debris-iya6s-a4hkr"},
            {"type": "roboflow", "workspace": "overflow-thaap", "project": "fallen-trees-with-palms"},
        ],
        "single_class": None,
        "model": "yolo11x.pt",
        "epochs": 100, "imgsz": 640, "batch": 64, "patience": 25, "cache": True,
    },
    # ----------------------------------------------------------------- 20% ---
    "streetlights": {
        "task": "detect",
        "datasets": [
            {"type": "roboflow", "workspace": "godspeed-yqpeo", "project": "damaged-lights"},
            {"type": "roboflow", "workspace": "streetlight-detection", "project": "sodioum-only-jkq3f"},
        ],
        # Single unified class: detect ALL street lights (easier task, more
        # instances -> higher mAP). ON/OFF + flickering come from the brightness
        # analytic in src/analytics/streetlight_state.py, not the detector.
        "unified": ["streetlight"],
        "merge_names": {
            "working": "streetlight", "sodium_on": "streetlight", "on": "streetlight",
            "not working": "streetlight", "sodium_off": "streetlight", "off": "streetlight",
            "light": "streetlight", "streetlight": "streetlight", "lamp": "streetlight",
            "damaged": "streetlight", "broken": "streetlight",
        },
        "single_class": None,
        "model": "yolo11l.pt",
        "epochs": 100, "imgsz": 768, "batch": 32, "patience": 25, "cache": True,
    },
    # ----------------------------------------------------------------- 10% ---
    "stray_animals": {
        "task": "detect",
        "datasets": [
            {"type": "kaggle", "slug": "bsridevi/modes-dataset-of-stray-animals"},
        ],
        "single_class": None,
        "model": "yolo11x.pt",
        # 400k images is huge -> do NOT cache to RAM, and a subset/fraction
        # trains fast while still being accurate. Raise 'fraction' for more data.
        "epochs": 60, "imgsz": 640, "batch": 64, "patience": 20,
        "cache": False, "fraction": 0.5,
    },
    # ----------------------------------------------------------------- 10% ---
    "accidents": {
        "task": "classify",       # accident vs no_accident from video frames
        "datasets": [
            {"type": "kaggle", "slug": "ckay16/accident-detection-from-cctv-footage"},
        ],
        # High-accuracy config for the H200. Mid-size model + higher resolution
        # + regularization generalises better on this smaller dataset than yolo11x.
        "model": "yolo11l-cls.pt",
        "epochs": 150, "imgsz": 384, "batch": 128, "patience": 30, "cache": True,
        "frame_stride": 8, "max_frames_per_video": 40,
    },
}


def get_config(category: str, **overrides) -> dict:
    if category not in CONFIGS:
        raise KeyError(f"Unknown category '{category}'. Choose from {list(CONFIGS)}")
    cfg = dict(CONFIGS[category])
    cfg["category"] = category
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg
