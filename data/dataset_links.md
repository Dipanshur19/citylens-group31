# Open-Access Datasets Used — Group 31 (KPI Group 4)

All datasets below are open-access / free-to-use, per the hackathon data rules.
This file is a required deliverable (#6 and bonus #9).

## Category 1 — Fire & Smoke (Priority 40%)
- ECO Group fire & smoke Object Detection (~10k images)
  https://universe.roboflow.com/eco-group/fire-smoke-yvnrc
- Wildfire Smoke Dataset (737 images)
  https://public.roboflow.com/object-detection/wildfire-smoke/1
- Domestic Fire and Smoke Dataset (indoor/outdoor samples)
  https://github.com/datacluster-labs/Domestic-Fire-and-Smoke-Dataset/tree/main/sample_datasets

## Category 2 — Collapsed Trees / Structures (20%)
- Fallen trees (with palms) Object Detection (8.7k images) — **USED (manual YOLOv8 export)**
  https://universe.roboflow.com/overflow-thaap/fallen-trees-with-palms
- Road Debris / obstacles (alternative, auto-download fallback)
  https://universe.roboflow.com/fallen-object/road-debris-iya6s-a4hkr

## Category 3 — Damaged Street Lights (20%)
- Damaged Lights Object Detection by GodSpeed (699 images) — **USED**
  https://universe.roboflow.com/godspeed-yqpeo/damaged-lights
- Sodium / street-light detection (sodium_on / sodium_off) — **USED**
  https://universe.roboflow.com/streetlight-detection/sodioum-only-jkq3f

## Category 4 — Dark Spots / Accidents (10%)
- Accident Detection from CCTV Footage (image frames) — **USED**
  https://www.kaggle.com/datasets/ckay16/accident-detection-from-cctv-footage

## Category 5 — Dead / Stray Animals on Road (10%)
- MoDES Dataset of Stray Animals (segmentation masks → converted to YOLO boxes) — **USED**
  https://www.kaggle.com/datasets/bsridevi/modes-dataset-of-stray-animals

## Notes
- Roboflow datasets are pulled with the `roboflow` Python package (YOLOv8 export format).
- Kaggle datasets are added to the notebook via "Add Input" or the `kaggle` CLI.
- All references and any LLM assistance are documented here and in GUIDE.md.
