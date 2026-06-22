# References, Tools & AI Assistance — Group 31 (Deliverable #6)

This document discloses every external tool, library, model, dataset, and AI
assistant used to build this submission, per hackathon rule #6.

## AI / LLM assistance
- An **AI coding assistant (Kiro)** was used to help write and debug the training
  pipeline (`configs.py`, `prepare_data.py`, `train.py`), the dataset converters
  (`tools/`), and the analytics modules (`src/analytics/`), and to guide setup on
  the GPU server. All code was reviewed, run, and validated by the team.

## Frameworks & libraries
- **Ultralytics YOLO11** — detection, classification, and tracking
  (https://github.com/ultralytics/ultralytics)
- **PyTorch / TorchVision** — deep-learning backend
- **OpenCV, NumPy, Pandas** — image processing, mask→bbox conversion, analytics
- **Roboflow** Python package — dataset download (YOLOv8 export)
- **Kaggle** CLI — dataset download
- **ByteTrack** (built into Ultralytics) — multi-object tracking for dwell time

## Pretrained weights
- COCO-pretrained `yolo11x.pt`, `yolo11l.pt`, `yolo11l-cls.pt` from Ultralytics.

## Datasets
See `data/dataset_links.md` for the full list of open-access datasets actually used.

## Hardware
- Single NVIDIA H200 NVL (143 GB), trained over SSH.

## Methods of note
- **Fire & Smoke:** messy 2-class labels collapsed into a single `fire_smoke`
  hazard class for higher, more stable mAP.
- **Street lights:** detector locates lights; ON/OFF + flickering derived from
  lamp brightness across frames (rule-based, explainable) — `streetlight_state.py`.
- **Accidents:** video frames extracted and classified (accident vs no-accident);
  accident events aggregated by location into ranked "dark spots".
- **Stray animals:** the MoDES segmentation masks were converted to YOLO bounding
  boxes (`tools/modes_masks_to_yolo.py`), giving a single-class animal detector;
  count + dwell-time + dead/alive come from tracking at inference.
