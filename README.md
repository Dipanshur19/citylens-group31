# CityLens AI Hackathon 2026 — Group 31

## KPI Group 4: Public Safety Hazards

This repository contains our detection models, analytics, and deliverables for the
CityLens AI Hackathon 2026. Every model is built on **YOLO (Ultralytics YOLO11)**,
trained on open-access datasets, and produces bounding boxes + the bonus analytics
feature set required by the scoring rubric.

> **Minimum bar:** > 85% accuracy across all assigned categories.
> **Our target:** mAP@50 ≥ 0.90 on each category.

---

## Assigned Categories & Weightage

| # | Category | Weight | Model | Bonus Analytics |
|---|----------|--------|-------|-----------------|
| 1 | Burning of waste, Smoke / Fire Detection | **40%** | `01_fire_smoke` | Severity (Mild/Moderate/Severe), Vulnerability (crowd/building density), Burning area (sq. m) |
| 2 | Collapsed trees or structures | **20%** | `02_collapsed_trees` | Obstacle type classification |
| 3 | Damaged Street Lights | **20%** | `03_streetlights` | OFF-state detection, Flickering detection (across video frames) |
| 4 | Dark Spots / Black Spots (accidents) | **10%** | `04_accidents_darkspots` | Accident detection, dark-spot location ranking by frequency |
| 5 | Dead or Stray Animals on Road | **10%** | `05_stray_animals` | Animal type, Dead/Alive, Count & dwell time |

Because category 1 is worth 40% of the score, **prioritise it** — spend the most
time getting its accuracy and analytics right.

---

## Repository Structure

```
citylens-group31/
├── README.md                     # <- this file (required deliverable)
├── requirements.txt
├── GUIDE.md                      # full step-by-step Kaggle workflow + 1-week plan
├── data/
│   └── dataset_links.md          # all open-access dataset sources (required deliverable)
├── configs/                      # auto-generated data.yaml files live here
├── src/
│   ├── common.py                 # shared helpers (model loader, plotting)
│   ├── export_predictions.py     # writes bbox coords + dimensions per frame (required)
│   └── analytics/
│       ├── fire_severity.py      # severity + vulnerability + area estimation
│       ├── streetlight_state.py  # OFF / flickering logic
│       ├── accident_darkspots.py # dark-spot frequency aggregation
│       └── animal_dwell.py       # count + dwell-time tracking
├── notebooks/                    # paste these into Kaggle cells (one per category)
│   ├── 01_fire_smoke.py
│   ├── 02_collapsed_trees.py
│   ├── 03_streetlights.py
│   ├── 04_accidents_darkspots.py
│   ├── 05_stray_animals.py
│   └── ipynb/                    # ready-to-UPLOAD Kaggle notebooks (.ipynb)
├── tools/build_notebooks.py      # regenerates the .ipynb files from the .py sources
└── outputs/                      # trained weights, predicted frames, CSVs (generated)
```

---

## Model Architecture (summary for judges)

- **Framework:** Ultralytics YOLO11, COCO-pretrained, fine-tuned per category.
- **Detectors:** `yolo11x` (fire, animals), `yolov8m` (collapsed trees), `yolo11l` (street lights).
- **Classifier:** `yolo11l-cls` for accident detection (accident vs no-accident).
- **Hardware:** trained on a single NVIDIA H200 (143 GB). Input sizes 640–1280px.
- **Augmentation:** mosaic, mixup, copy-paste, HSV, flips (Ultralytics) + tuned per category.
- **One model per category**; trained weights saved in `weights/<category>_best.pt`.

## Performance Metrics

Detection categories report **mAP@50**; the accident model reports **top-1 accuracy**.

| Category | Weight | Model | Metric | Score | Prize bar (>85%) |
|----------|--------|-------|--------|-------|------------------|
| Fire & Smoke      | 40% | yolo11x (single-class) | mAP@50 | **0.871** |  |
| Collapsed trees   | 20% | yolov8m | mAP@50 | **0.878** | pass |
| Street lights     | 20% | yolo11l (2-class) | mAP@50 (overall) | **0.730** |  data-limited¹ |
| Accidents         | 10% | yolo11l-cls | top-1 acc | **0.889** |  |
| Stray animals     | 10% | yolo11x | mAP@50 | **0.977** |  |

**Weighted average ≈ 0.857** (clears the 85% overall minimum).

¹ Street-light datasets are small/noisy; the `streetlight` class alone reaches
mAP@50 ≈ 0.85. OFF-state + flickering are delivered via the brightness analytic
in `src/analytics/streetlight_state.py`.

> Full training workflow on the H200: see `SETUP_GPU.md`. Kaggle workflow: `GUIDE.md`.

## Team — Group 31
Aryan Gupta · Udit Choudhary · Ashish Bairwa · Ayush Kiran Badgujar · Dipanshu Raj
