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

- **Backbone / detector:** Ultralytics YOLO11 (`yolo11s.pt` for fast training,
  `yolo11m.pt` when more accuracy is needed). YOLOv8 is a drop-in fallback.
- **Input size:** 640×640.
- **Transfer learning:** start from COCO-pretrained weights, fine-tune per category.
- **Augmentation:** mosaic, mixup, HSV, flip — handled by Ultralytics defaults plus
  tuned values in each script.
- **One model per category** (datasets are independent), each exported to ONNX +
  PyTorch `.pt`.

## Performance Metrics

Fill this table from each run's `results.csv` / validation output:

| Category | Model | mAP@50 | mAP@50-95 | Precision | Recall |
|----------|-------|--------|-----------|-----------|--------|
| Fire & Smoke      | yolo11s | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Collapsed trees   | yolo11s | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Street lights     | yolo11s | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Accidents         | yolo11s | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Stray animals     | yolo11s | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

> See `GUIDE.md` for the full step-by-step Kaggle workflow and the 1-week plan.

## Team — Group 31
Aryan Gupta · Udit Choudhary · Ashish Bairwa · Ayush Kiran Badgujar · Dipanshu Raj
