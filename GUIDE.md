# CityLens Hackathon — Complete Working Guide (Group 31, KPI Group 4)

This is your start-to-finish playbook. Read it once, then follow it.

---

## 0. The big picture (what you're actually being asked to do)

You must build computer-vision models that **detect and classify 5 types of public
safety hazards** from images/videos, draw bounding boxes, and output the box
coordinates. Hitting **> 85% accuracy** on each category makes you prize-eligible.
Scoring = 90% model performance + 10% ease of use (clean code + README).

Your 5 categories and their weight in the final score:

| Priority | Category | Weight | Task type |
|----------|----------|--------|-----------|
| 1 | Fire & Smoke | **40%** | image object detection + analytics |
| 2 | Collapsed trees / structures | 20% | image object detection |
| 2 | Damaged street lights | 20% | detection + video analytics |
| 3 | Accidents / dark spots | 10% | video classification + aggregation |
| 3 | Dead / stray animals | 10% | detection + video tracking |

**Spend your time in proportion to weight.** Fire & Smoke alone is 40% — nail it
first. The two 20% categories next. The 10% ones last.

---

## 1. Why YOLO (Ultralytics YOLO11)

- One framework solves every category: detection, classification, and tracking.
- COCO-pretrained -> fast fine-tuning, high accuracy on small datasets.
- Runs easily on Kaggle's free GPU. `yolo11s` trains fast; `yolo11m` if you need
  more accuracy and have time. It's open-source and free — satisfies the rules.

---

## 2. One-time Kaggle setup

1. Create a free account at kaggle.com and **verify your phone** (required for GPU
   + internet in notebooks).
2. New Notebook -> right sidebar:
   - **Accelerator:** GPU T4 x2 (or P100).
   - **Internet:** ON.
3. Get a **Roboflow API key**: roboflow.com -> Settings -> Roboflow API -> copy key.
   Paste it where each notebook says `PASTE_YOUR_ROBOFLOW_API_KEY`.
   (Better: store it in Kaggle "Add-ons -> Secrets" and read it, so you don't
   leak it in your submitted code.)

> GPU quota on Kaggle is ~30 hrs/week. That's plenty for 5 models if you use the
> fast settings below. Don't leave notebooks running idle.

---

## 3. Getting the datasets

**Roboflow datasets** (fire, trees, lights) — downloaded *inside* the notebook with
the `roboflow` package. The code is already in each notebook (CELL 2). Just paste
your API key. Each dataset's "Download this Dataset -> YOLOv8" page also shows the
exact `workspace().project().version()` snippet if a version number differs.

**Kaggle datasets** (accidents, MoDES animals) — don't download by hand:
- In your notebook, click **Add Input** (right sidebar) -> search the dataset name
  -> Add. It mounts read-only at `/kaggle/input/<dataset-slug>/`.
- Then point the notebook's input path at that folder.

---

## 4. How to run each model (same pattern for all)

Each file in `notebooks/` is split into cells marked `# %% CELL N`. To use it:
1. Open a fresh Kaggle notebook for that category.
2. Copy each `# %% CELL` block into its own notebook cell, in order.
3. Run top to bottom. Training cell shows live mAP; the validate cell prints the
   numbers you copy into the README metrics table.

Order to build them: `01_fire_smoke` -> `02_collapsed_trees` ->
`03_streetlights` -> `05_stray_animals` -> `04_accidents_darkspots`.

---

## 5. Hitting > 90% accuracy (and keeping training fast)

**Fast training:**
- Use `yolo11s.pt`, `imgsz=640`, `batch=16`, `cache=True`, `amp=True`. ~60 epochs
  on 10k images is roughly 1–2 hrs on a T4. The accident classifier (imgsz=224)
  is much faster.
- `patience=15` early-stops once mAP plateaus, so you never waste epochs.

**Pushing accuracy up (do these if a category is below target):**
1. **Train longer / bigger model:** bump to `yolo11m.pt` or `epochs=100`.
2. **More data:** merge an extra open dataset (use the remap helper in
   `01_fire_smoke.py` CELL 3). More variety = better generalisation.
3. **Clean the labels:** Roboflow sets sometimes have wrong/duplicate boxes.
   Drop obviously bad classes (the remap step already filters unused ones).
4. **Tune confidence at inference:** if recall is low, lower `conf` to 0.2; if
   precision is low, raise it to 0.4.
5. **Check the right metric:** "accuracy" for detection = **mAP@50**. For the
   accident *classifier* it's **top-1 accuracy**. Report both clearly so judges
   see you cleared 85%.

> Honest note: 90%+ mAP is realistic on the clean datasets (fire, lights, trees,
> animals). The accident video task is harder — if it lands at 85–90%, that's
> still prize-eligible and fine given it's only 10% weight.

---

## 6. Producing the required output files

For every category, after training you must produce (deliverables #4 and #5):
- **Annotated frames** with boxes drawn -> the `predict(..., save=True)` cells do
  this; find them in `runs/<name>/` (or use `src/export_predictions.py`).
- **A CSV of box coords + dimensions** (frame, class, confidence, x, y, width,
  height) -> `src/export_predictions.py` does exactly this. Run:
  ```python
  from src.export_predictions import export
  export("runs/fire_smoke/weights/best.pt",
         source="/kaggle/input/test-images", category="fire_smoke")
  ```
- **Bonus analytics CSVs** are produced by each notebook's analytics cell.
- **Download everything** from the Kaggle notebook's Output tab, or `model.export()`
  the weights, to assemble your final zip.

---

## 7. One-week plan (you have ~7 days)

| Day | Goal |
|-----|------|
| **Day 1** | Everyone sets up Kaggle + Roboflow key. Assign 1–2 categories per person. Get fire dataset downloading. Read this guide together. |
| **Day 2** | Train **Fire & Smoke** (40%). Get mAP@50 ≥ 0.90. Build its severity/vulnerability/area analytics. This is the most important day. |
| **Day 3** | Train **Collapsed trees** + **Street lights** (20% each) in parallel across teammates. |
| **Day 4** | Finish street-light OFF/flicker analytics. Train **Stray animals**, add count + dwell + dead/alive. |
| **Day 5** | Train **Accident** classifier + dark-spot aggregation. Re-train any model below 85%. |
| **Day 6** | Generate ALL output frames + CSVs. Fill the README metrics table. Clean and comment code. |
| **Day 7** | Assemble the single zip, double-check the deliverables checklist, submit before **22 June 2026, 24:00**. Don't wait for the last hour. |

**Divide work by category** (weights in brackets): give your strongest member the
fire model (40%), pair up on the 20% ones, and let two people handle the 10% video
tasks. One person owns the README + final zip.

---

## 8. Final submission checklist (single zip, due 22 June)

- [ ] **#1** Code repository (this whole folder) with access shared to judges
- [ ] **#2** README with trained weights info, model architecture, performance metrics — fill the table!
- [ ] **#3** Input frames/images used for training & testing
- [ ] **#4** Output frames/images with bounding boxes drawn
- [ ] **#5** CSV of bounding-box coords + dimensions per frame
- [ ] **#6** Details of any LLMs / external references used (this guide + dataset_links.md)
- [ ] **#7** Code comments explaining libraries, functions, sections — already in every file
- [ ] **#8 (bonus)** Analytics feature set CSVs — severity, area, OFF/flicker, dwell, dark spots
- [ ] **#9 (bonus)** Links to any extra datasets you sourced — add to `data/dataset_links.md`
- [ ] Trained `best.pt` weights for all 5 models included
- [ ] All 5 categories clear > 85% (report mAP@50 / top-1)

---

## 9. Connecting to the organizers' CCTV dataset (SSH)

The email gives SSH credentials (`citylens_user_31`) to their dataset. If you're
off-campus/non-IITK you must connect through the **Fortinet VPN** first, then SSH
in. Use that footage as extra **test** images/videos to prove your models work on
their real data — point the `predict`/`track` cells at those files. Training can
stay on the open datasets.

---

## 10. Notes on tools / references used (for deliverable #6)
- **Ultralytics YOLO11** — detection, classification, tracking framework.
- **Roboflow** — dataset hosting/download for fire, trees, lights.
- **ByteTrack** (built into Ultralytics) — multi-object tracking for dwell time.
- Analytics logic (severity, OFF/flicker, dark-spots, dwell) is our own rule-based
  code in `src/analytics/`, designed to be transparent and explainable.
