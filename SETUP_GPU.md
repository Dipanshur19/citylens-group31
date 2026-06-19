# H200 GPU Training Guide (SSH + VS Code) — CityLens Group 31

You have an **NVIDIA H200 (143 GB VRAM)**. This guide trains all 5 categories
from the command line — far faster and more accurate than the Kaggle notebooks.
The Kaggle notebooks still work; this is the better path now that you have a real GPU.

---

## What you'll run

```
python prepare_data.py --category fire_smoke   # download + prep one dataset
python train.py        --category fire_smoke   # train + validate + save weights
python train.py        --all                   # train every category back-to-back
```

Everything is driven by `configs.py` (models, epochs, batch sizes — already tuned
for the H200). Trained models are saved to `weights/<category>_best.pt` on disk,
so **a disconnect can never wipe them** like Kaggle did.

---

## STEP 1 — Connect VS Code to the H200 over SSH

1. Install the **"Remote - SSH"** extension in VS Code.
2. Press **F1** → "Remote-SSH: Connect to Host" → **Add New SSH Host** → enter:
   `ssh ras-c3ih@gpu-h200-10` (use the exact user@host from your terminal prompt).
3. Connect. VS Code opens a window running *on the GPU box*.
4. **File → Open Folder** → pick/create a working folder, e.g. `~/citylens`.
5. Open the integrated terminal (**Ctrl+`**). All commands below run there.

Verify the GPU is visible:
```bash
nvidia-smi          # should show the H200
```

## STEP 2 — Get this repository onto the box

```bash
cd ~
git clone https://github.com/Dipanshur19/citylens-group31.git
cd citylens-group31
```

## STEP 3 — Python environment + dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Confirm PyTorch sees the GPU:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: ... True NVIDIA H200 ...
```
If `cuda.is_available()` is False, install the CUDA build of torch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## STEP 4 — API keys for the datasets

**Roboflow** (fire, trees, streetlights):
```bash
export ROBOFLOW_API_KEY="your_roboflow_key"
```
(Add that line to `~/.bashrc` so it persists across logins.)

**Kaggle** (accidents, stray animals):
1. kaggle.com → your avatar → **Settings → API → Create New Token** → downloads `kaggle.json`.
2. Put it on the box:
   ```bash
   mkdir -p ~/.kaggle
   # upload kaggle.json (drag it into VS Code's file explorer), then:
   mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   ```

## STEP 5 — Train

Start with the 40%-weight category:
```bash
python train.py --category fire_smoke
```
You'll see the download, then a per-epoch table with `mAP50` climbing. On the
H200 with `yolo11x`, an epoch on ~7k images is roughly 20–40 s, so 120 epochs
finishes in well under an hour.

When done it prints, e.g.:
```
[fire_smoke] mAP@50=0.9xx mAP@50-95=0.6xx P=0.9xx R=0.9xx
[weights] saved -> weights/fire_smoke_best.pt
```

Then the rest:
```bash
python train.py --category collapsed_trees
python train.py --category streetlights
python train.py --category stray_animals
python train.py --category accidents
# or all at once:
python train.py --all
```

### Run long jobs so they survive a dropped SSH connection
Use `tmux` (or `nohup`) so training keeps going if VS Code disconnects:
```bash
tmux new -s train
python train.py --all 2>&1 | tee train.log
# detach: press Ctrl+b then d   |   reattach later: tmux attach -t train
```

## STEP 6 — Squeeze out more accuracy (you have the VRAM for it)

`configs.py` is already aggressive, but with 143 GB you can push further:
```bash
python train.py --category fire_smoke --batch 128 --imgsz 960 --epochs 150
```
- **Bigger `--imgsz`** (e.g. 960/1280) helps small fire/smoke and distant lights.
- **Bigger `--batch`** trains faster and stabilises training.
- **`--model yolo11x.pt`** is the most accurate; it's already the default for the
  high-value categories.
- For `stray_animals`, raise `fraction` in `configs.py` toward `1.0` to use more
  of the 400k images once you confirm the pipeline works.

## STEP 7 — Collect deliverables

After each run you have, under `runs/<category>/`:
- `train/` — loss/mAP plots, `results.csv`, confusion matrix
- `pred/` — annotated frames with boxes drawn (**deliverable #4**)
- `<category>_predictions.csv` — bbox coords + dimensions (**deliverable #5**)
- `weights/<category>_best.pt` (also copied to top-level `weights/`) — (**deliverable #2**)

Run the bonus analytics from `src/analytics/` (severity, OFF/flicker, dwell,
dark-spots) on these weights, then fill the metrics table in `README.md`.

## Troubleshooting
- **`kaggle: command not found`** → `pip install kaggle` inside the venv.
- **Roboflow download fails** → open the dataset's page → Download → YOLOv8 →
  copy the exact `workspace`/`project`/`version` into `configs.py`.
- **CUDA OOM** (unlikely on 143 GB) → lower `--batch`.
- **Slow data loading** → raise `workers` in `configs.py` (e.g. 24/32).
- **Want determinism** → set `cache=False` in `configs.py` for that category.
