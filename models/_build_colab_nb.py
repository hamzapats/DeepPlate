"""Build colab_train_plate.ipynb. Run once, then this file can stay as the generator."""
import json
from pathlib import Path

def md(text):
    lines = text.strip("\n").split("\n")
    src = [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(text):
    lines = text.strip("\n").split("\n")
    src = [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

cells = []

cells.append(md("""
# PlateGuard — train the plate detector

This notebook is the training run for the project. It produces one file: `best.pt`. That file is the YOLO model that draws a box around a number plate. PaddleOCR is a separate, already-trained reader. You do not train it here.

Run the cells from top to bottom with **Shift+Enter**. Do not use *Runtime → Run all*. The long training cell takes about 4–6 hours, and the last cell spends the test set. You only run that last cell once.

Your laptop does not do this job. Colab's free GPU does.
"""))

cells.append(md("""
## 0. What the words mean

**A weight file (`.pt`).** The model is a big list of numbers. Those numbers are the "weights". `yolov8s.pt` is that list, saved in one file. Training means nudging those numbers.

**COCO.** A public set of everyday photos (people, cars, dogs, buses, about 80 kinds of object). Ultralytics already trained YOLOv8 on it. The file you download already knows edges, corners, and what a car looks like. It has never been told the class "number plate".

**Fine-tuning.** You load that COCO file and keep training, this time on Indian plate photos with boxes. The early numbers stay useful (edges, cars). The last part of the network learns the single class `number_plate`. Starting from random numbers would need far more photos and far more time. The report calls this transfer learning.

**YOLOv8n and YOLOv8s.** Same network design, two sizes. The letter is the size:

| File | Size name | What it is for |
|---|---|---|
| `yolov8n.pt` | nano | The smallest. Use it for a 5-epoch check that the files are wired up. |
| `yolov8s.pt` | small | The real model. Accurate enough to catch a plate that is small in the frame, and it still finishes overnight on a free T4. |

Larger letters (`m`, `l`, `x`) are slower. You do not need them for this deadline.

**Image size 640.** Before the network looks at a photo, the photo is resized to 640×640 pixels. NPDS photos are already 640×640, so 640 keeps them as they are. 320 would throw away detail. 960 would stretch a 640 photo and invent no new detail.

**Epoch.** One pass through every training photo. Epoch 50 means each training photo has been shown about 50 times. Each time, colour and scale are jittered a little, so the repeats are not identical.

**Batch.** How many photos the GPU handles at once. 16 is the number that fits a T4 for YOLOv8s at 640.

**Loss.** One number that says how wrong the current boxes are. It should fall. You never quote the loss in the report. You quote precision, recall, and mAP.

**Precision.** Of the boxes the model drew, the fraction that really are plates.

**Recall.** Of the real plates, the fraction the model found.

**mAP@50.** A single score that combines precision and recall. A drawn box counts as correct when it overlaps the true plate by at least half. Higher is better. 1.00 would be perfect.

**Validation set.** The practice exam. You look at it while training, and you use it to pick the best epoch and the confidence threshold.

**Test set.** The final exam. You open it once, after every choice is frozen.

**`best.pt`.** After every epoch, Ultralytics scores the model on the validation photos. This file is overwritten only when the score improves. It is the epoch that did best on the practice exam, not the last epoch.
"""))

cells.append(md("""
## 1. Turn the GPU on

Before you run any cell:

1. Menu **Runtime → Change runtime type**.
2. Hardware accelerator: **T4 GPU**.
3. Save.

Then run the next cell. You want a line that names a Tesla T4. If it says `NONE`, the GPU is off. Change the runtime and run the cell again. Do not continue with `NONE`.
"""))

cells.append(code("""
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("GPU is off. Runtime → Change runtime type → T4 GPU, then run this cell again.")
"""))

cells.append(md("""
## 2. Install Ultralytics

Ultralytics is the Python package that loads YOLO, trains it, and saves `best.pt`. Colab already has PyTorch and OpenCV. This install takes about a minute. `WANDB` is an optional online logger. Turn it off so the cell does not stop and ask you for an account.
"""))

cells.append(code("""
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

!pip -q install ultralytics

import ultralytics, cv2, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("ultralytics", ultralytics.__version__)
print("opencv", cv2.__version__)
print("Write these four versions into the report (section 4.1.5 and Table 4.6).")
"""))

cells.append(md("""
## 3. Put the dataset on Google Drive

The dataset is **Indian Number Plate Dataset (IURS) – NPDS**, Mendeley, DOI `10.17632/sxrnr7hwtk.1`. It is 19,309 images of Indian vehicles. Plates are boxed in COCO format. Licence CC BY 4.0.

NPDS is not 19,309 different cars. Roboflow saved each original photo several times: flips, small rotations, shear, and a bit of noise. Those copies are the reason a later cell groups filenames before the split.

Download steps, on your laptop:

1. Open https://data.mendeley.com/datasets/sxrnr7hwtk/1 and sign in (a free account is enough).
2. Download version 1. You get one `.zip`. It is likely a few GB. Wait until the download finishes.
3. Open https://drive.google.com → **New → New folder** → name it `PlateGuard`.
4. Drag the zip into that folder. Wait until the upload icon finishes. A half-uploaded zip will fail in the next cells.

Come back here and run the next cell. A Google window asks permission to see your Drive. Allow it.
"""))

cells.append(code("""
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
DRIVE = Path("/content/drive/MyDrive/PlateGuard")
DRIVE.mkdir(parents=True, exist_ok=True)
zips = sorted(DRIVE.glob("*.zip"))
print("Folder:", DRIVE)
print("Zip files:", [z.name for z in zips] or "NONE — the zip is not in MyDrive/PlateGuard yet")
if len(zips) != 1:
    raise SystemExit("Put exactly one .zip in MyDrive/PlateGuard, then run this cell again.")
ZIP_PATH = zips[0]
print("Using:", ZIP_PATH, " size_GB=", round(ZIP_PATH.stat().st_size / 1e9, 2))
"""))

cells.append(md("""
## 4. Unzip onto the Colab disk

`/content` is the fast disk on the Colab machine. It is wiped when the session ends. The zip stays safe on Drive. Training images are read thousands of times, so they live on `/content`, not on Drive.
"""))

cells.append(code("""
import zipfile
RAW = Path("/content/npds_raw")
if not any(RAW.rglob("*.json")):
    RAW.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(RAW)
    print("unzipped")
else:
    print("already unzipped, skipping")

# Show the top of the tree so you can see what Mendeley actually shipped.
shown = 0
for p in sorted(RAW.rglob("*")):
    if p.is_file() and shown < 30:
        print(p.relative_to(RAW))
        shown += 1
print("...")
print("json files:", len(list(RAW.rglob("*.json"))))
print("images:", sum(1 for p in RAW.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}))
"""))

cells.append(md("""
## 5. Read the COCO boxes and look at four of them

A COCO file is a JSON with two lists that matter:

- `images`: file name, width, height, and an id.
- `annotations`: which image, and a box `[x, y, width, height]` in pixels. `x, y` is the **top-left** corner.

Roboflow often ships three JSON files (`train`, `valid`, `test`) whose image ids start again at 0. This cell joins them into one list and gives every image a new id, so nothing collides.

The picture below must show a rectangle on the plate. If the rectangle is on a wheel, a headlight, or off in a corner, stop and tell me what you see. Training on bad boxes produces a bad model.
"""))

cells.append(code("""
import json
from collections import Counter

def load_coco(root: Path):
    files = []
    for p in sorted(root.rglob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and "images" in data and "annotations" in data:
            files.append((p, data))
    if not files:
        raise SystemExit("No COCO json found. Paste the printed file tree into the chat.")
    print("COCO json files:")
    for p, data in files:
        names = [c.get("name") for c in data.get("categories", [])]
        print(f"  {p.relative_to(root)}  images={len(data['images'])}  boxes={len(data['annotations'])}  classes={names}")

    # One index of file name -> path. Built once. Searching per image would be very slow.
    by_name = {}
    for img_path in root.rglob("*"):
        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            by_name.setdefault(img_path.name, img_path)

    images, annotations = [], []
    next_id, missing = 1, 0
    for p, data in files:
        id_map = {}
        for im in data["images"]:
            name = Path(im["file_name"]).name
            hit = by_name.get(name)
            if hit is None:
                direct = p.parent / im["file_name"]
                hit = direct if direct.is_file() else None
            if hit is None:
                missing += 1
                continue
            id_map[im["id"]] = next_id
            images.append({
                "id": next_id,
                "file_name": im["file_name"],
                "width": im["width"],
                "height": im["height"],
                "abs_path": str(hit),
            })
            next_id += 1
        for ann in data["annotations"]:
            if ann["image_id"] not in id_map:
                continue
            annotations.append({
                "image_id": id_map[ann["image_id"]],
                "bbox": ann["bbox"],
                "category_id": ann.get("category_id"),
            })
    print(f"merged images={len(images)}  boxes={len(annotations)}  missing_files={missing}")
    if missing > 0.01 * max(len(images), 1):
        raise SystemExit("Too many images named in the json are not on disk. Stop.")
    return images, annotations

IMAGES, ANNOTATIONS = load_coco(RAW)
print("class ids:", Counter(a["category_id"] for a in ANNOTATIONS))
"""))

cells.append(code("""
import cv2
import random
from google.colab.patches import cv2_imshow

by_id = {im["id"]: im for im in IMAGES}
boxes_of = {}
for ann in ANNOTATIONS:
    boxes_of.setdefault(ann["image_id"], []).append(ann["bbox"])

random.seed(42)
sample_ids = random.sample(list(boxes_of), 4)
for iid in sample_ids:
    im = by_id[iid]
    frame = cv2.imread(im["abs_path"])
    for x, y, w, h in boxes_of[iid]:
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    print(im["file_name"], "boxes", boxes_of[iid])
    cv2_imshow(frame)
"""))

cells.append(md("""
## 6. Convert to YOLO and split

YOLO does not read COCO. It wants, next to each photo, a text file with one line per plate:

```
0  x_center  y_center  width  height
```

All four numbers are fractions of the image, from 0 to 1. Class `0` is `number_plate`.

Example: `0 0.51 0.62 0.18 0.07` means the plate's center is 51% of the way across and 62% of the way down, and the plate is 18% of the image wide and 7% of the image tall.

**The split.** Seed 42, about 70% train / 20% validation / 10% test. The split is by **original photo**, not by file. `car12.rf.aaaa.jpg` and `car12.rf.bbbb.jpg` share the key `car12`, so they land in the same split. The report's 13,516 / 3,862 / 1,931 counts are 70/20/10 of 19,309 files. Your counts will differ, because a group of three copies moves together. That is the correct result. The report says "about 70%".

Only images that have at least one box are kept. A photo with no plate cannot teach the detector.
"""))

cells.append(code("""
import os, random, shutil
from collections import defaultdict

OUT = Path("/content/plates")
if OUT.exists():
    shutil.rmtree(OUT)

def group_key(file_name: str) -> str:
    stem = os.path.splitext(os.path.basename(file_name))[0]
    if ".rf." in stem:
        stem = stem.split(".rf.")[0]
    return stem

labels = defaultdict(list)
for ann in ANNOTATIONS:
    im = by_id[ann["image_id"]]
    x, y, w, h = ann["bbox"]
    if w <= 0 or h <= 0:
        continue
    cx = (x + w / 2) / im["width"]
    cy = (y + h / 2) / im["height"]
    nw, nh = w / im["width"], h / im["height"]
    # A fraction outside 0..1 is a broken box. Skip that box.
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < nw <= 1 and 0 < nh <= 1):
        continue
    labels[im["id"]].append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

groups = defaultdict(list)
for iid, im in by_id.items():
    if iid in labels:
        groups[group_key(im["file_name"])].append(iid)

keys = sorted(groups)
random.Random(42).shuffle(keys)
n = len(keys)
a, b = int(0.70 * n), int(0.90 * n)
split_keys = {"train": keys[:a], "val": keys[a:b], "test": keys[b:]}

sizes = [len(v) for v in groups.values()]
print(f"original photos (groups) = {len(groups)}")
print(f"copies per photo: min {min(sizes)}, max {max(sizes)}, most common {Counter(sizes).most_common(3)}")
if max(sizes) == 1:
    print("WARNING: no .rf. copies were found. Training can continue, but paste 8 file names")
    print("from the unzip listing into the chat before the long run so the grouping can be checked.")

for split, ks in split_keys.items():
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    for k in ks:
        for iid in groups[k]:
            im = by_id[iid]
            src = Path(im["abs_path"])
            stem = os.path.splitext(im["file_name"])[0].replace("/", "__").replace(" ", "_")
            dst = OUT / "images" / split / f"{stem}{src.suffix.lower()}"
            if dst.exists():
                raise SystemExit(f"Name collision: {dst}")
            shutil.copy(src, dst)
            (OUT / "labels" / split / f"{stem}.txt").write_text("\\n".join(labels[iid]), encoding="utf-8")

for split in ("train", "val", "test"):
    n_img = len(list((OUT / "images" / split).iterdir()))
    n_lbl = len(list((OUT / "labels" / split).iterdir()))
    print(f"{split:5}  images={n_img}  labels={n_lbl}")

# Show one label file so you can see the four fractions.
sample_label = next((OUT / "labels" / "train").iterdir())
print("example label", sample_label.name)
print(sample_label.read_text())
"""))

cells.append(md("""
## 7. Tell YOLO where the folders are

This small text file is the dataset config. `names: 0: number_plate` is why the trained model has one class. Ultralytics reads `train`, `val`, and `test` relative to `path`.
"""))

cells.append(code("""
YAML = OUT / "plate_data.yaml"
YAML.write_text(
    "path: /content/plates\\n"
    "train: images/train\\n"
    "val: images/val\\n"
    "test: images/test\\n"
    "names:\\n"
    "  0: number_plate\\n",
    encoding="utf-8",
)
print(YAML.read_text())
"""))

cells.append(md("""
## 8. Five-epoch check on the nano model

This is a wiring check, not the model you keep.

`YOLO("yolov8n.pt")` downloads the COCO nano weights (a few megabytes) the first time. Training then shows every **training** photo about 5 times and, after each pass, scores the **validation** photos.

What you want to see:

- The run starts and prints a table per epoch. `box_loss` on the training side goes down across the 5 epochs.
- Two pictures appear below the cell: the true boxes, and the boxes the nano model drew. The predicted boxes should sit roughly on plates, even if they are a bit loose. Five epochs is a short look.

If the run crashes with "no labels found", stop. If the predicted picture has boxes all over the road and none on plates, stop and send me that picture. Do not start the overnight run.

The weights from this cell are thrown away.
"""))

cells.append(code("""
from ultralytics import YOLO
from IPython.display import Image, display

smoke = YOLO("yolov8n.pt")
smoke.train(
    data=str(YAML),
    epochs=5,
    batch=16,
    imgsz=640,
    lr0=0.01,
    optimizer="auto",
    seed=42,
    device=0,
    project="/content/drive/MyDrive/PlateGuard/runs",
    name="smoke_nano",
    exist_ok=True,
    patience=5,
    workers=2,
    cache=True,
    plots=True,
    deterministic=True,
    hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
    degrees=5, translate=0.1, scale=0.4,
    fliplr=0.0,
)
smoke_dir = Path("/content/drive/MyDrive/PlateGuard/runs/smoke_nano")
for pic in ("val_batch0_labels.jpg", "val_batch0_pred.jpg"):
    path = smoke_dir / pic
    print(pic, "exists" if path.exists() else "MISSING")
    if path.exists():
        display(Image(str(path)))
"""))

cells.append(md("""
## 9. The real train: YOLOv8s, up to 50 epochs

Run this cell only after the pictures in cell 8 show boxes on plates.

What each argument does:

- `yolov8s.pt` — download the COCO **small** weights and start from them.
- `epochs=50` — upper limit. `patience=15` stops early if validation mAP has not improved for 15 epochs, and `best.pt` stays on the best epoch.
- `batch=16` — 16 photos at a time. If the cell dies with "CUDA out of memory", change 16 to 8 and run this cell again.
- `imgsz=640` — network input size. Matches the dataset.
- `lr0=0.01` — how big each nudge to the weights is. This is the Ultralytics starting value. Leave it.
- `optimizer="auto"` — Ultralytics picks AdamW for this setup. Write whichever name appears in the log into Table 4.6.
- `seed=42` and `deterministic=True` — a second run stays as close as possible to this one. The report asks for a recorded seed.
- `fliplr=0.0` — do not mirror photos again during training. A mirrored plate is a different character arrangement. NPDS already contains flipped copies, and those still teach the box location.
- `degrees=5, translate=0.1, scale=0.4` — small rotation, shift, and zoom. A roadside camera is not perfectly straight-on.
- `hsv_h/s/v` — small colour, saturation, and brightness changes, so day and evening look less different.
- `cache=True` — keep images in RAM so the GPU is not waiting on the disk.
- `project=.../PlateGuard/runs` — write `best.pt` onto Drive. `/content` is deleted when Colab disconnects. Drive is not.

Leave this browser tab open. A free session can still drop. Because the weights are on Drive, a dropped run keeps the last `best.pt`. Look at the epoch table. Training is about 4–6 hours.

The file you are making:

`MyDrive/PlateGuard/runs/plate_detector/weights/best.pt`
"""))

cells.append(code("""
model = YOLO("yolov8s.pt")
model.train(
    data=str(YAML),
    epochs=50,
    batch=16,
    imgsz=640,
    lr0=0.01,
    optimizer="auto",
    seed=42,
    device=0,
    project="/content/drive/MyDrive/PlateGuard/runs",
    name="plate_detector",
    exist_ok=True,
    patience=15,
    workers=2,
    cache=True,
    plots=True,
    deterministic=True,
    hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
    degrees=5, translate=0.1, scale=0.4,
    fliplr=0.0,
)
print("best weights:", "/content/drive/MyDrive/PlateGuard/runs/plate_detector/weights/best.pt")
"""))

cells.append(md("""
## 10. Read the validation score and pick the confidence threshold

Open `MyDrive/PlateGuard/runs/plate_detector/results.png` as well. You want the validation mAP line to rise and then flatten. The epoch where it is highest is the one saved as `best.pt`.

The app ignores any box whose confidence is below a threshold. The code default is **0.25**. This cell tries four thresholds on the **validation** set only and prints precision, recall, and F1.

F1 is the balance of precision and recall: `2 * P * R / (P + R)`. Use the threshold with the highest F1 in the app later. Write that number down. This does not touch the test set.

mAP for the report is **not** measured here. The official mAP uses a very low confidence (0.001) so the full precision-recall curve is scored. That happens in the next cell.
"""))

cells.append(code("""
import csv
best_path = "/content/drive/MyDrive/PlateGuard/runs/plate_detector/weights/best.pt"
run_dir = Path("/content/drive/MyDrive/PlateGuard/runs/plate_detector")
rows = list(csv.DictReader((run_dir / "results.csv").open()))
# Column names contain spaces in current Ultralytics. Strip them.
rows = [{k.strip(): v.strip() for k, v in row.items()} for row in rows]
last = rows[-1]
print("epochs completed:", len(rows))
print("last epoch precision", last.get("metrics/precision(B)"))
print("last epoch recall   ", last.get("metrics/recall(B)"))
print("last epoch mAP@50   ", last.get("metrics/mAP50(B)"))
print("last epoch mAP@50-95", last.get("metrics/mAP50-95(B)"))
print("The epoch inside best.pt is the best validation epoch, which may be earlier than the last row.")
display(Image(str(run_dir / "results.png")))

best = YOLO(best_path)
print("\\nconfidence threshold sweep on VALIDATION")
for conf in (0.15, 0.25, 0.35, 0.45):
    m = best.val(data=str(YAML), split="val", conf=conf, iou=0.45, plots=False, verbose=False)
    p, r = m.box.mp, m.box.mr
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    print(f"conf {conf:.2f}  precision {p:.4f}  recall {r:.4f}  F1 {f1:.4f}")
"""))

cells.append(md("""
## 11. Test set, once

Run this cell **one time**, after the threshold sweep. Running it again does not make the number more honest, and every extra look is a chance to change something and quietly fit the test set.

`conf=0.001` is only for this measurement. It is not the threshold you put in the app. The app threshold is the F1 winner from cell 10.

Copy the four printed numbers into the report. Also fill Table 4.6:

| Row | Value |
|---|---|
| YOLO version | YOLOv8s |
| Pretrained model | yolov8s.pt (COCO) |
| Epochs | the "epochs completed" number from cell 10 |
| Batch size | 16 (or 8 if you had to lower it) |
| Image size | 640 |
| Learning rate | 0.01 |
| Optimizer | the name printed in the training log |
| Confidence threshold | the F1 winner from cell 10 |
| NMS IoU threshold | 0.45 |
| Random seed | 42 |
| Training environment | Google Colab, Tesla T4 |

`best.pt` is already on your Drive. Download it from `PlateGuard/runs/plate_detector/weights/best.pt`. That is the model.
"""))

cells.append(code("""
metrics = best.val(data=str(YAML), split="test", conf=0.001, iou=0.7, plots=True,
                   project="/content/drive/MyDrive/PlateGuard/runs", name="test_once")
print(f"TEST  Precision {metrics.box.mp:.4f}  Recall {metrics.box.mr:.4f}  "
      f"mAP@50 {metrics.box.map50:.4f}  mAP@50-95 {metrics.box.map:.4f}")
print("best.pt:", best_path)
"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": "colab_train_plate.ipynb", "provenance": []},
    },
    "cells": cells,
}
out = Path(__file__).with_name("colab_train_plate.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out, "cells", len(cells))
