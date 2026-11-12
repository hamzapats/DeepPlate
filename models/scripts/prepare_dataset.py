"""Dataset preparation helpers (Section 4.1.2.6): COCO -> YOLO conversion and
leakage-free group-level splitting. Dataset paths/names are left blank on purpose."""
import json
import os
import random
import shutil
from collections import defaultdict

# ---- fill in -----------------------------------------------------------------
COCO_JSON   = "____________________"    # annotation file of the IURS/NPDS dataset
IMAGES_DIR  = "____________________"
OUT_DIR     = "____________________"
SPLIT       = (0.7, 0.2, 0.1)           # train / val / test
SEED        = 42
PLATE_CAT_IDS = None                    # None = treat every annotation as the plate class


def group_key(file_name: str) -> str:
    """Same original photo must always get the same key, so its copies stay in one split.

    NPDS was exported from Roboflow. Each photo was saved several times (flips,
    rotations, noise) under names like car12.rf.9f3ab1c0.jpg. The text before
    ".rf." is the original photo. If one copy is in train and another is in test,
    the test score is dishonest: the model has already seen that plate.
    """
    stem = os.path.splitext(os.path.basename(file_name))[0]
    if ".rf." in stem:
        stem = stem.split(".rf.")[0]
    return stem
# ------------------------------------------------------------------------------


def unique_stem(file_name: str) -> str:
    """Flat, collision-free name from the file's relative path, so cam1/001.jpg and
    cam2/001.jpg become cam1__001 and cam2__001 instead of overwriting each other."""
    rel = os.path.splitext(os.path.normpath(file_name))[0]
    return rel.replace(os.sep, "__").replace("/", "__").replace(" ", "_")


def coco_to_yolo():
    with open(COCO_JSON, encoding="utf-8") as f:
        coco = json.load(f)
    imgs = {i["id"]: i for i in coco["images"]}
    labels = defaultdict(list)
    for a in coco["annotations"]:
        if PLATE_CAT_IDS and a["category_id"] not in PLATE_CAT_IDS:
            continue
        im = imgs[a["image_id"]]
        x, y, w, h = a["bbox"]           # COCO: top-left x, y, width, height (pixels)
        if w <= 0 or h <= 0:
            continue
        cx, cy = (x + w / 2) / im["width"], (y + h / 2) / im["height"]
        labels[im["id"]].append(f"0 {cx:.6f} {cy:.6f} {w / im['width']:.6f} {h / im['height']:.6f}")
    return imgs, labels


def group_split(imgs, labels):
    groups = defaultdict(list)
    for iid, im in imgs.items():
        if iid in labels:
            groups[group_key(im["file_name"])].append(iid)
    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)
    n = len(keys)
    a, b = int(SPLIT[0] * n), int((SPLIT[0] + SPLIT[1]) * n)
    return {"train": keys[:a], "val": keys[a:b], "test": keys[b:]}, groups


def write_dataset():
    imgs, labels = coco_to_yolo()
    split_keys, groups = group_split(imgs, labels)
    for split, keys in split_keys.items():
        os.makedirs(f"{OUT_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUT_DIR}/labels/{split}", exist_ok=True)
        for k in keys:
            for iid in groups[k]:
                fn = imgs[iid]["file_name"]
                stem = unique_stem(fn)
                ext = os.path.splitext(fn)[1]
                dst_img = f"{OUT_DIR}/images/{split}/{stem}{ext}"
                if os.path.exists(dst_img):                  # never silently overwrite
                    raise FileExistsError(f"Name collision: {dst_img}")
                shutil.copy(os.path.join(IMAGES_DIR, fn), dst_img)
                with open(f"{OUT_DIR}/labels/{split}/{stem}.txt", "w", encoding="utf-8") as lf:
                    lf.write("\n".join(labels[iid]))
    print({s: sum(len(groups[k]) for k in ks) for s, ks in split_keys.items()})


if __name__ == "__main__":
    write_dataset()
