"""Evaluation (Section 4.2.4).
 1) Detection : precision, recall, mAP@50, mAP@50-95  (ultralytics val on the test split)
 2) OCR       : exact-match accuracy and CER on a manifest CSV (image_path,text)
 3) System    : latency + FPS on a video
"""
import argparse
import csv
import time

import cv2

from models_pipeline.config import PlateGuardConfig
from models_pipeline.ocr_engine import PlateOCR
from models_pipeline.preprocess import preprocess
from models_pipeline.results import merge_lines, normalise_text


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def eval_detection(weights, data_yaml):
    from ultralytics import YOLO
    m = YOLO(weights).val(data=data_yaml, split="test", conf=0.001)
    print(f"Precision {m.box.mp:.4f} | Recall {m.box.mr:.4f} | "
          f"mAP@50 {m.box.map50:.4f} | mAP@50-95 {m.box.map:.4f}")


def eval_ocr(manifest):
    cfg = PlateGuardConfig()
    ocr = PlateOCR(cfg.ocr)
    n = correct = edits = chars = 0
    t_ocr = 0.0
    with open(manifest, newline="", encoding="utf-8") as mf:
        rows = list(csv.reader(mf))
    for row in rows:
        path, gt = row[0], normalise_text(row[1])
        img = cv2.imread(path)
        if img is None or not gt:
            continue
        t = time.time()
        res = merge_lines(ocr.recognise(preprocess(img, cfg.preprocess)))
        t_ocr += time.time() - t
        pred = res.plate_text if res else ""
        n += 1
        correct += pred == gt
        edits += levenshtein(pred, gt)
        chars += len(gt)
    print(f"Samples {n} | Exact accuracy {correct / max(n, 1):.4f} | "
          f"CER {edits / max(chars, 1):.4f} | OCR time/plate {1000 * t_ocr / max(n, 1):.1f} ms")


def eval_system(weights, video):
    from models_pipeline import PlateGuard
    cfg = PlateGuardConfig()
    cfg.detector.weights_path = weights
    pg = PlateGuard(cfg)
    t = time.time()
    res = pg.process_video(video)
    print(f"{len(res)} unique plates | total time {time.time() - t:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["detection", "ocr", "system"])
    ap.add_argument("--weights")
    ap.add_argument("--data", default="configs/plate_data.yaml")
    ap.add_argument("--manifest", help="CSV: image_path,text")
    ap.add_argument("--video")
    a = ap.parse_args()
    {"detection": lambda: eval_detection(a.weights, a.data),
     "ocr": lambda: eval_ocr(a.manifest),
     "system": lambda: eval_system(a.weights, a.video)}[a.task]()
