"""Usage:
    python main.py --video traffic.mp4 --weights weights/plate_best.pt
    python main.py --image car.jpg    --weights weights/plate_best.pt
"""
import argparse
import json
import logging

from models_pipeline import PlateGuard
from models_pipeline.config import PlateGuardConfig


def main():
    ap = argparse.ArgumentParser(description="PlateGuard - ANPR")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--image")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--every", type=int, default=5, help="process every k-th frame")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--ocr-device", default="cpu", help="cpu | gpu:0")
    ap.add_argument("--debug-crops", default=None)
    ap.add_argument("--out", default=None, help="save results as JSON")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler("models.log")])

    cfg = PlateGuardConfig()
    cfg.detector.weights_path = a.weights
    cfg.detector.conf_threshold = a.conf
    cfg.video.frame_interval = a.every
    cfg.ocr.device = a.ocr_device
    cfg.save_debug_crops_dir = a.debug_crops

    pg = PlateGuard(cfg)
    results = pg.process_video(a.video) if a.video else [r.to_dict() for r in pg.read_image(a.image)]

    print(f"{'Number Plate Text':<20}OCR Confidence")
    for r in results:
        print(f"{r['plate_text']:<20}{r['ocr_confidence']:.2f}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
