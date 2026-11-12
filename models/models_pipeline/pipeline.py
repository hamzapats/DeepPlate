"""Integrated pipeline (Algorithm 4.3):
Video -> OpenCV frames -> YOLO -> crop -> preprocess -> PaddleOCR -> {plate_text, ocr_confidence}"""
import logging
import os
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from .config import PlateGuardConfig
from .cropping import crop_plate
from .detector import PlateDetector
from .ocr_engine import PlateOCR
from .preprocess import preprocess
from .results import PlateResult, deduplicate, merge_lines
from .video_input import extract_frames

log = logging.getLogger("models")


class PlateGuard:
    def __init__(self, cfg: Optional[PlateGuardConfig] = None):
        self.cfg = cfg or PlateGuardConfig()
        self.detector = PlateDetector(self.cfg.detector)   # models loaded once
        self.ocr = PlateOCR(self.cfg.ocr)

    # ---- single image / frame ----------------------------------------------
    def read_frame(self, frame: np.ndarray, frame_idx: int = 0) -> List[PlateResult]:
        detections = self.detector.detect(frame)
        if not detections:
            log.debug("no plate in frame %d", frame_idx)          # T11
            return []
        out: List[PlateResult] = []
        for n, det in enumerate(detections):                      # T05 / FR4
            try:
                crop = crop_plate(frame, det.box, self.cfg.crop)
                if crop is None:
                    log.warning("invalid box %s in frame %d", det.box, frame_idx)
                    continue
                img = preprocess(crop, self.cfg.preprocess)
                self._save_debug(img, frame_idx, n)
                lines = self.ocr.recognise(img)
                if not lines:
                    log.warning("OCR failure in frame %d", frame_idx)
                    continue
                res = merge_lines(lines, self.cfg.ocr.min_plate_chars)
                if res is None or res.ocr_confidence < self.cfg.ocr.min_confidence:
                    continue
                out.append(res)
            except Exception:                                      # one bad crop never stops the video
                log.exception("error on frame %d, detection %d", frame_idx, n)
        return out

    def read_image(self, path: str) -> List[PlateResult]:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return self.read_frame(img)

    # ---- video ----------------------------------------------------------------
    def process_video(self, video_path: str) -> List[Dict]:
        t0, n_frames = time.time(), 0
        results: List[PlateResult] = []
        for idx, ts, frame in extract_frames(video_path, self.cfg.video):
            n_frames += 1
            for r in self.read_frame(frame, idx):
                log.info("frame %d (%.2fs): %s (%.2f)", idx, ts, r.plate_text, r.ocr_confidence)
                results.append(r)
        if self.cfg.deduplicate:
            results = deduplicate(results)
        elapsed = time.time() - t0
        log.info("processed %d frames in %.1fs (%.2f fps)", n_frames, elapsed, n_frames / max(elapsed, 1e-6))
        return [r.to_dict() for r in results]

    def _save_debug(self, img, frame_idx, n):
        d = self.cfg.save_debug_crops_dir
        if d:
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, f"f{frame_idx}_p{n}.png"), img)
