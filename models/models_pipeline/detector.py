"""YOLO number-plate detector. Output matches the expected format:
x1, y1, x2, y2, confidence, class_id, class_name."""
from dataclasses import dataclass
from typing import List

import numpy as np
from ultralytics import YOLO

from .config import DetectorConfig


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str

    @property
    def box(self):
        return (self.x1, self.y1, self.x2, self.y2)


class PlateDetector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.model = YOLO(cfg.weights_path)       # loaded once, reused for every frame

    def detect(self, frame: np.ndarray) -> List[Detection]:
        res = self.model.predict(
            frame,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.nms_iou_threshold,       # NMS IoU
            imgsz=self.cfg.img_size,
            device=self.cfg.device,
            verbose=False,
        )[0]
        names = res.names
        out: List[Detection] = []
        if res.boxes is None:
            return out
        for xyxy, conf, cls in zip(res.boxes.xyxy.cpu().numpy(),
                                   res.boxes.conf.cpu().numpy(),
                                   res.boxes.cls.cpu().numpy()):
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]   # original-frame coords
            cid = int(cls)
            out.append(Detection(x1, y1, x2, y2, float(conf), cid, names.get(cid, str(cid))))
        return out
