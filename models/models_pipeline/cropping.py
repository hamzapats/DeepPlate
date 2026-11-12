"""Plate cropping with margin and boundary clamping (Eq. 4.1, Algorithm 4.2)."""
from typing import Optional, Tuple

import numpy as np

from .config import CropConfig


def clamp_box(box: Tuple[int, int, int, int], margin: float,
              frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - margin * w))
    x2 = int(min(frame_w, x2 + margin * w))
    y1 = int(max(0, y1 - margin * h))
    y2 = int(min(frame_h, y2 + margin * h))
    return x1, y1, x2, y2


def crop_plate(frame: np.ndarray, box: Tuple[int, int, int, int],
               cfg: CropConfig) -> Optional[np.ndarray]:
    """Crop from the ORIGINAL full-resolution frame. Returns None for invalid boxes."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, cfg.margin, W, H)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[1] < cfg.min_width or crop.shape[0] < cfg.min_height:
        return None
    return crop.copy()
