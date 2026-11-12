"""Minimal, configurable preprocessing of a plate crop (Table 4.1).
Every step is optional so you can ablate them; the output is always 3-channel BGR
because PaddleOCR expects a colour image array."""
from typing import Optional

import cv2
import numpy as np

from .config import PreprocessConfig


def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype("float32")
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")


def correct_perspective(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Optional: warp to fronto-parallel view when 4 plate corners are known."""
    tl, tr, br, bl = order_corners(corners)
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype="float32"), dst)
    return cv2.warpPerspective(img, M, (w, h))


def preprocess(crop: np.ndarray, cfg: PreprocessConfig,
               corners: Optional[np.ndarray] = None) -> np.ndarray:
    img = crop
    if cfg.perspective_correction and corners is not None:
        img = correct_perspective(img, corners)

    if cfg.resize:
        h = img.shape[0]
        if not (cfg.only_upscale_small and h >= cfg.target_height):
            scale = cfg.target_height / h
            interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)

    if cfg.grayscale or cfg.clahe or cfg.denoise:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        if cfg.clahe:
            g = cfg.clahe_tile_grid
            gray = cv2.createCLAHE(cfg.clahe_clip_limit, (g, g)).apply(gray)
        if cfg.denoise == "gaussian":
            k = cfg.gaussian_kernel | 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        elif cfg.denoise == "bilateral":
            gray = cv2.bilateralFilter(gray, cfg.bilateral_d, cfg.bilateral_sigma, cfg.bilateral_sigma)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return np.ascontiguousarray(img, dtype=np.uint8)   # PaddleOCR normalises internally
