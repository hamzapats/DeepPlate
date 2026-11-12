"""Video validation and frame extraction (Algorithm 4.1)."""
import logging
import os
from typing import Iterator, Tuple

import cv2
import numpy as np

from .config import VideoConfig

log = logging.getLogger("models")


class VideoOpenError(Exception):
    pass


def validate_video(path: str, cfg: VideoConfig) -> None:
    """T01 / T02: accept valid video, raise clear error otherwise."""
    if not os.path.isfile(path):
        raise VideoOpenError("Video file not found.")
    if os.path.splitext(path)[1].lower() not in cfg.allowed_ext:
        raise VideoOpenError(f"Unsupported file type. Allowed: {cfg.allowed_ext}")
    if os.path.getsize(path) > cfg.max_file_size_mb * 1024 * 1024:
        raise VideoOpenError("Video file is too large.")


def extract_frames(path: str, cfg: VideoConfig) -> Iterator[Tuple[int, float, np.ndarray]]:
    """Yield (frame_index, timestamp_seconds, BGR frame) for every k-th frame."""
    validate_video(path, cfg)
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise VideoOpenError("Video could not be opened (corrupt or unsupported codec).")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        # Some formats (MKV, streams) report FRAME_COUNT=0; that is not an error.
        if total == 0:
            log.warning("Video FRAME_COUNT unavailable — will read until EOF (streaming/container format).")
        log.info("Video: %dx%d, %.2f fps, %d frames",
                 int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), fps, total)
        i = 0
        k = max(1, cfg.frame_interval)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % k == 0:
                yield i, i / fps, frame          # Eq. 2.4: t = i / fps
            i += 1
    finally:
        cap.release()
