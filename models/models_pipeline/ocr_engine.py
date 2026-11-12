"""PaddleOCR wrapper. Works with PaddleOCR 3.x (predict) and falls back to 2.x (ocr)."""
import os
import tempfile

os.environ.setdefault("PADDLE_PDX_CACHE_HOME", os.path.join(tempfile.gettempdir(), "paddlex"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from dataclasses import dataclass
from typing import List

import numpy as np

from .config import OCRConfig


@dataclass
class OCRLine:
    text: str
    confidence: float
    x_center: float
    y_center: float
    height: float


class PlateOCR:
    def __init__(self, cfg: OCRConfig):
        self.cfg = cfg
        self.engine = None
        self.v3 = False
        try:
            from paddleocr import PaddleOCR
            try:    # PaddleOCR 3.x
                self.engine = PaddleOCR(
                    lang=cfg.lang,
                    use_doc_orientation_classify=cfg.use_doc_orientation_classify,
                    use_doc_unwarping=cfg.use_doc_unwarping,
                    use_textline_orientation=cfg.use_textline_orientation,
                    device=cfg.device,
                    enable_mkldnn=cfg.enable_mkldnn,
                )
                self.v3 = True
            except TypeError:   # PaddleOCR 2.x
                self.engine = PaddleOCR(lang=cfg.lang, use_angle_cls=cfg.use_textline_orientation,
                                        use_gpu=cfg.device.startswith("gpu"), show_log=False)
                self.v3 = False
        except (ImportError, Exception) as e:
            import logging
            logging.getLogger("plateguard.ocr").warning("PaddleOCR not available (%s).", e)

    def recognise(self, image: np.ndarray) -> List[OCRLine]:
        if self.engine is None:
            return []
        lines: List[OCRLine] = []
        if self.v3:
            for res in self.engine.predict(image):
                # In PaddleOCR 3.x, res.json is a callable method, not a dict.
                if hasattr(res, "json") and callable(res.json):
                    data = res.json().get("res", {})
                elif hasattr(res, "json"):
                    data = res.json.get("res", {}) if isinstance(res.json, dict) else {}
                else:
                    data = res
                texts = data.get("rec_texts", [])
                scores = data.get("rec_scores", [])
                polys = data.get("rec_polys", data.get("dt_polys", []))
                for t, s, p in zip(texts, scores, polys):
                    lines.append(self._line(t, s, p))
        else:
            out = self.engine.ocr(image, cls=self.cfg.use_textline_orientation)
            for page in out or []:
                for poly, (t, s) in page or []:
                    lines.append(self._line(t, s, poly))
        return lines

    @staticmethod
    def _line(text, score, poly) -> OCRLine:
        p = np.asarray(poly, dtype=float).reshape(-1, 2)
        return OCRLine(str(text), float(score), float(p[:, 0].mean()),
                       float(p[:, 1].mean()), float(p[:, 1].max() - p[:, 1].min()))
