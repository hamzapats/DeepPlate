"""Line ordering, text normalisation and confidence aggregation (Eq. 4.2)."""
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .ocr_engine import OCRLine


@dataclass
class PlateResult:
    plate_text: str
    ocr_confidence: float

    def to_dict(self) -> Dict:
        return {"plate_text": self.plate_text, "ocr_confidence": round(self.ocr_confidence, 4)}


def normalise_text(text: str) -> str:
    """Upper case; whitespace and punctuation removed."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def order_lines(lines: List[OCRLine]) -> List[OCRLine]:
    """Top-to-bottom, then left-to-right, so two-line plates read as one string."""
    if not lines:
        return []
    med_h = max(float(np.median([l.height for l in lines])), 1.0)
    lines = sorted(lines, key=lambda l: l.y_center)
    rows, cur = [], [lines[0]]
    for l in lines[1:]:
        if abs(l.y_center - np.mean([c.y_center for c in cur])) < 0.5 * med_h:
            cur.append(l)
        else:
            rows.append(cur)
            cur = [l]
    rows.append(cur)
    return [l for r in rows for l in sorted(r, key=lambda x: x.x_center)]


def merge_lines(lines: List[OCRLine], min_chars: int = 1) -> Optional[PlateResult]:
    """Join ordered lines; confidence = character-weighted mean (Eq. 4.2)."""
    texts, weights, confs = [], [], []
    for l in order_lines(lines):
        t = normalise_text(l.text)
        if t and t != "IND":
            texts.append(t)
            weights.append(len(t))
            confs.append(l.confidence)
    text = "".join(texts)
    # Strip leading "IND" country identifier if followed by state code (e.g. INDRJ14CV0002 -> RJ14CV0002)
    cleaned_text = re.sub(r"^IND(?=[A-Z]{2}\d)", "", text)
    if cleaned_text:
        text = cleaned_text
    if len(text) < min_chars:
        return None
    conf = float(np.average(confs, weights=weights))
    return PlateResult(text, conf)


def deduplicate(results: List[PlateResult]) -> List[PlateResult]:
    """Optional: keep the highest-confidence record per plate text
    (a simple stand-in for tracking / multi-frame aggregation)."""
    best: Dict[str, PlateResult] = {}
    for r in results:
        if r.plate_text not in best or r.ocr_confidence > best[r.plate_text].ocr_confidence:
            best[r.plate_text] = r
    return list(best.values())
