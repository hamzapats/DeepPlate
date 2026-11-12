"""HTTP API for number-plate image inference."""
import logging
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Annotated, Optional

os.environ.setdefault("PADDLE_PDX_CACHE_HOME", os.path.join(tempfile.gettempdir(), "paddlex"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# Ensure models directory is on sys.path so models_pipeline is always importable
MODELS_DIR = Path(__file__).resolve().parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

log = logging.getLogger("plateguard.api")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "best.pt"


class PlateInference:
    """Loads detector and OCR once, then reuses them for each request."""

    def __init__(self, weights_path: Optional[Path] = None):
        weights = weights_path or Path(os.environ.get("PLATEGUARD_WEIGHTS", DEFAULT_WEIGHTS))
        if not weights.is_file():
            raise FileNotFoundError(
                f"YOLO weights not found: {weights}. "
                "Place best.pt in models/weights/ or set PLATEGUARD_WEIGHTS."
            )

        from models_pipeline.config import (
            CropConfig,
            DetectorConfig,
            OCRConfig,
            PlateGuardConfig,
            PreprocessConfig,
        )
        from models_pipeline.detector import PlateDetector
        from models_pipeline.ocr_engine import PlateOCR

        config = PlateGuardConfig(
            detector=DetectorConfig(weights_path=str(weights), conf_threshold=0.25),
            crop=CropConfig(),
            preprocess=PreprocessConfig(),
            ocr=OCRConfig(device=os.environ.get("PLATEGUARD_OCR_DEVICE", "cpu")),
        )
        self.detector = PlateDetector(config.detector)
        self.ocr = PlateOCR(config.ocr)
        self.crop_config = config.crop
        self.preprocess_config = config.preprocess
        self.ocr_config = config.ocr
        self._inference_lock = Lock()

    def predict(self, frame: np.ndarray) -> list[dict]:
        from models_pipeline.cropping import crop_plate
        from models_pipeline.preprocess import preprocess
        from models_pipeline.results import merge_lines

        plates = []
        with self._inference_lock:
            detections = self.detector.detect(frame)
            for detection in detections:
                plate = {
                    "plate_text": None,
                    "detection_confidence": round(detection.confidence, 4),
                    "ocr_confidence": None,
                    "bbox": {
                        "x1": detection.x1,
                        "y1": detection.y1,
                        "x2": detection.x2,
                        "y2": detection.y2,
                    },
                }
                crop = crop_plate(frame, detection.box, self.crop_config)
                if crop is not None:
                    lines = self.ocr.recognise(preprocess(crop, self.preprocess_config))
                    result = merge_lines(lines, self.ocr_config.min_plate_chars)
                    if result is not None and result.ocr_confidence >= self.ocr_config.min_confidence:
                        plate["plate_text"] = result.plate_text
                        plate["ocr_confidence"] = round(result.ocr_confidence, 4)
                plates.append(plate)
        return plates


def create_app(inference: Optional[PlateInference] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.inference = inference if inference is not None else PlateInference()
        yield

    app = FastAPI(title="PlateGuard Inference API", version="1.0.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"status": "ok", "model_loaded": hasattr(app.state, "inference")}

    @app.post("/api/predict")
    async def predict(file: Annotated[UploadFile, File(...)]):
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded image is empty.")
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds the 10 MB upload limit.")

        image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Upload must be a valid JPEG, PNG, or WebP image.")

        height, width = image.shape[:2]
        if height * width > MAX_IMAGE_PIXELS:
            raise HTTPException(status_code=413, detail="Image exceeds the 20-megapixel limit.")

        plates = await run_in_threadpool(app.state.inference.predict, image)
        return {
            "filename": file.filename,
            "image_size": {"width": width, "height": height},
            "plates": plates,
        }

    return app


app = create_app()
