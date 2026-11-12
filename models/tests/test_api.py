import io

import cv2
import numpy as np
from fastapi.testclient import TestClient

from api import create_app


class FakeInference:
    def predict(self, frame):
        if frame.mean() == 0:
            return []
        return [{
            "plate_text": "OD02AB1234",
            "detection_confidence": 0.98,
            "ocr_confidence": 0.96,
            "bbox": {"x1": 10, "y1": 20, "x2": 90, "y2": 50},
        }]


def encoded_image(value):
    image = np.full((80, 120, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_health_reports_loaded_model():
    with TestClient(create_app(FakeInference())) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_detection_and_ocr_fields():
    with TestClient(create_app(FakeInference())) as client:
        response = client.post(
            "/api/predict",
            files={"file": ("vehicle.jpg", encoded_image(200), "image/jpeg")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "vehicle.jpg"
    assert body["image_size"] == {"width": 120, "height": 80}
    assert body["plates"][0]["plate_text"] == "OD02AB1234"
    assert body["plates"][0]["detection_confidence"] == 0.98
    assert body["plates"][0]["ocr_confidence"] == 0.96
    assert body["plates"][0]["bbox"] == {"x1": 10, "y1": 20, "x2": 90, "y2": 50}


def test_predict_returns_empty_list_when_no_plate_detected():
    with TestClient(create_app(FakeInference())) as client:
        response = client.post(
            "/api/predict",
            files={"file": ("empty.jpg", encoded_image(0), "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.json()["plates"] == []


def test_predict_rejects_non_image_upload():
    with TestClient(create_app(FakeInference())) as client:
        response = client.post(
            "/api/predict",
            files={"file": ("not-image.txt", io.BytesIO(b"not an image"), "text/plain")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Upload must be a valid JPEG, PNG, or WebP image."
