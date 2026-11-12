"""Unit tests that need no models or datasets (T03, T06, T12 helpers)."""
import numpy as np
import pytest

from models_pipeline.config import CropConfig, PreprocessConfig, VideoConfig
from models_pipeline.cropping import clamp_box, crop_plate
from models_pipeline.ocr_engine import OCRLine
from models_pipeline.preprocess import preprocess
from models_pipeline.results import deduplicate, merge_lines, normalise_text, PlateResult
from models_pipeline.video_input import VideoOpenError, validate_video


def test_clamp_inside_frame():
    assert clamp_box((0, 0, 100, 50), 0.1, 120, 60) == (0, 0, 110, 55)
    x1, y1, x2, y2 = clamp_box((90, 40, 120, 60), 0.5, 120, 60)
    assert x2 <= 120 and y2 <= 60 and x1 >= 0 and y1 >= 0


def test_crop_invalid_and_valid():
    frame = np.zeros((100, 200, 3), np.uint8)
    cfg = CropConfig()
    assert crop_plate(frame, (50, 50, 50, 60), cfg) is None
    assert crop_plate(frame, (10, 10, 110, 50), cfg).shape[0] > 0


def test_normalise():
    assert normalise_text("od 02-ab.1234") == "OD02AB1234"


def test_two_line_merge_and_weighted_conf():
    lines = [OCRLine("2519", 0.80, 50, 70, 20), OCRLine("MH 05", 0.90, 50, 20, 20),
             OCRLine("AJ", 0.70, 90, 20, 20)]
    r = merge_lines(lines)
    assert r.plate_text == "MH05AJ2519"
    assert r.ocr_confidence == pytest.approx((4 * .9 + 2 * .7 + 4 * .8) / 10)


def test_empty_merge_none():
    assert merge_lines([]) is None


def test_merge_lines_excludes_country_mark():
    lines = [OCRLine("HR26DN5018", 1.0, 60, 30, 12), OCRLine("IND", 1.0, 60, 45, 8)]
    result = merge_lines(lines)
    assert result.plate_text == "HR26DN5018"
    assert result.ocr_confidence == pytest.approx(1.0)


def test_dedup_keeps_best():
    out = deduplicate([PlateResult("A1", .5), PlateResult("A1", .9), PlateResult("B2", .6)])
    assert {r.plate_text: r.ocr_confidence for r in out} == {"A1": .9, "B2": .6}


def test_preprocess_returns_bgr_uint8():
    img = np.random.randint(0, 255, (20, 80, 3), np.uint8)
    cfg = PreprocessConfig(grayscale=True, clahe=True, denoise="bilateral")
    out = preprocess(img, cfg)
    assert out.ndim == 3 and out.shape[2] == 3 and out.dtype == np.uint8
    assert out.shape[0] == cfg.target_height          # small crop upscaled


def test_invalid_video():
    with pytest.raises(VideoOpenError):
        validate_video("does_not_exist.mp4", VideoConfig())
    with pytest.raises(VideoOpenError):
        validate_video(__file__, VideoConfig())          # .py not allowed
