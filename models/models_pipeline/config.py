"""Central configuration (Tables 4.1, 4.6, 4.7 of the report). Fill the blanks later."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VideoConfig:
    frame_interval: int = 5                       # k : process every k-th frame
    allowed_ext: tuple = (".mp4", ".avi", ".mov", ".mkv")
    max_file_size_mb: int = 2048


@dataclass
class DetectorConfig:
    weights_path: str = "____________________"    # e.g. "weights/plate_best.pt"
    conf_threshold: float = 0.25                  # choose on validation set
    nms_iou_threshold: float = 0.45
    img_size: int = 640
    device: Optional[str] = None                  # None = auto, "cpu", "0", ...


@dataclass
class CropConfig:
    margin: float = 0.05                          # alpha in Eq. 4.1
    min_width: int = 20
    min_height: int = 8


@dataclass
class PreprocessConfig:
    """Pipeline diagram: 'minimal OpenCV/NumPy processing -- additional correction
    only when needed'. So everything beyond the crop is OFF by default and can be
    switched on after ablation tests (Table 4.1)."""
    resize: bool = True
    target_height: int = 64                       # only used when the crop is smaller
    only_upscale_small: bool = True
    grayscale: bool = False
    clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8
    denoise: Optional[str] = None                 # None | "gaussian" | "bilateral"
    gaussian_kernel: int = 3
    bilateral_d: int = 5
    bilateral_sigma: float = 50.0
    perspective_correction: bool = False          # needs 4 corners; optional


@dataclass
class OCRConfig:
    lang: str = "en"
    use_textline_orientation: bool = True         # PaddleOCR orientation module
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    device: str = "cpu"                           # "cpu" | "gpu:0"
    enable_mkldnn: bool = False
    min_confidence: float = 0.50                  # reject readings below this
    min_plate_chars: int = 4


@dataclass
class PlateGuardConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    deduplicate: bool = True                      # keep best reading per plate text
    save_debug_crops_dir: Optional[str] = None
