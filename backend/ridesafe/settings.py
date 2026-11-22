"""API settings, read from environment variables (prefix RIDESAFE_), with defaults for a local run.

    RIDESAFE_MODEL              helmet model (ONNX)                         models/best.onnx
    RIDESAFE_VEHICLE_MODEL      COCO vehicle model for plate reading        models/vehicles.onnx
    RIDESAFE_THRESHOLDS         per-class thresholds                        configs/thresholds.yaml
    RIDESAFE_TRACKER            ByteTrack settings                          configs/bytetrack.yaml
    RIDESAFE_DATA_DIR           uploads, job results and the SQLite log    results/api
    RIDESAFE_SAMPLE_DIR         a pre-computed, read-only sample job        (none)
    RIDESAFE_DEMO               public demo: masked plates, visitors review only their own uploads   false
    RIDESAFE_PLATES             read number plates (needs rapidocr_onnxruntime + the vehicle model)   true
    RIDESAFE_MAX_IMAGE_MB       upload limit for /detect                    10
    RIDESAFE_MAX_VIDEO_MB       upload limit for /jobs/video                100
    RIDESAFE_MAX_VIDEO_SECONDS  longest video accepted                      120
    RIDESAFE_MAX_QUEUED         videos allowed to wait at once (more: 429) 10
    RIDESAFE_VID_STRIDE         analyse every Nth video frame               2
    RIDESAFE_UPLOAD_TTL_S       delete uploads and their results after this many seconds (0 = keep)   3600
    RIDESAFE_CORS_ORIGINS       comma-separated origins allowed to call the API                       *
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name, default) -> str:
    return os.environ.get(f"RIDESAFE_{name}", str(default))


def _flag(name, default):
    return _env(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    model: str = field(default_factory=lambda: _env("MODEL", "models/best.onnx"))
    vehicle_model: str = field(default_factory=lambda: _env("VEHICLE_MODEL", "models/vehicles.onnx"))
    thresholds: str = field(default_factory=lambda: _env("THRESHOLDS", "configs/thresholds.yaml"))
    tracker: str = field(default_factory=lambda: _env("TRACKER", "configs/bytetrack.yaml"))
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "results/api")))
    sample_dir: str = field(default_factory=lambda: _env("SAMPLE_DIR", ""))
    demo: bool = field(default_factory=lambda: _flag("DEMO", False))
    plates: bool = field(default_factory=lambda: _flag("PLATES", True))
    max_image_mb: float = field(default_factory=lambda: float(_env("MAX_IMAGE_MB", 10)))
    max_video_mb: float = field(default_factory=lambda: float(_env("MAX_VIDEO_MB", 100)))
    max_video_seconds: float = field(default_factory=lambda: float(_env("MAX_VIDEO_SECONDS", 120)))
    max_queued: int = field(default_factory=lambda: int(_env("MAX_QUEUED", 10)))
    vid_stride: int = field(default_factory=lambda: int(_env("VID_STRIDE", 2)))
    upload_ttl_s: int = field(default_factory=lambda: int(_env("UPLOAD_TTL_S", 3600)))
    cors_origins: tuple = field(default_factory=lambda: tuple(
        o.strip() for o in _env("CORS_ORIGINS", "*").split(",") if o.strip()))
