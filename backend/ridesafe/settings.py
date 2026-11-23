"""API settings, read from environment variables (prefix RIDESAFE_), with defaults for a local run.

    RIDESAFE_MODEL              helmet model (ONNX)                         models/best.onnx
    RIDESAFE_VEHICLE_MODEL      COCO vehicle model for plate reading        models/vehicles.onnx
    RIDESAFE_THRESHOLDS         per-class thresholds                        configs/thresholds.yaml
    RIDESAFE_TRACKER            ByteTrack settings                          configs/bytetrack.yaml
    RIDESAFE_DATA_DIR           uploads, job results and the SQLite log    results/api
    RIDESAFE_SAMPLE_DIR         a pre-computed, read-only sample job        (none)
    RIDESAFE_ADMIN_TOKEN        the author's secret key (32+ characters): sees and edits everything   (none)
    RIDESAFE_DEMO               public site: plates masked and plate crops hidden for visitors       true
    RIDESAFE_PLATES             read number plates (needs rapidocr_onnxruntime + the vehicle model)   true
    RIDESAFE_RIDING_CHECK       only heads on a motorcycle can be violators (vehicle model; pedestrians and cyclists
                                are labelled 'not riding' / 'cyclist')                                  true
    RIDESAFE_DEDUPE_GAP         a violator lost for up to this many analysed frames and picked up again at the same
                                place is linked to the first log, not logged twice (0 = off)            60
    RIDESAFE_MAX_IMAGE_MB       upload limit for /detect                    10
    RIDESAFE_MAX_VIDEO_MB       upload limit for /jobs/video                100
    RIDESAFE_MAX_VIDEO_SECONDS  longest video accepted                      120
    RIDESAFE_MAX_QUEUED         videos allowed to wait at once (more: 429) 10
    RIDESAFE_VID_STRIDE         analyse every Nth video frame               2
    RIDESAFE_UPLOAD_TTL_S       delete uploads and their results after this many seconds (0 = keep)   3600
    RIDESAFE_CORS_ORIGINS       comma-separated origins allowed to call the API                       *
    RIDESAFE_CORS_ORIGIN_REGEX  also allow origins matching this regex (e.g. the web app's PR previews)  (none)
    RIDESAFE_RATE_DETECT        photo checks per minute per visitor (0 = no limit)                    30
    RIDESAFE_RATE_UPLOADS       video uploads per hour per visitor (0 = no limit)                     10
    RIDESAFE_URL_SECRET         key that signs image links (default: random at each start)
    RIDESAFE_TRUSTED_PROXY      behind a reverse proxy (nginx, Azure ingress): take the client address from
                                the last X-Forwarded-For entry, which the proxy itself adds           false
    RIDESAFE_LOG_LEVEL          log detail on stdout: DEBUG / INFO / WARNING                         INFO

Visitors only ever see and change their own uploads (and the read-only sample), whatever the mode; the
author, identified by RIDESAFE_ADMIN_TOKEN, can see and change everything and has no rate limits.
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
    admin_token: str = field(default_factory=lambda: _env("ADMIN_TOKEN", ""))
    demo: bool = field(default_factory=lambda: _flag("DEMO", True))
    plates: bool = field(default_factory=lambda: _flag("PLATES", True))
    riding_check: bool = field(default_factory=lambda: _flag("RIDING_CHECK", True))
    dedupe_gap: int = field(default_factory=lambda: int(_env("DEDUPE_GAP", 60)))
    max_image_mb: float = field(default_factory=lambda: float(_env("MAX_IMAGE_MB", 10)))
    max_video_mb: float = field(default_factory=lambda: float(_env("MAX_VIDEO_MB", 100)))
    max_video_seconds: float = field(default_factory=lambda: float(_env("MAX_VIDEO_SECONDS", 120)))
    max_queued: int = field(default_factory=lambda: int(_env("MAX_QUEUED", 10)))
    vid_stride: int = field(default_factory=lambda: int(_env("VID_STRIDE", 2)))
    upload_ttl_s: int = field(default_factory=lambda: int(_env("UPLOAD_TTL_S", 3600)))
    rate_detect_per_min: int = field(default_factory=lambda: int(_env("RATE_DETECT", 30)))
    rate_uploads_per_hour: int = field(default_factory=lambda: int(_env("RATE_UPLOADS", 10)))
    url_secret: str = field(default_factory=lambda: _env("URL_SECRET", ""))
    trusted_proxy: bool = field(default_factory=lambda: _flag("TRUSTED_PROXY", False))
    cors_origins: tuple = field(default_factory=lambda: tuple(
        o.strip() for o in _env("CORS_ORIGINS", "*").split(",") if o.strip()))
    cors_origin_regex: str = field(default_factory=lambda: _env("CORS_ORIGIN_REGEX", ""))
