"""Small helpers shared by detect.py and app.py: thresholds and image listing."""

import os
from pathlib import Path

import torch
import yaml

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_THRESHOLDS = "configs/thresholds.yaml"


def load_thresholds(path=DEFAULT_THRESHOLDS):
    """Per-class confidence thresholds {class name: threshold}, or None if the file doesn't exist."""
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return {str(k): float(v) for k, v in yaml.safe_load(f)["thresholds"].items()}


def apply_thresholds(result, thresholds, default=0.25):
    """Keep only boxes whose confidence >= their class threshold. Returns a filtered Ultralytics Results."""
    if not len(result.boxes):
        return result
    thr = torch.tensor(
        [thresholds.get(result.names[int(c)], default) for c in result.boxes.cls],
        device=result.boxes.conf.device,
    )
    return result[result.boxes.conf >= thr]


def list_images(source, limit=0):
    """Return image paths from a file or folder (sorted), optionally limited to the first `limit` images."""
    source = Path(source)
    if source.is_file():
        return [str(source)]
    if not source.is_dir():
        raise FileNotFoundError(f"Source not found: {source}")
    images = sorted(str(p) for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise FileNotFoundError(f"No images found in: {source}")
    return images[:limit] if limit > 0 else images
