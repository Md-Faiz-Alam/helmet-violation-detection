"""Per-class confidence thresholds (configs/thresholds.yaml, written by scripts/tune_thresholds.py)."""

import os

import yaml

DEFAULT_THRESHOLDS = "configs/thresholds.yaml"


def load_thresholds(path=DEFAULT_THRESHOLDS):
    """Per-class confidence thresholds {class name: threshold}, or None if the file doesn't exist."""
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return {str(k): float(v) for k, v in yaml.safe_load(f)["thresholds"].items()}
