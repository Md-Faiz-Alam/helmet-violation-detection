"""Train a YOLO helmet detector from a YAML experiment config.

    python scripts/train.py --config configs/experiments/yolo26s_640.yaml
    python scripts/train.py --config configs/experiments/baseline_yolov8s_640.yaml --epochs 1 --device cpu   # quick test
    python scripts/train.py --config configs/experiments/yolo26s_640.yaml --promote   # also copy best.pt to models/

Each run is saved to results/train/<experiment name>/ together with the merged config.
If MLflow is installed (pip install mlflow), Ultralytics logs params/metrics to it automatically.
"""

import argparse
import os
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


def deep_merge(base, override):
    out = dict(base)
    for k, v in override.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path):
    """Load a config, recursively merging the file named by `_base_` (relative to this file)."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    base = cfg.pop("_base_", None)
    if base:
        cfg = deep_merge(load_config(path.parent / base), cfg)
    return cfg


def run_training(cfg, name, output_dir="results/train", promote=False, model_dir="models"):
    model = YOLO(cfg["weights"])
    model.train(
        data=cfg["data"],
        project=os.path.abspath(output_dir),
        name=name,
        exist_ok=False,  # never overwrite a previous run (Ultralytics appends 2, 3, ...)
        **cfg["train"],
    )

    save_dir = model.trainer.save_dir if model.trainer is not None else None
    if save_dir is None:
        raise RuntimeError("Training did not produce a run folder")
    run_dir = Path(save_dir)
    with open(run_dir / "experiment.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"name": name, **cfg}, f, sort_keys=False)

    best = run_dir / "weights" / "best.pt"
    print(f"✅ Run saved in: {run_dir}")
    if promote and best.exists():
        os.makedirs(model_dir, exist_ok=True)
        shutil.copy2(best, Path(model_dir) / "best.pt")
        print(f"✅ Promoted {best} -> {model_dir}/best.pt (used by the app)")
    return run_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train a YOLO helmet detector from a config")
    ap.add_argument("--config", required=True, help="Experiment YAML, e.g. configs/experiments/yolo26s_640.yaml")
    ap.add_argument("--name", default=None, help="Run name (default: config file name)")
    ap.add_argument("--promote", action="store_true", help="Copy best.pt to models/best.pt after training")
    # Optional overrides for quick tests
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--device", type=str, help="cuda device (0, 0,1) or 'cpu'; default: auto")
    ap.add_argument("--fraction", type=float, help="Use only this fraction of the train set (quick tests)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    for key in ("epochs", "batch", "imgsz", "device", "fraction"):
        if getattr(args, key) is not None:
            cfg["train"][key] = getattr(args, key)

    run_training(cfg, name=args.name or Path(args.config).stem, promote=args.promote)
