"""Run helmet detection on an image or a folder of images and save annotated copies.

    python scripts/detect.py --source data/test/images --n_samples 5   # tuned per-class thresholds
    python scripts/detect.py --source samples/image1.jpg --conf 0.4      # one threshold for all classes

By default it uses the per-class thresholds in configs/thresholds.yaml (from tune_thresholds.py).
For video, webcam or a camera stream use scripts/track_video.py: it tracks each rider and decides
once per rider instead of labelling every frame on its own.
"""

import argparse
import os
from collections import Counter
from datetime import datetime

import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results

from utils import DEFAULT_THRESHOLDS, apply_thresholds, list_images, load_thresholds


def run_detection(weights, source, conf=None, thresholds_path=DEFAULT_THRESHOLDS, n_samples=0, output_dir="results/detect"):
    run_folder = os.path.join(output_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_folder, exist_ok=True)

    # An explicit --conf wins; otherwise use tuned per-class thresholds if available.
    thresholds = None if conf is not None else load_thresholds(thresholds_path)
    if thresholds:
        print("Using per-class thresholds:", ", ".join(f"{k}={v:.2f}" for k, v in thresholds.items()))
        predict_conf = min(thresholds.values())
    else:
        conf = 0.25 if conf is None else conf
        print(f"Using one threshold for all classes: {conf}")
        thresholds, predict_conf = {}, conf

    model = YOLO(weights)
    images = list_images(source, limit=n_samples)
    counts = Counter()
    for path in images:
        for raw in model.predict(source=path, conf=predict_conf, stream=True, verbose=False):
            if not isinstance(raw, Results):
                raise TypeError(f"Expected a detection result, got {type(raw).__name__}")
            r = apply_thresholds(raw, thresholds, default=predict_conf)
            if r.boxes is not None:
                counts.update(r.names[int(c)] for c in r.boxes.cls)
            cv2.imwrite(os.path.join(run_folder, os.path.basename(path)), r.plot())

    print(f"✅ Processed {len(images)} images. Results saved in: {os.path.normpath(run_folder)}")
    for name, n in counts.most_common():
        print(f"   {name}: {n}")
    return run_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helmet detection on images (use track_video.py for video)")
    parser.add_argument("--weights", type=str, default="models/best.pt", help="Path to model weights")
    parser.add_argument("--source", type=str, required=True, help="Image file or folder of images")
    parser.add_argument("--conf", type=float, default=None, help="One threshold for all classes (overrides --thresholds)")
    parser.add_argument("--thresholds", type=str, default=DEFAULT_THRESHOLDS, help="Per-class thresholds YAML")
    parser.add_argument("--n_samples", type=int, default=0, help="Max number of images to process (0 = all)")
    args = parser.parse_args()

    run_detection(args.weights, args.source, args.conf, args.thresholds, args.n_samples)
