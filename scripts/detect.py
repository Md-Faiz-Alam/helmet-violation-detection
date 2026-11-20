"""Run helmet detection on an image, a folder of images, or a video.

    python scripts/detect.py --source data/test/images --n_samples 5   # tuned per-class thresholds
    python scripts/detect.py --source samples/image1.jpg --conf 0.4      # one threshold for all classes
    python scripts/detect.py --source path/to/video.mp4

By default it uses the per-class thresholds in configs/thresholds.yaml (from tune_thresholds.py).
"""

import argparse
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

from utils import DEFAULT_THRESHOLDS, VIDEO_EXTS, apply_thresholds, create_light_bundle, list_images, load_thresholds


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
    is_video = Path(source).suffix.lower() in VIDEO_EXTS
    inputs = [source] if is_video else list_images(source, limit=n_samples)

    counts, n_frames, writer = Counter(), 0, None
    for item in inputs:
        for r in model.predict(source=item, conf=predict_conf, stream=True, verbose=False):
            r = apply_thresholds(r, thresholds, default=predict_conf)
            n_frames += 1
            counts.update(r.names[int(c)] for c in r.boxes.cls)
            frame = r.plot()
            if is_video:
                if writer is None:
                    fps = cv2.VideoCapture(source).get(cv2.CAP_PROP_FPS) or 25
                    out_path = os.path.join(run_folder, Path(source).stem + "_pred.mp4")
                    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
                writer.write(frame)
            else:
                cv2.imwrite(os.path.join(run_folder, os.path.basename(r.path)), frame)
    if writer is not None:
        writer.release()

    print(f"✅ Processed {n_frames} {'frames' if is_video else 'images'}. Results saved in: {os.path.normpath(run_folder)}")
    for name, n in counts.most_common():
        print(f"   {name}: {n}")

    bundle_path = create_light_bundle(run_folder)
    print(f"✅ Light bundle created at: {os.path.normpath(bundle_path)}")
    return run_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Helmet Detection Script")
    parser.add_argument("--weights", type=str, default="models/best.pt", help="Path to model weights")
    parser.add_argument("--source", type=str, required=True, help="Image file, image folder, or video file")
    parser.add_argument("--conf", type=float, default=None, help="One threshold for all classes (overrides --thresholds)")
    parser.add_argument("--thresholds", type=str, default=DEFAULT_THRESHOLDS, help="Per-class thresholds YAML")
    parser.add_argument("--n_samples", type=int, default=0, help="Max number of images to process (0 = all)")
    args = parser.parse_args()

    run_detection(args.weights, args.source, args.conf, args.thresholds, args.n_samples)
