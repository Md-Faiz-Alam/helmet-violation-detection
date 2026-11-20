"""Evaluate trained weights on a dataset split and print per-class metrics.

    python scripts/evaluate.py --weights models/best.pt --data data/data.yaml --split test
"""

import argparse
import os

from ultralytics import YOLO


def run_evaluation(weights, data_yaml, split="test", img_size=640, device=None, output_dir="results/eval"):
    model = YOLO(weights)
    metrics = model.val(
        data=data_yaml,
        split=split,
        imgsz=img_size,
        device=device,
        plots=True,  # confusion matrix + PR curves
        project=os.path.abspath(output_dir),
        name=split,
        exist_ok=True,  # overwrite results/eval/<split> instead of creating <split>-2, -3, ...
        verbose=False,
    )

    header = f"{'class':<18}{'precision':>10}{'recall':>10}{'mAP50':>10}{'mAP50-95':>10}"
    print(f"\n📊 Metrics on '{split}' split\n{header}\n{'-' * len(header)}")
    for i, c in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(i)
        print(f"{metrics.names[int(c)]:<18}{p:>10.3f}{r:>10.3f}{ap50:>10.3f}{ap:>10.3f}")
    box = metrics.box
    print(f"{'all':<18}{box.mp:>10.3f}{box.mr:>10.3f}{box.map50:>10.3f}{box.map:>10.3f}")
    print(f"\nPlots saved in: {metrics.save_dir}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO helmet model")
    parser.add_argument("--weights", type=str, default="models/best.pt", help="Path to model weights")
    parser.add_argument("--data", type=str, default="data/data.yaml", help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--img", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default=None, help="cuda device or 'cpu'; default: auto")
    args = parser.parse_args()

    run_evaluation(args.weights, args.data, args.split, args.img, args.device)
