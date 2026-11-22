"""PyTorch vs ONNX Runtime (FP32, INT8) on CPU: same accuracy? how much faster?

    python scripts/export_onnx.py --int8         # first: models/best.onnx + models/best.int8.onnx
    python scripts/benchmark_onnx.py

Every backend is scored by the same code (scripts/metrics.py) on the same images:
- mAP50 per split (predictions at conf 0.001, COCO 101-point AP, IoU 0.5),
- no-helmet precision / recall at the deployed per-class thresholds (configs/thresholds.yaml),
- agreement with PyTorch at the deployed thresholds (share of PyTorch boxes found with the same class
  at IoU >= 0.5, and their mean IoU).
Latency: batch 1 on CPU, images already decoded in memory, timed end to end (letterbox, inference,
NMS, boxes back in image pixels), after warm-up.
"""

import argparse
import os
import time
from collections.abc import Callable
from datetime import date

import cv2
import numpy as np

from metrics import average_precision, box_iou, counts_at, load_split, match_image, prf, predict_array
from onnx_detector import OnnxDetector
from utils import load_thresholds

VIOLATION = "Without Helmet"
SPLITS = [("test", "data/data.yaml", "test"), ("headwear test", "data/headwear_test.yaml", "test")]


class TorchBackend:
    def __init__(self, weights, imgsz):
        from ultralytics import YOLO

        self.model, self.imgsz = YOLO(weights), imgsz

    def __call__(self, img, conf):
        return predict_array(self.model, img, imgsz=self.imgsz, conf=conf, device="cpu")


class OnnxBackend:
    def __init__(self, path, threads):
        self.det = OnnxDetector(path, threads=threads)

    def __call__(self, img, conf):
        return self.det(img, conf=conf)


def agreement(ref_items, items, thr):
    """Share of reference (PyTorch) boxes that the other backend also finds, and their mean IoU."""
    from metrics import filter_preds

    found = total = 0
    ious = []
    for ref, it in zip(ref_items, items, strict=True):
        a, b = filter_preds(ref["pred"], thr), filter_preds(it["pred"], thr)
        gt = np.concatenate([a[:, 5:6], a[:, :4]], axis=1) if len(a) else np.zeros((0, 5))
        tp, _, missed = match_image(gt, b)
        total += len(a)
        found += int((~missed).sum())
        if tp.any():
            ious.extend(box_iou(b[tp, :4], a[:, :4]).max(axis=1))
    return found / max(total, 1), float(np.mean(ious)) if ious else 0.0


def latency(backend, images, conf, repeats, warmup=10):
    for img in images[:warmup]:
        backend(img, conf)
    times = []
    for _ in range(repeats):
        for img in images:
            t0 = time.perf_counter()
            backend(img, conf)
            times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.percentile(times, 50), np.percentile(times, 95)


def main(args):
    thresholds = load_thresholds(args.thresholds) or {}
    backends: list[tuple[str, str, Callable[[], TorchBackend | OnnxBackend]]] = [("PyTorch", args.weights, lambda: TorchBackend(args.weights, args.imgsz))]
    for name, path in (("ONNX Runtime FP32", args.onnx), ("ONNX Runtime INT8", args.onnx.replace(".onnx", ".int8.onnx"))):
        if os.path.exists(path):
            backends.append((name, path, lambda p=path: OnnxBackend(p, args.threads)))
        else:
            print(f"Skipping {name}: {path} not found (run scripts/export_onnx.py --int8)")

    splits = {}
    for label, data, split in SPLITS:
        names, items = load_split(data, split)
        for it in items:
            it["img"] = cv2.imread(it["path"])
        splits[label] = (names, items)
    names = splits["test"][0]
    v = names.index(VIOLATION)
    thr = {i: thresholds.get(n, 0.25) for i, n in enumerate(names)}

    rows, ref = [], {}
    for name, path, make in backends:
        print(f"Evaluating {name} ({path}) ...")
        backend = make()
        row = {"backend": name, "file": path, "size_MB": os.path.getsize(path) / 1e6}
        for label, (_, items) in splits.items():
            preds = [{"gt": it["gt"], "pred": backend(it["img"], 0.001)} for it in items]
            aps = average_precision(preds, len(names))
            tp, fp, fn = counts_at(preds, thr, len(names))
            p, r, _ = prf(tp[v], fp[v], fn[v])
            row[f"{label}|mAP50"], row[f"{label}|nh_p"], row[f"{label}|nh_r"] = aps.mean(), p, r
            if name == "PyTorch":
                ref[label] = preds
            row[f"{label}|agree"], row[f"{label}|iou"] = agreement(ref[label], preds, thr)
        test_imgs = [it["img"] for it in splits["test"][1]]
        row["mean_ms"], row["p50_ms"], row["p95_ms"] = latency(backend, test_imgs, 0.25, args.repeats)
        rows.append(row)

    base = rows[0]["mean_ms"]
    lines = ["| Backend | Size (MB) | Mean ms/img | p50 | p95 | CPU FPS | Speed-up | test mAP50 | test no-helmet P / R "
             "| headwear mAP50 | headwear no-helmet P / R | Same boxes as PyTorch (test / headwear) |",
             "|" + "---|" * 12]
    for r in rows:
        lines.append(
            f"| {r['backend']} | {r['size_MB']:.1f} | {r['mean_ms']:.1f} | {r['p50_ms']:.1f} | {r['p95_ms']:.1f} "
            f"| {1000 / r['mean_ms']:.1f} | {base / r['mean_ms']:.2f}x | {r['test|mAP50']:.3f} "
            f"| {r['test|nh_p']:.3f} / {r['test|nh_r']:.3f} | {r['headwear test|mAP50']:.3f} "
            f"| {r['headwear test|nh_p']:.3f} / {r['headwear test|nh_r']:.3f} "
            f"| {r['test|agree']:.1%} / {r['headwear test|agree']:.1%} (IoU {r['test|iou']:.3f}) |")

    import onnxruntime
    import torch

    n_test, n_head = len(splits["test"][1]), len(splits["headwear test"][1])
    report = f"""# ONNX Runtime vs PyTorch on CPU

_Generated by `scripts/benchmark_onnx.py` on {date.today()} on {args.cpu}.
PyTorch {torch.__version__} ({torch.get_num_threads()} threads), ONNX Runtime {onnxruntime.__version__}
({args.threads or 'default'} threads). Input {args.imgsz}x{args.imgsz}, batch 1._

{chr(10).join(lines)}

- **Latency:** {n_test} test images x {args.repeats} passes, decoded in memory, timed end to end (resize/letterbox,
  inference, NMS, boxes in image pixels), after 10 warm-up runs. FPS = 1000 / mean ms (one image at a time).
- **Accuracy:** test = {n_test} images (data/data.yaml), headwear test = {n_head} held-out images
  (data/headwear_test.yaml). mAP50 from predictions at conf 0.001 (COCO 101-point AP, IoU 0.5), scored by
  `scripts/metrics.py` for every backend, so the columns compare backends, not evaluation code. The absolute
  values are a little below `ultralytics val` (test mAP50 0.937 for PyTorch) because the matching differs
  (greedy per image here); use them to compare rows, and the model-comparison reports for the model's mAP.
  No-helmet precision / recall at the deployed thresholds ({', '.join(f'{k} {v}' for k, v in thresholds.items())}).
- **Same boxes as PyTorch:** share of PyTorch detections (deployed thresholds) that the backend also finds with
  the same class at IoU >= 0.5; IoU = mean IoU of those boxes on the test split.
- The ONNX backends run `scripts/onnx_detector.py` (numpy + OpenCV + onnxruntime, no PyTorch), the same code as the API.
  PyTorch runs through Ultralytics `predict`, which letterboxes to the smallest stride-32 rectangle; the ONNX model
  has a fixed square input, which explains small box differences.
- INT8: static quantization (QDQ, per-channel weights, MinMax calibration on 200 training images), `scripts/export_onnx.py --int8`.
"""
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"Saved {args.report}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Benchmark PyTorch vs ONNX Runtime on CPU")
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--onnx", default="models/best.onnx", help="FP32 model; <name>.int8.onnx is used if present")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--threads", type=int, default=0, help="ONNX Runtime intra-op threads (0 = default)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--thresholds", default="configs/thresholds.yaml")
    ap.add_argument("--cpu", default="Intel Core i7-7700K (4 cores / 8 threads)", help="CPU name for the report")
    ap.add_argument("--report", default="reports/onnx_benchmark.md")
    main(ap.parse_args())
