"""Error analysis: why does the model fail?

Matches predictions to ground truth at the chosen operating thresholds and groups every error:
  False negatives (missed objects)
    - confused  : a prediction of the OTHER class covers it (helmet <-> no helmet mix-up)
    - missed    : nothing detected there at all
  False positives (false alarms)
    - wrong_class : box is on a real object of the other class
    - duplicate   : second box on an object that was already detected
    - background  : no object there
It also reports recall by object size (small objects are a classic weakness),
and saves an image gallery per error type to reports/error_analysis/.

Defaults to the VALIDATION split so the test set stays untouched for final reporting.

    python scripts/error_analysis.py --weights models/best.pt
"""

import argparse
import os
import shutil
from collections import Counter, defaultdict
from datetime import date

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from metrics import box_iou, filter_preds, load_split, match_image, predict_items

SIZE_BUCKETS = [("small (<32px)", 0, 32**2), ("medium (32-96px)", 32**2, 96**2), ("large (>96px)", 96**2, float("inf"))]
COLORS = {0: (60, 180, 75), 1: (40, 40, 230)}  # BGR: With Helmet green, Without Helmet red
HIGHLIGHT = (0, 215, 255)  # yellow


def size_bucket(box):
    area = (box[2] - box[0]) * (box[3] - box[1])
    for name, lo, hi in SIZE_BUCKETS:
        if lo <= area < hi:
            return name
    return SIZE_BUCKETS[-1][0]


def load_thresholds(path, names, fallback):
    if path and os.path.exists(path):
        t = yaml.safe_load(open(path, encoding="utf-8"))["thresholds"]
        return {i: float(t[n]) for i, n in enumerate(names)}
    return {i: fallback for i in range(len(names))}


def analyse(items, names, thresholds):
    errors = []  # dicts: kind, category, cls, box, conf, item
    size_stats = defaultdict(lambda: Counter())  # (cls, bucket) -> {"tp":, "fn":}
    for item in items:
        gt, pred = item["gt"], filter_preds(item["pred"], thresholds)
        tp, fp, fn = match_image(gt, pred)
        ious = box_iou(pred[:, :4], gt[:, 1:5]) if len(pred) and len(gt) else np.zeros((len(pred), len(gt)))

        for j, g in enumerate(gt):
            c = int(g[0])
            size_stats[(c, size_bucket(g[1:5]))]["fn" if fn[j] else "tp"] += 1
            if fn[j]:
                other = (ious[:, j] >= 0.5) & (pred[:, 5] != c) if len(pred) else np.array([], bool)
                cat = "confused" if other.any() else "missed"
                errors.append({"kind": "FN", "category": cat, "cls": c, "box": g[1:5], "conf": None, "item": item})

        for i, p in enumerate(pred):
            if not fp[i]:
                continue
            c = int(p[5])
            if len(gt) and ((ious[i] >= 0.5) & (gt[:, 0] != c)).any():
                cat = "wrong_class"
            elif len(gt) and ((ious[i] >= 0.5) & (gt[:, 0] == c)).any():
                cat = "duplicate"
            else:
                cat = "background"
            errors.append({"kind": "FP", "category": cat, "cls": c, "box": p[:4], "conf": float(p[4]), "item": item})
    return errors, size_stats


def draw(error, names, thresholds, scale=1.5):
    item = error["item"]
    img = cv2.imread(item["path"])
    if img is None:  # cv2.imread returns None instead of raising
        raise FileNotFoundError(f"Could not read image: {item['path']}")
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    for g in item["gt"]:  # ground truth: thin
        x1, y1, x2, y2 = (g[1:5] * scale).astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[int(g[0])], 1)
    for p in filter_preds(item["pred"], thresholds):  # predictions: thick + confidence
        x1, y1, x2, y2 = (p[:4] * scale).astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[int(p[5])], 2)
        cv2.putText(img, f"{p[4]:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[int(p[5])], 1)
    x1, y1, x2, y2 = (np.asarray(error["box"]) * scale).astype(int)  # the error itself
    cv2.rectangle(img, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), HIGHLIGHT, 2)
    conf = f" conf={error['conf']:.2f}" if error["conf"] is not None else ""
    label = f"{error['kind']} {error['category']}: {names[error['cls']]}{conf}"
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img


def main(args):
    model = YOLO(args.weights)
    names, items = load_split(args.data, args.split)
    thresholds = load_thresholds(args.thresholds, names, args.conf)
    print(f"Predicting {len(items)} {args.split} images...")
    predict_items(model, items, imgsz=args.img, device=args.device)
    errors, size_stats = analyse(items, names, thresholds)

    # gallery
    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    by_cat = defaultdict(list)
    for e in errors:
        by_cat[(e["kind"], e["category"], e["cls"])].append(e)
    gallery_lines = []
    for (kind, cat, c), errs in sorted(by_cat.items()):
        folder = os.path.join(args.out, f"{kind}_{cat}_{names[c].replace(' ', '_').lower()}")
        os.makedirs(folder, exist_ok=True)
        errs = sorted(errs, key=lambda e: -(e["conf"] or 0))[: args.per_category]
        for k, e in enumerate(errs):
            fname = f"{k:02d}_{os.path.splitext(os.path.basename(e['item']['path']))[0][:40]}.jpg"
            cv2.imwrite(os.path.join(folder, fname), draw(e, names, thresholds), [cv2.IMWRITE_JPEG_QUALITY, 80])
        gallery_lines.append(f"- `{os.path.relpath(folder, os.path.dirname(args.report)).replace(os.sep, '/')}/` ({len(errs)} examples)")

    # summary tables
    cnt = Counter((e["kind"], e["category"], e["cls"]) for e in errors)
    err_rows = ["| Error | Type | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for kind, cat, desc in [("FN", "missed", "not detected at all"), ("FN", "confused", "detected as the other class"),
                            ("FP", "wrong_class", "box on an object of the other class"),
                            ("FP", "duplicate", "second box on the same object"), ("FP", "background", "nothing there")]:
        err_rows.append(f"| {kind} {cat} | {desc} | " + " | ".join(str(cnt[(kind, cat, c)]) for c in range(len(names))) + " |")

    size_rows = ["| Class | Size | Objects | Recall |", "|---|---|---|---|"]
    for c, name in enumerate(names):
        for b, _, _ in SIZE_BUCKETS:
            s = size_stats[(c, b)]
            total = s["tp"] + s["fn"]
            if total:
                size_rows.append(f"| {name} | {b} | {total} | {s['tp'] / total:.2f} |")

    thr_txt = ", ".join(f"{names[c]} = {thresholds[c]:.2f}" for c in range(len(names)))
    report = f"""# Error analysis

_Generated by `scripts/error_analysis.py` on {date.today()} · weights: `{args.weights}` · split: **{args.split}** ({len(items)} images) · thresholds: {thr_txt} · IoU 0.5_

## Error breakdown

{chr(10).join(err_rows)}

## Recall by object size

Object size is in original image pixels (images are 416×416).

{chr(10).join(size_rows)}

## Gallery

Ground truth is drawn with thin boxes and predictions with thick boxes plus confidence; green = With Helmet, red = Without Helmet. The yellow box marks the error.

{chr(10).join(gallery_lines)}
"""
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"Report: {args.report}  Gallery: {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Error analysis with gallery")
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--data", default="data/data.yaml")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--thresholds", default="configs/thresholds.yaml", help="Per-class thresholds (falls back to --conf)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--per_category", type=int, default=8, help="Max gallery images per error type")
    ap.add_argument("--out", default="reports/error_analysis")
    ap.add_argument("--report", default="reports/error_analysis.md")
    main(ap.parse_args())
