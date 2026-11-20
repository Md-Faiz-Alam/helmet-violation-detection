"""Shared helpers: load ground truth, run predictions once, and match predictions to ground truth.

Used by tune_thresholds.py and error_analysis.py. Matching is class-aware and greedy
(highest-confidence prediction first), at IoU >= 0.5, which mirrors how mAP50 counts TP/FP.
"""

from pathlib import Path

import numpy as np
import yaml
from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ----------------------------------------------------------------------------- data

def load_split(data_yaml, split):
    """Return (class_names, items) where each item is {"path", "w", "h", "gt"}.

    gt is an (N, 5) float array of [cls, x1, y1, x2, y2] in pixels.
    """
    data_yaml = Path(data_yaml)
    cfg = yaml.safe_load(data_yaml.read_text())
    names = cfg["names"]
    key = {"valid": "val"}.get(split, split)
    img_dir = (data_yaml.parent / cfg[key]).resolve()
    label_dir = img_dir.parent / "labels"  # YOLO layout: <split>/images + <split>/labels

    items = []
    for img_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        with Image.open(img_path) as im:
            w, h = im.size
        gt = np.zeros((0, 5), dtype=float)
        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            rows = [list(map(float, ln.split())) for ln in label_path.read_text().splitlines() if ln.strip()]
            if rows:
                a = np.array(rows, dtype=float)
                cx, cy, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
                gt = np.stack([a[:, 0], cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        items.append({"path": str(img_path), "w": w, "h": h, "gt": gt})
    return names, items


def predict_items(model, items, imgsz=640, conf=0.001, device=None):
    """Attach raw predictions to each item: item["pred"] = (M, 6) [x1, y1, x2, y2, conf, cls]."""
    for item in items:
        r = model.predict(item["path"], imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        b = r.boxes
        if len(b):
            item["pred"] = np.concatenate(
                [b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]], axis=1
            )
        else:
            item["pred"] = np.zeros((0, 6))
    return items


# ----------------------------------------------------------------------------- matching

def box_iou(a, b):
    """IoU between boxes a (N, 4) and b (M, 4) in xyxy format -> (N, M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def filter_preds(pred, thresholds):
    """Keep predictions whose confidence >= the threshold of their class. thresholds: {cls_id: thr}."""
    if len(pred) == 0:
        return pred
    thr = np.array([thresholds[int(c)] for c in pred[:, 5]])
    return pred[pred[:, 4] >= thr]


def match_image(gt, pred, iou_thr=0.5):
    """Class-aware greedy matching.

    Returns (tp_pred, fp_pred, fn_gt): boolean masks over pred/pred/gt.
    """
    tp = np.zeros(len(pred), dtype=bool)
    gt_used = np.zeros(len(gt), dtype=bool)
    if len(pred) and len(gt):
        ious = box_iou(pred[:, :4], gt[:, 1:5])
        for i in np.argsort(-pred[:, 4]):
            same_cls = gt[:, 0] == pred[i, 5]
            cand = np.where(same_cls & ~gt_used & (ious[i] >= iou_thr))[0]
            if len(cand):
                j = cand[np.argmax(ious[i, cand])]
                tp[i] = True
                gt_used[j] = True
    return tp, ~tp, ~gt_used


def counts_at(items, thresholds, n_classes, iou_thr=0.5):
    """Per-class TP / FP / FN totals at the given per-class thresholds."""
    tp = np.zeros(n_classes, int)
    fp = np.zeros(n_classes, int)
    fn = np.zeros(n_classes, int)
    for item in items:
        pred = filter_preds(item["pred"], thresholds)
        t, f, m = match_image(item["gt"], pred, iou_thr)
        for c in range(n_classes):
            tp[c] += int((t & (pred[:, 5] == c)).sum()) if len(pred) else 0
            fp[c] += int((f & (pred[:, 5] == c)).sum()) if len(pred) else 0
            fn[c] += int((m & (item["gt"][:, 0] == c)).sum()) if len(item["gt"]) else 0
    return tp, fp, fn


def prf(tp, fp, fn, beta=1.0):
    """Precision, recall and F-beta (beta > 1 weighs recall more)."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    b2 = beta**2
    f = (1 + b2) * p * r / (b2 * p + r) if p + r else 0.0
    return p, r, f
