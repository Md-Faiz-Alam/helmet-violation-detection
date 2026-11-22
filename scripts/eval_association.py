"""Does requiring a vehicle under each head reduce false alarms, and what does it cost in recall?

Tunes the head-to-vehicle rule (backend/ridesafe/association.py) on the VALIDATION set, then reports it once
on the TEST set, against the same model with no vehicle filter, and shows where the remaining
false alarms are. Report: reports/association.md.

    python scripts/eval_association.py
"""

import argparse
import itertools
import os
from datetime import date

import numpy as np
from ultralytics import YOLO

from ridesafe.association import VEHICLE_CLASSES, AssociationRule, associate

from metrics import counts_at, filter_preds, load_split, match_image, predict_items, prf
from utils import load_thresholds

VIOLATION_CLASS = "Without Helmet"

GRID = {
    "max_dx": [0.5, 0.75, 1.0, 1.5],
    "min_dy": [-1.0, -1.5, -2.0, -3.0],
    "max_dy": [0.25, 0.5, 1.0],
    "max_head_ratio": [1.2, 2.0],
    "vehicle_conf": [0.15, 0.25, 0.4],
}


def detect_vehicles(coco, items, device):
    for it in items:
        r = coco.predict(it["path"], conf=0.1, device=device, verbose=False)[0]
        it["vehicles"] = [(tuple(b), float(s)) for c, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy.tolist(), strict=True)
                          if r.names[int(c)] in VEHICLE_CLASSES]


def filtered(items, rule, min_conf):
    """Copies of items whose predictions are kept only if the head sits on a vehicle."""
    out = []
    for it in items:
        pred = it["pred"][it["pred"][:, 4] >= min_conf]  # lower-confidence boxes never pass the thresholds anyway
        keep = np.array(associate([tuple(p[:4]) for p in pred], it["vehicles"], rule), dtype=int) >= 0 \
            if len(pred) else np.zeros(0, bool)
        out.append({**it, "pred": pred[keep] if len(pred) else pred})
    return out


def gt_coverage(items, rule):
    """Share of ground-truth heads (all riders) that the rule associates with a vehicle: the recall ceiling."""
    heads = [(it, tuple(g[1:5])) for it in items for g in it["gt"]]
    kept = sum(associate([h], it["vehicles"], rule)[0] >= 0 for it, h in heads)
    return kept / max(len(heads), 1)


def where_are_the_errors(items, thresholds, cls, rule):
    """For one class at the operating thresholds: how many false alarms / correct detections sit on a vehicle."""
    fp_on = fp_off = tp_on = tp_off = 0
    for it in items:
        pred = filter_preds(it["pred"], thresholds)
        if not len(pred):
            continue
        tp, _, _ = match_image(it["gt"], pred)
        on = np.array(associate([tuple(p[:4]) for p in pred], it["vehicles"], rule)) >= 0
        mine = pred[:, 5] == cls
        fp_on += int((mine & ~tp & on).sum())
        fp_off += int((mine & ~tp & ~on).sum())
        tp_on += int((mine & tp & on).sum())
        tp_off += int((mine & tp & ~on).sum())
    return fp_on, fp_off, tp_on, tp_off


def scores(items, thresholds, names, beta=2.0):
    tp, fp, fn = counts_at(items, thresholds, len(names))
    res = {}
    for c, name in enumerate(names):
        b = beta if name == VIOLATION_CLASS else 1.0
        p, r, f = prf(tp[c], fp[c], fn[c], beta=b)
        res[name] = {"P": p, "R": r, "F": f, "beta": b, "TP": int(tp[c]), "FP": int(fp[c]), "FN": int(fn[c])}
    return res


def table(rows):
    out = ["| Setting | Class | Precision | Recall | F-score | TP | FP | FN |", "|---|---|---|---|---|---|---|---|"]
    for label, res in rows:
        for name, s in res.items():
            out.append(f"| {label} | {name} | {s['P']:.3f} | {s['R']:.3f} | {s['F']:.3f} (F{s['beta']:g}) "
                       f"| {s['TP']} | {s['FP']} | {s['FN']} |")
    return "\n".join(out)


def main(args):
    helmet, coco = YOLO(args.weights), YOLO(args.coco_weights)
    thr_by_name = load_thresholds(args.thresholds) or {}
    data = {}
    for split in ("val", "test"):
        names, items = load_split(args.data, split)
        print(f"{split}: helmet + vehicle predictions on {len(items)} images...")
        predict_items(helmet, items, imgsz=args.img, device=args.device)
        detect_vehicles(coco, items, args.device)
        data[split] = items
    thresholds = {i: thr_by_name.get(n, 0.25) for i, n in enumerate(names)}
    min_conf = min(thresholds.values())
    v = names.index(VIOLATION_CLASS)
    safe = 1 - v

    # Tune on val: best no-helmet F2, ties broken by With Helmet F1
    best = None
    for combo in itertools.product(*GRID.values()):
        rule = AssociationRule(**dict(zip(GRID, combo, strict=True)))
        s = scores(filtered(data["val"], rule, min_conf), thresholds, names)
        key = (round(s[names[v]]["F"], 4), round(s[names[safe]]["F"], 4))
        if best is None or key > best[0]:
            best = (key, rule)
    assert best is not None, "empty search grid"
    rule = best[1]
    print("Chosen rule:", rule)

    sections = []
    for split, title in (("val", "Validation (used to choose the rule)"), ("test", "Test (held out)")):
        base = scores(data[split], thresholds, names)
        filt = scores(filtered(data[split], rule, min_conf), thresholds, names)
        cov = gt_coverage(data[split], rule)
        sections.append(f"### {title}\n\nGround-truth riders whose head the rule places on a detected vehicle: "
                        f"**{cov:.1%}** (the recall ceiling with the filter on).\n\n"
                        f"{table([('No filter', base), ('Vehicle filter', filt)])}\n")

    geo = AssociationRule()  # geometry measured on train, before any tuning
    fp_on, fp_off, tp_on, tp_off = where_are_the_errors(data["val"], thresholds, v, geo)
    breakdown = (f"With the train-measured geometry (`{geo}`), on val at the operating thresholds:\n\n"
                 f"| Without Helmet | On a detected vehicle | Not on one |\n|---|---|---|\n"
                 f"| False alarms ({fp_on + fp_off}) | {fp_on} | {fp_off} |\n"
                 f"| Correct detections ({tp_on + tp_off}) | {tp_on} | {tp_off} |\n")

    thr_txt = ", ".join(f"{k} = {v_:.2f}" for k, v_ in thr_by_name.items())
    report = f"""# Head-to-vehicle association

_Generated by `scripts/eval_association.py` on {date.today()} · helmet weights: `{args.weights}` · vehicle detector: `{args.coco_weights}` (COCO) · thresholds: {thr_txt} · IoU 0.5_

**Why:** the dataset only labels riders, so the helmet model also flags pedestrians and bystanders without a helmet. A head now counts only if it sits on a detected bicycle or motorcycle ([backend/ridesafe/association.py](../backend/ridesafe/association.py)).

**Rule** (tuned on val over {np.prod([len(x) for x in GRID.values()])} combinations for Without Helmet F2, ties by With Helmet F1):
`{rule}`

{chr(10).join(sections)}
### Where the remaining false alarms are

{breakdown}
**How to read this:** a head-on-vehicle filter can only remove false alarms on people who are *not* on a vehicle, and it also drops real riders whose vehicle is not detected (close-ups with the bike out of frame, heavy occlusion). If most false alarms already sit on or right next to a bike (crowds at bike events, people standing with their bicycles, unlabeled pillion riders), single-image geometry cannot separate them, and the filter is not worth its recall cost.
"""
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"Saved {args.report}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tune and evaluate head-to-vehicle association")
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--coco_weights", default="yolo26s.pt", help="COCO-pretrained detector for bicycles/motorcycles")
    ap.add_argument("--data", default="data/data.yaml")
    ap.add_argument("--thresholds", default="configs/thresholds.yaml")
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--report", default="reports/association.md")
    main(ap.parse_args())
