"""Prepare extra training data for the headwear failure mode: clean, de-duplicate, mine hard examples.

The video benchmark showed riders wearing caps, skullcaps or a cloth on the head being read as
"With Helmet". This script takes a second dataset (default: "Helmet, Rider, ANPR Detection" v12,
CC BY 4.0, downloaded with scripts/download_roboflow.py) and:

  1. converts its labels to ours (with helmet -> 0, without helmet -> 1; polygons -> boxes; rider and
     number-plate boxes are dropped) and keeps ONE copy per original photo (Roboflow stored 2-3
     augmented copies; the one with the tightest boxes is the un-rotated one);
  2. drops images where a labelled rider has no labelled head (an unlabelled head would teach the
     model to ignore heads);
  3. drops anything that could leak into our evaluation: images whose source name also appears in our
     dataset, and images that look like one of ours (perceptual hash, ignoring the black letterbox bars);
  4. runs the CURRENT model and marks every no-helmet head as caught / confused (predicted "With
     Helmet": the headwear failure) / missed;
  5. splits by source family (a video's frames never land on both sides): ~80% extra-train,
     ~20% held-out "headwear test".

    python scripts/mine_headwear.py
Writes data/extra/headwear/{images,labels}/, manifest.csv, summary.json, review/confused_*.jpg, the image
lists extra_train.txt / headwear_test.txt, and two dataset configs next to data/data.yaml:
    data/data_headwear.yaml   train = our oversampled train + extra_train; val/test = ours (unchanged)
    data/headwear_test.yaml   val = test = the held-out headwear test (for evaluate.py / benchmark.py)
`--no-mine` skips the model step (the split does not depend on it), e.g. to rebuild the same data on Kaggle.
"""

import argparse
import csv
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import yaml

from metrics import box_iou, predict_array, read_image

EXTRA_CLASSES = {2: 0, 3: 1}  # extra dataset: 0 plate, 1 rider, 2 with helmet, 3 without helmet
RIDER = 1


def source_name(name):
    """Original-photo name shared by Roboflow's augmented copies (and by both datasets' exports)."""
    return re.split(r"_png_jpg|_jpg\.rf\.|_jpeg\.rf\.|_png\.rf\.|\.rf\.", name)[0]


def source_family(name):
    """Coarser group for splitting: all frames of one video / numbered series share it."""
    s = source_name(name)
    s = re.sub(r"(_mp4|_avi|_mov)[-_].*$", r"\1", s, flags=re.I)  # VID_..._mp4-0012 -> VID_..._mp4
    s = re.sub(r"(frame)[-_]?\d+.*$", r"\1", s, flags=re.I)
    return re.sub(r"[-_ ()]*\d+[-_ ()]*$", "", s) or s       # Without_Helmet-44- -> Without_Helmet


def read_labels(path, w, h):
    """YOLO boxes or polygons -> list of (cls, x1, y1, x2, y2) in pixels."""
    out = []
    for ln in path.read_text().splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        c, v = int(p[0]), list(map(float, p[1:]))
        if len(v) == 4:
            cx, cy, bw, bh = v
            out.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
        else:
            xs, ys = v[0::2], v[1::2]
            out.append((c, min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h))
    return out


def trimmed_gray(img):
    """Remove black letterbox bars, so a 640x640 'fit' export hashes like the original photo."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(g > 12)
    return g[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if len(xs) else g


def dhash(img, n=16):
    g = cv2.resize(trimmed_gray(img), (n + 1, n))
    return (g[:, 1:] > g[:, :-1]).flatten()


def riders_have_heads(labels):
    heads = [b for b in labels if b[0] in EXTRA_CLASSES]
    for c, x1, y1, x2, y2 in labels:
        if c != RIDER:
            continue
        top = y1 + 0.6 * (y2 - y1)  # a rider's head sits in the upper part of the rider box
        if not any(hx1 < x2 and hx2 > x1 and hy1 < top and hy2 > y1 for _, hx1, hy1, hx2, hy2 in heads):
            return False
    return True


def main(args):
    src, out = Path(args.source), Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir()
    (out / "review").mkdir()
    stats = Counter()

    # 1-2. collect, convert, one copy per original, completeness filter
    copies = defaultdict(list)
    for split in ("train", "valid", "test"):
        for img_path in sorted((src / split / "images").glob("*")):
            lab = src / split / "labels" / f"{img_path.stem}.txt"
            if lab.exists():
                copies[source_name(img_path.name)].append((img_path, lab))
    stats["originals"] = len(copies)
    items = []
    for name, cps in copies.items():
        best = None
        for img_path, lab in cps:
            w, h = Image.open(img_path).size
            labels = read_labels(lab, w, h)
            heads = [b for b in labels if b[0] in EXTRA_CLASSES]
            if not heads:
                continue
            area = np.mean([(b[3] - b[1]) * (b[4] - b[2]) for b in heads])
            if best is None or area < best[0]:
                best = (area, img_path, labels)
        if best is None:
            stats["dropped: no head labels"] += 1
            continue
        _, img_path, labels = best
        img_w, img_h = Image.open(img_path).size
        if max((b[3] - b[1]) * (b[4] - b[2]) for b in labels if b[0] in EXTRA_CLASSES) > args.max_head_frac * img_w * img_h:
            stats["dropped: a head box covers most of the image (product photo / close-up)"] += 1
            continue
        if not riders_have_heads(labels):
            stats["dropped: a rider has no labelled head"] += 1
            continue
        items.append({"name": name, "path": img_path, "family": source_family(img_path.name),
                      "gt": np.array([[EXTRA_CLASSES[c], *b] for c, *b in labels if c in EXTRA_CLASSES], float)})

    # 3. leakage filter against our own dataset (all splits)
    ours = [p for s in ("train", "valid", "test") for p in sorted(Path(args.ours, s, "images").glob("*"))]
    our_names = {source_name(p.name) for p in ours}
    our_hashes = np.array([dhash(read_image(p)) for p in ours])
    dists = []
    kept = []
    for it in items:
        d = int((our_hashes != dhash(read_image(it["path"]))).sum(1).min())
        dists.append(d)
        if it["name"] in our_names:
            stats["dropped: same source name as our dataset"] += 1
        elif d <= args.hash_threshold:
            stats["dropped: looks like one of our images"] += 1
        else:
            kept.append(it)
    hist = np.histogram(dists, bins=[0, 10, 20, 30, 40, 50, 60, 80, 256])
    stats["kept"] = len(kept)

    # 4. mine with the current model
    model = None
    if not args.no_mine:
        from ultralytics import YOLO
        model = YOLO(args.weights)
    rows, confused_crops = [], []
    for k, it in enumerate(kept):
        if model is None:
            gt = it["gt"]
            rows.append({"image": it["path"].name, "source": it["name"], "family": it["family"],
                         "n_without": int((gt[:, 0] == 1).sum()), "n_with": int((gt[:, 0] == 0).sum()),
                         "wo_caught": 0, "wo_confused": 0, "wo_missed": 0, "with_confused": 0})
            continue
        img = read_image(it["path"])  # images are re-read instead of all held in memory
        pred = predict_array(model, img, conf=0.05, imgsz=args.img, device=args.device)
        gt = it["gt"]
        ious = box_iou(gt[:, 1:5], pred[:, :4]) if len(pred) else np.zeros((len(gt), 0))
        outcome = Counter()
        for j, g in enumerate(gt):
            cls = int(g[0])
            hit = ious[j] >= 0.5 if len(pred) else np.zeros(0, bool)
            same = hit & (pred[:, 5] == cls) & (pred[:, 4] >= (args.thr_without if cls == 1 else args.thr_with)) \
                if len(pred) else hit
            other = hit & (pred[:, 5] != cls) & (pred[:, 4] >= (args.thr_with if cls == 1 else args.thr_without)) \
                if len(pred) else hit
            tag = "caught" if same.any() else "confused" if other.any() else "missed"
            outcome[f"{'wo' if cls == 1 else 'with'}_{tag}"] += 1
            if cls == 1 and tag == "confused" and len(confused_crops) < args.gallery:
                x1, y1, x2, y2 = (int(v) for v in g[1:5])
                pad = max(x2 - x1, y2 - y1)
                c = img[max(0, y1 - pad):y2 + 2 * pad, max(0, x1 - pad):x2 + pad].copy()
                cv2.rectangle(c, (x1 - max(0, x1 - pad), y1 - max(0, y1 - pad)),
                              (x2 - max(0, x1 - pad), y2 - max(0, y1 - pad)), (0, 215, 255), 2)
                confused_crops.append(cv2.resize(c, (160, 160)))
        stats.update(outcome)
        rows.append({"image": it["path"].name, "source": it["name"], "family": it["family"],
                     "n_without": int((gt[:, 0] == 1).sum()), "n_with": int((gt[:, 0] == 0).sum()),
                     "wo_caught": outcome["wo_caught"], "wo_confused": outcome["wo_confused"],
                     "wo_missed": outcome["wo_missed"], "with_confused": outcome["with_confused"]})
        if (k + 1) % 500 == 0:
            print(f"  mined {k + 1}/{len(kept)}")

    # 5. random split by source family (frames of one video never land on both sides)
    fams = sorted({r["family"] for r in rows})
    random.Random(args.seed).shuffle(fams)
    fam_size = Counter(r["family"] for r in rows)
    test_fams, n_test, target = set(), 0, args.test_share * len(rows)
    for f in fams:
        if n_test >= target:
            break
        if fam_size[f] > 0.25 * target:  # never let one huge video dominate the test set
            continue
        test_fams.add(f)
        n_test += fam_size[f]
    for k, (r, it) in enumerate(zip(rows, kept, strict=True)):
        r["split"] = "headwear_test" if r["family"] in test_fams else "extra_train"
        stem = f"x{k:05d}_{it['path'].stem.split('.rf.')[0][:60]}"  # unique, short, marks extra data
        r["file"] = stem + ".jpg"
        shutil.copy2(it["path"], out / "images" / r["file"])
        w, h = Image.open(it["path"]).size
        with open(out / "labels" / f"{stem}.txt", "w") as f:
            for c, x1, y1, x2, y2 in it["gt"]:
                f.write(f"{int(c)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")

    # image lists + dataset configs (absolute paths, as written on the machine that runs this)
    for split in ("extra_train", "headwear_test"):
        paths = [str((out / "images" / r["file"]).resolve()).replace("\\", "/") for r in rows if r["split"] == split]
        (out / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    ours_yaml = Path(args.ours) / "data.yaml"
    names = yaml.safe_load(ours_yaml.read_text())["names"]
    extra_rel = Path(os.path.relpath(out, ours_yaml.parent)).as_posix()
    (ours_yaml.parent / "data_headwear.yaml").write_text(
        "# Generated by scripts/mine_headwear.py: our (oversampled) train + extra headwear data; val/test unchanged.\n"
        + yaml.safe_dump({"train": ["train_oversampled.txt", f"{extra_rel}/extra_train.txt"], "val": "valid/images",
                          "test": "test/images", "nc": len(names), "names": names}, sort_keys=False))
    (ours_yaml.parent / "headwear_test.yaml").write_text(
        "# Generated by scripts/mine_headwear.py: held-out headwear test set (never used for training).\n"
        + yaml.safe_dump({"train": f"{extra_rel}/extra_train.txt", "val": f"{extra_rel}/headwear_test.txt",
                          "test": f"{extra_rel}/headwear_test.txt", "nc": len(names), "names": names}, sort_keys=False))

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    for i in range(0, len(confused_crops), 48):
        tiles = confused_crops[i:i + 48] + [np.zeros((160, 160, 3), np.uint8)] * (-len(confused_crops[i:i + 48]) % 8)
        sheet = np.vstack([np.hstack(tiles[j:j + 8]) for j in range(0, len(tiles), 8)])
        cv2.imwrite(str(out / "review" / f"confused_{i // 48 + 1}.jpg"), sheet)

    split_stats = {s: Counter() for s in ("extra_train", "headwear_test")}
    for r in rows:
        split_stats[r["split"]].update({"images": 1, **{k: r[k] for k in ("n_without", "n_with", "wo_confused", "wo_missed")}})
    summary = {"source": str(src), "weights": args.weights, "filter": dict(stats),
               "hash_distance_histogram": {f"{a}-{b}": int(n) for a, b, n in zip(hist[1][:-1], hist[1][1:], hist[0], strict=True)},
               "splits": {k: dict(v) for k, v in split_stats.items()}, "test_families": sorted(test_fams)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("filter", "hash_distance_histogram", "splits")}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Clean, de-duplicate and mine a second helmet dataset")
    ap.add_argument("--source", default="data/extra/helmet-rider-anpr-detection_v12")
    ap.add_argument("--ours", default="data", help="Our dataset root (train/valid/test)")
    ap.add_argument("--out", default="data/extra/headwear")
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--thr-with", type=float, default=0.25, help="Operating threshold for With Helmet")
    ap.add_argument("--thr-without", type=float, default=0.16, help="Operating threshold for Without Helmet")
    ap.add_argument("--hash-threshold", type=int, default=30, help="Max differing bits (of 256) to call it a duplicate")
    ap.add_argument("--test-share", type=float, default=0.2)
    ap.add_argument("--gallery", type=int, default=144, help="Confused heads to save for review")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-head-frac", type=float, default=0.2, help="Drop images where a head box covers more than this")
    ap.add_argument("--no-mine", action="store_true", help="Skip running the model (split and files are identical)")
    main(ap.parse_args())
