"""Render a video ground-truth file as review sheets: one row per rider, showing the head at
3 moments plus the stored label, so a person can check exactly what eval_video.py will use.

    python scripts/review_gt.py --gt eval/video_gt/indian_highway_riders.yaml --video data/videos/indian_highway_riders.mp4

Writes eval/video_gt/review_<clip>/sheet_N.jpg (git-ignored: it shows real faces). Optional --hires
takes the crops from a higher-resolution copy of the same clip (e.g. the 1080p download), which makes
small riders and plates much easier to judge; the boxes are rescaled automatically.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

GREEN, RED, GREY = (60, 180, 60), (40, 40, 230), (150, 150, 150)


def rider_tiles(cap, scale, rider, size=240):
    kf = sorted(rider["keyframes"].items())
    largest = max(kf, key=lambda it: (it[1][2] - it[1][0]) * (it[1][3] - it[1][1]))
    picks = sorted({kf[0][0]: kf[0], kf[len(kf) // 2][0]: kf[len(kf) // 2], largest[0]: largest}.values())
    tiles = []
    for f, box in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f) - 1)
        ok, img = cap.read()
        if not ok:
            continue
        H, W = img.shape[:2]
        x1, y1, x2, y2 = (int(v * scale) for v in box)
        bw, bh = x2 - x1, y2 - y1
        X1, Y1, X2, Y2 = max(0, x1 - 2 * bw), max(0, y1 - bh), min(W, x2 + 2 * bw), min(H, y2 + 5 * bh)
        c = img[Y1:Y2, X1:X2].copy()
        cv2.rectangle(c, (x1 - X1, y1 - Y1), (x2 - X1, y2 - Y1), (0, 215, 255), 2)
        c = cv2.resize(c, (max(1, int(c.shape[1] * size / max(c.shape[0], 1))), size), interpolation=cv2.INTER_CUBIC)
        cv2.putText(c, f"frame {f}", (4, size - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.pad(c, ((0, 0), (0, 4), (0, 0))))
    return tiles


def caption(width, rider):
    helmet = rider["helmet"]
    color = GREEN if helmet is True else RED if helmet is False else GREY
    text = {True: "HELMET", False: "NO HELMET"}.get(helmet, f"helmet: {helmet}")
    bar = np.full((52, width, 3), 255, np.uint8)
    cv2.rectangle(bar, (0, 0), (8, 52), color, -1)
    cv2.putText(bar, f"{rider['id']}  {text}   plate: {rider.get('plate') or '-'}", (14, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.putText(bar, rider.get("description", "")[:110], (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1,
                cv2.LINE_AA)
    return bar


def main(args):
    gt = yaml.safe_load(Path(args.gt).read_text(encoding="utf-8"))
    cap = cv2.VideoCapture(args.hires or args.video)
    scale = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / gt["resolution"][0]
    rows = []
    for r in gt["riders"]:
        tiles = rider_tiles(cap, scale, r)
        strip = np.hstack(tiles) if tiles else np.zeros((240, 400, 3), np.uint8)
        rows.append(np.vstack([caption(max(strip.shape[1], 700), r),
                               np.pad(strip, ((0, 8), (0, max(0, 700 - strip.shape[1])), (0, 0)), constant_values=255)]))
    cap.release()

    out = Path(args.out or Path(args.gt).parent / f"review_{Path(gt['video']).stem}")
    out.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(rows), args.per_sheet):
        chunk = rows[i:i + args.per_sheet]
        W = max(r.shape[1] for r in chunk)
        sheet = np.vstack([np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0)), constant_values=255) for r in chunk])
        cv2.imwrite(str(out / f"sheet_{i // args.per_sheet + 1}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"{len(rows)} riders -> {out}\\sheet_*.jpg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Render video ground truth for human review")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--video", required=True, help="The clip the ground truth was made on")
    ap.add_argument("--hires", default=None, help="Optional higher-resolution copy of the same clip for sharper crops")
    ap.add_argument("--out", default=None)
    ap.add_argument("--per-sheet", type=int, default=5)
    main(ap.parse_args())
