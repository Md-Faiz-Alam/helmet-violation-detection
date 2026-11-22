"""Draft a video ground-truth file from a tracker run, for a human to correct (tracker-drafted labelling).

    python scripts/track_video.py --source data/videos/clip.mp4 --vid-stride 2 --no-plates
    python scripts/video_gt_draft.py --video data/videos/clip.mp4 --run results/video/clip_<timestamp>

Writes to eval/video_gt/<clip>_draft.yaml plus review images in eval/video_gt/review_<clip>/:
  track_XXXX.jpg      the head at 3 moments of each track (first seen, middle, closest)
  overview_XXs.jpg    the full frame every --every seconds with all tracked boxes, to spot missed riders

The draft lists one rider per track with `helmet: null`. The annotator must, for every rider:
  * set helmet: true / false / unclear (unclear = cannot tell even when looking closely);
  * set plate: the plate text if a person can read it, else null;
  * delete entries that are not riders (pedestrians, ghosts) and merge entries that are the same
    rider (copy all keyframes into one entry): the tracker's own mistakes must not become ground truth;
  * add riders the tracker missed, with at least one keyframe (frame: [x1, y1, x2, y2] head box).
Then save it as eval/video_gt/<clip>.yaml (without _draft) and fill in `annotation.reviewed_by`.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml


def read_boxes(run_dir):
    tracks = defaultdict(list)  # track id -> [(frame, (x1, y1, x2, y2), class, conf)]
    with open(Path(run_dir) / "track_boxes.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            box = tuple(int(float(r[k])) for k in ("x1", "y1", "x2", "y2"))
            tracks[int(r["track_id"])].append((int(r["frame"]), box, r["class"], float(r["conf"])))
    return tracks


def keyframes(obs, k=3):
    """First seen, middle, and largest (closest) observation of a track."""
    obs = sorted(obs)
    largest = max(obs, key=lambda o: (o[1][2] - o[1][0]) * (o[1][3] - o[1][1]))
    picks = {obs[0][0]: obs[0], obs[len(obs) // 2][0]: obs[len(obs) // 2], largest[0]: largest}
    return [picks[f] for f in sorted(picks)][:k]


def read_frames(video, frames):
    cap, out = cv2.VideoCapture(str(video)), {}
    for f in sorted(set(frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)  # track_boxes.csv frames are 1-based
        ok, img = cap.read()
        if ok:
            out[f] = img
    cap.release()
    return out


def crop(img, box, pad=1.5, size=220):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    X1, Y1 = max(0, int(x1 - pad * bw)), max(0, int(y1 - pad * bh))
    X2, Y2 = min(w, int(x2 + pad * bw)), min(h, int(y2 + 3 * pad * bh))
    c = img[Y1:Y2, X1:X2].copy()
    cv2.rectangle(c, (x1 - X1, y1 - Y1), (x2 - X1, y2 - Y1), (0, 215, 255), 2)
    return cv2.resize(c, (int(c.shape[1] * size / max(c.shape[0], 1)), size))


def caption(img, text):
    bar = np.full((24, img.shape[1], 3), 0, np.uint8)
    cv2.putText(bar, text, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main(args):
    video, run = Path(args.video), Path(args.run)
    cap = cv2.VideoCapture(str(video))
    fps, n_frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    tracks = {t: o for t, o in read_boxes(run).items() if len(o) >= args.min_obs}
    kfs = {t: keyframes(o) for t, o in tracks.items()}
    every = max(1, int(round(args.every * fps)))
    overview_frames = list(range(1, n_frames + 1, every))
    frames = read_frames(video, [kf[0] for ks in kfs.values() for kf in ks] + overview_frames)

    review = Path(args.out) / f"review_{video.stem}"
    review.mkdir(parents=True, exist_ok=True)
    riders = []
    for t, ks in sorted(kfs.items()):
        tiles = [caption(crop(frames[f], box), f"#{t} f{f} {f / fps:.1f}s {cls[:4]} {conf:.2f}")
                 for f, box, cls, conf in ks if f in frames]
        hmax = max(x.shape[0] for x in tiles)
        tiles = [np.pad(x, ((0, hmax - x.shape[0]), (0, 4), (0, 0))) for x in tiles]
        cv2.imwrite(str(review / f"track_{t:04d}.jpg"), np.hstack(tiles))
        riders.append({"id": f"R{t}", "helmet": None, "plate": None, "note": f"from track {t}",
                       "keyframes": {f: list(box) for f, box, _, _ in ks}})

    boxes_at = defaultdict(list)
    for t, obs in tracks.items():
        for f, box, _, _ in obs:
            boxes_at[f].append((t, box))
    for f in overview_frames:
        if f not in frames:
            continue
        img = frames[f].copy()
        near = min(boxes_at, key=lambda g: abs(g - f)) if boxes_at else None  # closest processed frame
        for t, (x1, y1, x2, y2) in boxes_at.get(near, []):
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 215, 255), 2)
            cv2.putText(img, f"#{t}", (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)
        cv2.imwrite(str(review / f"overview_{f / fps:05.1f}s.jpg"), caption(img, f"frame {f}  {f / fps:.1f}s"))

    gt = {
        "video": video.name,
        "frames": n_frames, "fps": round(fps, 3), "resolution": [W, H],
        "source": args.source_url, "download": args.download_url,
        "annotation": {"method": "drafted from tracker boxes by scripts/video_gt_draft.py, corrected by hand",
                       "annotated_by": None, "reviewed_by": None,
                       "rules": "rider = person on a moving two-wheeler; helmet: true/false/unclear; "
                                "plate only if a person can read it; keyframes are head boxes in pixels"},
        "riders": riders,
    }
    out_yaml = Path(args.out) / f"{video.stem}_draft.yaml"
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(gt, f, sort_keys=False, default_flow_style=None, width=140)
    print(f"Draft: {out_yaml} ({len(riders)} tracks) | review images: {review}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Draft video ground truth from a tracker run")
    ap.add_argument("--video", required=True)
    ap.add_argument("--run", required=True, help="results/video/<clip>_<timestamp> folder from track_video.py")
    ap.add_argument("--out", default="eval/video_gt")
    ap.add_argument("--min-obs", type=int, default=3, help="Skip tracks seen in fewer frames")
    ap.add_argument("--every", type=float, default=1.0, help="Seconds between overview frames")
    ap.add_argument("--source-url", default=None)
    ap.add_argument("--download-url", default=None)
    main(ap.parse_args())
