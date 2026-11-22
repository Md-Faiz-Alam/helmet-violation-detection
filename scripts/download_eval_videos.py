"""Download the video benchmark clips listed in eval/video_gt/videos.yaml into data/videos/.

    python scripts/download_eval_videos.py              # all clips (skips the ones already there)
    python scripts/download_eval_videos.py --split test # only the test clips

Clips with `section_s: [start, end]` are cut to that section after downloading (the full file is
kept as <name>_full.mp4), so everyone evaluates on exactly the same frames.
"""

import argparse
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import yaml


def cut_section(src, dst, start_s, end_s):
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    first, last = round(start_s * fps), round(end_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    out = cv2.VideoWriter(str(dst), cv2.VideoWriter.fourcc(*"mp4v"), fps, size)
    n = 0
    for _ in range(first, last):
        ok, frame = cap.read()
        if not ok:
            break
        out.write(frame)
        n += 1
    cap.release()
    out.release()
    return n


def download(url, dst):
    """Download a file; some hosts refuse Python's HTTP client (HTTP 403), so fall back to curl, if installed."""
    try:
        with urllib.request.urlopen(url, timeout=600) as r, open(dst, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
    except urllib.error.HTTPError as e:
        Path(dst).unlink(missing_ok=True)
        if e.code != 403 or not shutil.which("curl"):
            raise
        subprocess.run(["curl", "-sSL", "--fail", "-o", str(dst), url], check=True, timeout=600)


def main(args):
    manifest = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for v in manifest["videos"]:
        if args.split and v["split"] != args.split:
            continue
        final = out_dir / f"{v['name']}.mp4"
        if final.exists():
            print(f"✓ {final.name} (already there)")
            continue
        raw = out_dir / (f"{v['name']}_full.mp4" if "section_s" in v else final.name)
        if not raw.exists():
            print(f"↓ {v['name']} from {v['page']} ({v['creator']}, Pexels license)")
            download(v["download"], raw)
        if "section_s" in v:
            start, end = v["section_s"]
            n = cut_section(raw, final, start, end)
            print(f"✂ {final.name}: {start}-{end} s ({n} frames)")
        else:
            print(f"✓ {final.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download the video benchmark clips")
    ap.add_argument("--manifest", default="eval/video_gt/videos.yaml")
    ap.add_argument("--out", default="data/videos")
    ap.add_argument("--split", choices=["test", "val"], default=None)
    main(ap.parse_args())
