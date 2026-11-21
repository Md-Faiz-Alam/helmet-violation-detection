"""Download a Roboflow Universe dataset version in YOLO format, without the `roboflow` package
(it pins opencv-python-headless, which breaks OpenCV windows next to opencv-python).

    $env:ROBOFLOW_API_KEY="your_private_key"        # PowerShell; never commit or print it
    python scripts/download_roboflow.py --info       # show size and splits only
    python scripts/download_roboflow.py              # download + extract into data/extra/<project>_v<version>/

Defaults: "Helmet, Rider, ANPR Detection" v12 by OnlyTusik (CC BY 4.0), used to add non-helmet
headwear examples (scripts/mine_headwear.py). The API key is read from ROBOFLOW_API_KEY only.
"""

import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

API = "https://api.roboflow.com"


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def export_info(workspace, project, version, fmt, key):
    """Ask Roboflow for the export; it may need a little time to be generated the first time."""
    url = f"{API}/{workspace}/{project}/{version}/{fmt}?" + urllib.parse.urlencode({"api_key": key})
    for attempt in range(30):
        try:
            data = get_json(url)
        except urllib.error.HTTPError as e:
            raise SystemExit(f"Roboflow API error {e.code}: check the key, workspace, project and version.") from None
        link = (data.get("export") or {}).get("link")
        if link:
            return data, link
        print(f"  export not ready yet, waiting... ({attempt + 1}/30)")
        time.sleep(10)
    raise SystemExit("Export was not ready after 5 minutes; try again later.")


def safe_extract(zip_path, dest):
    """Extract, refusing any member that would land outside `dest` (zip path traversal)."""
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            target = (dest / m.filename).resolve()
            if dest not in target.parents and target != dest:
                raise SystemExit(f"Unsafe path in archive: {m.filename}")
        z.extractall(dest)


def main(args):
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit('Set your key first, e.g. PowerShell: $env:ROBOFLOW_API_KEY="..."')
    data, link = export_info(args.workspace, args.project, args.version, args.format, key)
    version = data.get("version") or {}
    splits = version.get("splits") or {}
    with urllib.request.urlopen(urllib.request.Request(link, method="HEAD"), timeout=60) as r:
        size = int(r.headers.get("Content-Length") or 0)
    print(f"{args.workspace}/{args.project} v{args.version} ({args.format}): {version.get('images', '?')} images, "
          f"splits {splits or '?'}, download {size / 1e6:.0f} MB")
    if args.info:
        return

    out = Path(args.out or f"data/extra/{args.project}_v{args.version}")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} already exists and is not empty; delete it to download again.")
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out.with_suffix(".zip")
    print(f"Downloading to {zip_path} ...")
    with urllib.request.urlopen(link, timeout=600) as r, open(zip_path, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    safe_extract(zip_path, out)
    zip_path.unlink()
    n = sum(1 for _ in out.rglob("*.jpg"))
    print(f"Extracted {n} images into {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download a Roboflow dataset version (YOLO format)")
    ap.add_argument("--workspace", default="onlytusik")
    ap.add_argument("--project", default="helmet-rider-anpr-detection")
    ap.add_argument("--version", type=int, default=12)
    ap.add_argument("--format", default="yolov8")
    ap.add_argument("--out", default=None)
    ap.add_argument("--info", action="store_true", help="Only print size and splits")
    sys.exit(main(ap.parse_args()))
