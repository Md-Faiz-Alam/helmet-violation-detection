"""End-to-end smoke tests: the real model through each entry point (detect.py, track_video.py, app.py).

Slow (they load the model), so they are excluded from a plain `pytest` run. Run them with:
    pytest -m smoke
They check that everything runs and produces sane output, not accuracy (see reports/ for that).
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("ultralytics")
cv2 = pytest.importorskip("cv2")

pytestmark = pytest.mark.smoke
ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "image1.jpg"  # 7 riders with a helmet, 1 without


def run_script(*args, cwd=ROOT):
    out = subprocess.run([sys.executable, *map(str, args)], cwd=cwd, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=600)
    assert out.returncode == 0, f"{args[0]} failed:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}"
    return out.stdout


def make_clip(path, frames=30):
    """A short pan across a sample photo: enough motion for the tracker, no video download needed."""
    img = cv2.imread(str(SAMPLE))
    big = cv2.resize(img, None, fx=2, fy=2)
    h, w = big.shape[:2]
    cw, ch = int(w * 0.8), int(h * 0.8)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 25, (640, 360))
    for i in range(frames):
        x = int((w - cw) * i / (frames - 1))
        vw.write(cv2.resize(big[(h - ch) // 2:(h - ch) // 2 + ch, x:x + cw], (640, 360)))
    vw.release()


def test_detect_on_sample_image():
    out = run_script("scripts/detect.py", "--source", SAMPLE)
    assert "Processed 1 images" in out
    assert "Without Helmet" in out and "With Helmet" in out


def test_track_video_end_to_end(tmp_path):
    clip = tmp_path / "clip.mp4"
    make_clip(clip)
    db = tmp_path / "violations.db"
    run_script("scripts/track_video.py", "--source", clip, "--no-plates", "--out", tmp_path / "out", "--db", db)
    run = next((tmp_path / "out").iterdir())
    for name in ("annotated.mp4", "tracks.csv", "track_boxes.csv", "summary.json"):
        assert (run / name).exists(), name
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["frames_processed"] == 30
    assert summary["riders"] >= 3  # several heads stay in view for the whole pan
    with open(run / "track_boxes.csv", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) > 30

    # the SQLite log has a finished run, and one row per logged violation
    from violation_log import ViolationLog
    log = ViolationLog(db)
    [r] = log.runs()
    assert r["status"] == "finished" and r["frames_processed"] == 30
    assert len(log.violations(r["id"])) == summary["violations"]
    log.close()


def test_streamlit_app_on_sample(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(ROOT)  # the app uses paths relative to the project root
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
    assert not at.exception
    [x for x in at.radio if x.label == "Input"][0].set_value("Sample").run()
    at.selectbox[0].set_value("image1.jpg").run()
    assert not at.exception
    metrics = {m.label: int(m.value) for m in at.metric}
    assert metrics["🚨 Without helmet"] >= 1
    assert metrics["✅ With helmet"] >= 1
