"""End-to-end smoke tests: the real model through each entry point (detect.py, track_video.py, the API).

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
    run_script("scripts/track_video.py", "--source", clip, "--vid-stride", "1", "--no-plates", "--out", tmp_path / "out",
               "--db", db)
    run = next((tmp_path / "out").iterdir())
    for name in ("annotated.mp4", "tracks.csv", "track_boxes.csv", "summary.json"):
        assert (run / name).exists(), name
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["frames_processed"] == 30
    assert summary["riders"] >= 3  # several heads stay in view for the whole pan
    with open(run / "track_boxes.csv", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) > 30

    # the SQLite log has a finished run, and one row per logged violation
    from ridesafe.violation_log import ViolationLog
    log = ViolationLog(db)
    [r] = log.runs()
    assert r["status"] == "finished" and r["frames_processed"] == 30
    assert len(log.violations(r["id"])) == summary["violations"]
    log.close()


def test_onnx_export_matches_pytorch(tmp_path):
    """Export the real model, run the torch-free detector and compare with PyTorch on the sample photo."""
    onnx_path = tmp_path / "best.onnx"
    run_script("scripts/export_onnx.py", "--out", onnx_path)

    from ultralytics import YOLO

    from metrics import box_iou, predict_array
    from ridesafe.detector import OnnxDetector

    img = cv2.imread(str(SAMPLE))
    ref = predict_array(YOLO(str(ROOT / "models" / "best.pt")), img, conf=0.25, device="cpu")
    got = OnnxDetector(str(onnx_path))(img, conf=0.25)
    assert sorted(got[:, 5].tolist()) == sorted(ref[:, 5].tolist())  # same boxes per class
    assert box_iou(got[:, :4], ref[:, :4]).max(axis=1).min() > 0.99  # dynamic export: same input as PyTorch

    # The dynamic export also runs at a larger input size (for far-away riders), in the same image pixels
    big = OnnxDetector(str(onnx_path), imgsz=960)
    assert big.imgsz == 960
    got_big = big(img, conf=0.25)
    assert len(got_big) and got_big[:, [0, 2]].max() <= img.shape[1] and got_big[:, [1, 3]].max() <= img.shape[0]
    with pytest.raises(ValueError, match="multiple of"):
        OnnxDetector(str(onnx_path), imgsz=1000)


def test_api_with_real_models(tmp_path, monkeypatch):
    """The API on the real ONNX models (exported by CI before the smoke tests): /detect gives the same
    boxes as PyTorch at the deployed thresholds, and a video job runs end to end with plate reading on."""
    import time
    from dataclasses import replace

    from fastapi.testclient import TestClient
    from ultralytics import YOLO

    from metrics import predict_array
    from ridesafe.api import create_app
    from ridesafe.settings import Settings
    from utils import load_thresholds

    monkeypatch.chdir(ROOT)
    settings = replace(Settings(), data_dir=tmp_path / "api", plates=True)
    with TestClient(create_app(settings)) as c:
        assert c.get("/model").json()["plates_enabled"]
        body = c.post("/detect", files={"file": ("image1.jpg", SAMPLE.read_bytes(), "image/jpeg")}).json()
        thr = load_thresholds() or {}
        model = YOLO(str(ROOT / "models" / "best.pt"))
        ref = [p for p in predict_array(model, str(SAMPLE), conf=0.1, device="cpu")
               if p[4] >= thr[model.names[int(p[5])]]]
        assert len(body["detections"]) == len(ref)
        assert body["without_helmet"] == sum(int(p[5]) == 1 for p in ref) >= 1

        clip = tmp_path / "clip.mp4"
        make_clip(clip)
        with open(clip, "rb") as f:
            job_id = c.post("/jobs/video", files={"file": ("clip.mp4", f, "video/mp4")}).json()["job"]["id"]
        job = {}
        for _ in range(300):
            job = c.get(f"/jobs/{job_id}").json()
            if job["status"] in ("finished", "failed"):
                break
            time.sleep(0.5)
        assert job["status"] == "finished", job["error"]
        assert job["summary"]["riders"] >= 3
        tracks = c.get(f"/jobs/{job_id}/tracks").json()
        assert len(tracks["frames"]) == 15  # 30 frames, every 2nd
