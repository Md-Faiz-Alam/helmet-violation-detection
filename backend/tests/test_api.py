"""API tests with a fake detector: no model files, no PyTorch (runs in the fast CI job).

The fake "detector" finds the bright square in a synthetic image or video frame and calls it a head
without a helmet, so the whole path (upload -> job -> tracking -> violation log -> review) runs for real.
"""

import time
from dataclasses import replace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("lap")

from fastapi.testclient import TestClient  # noqa: E402

import ridesafe.api as api  # noqa: E402
from ridesafe.jobs import make_sample  # noqa: E402
from ridesafe.settings import Settings  # noqa: E402
from ridesafe.violation_log import ViolationLog  # noqa: E402

THRESHOLDS = {"With Helmet": 0.25, "Without Helmet": 0.31}


class FakeDetector:
    names = {0: "With Helmet", 1: "Without Helmet"}
    imgsz, dynamic = 640, True

    def __call__(self, img, conf=None, agnostic=False):
        gray = img.max(axis=2)
        ys, xs = np.nonzero(gray > 200)
        if not len(xs):
            return np.zeros((0, 6), np.float32)
        return np.array([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, 0.9, 1]], np.float32)


def square_image(x=100, y=80, size=40, w=320, h=240):
    img = np.zeros((h, w, 3), np.uint8)
    img[y:y + size, x:x + size] = 255
    return img


def write_video(path, frames=40, fps=20):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (320, 240))
    for i in range(frames):
        vw.write(square_image(x=60 + 3 * i))
    vw.release()
    return path


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    def fake_components(settings):
        return FakeDetector(), ViolationLog(settings.data_dir / "violations.db"), THRESHOLDS, None

    monkeypatch.setattr(api, "build_components", fake_components)
    clients = []

    def make(**overrides):
        s = replace(Settings(), data_dir=tmp_path / f"data{len(clients)}", plates=False, vid_stride=2, **overrides)
        c = TestClient(api.create_app(s))
        c.__enter__()
        clients.append(c)
        return c

    yield make
    for c in clients:
        c.__exit__(None, None, None)


def upload_video(client, path, name="clip.mp4"):
    with open(path, "rb") as f:
        r = client.post("/jobs/video", files={"file": (name, f, "video/mp4")})
    return r


def wait_finished(client, job_id, timeout=60):
    t0, job = time.time(), {}
    while time.time() - t0 < timeout:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("finished", "failed"):
            return job
        time.sleep(0.2)
    raise TimeoutError(job)


def png(img):
    return cv2.imencode(".png", img)[1].tobytes()


def test_health_and_model(make_client):
    c = make_client()
    assert c.get("/health").json()["status"] == "ok"
    m = c.get("/model").json()
    assert m["classes"] == ["With Helmet", "Without Helmet"] and m["thresholds"] == THRESHOLDS
    assert m["demo"] is False and m["limits"]["vid_stride"] == 2


def test_detect(make_client):
    c = make_client(max_image_mb=0.01)
    r = c.post("/detect", files={"file": ("a.png", png(square_image()), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert (body["width"], body["height"], body["without_helmet"], body["with_helmet"]) == (320, 240, 1, 0)
    assert body["detections"][0]["box"] == [100, 80, 140, 120]
    assert c.post("/detect", files={"file": ("a.gif", b"GIF89a", "image/gif")}).status_code == 415
    assert c.post("/detect", files={"file": ("a.png", b"not an image", "image/png")}).status_code == 422
    big = np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8)  # ~270 kB as PNG
    assert c.post("/detect", files={"file": ("big.png", png(big), "image/png")}).status_code == 413


def test_video_job_end_to_end(make_client, tmp_path):
    c = make_client()
    r = upload_video(c, write_video(tmp_path / "clip.mp4"))
    assert r.status_code == 202
    job_id, token = r.json()["job"]["id"], r.json()["owner_token"]
    job = wait_finished(c, job_id)
    assert job["status"] == "finished", job["error"]
    assert job["summary"]["violations"] == 1 and job["progress"]["frames_done"] == 40

    tracks = c.get(f"/jobs/{job_id}/tracks").json()
    assert len(tracks["frames"]) == 20 and tracks["fps"] == pytest.approx(20)
    [(tid, track)] = tracks["tracks"].items()
    assert track["label"] == "violation" and track["violation_id"] is not None
    frame_idx, boxes = tracks["frames"][10]
    assert frame_idx == 21 and boxes[0][0] == int(tid) and boxes[0][5] == "violation"

    video = c.get(f"/jobs/{job_id}/video", headers={"Range": "bytes=0-9"})
    assert video.status_code == 206 and len(video.content) == 10

    [v] = c.get("/violations", params={"job_id": job_id}).json()
    assert v["job_id"] == job_id and v["review_status"] == "pending" and v["editable"]
    assert c.get(v["snapshot_url"]).headers["content-type"] == "image/jpeg"
    r = c.patch(f"/violations/{v['id']}", json={"review_status": "confirmed"})
    assert r.json()["review_status"] == "confirmed"
    assert c.get("/violations", params={"review_status": "confirmed"}).json()[0]["id"] == v["id"]
    assert c.patch(f"/violations/{v['id']}", json={"review_status": "maybe"}).status_code == 422

    assert c.get("/metrics").json()["jobs"] == {"finished": 1}
    assert c.delete(f"/jobs/{job_id}", headers={"X-Owner-Token": token}).status_code == 204
    assert c.get(f"/jobs/{job_id}").status_code == 404
    assert c.get(f"/violations/{v['id']}").status_code == 404


def test_video_upload_validation(make_client, tmp_path):
    c = make_client(max_video_seconds=1)
    assert upload_video(c, write_video(tmp_path / "a.mp4"), name="a.txt").status_code == 415
    (tmp_path / "bad.mp4").write_bytes(b"\x00" * 1000)
    assert upload_video(c, tmp_path / "bad.mp4").status_code == 422
    r = upload_video(c, write_video(tmp_path / "long.mp4"))  # 2 s at 20 FPS, limit 1 s
    assert r.status_code == 422 and "too long" in r.json()["detail"]


def test_demo_mode_ownership_and_masking(make_client, tmp_path):
    assert api.mask_plate("MP04ZD7116") == "MP04 •• 7116"
    c = make_client(demo=True)
    r = upload_video(c, write_video(tmp_path / "clip.mp4"))
    job_id, token = r.json()["job"]["id"], r.json()["owner_token"]
    wait_finished(c, job_id)
    mine, other = {"X-Owner-Token": token}, {"X-Owner-Token": "someone-else"}

    assert [j["id"] for j in c.get("/jobs", headers=mine).json()] == [job_id]
    assert c.get("/jobs", headers=other).json() == [] and c.get("/jobs").json() == []
    assert c.get("/violations", headers=other).json() == []
    [v] = c.get("/violations", headers=mine).json()
    assert v["editable"] and not c.get(f"/violations/{v['id']}", headers=other).json()["editable"]
    assert c.patch(f"/violations/{v['id']}", json={"review_status": "rejected"}, headers=other).status_code == 403
    assert c.patch(f"/violations/{v['id']}", json={"review_status": "rejected"}).status_code == 403
    assert c.patch(f"/violations/{v['id']}", json={"review_status": "rejected"}, headers=mine).status_code == 200
    assert c.get(f"/violations/{v['id']}/plate").status_code == 404
    assert c.delete(f"/jobs/{job_id}", headers=other).status_code == 403


def test_one_client_token_owns_all_its_uploads(make_client, tmp_path):
    c = make_client(demo=True)
    me = {"X-Owner-Token": "browser-token-0123456789"}
    clip = write_video(tmp_path / "clip.mp4")
    ids = []
    for _ in range(2):
        with open(clip, "rb") as f:
            r = c.post("/jobs/video", files={"file": ("clip.mp4", f, "video/mp4")}, headers=me)
        assert r.json()["owner_token"] == me["X-Owner-Token"]
        ids.append(r.json()["job"]["id"])
    for job_id in ids:
        wait_finished(c, job_id)
    assert sorted(j["id"] for j in c.get("/jobs", headers=me).json()) == sorted(ids)
    assert len(c.get("/violations", headers=me).json()) == 2
    with open(clip, "rb") as f:
        bad = c.post("/jobs/video", files={"file": ("clip.mp4", f, "video/mp4")}, headers={"X-Owner-Token": "short"})
    assert bad.status_code == 422


def test_uploads_expire_with_their_violations(make_client, tmp_path):
    c = make_client(upload_ttl_s=60)
    job_id = upload_video(c, write_video(tmp_path / "clip.mp4")).json()["job"]["id"]
    wait_finished(c, job_id)
    assert len(c.get("/violations").json()) == 1
    manager = c.app.state.jobs
    assert manager.expire(now=time.time() + 30) == 0
    assert manager.expire(now=time.time() + 120) == 1
    assert c.get(f"/jobs/{job_id}").status_code == 404 and c.get("/violations").json() == []


def test_read_only_sample_job(make_client, tmp_path, monkeypatch):
    def fake_components(settings):
        return FakeDetector(), ViolationLog(settings.data_dir / "violations.db"), THRESHOLDS, None

    monkeypatch.setattr(api, "build_components", fake_components)
    sample = tmp_path / "sample"
    make_sample(write_video(tmp_path / "clip.mp4"), sample, replace(Settings(), plates=False, vid_stride=2))
    assert (sample / "job.json").exists() and (sample / "tracks.json").exists()

    for _ in range(2):  # a restart must not import the sample's violations twice
        c = make_client(demo=True, sample_dir=str(sample), upload_ttl_s=1)
        [job] = c.get("/jobs").json()
        assert job["readonly"] and not job["editable"] and job["expires_in_s"] is None
        assert c.get(f"/jobs/{job['id']}/tracks").status_code == 200
        [v] = c.get("/violations").json()
        assert c.patch(f"/violations/{v['id']}", json={"review_status": "confirmed"}).status_code == 403
        assert c.get(v["snapshot_url"]).status_code == 200
        assert c.app.state.jobs.expire(now=time.time() + 10_000) == 0
