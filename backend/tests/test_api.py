"""API tests with a fake detector: no model files, no PyTorch (runs in the fast CI job).

The fake "detector" finds the bright square in a synthetic image or video frame and calls it a head
without a helmet, so the whole path (upload -> job -> tracking -> violation log -> review) runs for real.
Clients act as a visitor's browser (their own X-Owner-Token) unless a test says otherwise; AUTHOR adds
the author's key.
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
ADMIN_KEY = "author-key-" + "x" * 32
AUTHOR = {"Authorization": f"Bearer {ADMIN_KEY}"}
ME = {"X-Owner-Token": "visitor-me-0123456789"}
OTHER = {"X-Owner-Token": "visitor-other-0123456789"}


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
        base = {"plates": False, "riding_check": False, "vid_stride": 2, "admin_token": ADMIN_KEY, "rate_detect_per_min": 0,
                "rate_uploads_per_hour": 0}
        s = replace(Settings(), data_dir=tmp_path / f"data{len(clients)}", **{**base, **overrides})
        c = TestClient(api.create_app(s), headers=ME)  # requests come from "my" browser by default
        c.__enter__()
        clients.append(c)
        return c

    yield make
    for c in clients:
        c.__exit__(None, None, None)


def upload_video(client, path, name="clip.mp4", headers=None):
    with open(path, "rb") as f:
        return client.post("/jobs/video", files={"file": (name, f, "video/mp4")}, headers=headers)


def wait_finished(client, job_id, timeout=60):
    t0, job = time.time(), {}
    while time.time() - t0 < timeout:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("finished", "failed"):
            return job
        time.sleep(0.2)
    raise TimeoutError(job)


def finished_job(client, tmp_path, headers=None):
    """Upload the synthetic clip and wait: returns (job id, the one violation it logs)."""
    job_id = upload_video(client, write_video(tmp_path / "clip.mp4"), headers=headers).json()["job"]["id"]
    assert wait_finished(client, job_id)["status"] == "finished"
    [v] = client.get("/violations", params={"job_id": job_id}, headers=headers).json()
    return job_id, v


def png(img):
    return cv2.imencode(".png", img)[1].tobytes()


# ----------------------------------------------------------------------------- service

def test_health_model_and_whoami(make_client):
    c = make_client()
    assert c.get("/health").json()["status"] == "ok"
    m = c.get("/model").json()
    assert m["classes"] == ["With Helmet", "Without Helmet"] and m["thresholds"] == THRESHOLDS
    assert m["demo"] is True and m["author_enabled"] is True  # safe default: the public-site rules
    assert c.get("/me").json() == {"role": "visitor"}
    assert c.get("/me", headers=AUTHOR).json() == {"role": "author"}
    assert c.get("/me", headers={"Authorization": "Bearer wrong-key"}).status_code == 401
    assert c.get("/me", headers={"Authorization": "Basic abc"}).status_code == 401


def test_metrics_are_for_the_author_only(make_client):
    c = make_client()
    assert c.get("/metrics").status_code == 403
    assert "routes" in c.get("/metrics", headers=AUTHOR).json()


def test_weak_or_missing_author_key(make_client):
    with pytest.raises(ValueError, match="at least 32"):
        api.create_app(replace(Settings(), admin_token="short"))
    c = make_client(admin_token="")  # no author configured: nobody can claim to be one
    assert c.get("/me", headers=AUTHOR).status_code == 401


def test_cors_allows_the_web_app_and_its_previews_only(make_client):
    c = make_client(cors_origins=("https://app.example.net",),
                    cors_origin_regex=r"https://app-\d+\.example\.net")
    allowed = "access-control-allow-origin"
    for origin, ok in [("https://app.example.net", True), ("https://app-12.example.net", True),
                       ("https://evil.example.com", False), ("https://app-12.example.net.evil.com", False)]:
        r = c.get("/health", headers={"Origin": origin})
        assert (r.headers.get(allowed) == origin) is ok, origin


def test_docs_work_behind_a_path_prefix(make_client):
    """The web app proxies the API under /api: the docs must not ask for /openapi.json at the site root."""
    c = make_client()
    page = c.get("/docs").text
    assert "url: 'openapi.json'" in page and "'/openapi.json'" not in page
    spec = c.get("/openapi.json").json()
    assert spec["servers"][0]["url"] == "." and "/detect" in spec["paths"]


# ----------------------------------------------------------------------------- images

def test_detect(make_client):
    c = make_client(max_image_mb=0.01)
    r = c.post("/detect", files={"file": ("a.png", png(square_image()), "image/png")})
    assert r.status_code == 200 and r.headers["x-content-type-options"] == "nosniff"
    body = r.json()
    assert (body["width"], body["height"], body["without_helmet"], body["with_helmet"]) == (320, 240, 1, 0)
    assert body["detections"][0]["box"] == [100, 80, 140, 120]
    assert c.post("/detect", files={"file": ("a.gif", b"GIF89a", "image/gif")}).status_code == 415
    assert c.post("/detect", files={"file": ("a.png", b"not an image", "image/png")}).status_code == 422
    big = np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8)  # ~270 kB as PNG
    assert c.post("/detect", files={"file": ("big.png", png(big), "image/png")}).status_code == 413


def test_rate_limits_per_visitor_not_for_the_author(make_client):
    c = make_client(rate_detect_per_min=2)
    photo = {"file": ("a.png", png(square_image()), "image/png")}
    assert [c.post("/detect", files=photo).status_code for _ in range(3)] == [200, 200, 429]
    r = c.post("/detect", files=photo)
    assert r.status_code == 429 and int(r.headers["retry-after"]) > 0
    assert all(c.post("/detect", files=photo, headers=AUTHOR).status_code == 200 for _ in range(3))


def test_rate_limits_behind_a_proxy_cannot_be_dodged(make_client):
    """Behind a trusted proxy the client is the LAST X-Forwarded-For entry (added by the proxy)."""
    c = make_client(rate_detect_per_min=1, trusted_proxy=True)
    photo = {"file": ("a.png", png(square_image()), "image/png")}

    def send(xff):
        return c.post("/detect", files=photo, headers={"X-Forwarded-For": xff}).status_code

    assert send("10.0.0.1") == 200 and send("10.0.0.2") == 200  # two real visitors
    assert send("1.2.3.4, 10.0.0.1") == 429  # a faked first entry does not make a new visitor


def test_upload_rate_limit(make_client, tmp_path):
    c = make_client(rate_uploads_per_hour=1)
    clip = write_video(tmp_path / "clip.mp4")
    assert upload_video(c, clip).status_code == 202
    assert upload_video(c, clip).status_code == 429
    assert upload_video(c, clip, headers=AUTHOR).status_code == 202


# ----------------------------------------------------------------------------- video jobs

def test_video_job_end_to_end(make_client, tmp_path):
    c = make_client()
    r = upload_video(c, write_video(tmp_path / "clip.mp4"))
    assert r.status_code == 202 and r.json()["owner_token"] == ME["X-Owner-Token"]
    job_id = r.json()["job"]["id"]
    job = wait_finished(c, job_id)
    assert job["status"] == "finished", job["error"]
    assert job["summary"]["violations"] == 1 and job["progress"]["frames_done"] == 40 and job["editable"]

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
    r = c.patch(f"/violations/{v['id']}", json={"review_status": "confirmed"}).json()
    assert r["review_status"] == "confirmed" and r["reviewed_by"] == "uploader" and r["reviewed_at"]
    assert c.get("/violations", params={"review_status": "confirmed"}).json()[0]["id"] == v["id"]
    assert c.patch(f"/violations/{v['id']}", json={"review_status": "maybe"}).status_code == 422

    assert c.delete(f"/jobs/{job_id}").status_code == 204
    assert c.get(f"/jobs/{job_id}").status_code == 404
    assert c.get(f"/violations/{v['id']}").status_code == 404


def test_video_upload_validation(make_client, tmp_path):
    c = make_client(max_video_seconds=1)
    assert upload_video(c, write_video(tmp_path / "a.mp4"), name="a.txt").status_code == 415
    (tmp_path / "bad.mp4").write_bytes(b"\x00" * 1000)
    assert upload_video(c, tmp_path / "bad.mp4").status_code == 422
    r = upload_video(c, write_video(tmp_path / "long.mp4"))  # 2 s at 20 FPS, limit 1 s
    assert r.status_code == 422 and "too long" in r.json()["detail"]
    assert upload_video(c, write_video(tmp_path / "a.mp4"), headers={"X-Owner-Token": "short"}).status_code == 422


# ----------------------------------------------------------------------------- who may do what

def test_visitors_only_see_and_change_their_own_uploads(make_client, tmp_path):
    assert api.mask_plate("MP04ZD7116") == "MP04 •• 7116"
    c = make_client()
    job_id, v = finished_job(c, tmp_path)

    assert [j["id"] for j in c.get("/jobs").json()] == [job_id]
    assert c.get("/jobs", headers=OTHER).json() == []
    assert c.get("/violations", headers=OTHER).json() == []
    assert c.get(f"/violations/{v['id']}", headers=OTHER).status_code == 404  # guessing the number reveals nothing
    for status in ("confirmed", "rejected"):
        assert c.patch(f"/violations/{v['id']}", json={"review_status": status}, headers=OTHER).status_code == 404
    assert not c.get(f"/jobs/{job_id}", headers=OTHER).json()["editable"]
    assert c.delete(f"/jobs/{job_id}", headers=OTHER).status_code == 403
    assert c.patch(f"/violations/{v['id']}", json={"review_status": "rejected"}).status_code == 200


def test_the_author_sees_and_changes_everything(make_client, tmp_path):
    c = make_client()
    job_id, v = finished_job(c, tmp_path, headers=OTHER)  # someone else's upload
    c.app.state.log.set_plate(v["id"], "MP04ZD7116", 0.9, 3, 5, 120, None)

    assert c.get(f"/violations/{v['id']}", headers=OTHER).json()["plate"] == "MP04 •• 7116"  # masked for visitors
    seen = c.get("/violations", headers=AUTHOR).json()
    assert [x["id"] for x in seen] == [v["id"]] and seen[0]["plate"] == "MP04ZD7116" and seen[0]["editable"]
    assert [j["id"] for j in c.get("/jobs", headers=AUTHOR).json()] == [job_id]

    r = c.patch(f"/violations/{v['id']}", json={"review_status": "rejected"}, headers=AUTHOR)
    assert r.status_code == 200 and r.json()["reviewed_by"] == "author"  # the audit trail shows who
    assert c.get(f"/violations/{v['id']}", headers=OTHER).json()["reviewed_by"] == "author"
    assert c.delete(f"/jobs/{job_id}", headers=AUTHOR).status_code == 204


def test_image_links_are_signed(make_client, tmp_path):
    c = make_client()
    _, v = finished_job(c, tmp_path)
    url = v["snapshot_url"]
    assert "?sig=" in url and c.get(url).status_code == 200
    base = url.split("?")[0]
    assert c.get(base).status_code == 404
    assert c.get(base + "?sig=" + "0" * 32).status_code == 404
    assert v["plate_image_url"] is None


def test_local_install_shows_full_plates(make_client, tmp_path):
    c = make_client(demo=False)
    _, v = finished_job(c, tmp_path)
    c.app.state.log.set_plate(v["id"], "MP04ZD7116", 0.9, 3, 5, 120, None)
    assert c.get(f"/violations/{v['id']}").json()["plate"] == "MP04ZD7116"
    assert c.get(f"/violations/{v['id']}", headers=OTHER).status_code == 404  # still only the uploader's


def test_one_client_token_owns_all_its_uploads(make_client, tmp_path):
    c = make_client()
    clip = write_video(tmp_path / "clip.mp4")
    ids = [upload_video(c, clip).json()["job"]["id"] for _ in range(2)]
    for job_id in ids:
        wait_finished(c, job_id)
    assert sorted(j["id"] for j in c.get("/jobs").json()) == sorted(ids)
    assert len(c.get("/violations").json()) == 2


# ----------------------------------------------------------------------------- robustness

def test_a_failing_job_is_reported_and_the_worker_keeps_going(make_client, tmp_path, caplog):
    c = make_client()
    real_call = FakeDetector.__call__
    calls = {"n": 0}

    def flaky(self, img, conf=None, agnostic=False):
        calls["n"] += 1
        if calls["n"] == 5:
            raise RuntimeError("model crashed")
        return real_call(self, img, conf, agnostic)

    clip = write_video(tmp_path / "clip.mp4")
    try:
        FakeDetector.__call__ = flaky
        failed = wait_finished(c, upload_video(c, clip).json()["job"]["id"])
    finally:
        FakeDetector.__call__ = real_call
    assert failed["status"] == "failed" and "model crashed" in failed["error"]
    run = next(r for r in c.app.state.log.runs() if r["source"] == f"job:{failed['id']}")
    assert run["status"] == "failed"
    logged = [r for r in caplog.records if r.name == "ridesafe.jobs" and r.levelname == "ERROR"]
    assert logged and failed["id"] in logged[0].getMessage() and "model crashed" in logged[0].exc_text
    assert wait_finished(c, upload_video(c, clip).json()["job"]["id"])["status"] == "finished"


def test_too_many_waiting_videos_are_refused(make_client, tmp_path):
    c = make_client(max_queued=0)
    assert upload_video(c, write_video(tmp_path / "clip.mp4")).status_code == 429


def test_uploads_expire_with_their_violations(make_client, tmp_path):
    c = make_client(upload_ttl_s=60)
    job_id, _ = finished_job(c, tmp_path)
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
    make_sample(write_video(tmp_path / "clip.mp4"), sample, replace(Settings(), plates=False, riding_check=False, vid_stride=2))
    assert (sample / "job.json").exists() and (sample / "tracks.json").exists()

    for _ in range(2):  # a restart must not import the sample's violations twice
        c = make_client(sample_dir=str(sample), upload_ttl_s=1)
        [job] = c.get("/jobs").json()
        assert job["readonly"] and not job["editable"] and job["expires_in_s"] is None
        assert c.get(f"/jobs/{job['id']}/tracks").status_code == 200
        [v] = c.get("/violations").json()
        for headers in (None, AUTHOR):  # the sample is read-only for everyone
            r = c.patch(f"/violations/{v['id']}", json={"review_status": "confirmed"}, headers=headers)
            assert r.status_code == 403
        assert c.get(v["snapshot_url"]).status_code == 200
        assert c.app.state.jobs.expire(now=time.time() + 10_000) == 0


class NoVehicles:
    """A COCO detector stand-in that never finds a motorcycle or bicycle."""
    names = {0: "person", 1: "bicycle", 3: "motorcycle"}

    def __call__(self, img, conf=None, agnostic=False):
        return np.zeros((0, 6), np.float32)


def test_riding_check_stops_logging_people_not_on_a_motorcycle(make_client, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "build_vehicle_detector", lambda settings, plate_reader: NoVehicles())
    c = make_client(riding_check=True)
    job_id = upload_video(c, write_video(tmp_path / "clip.mp4")).json()["job"]["id"]
    assert wait_finished(c, job_id)["status"] == "finished"
    assert c.get("/violations", params={"job_id": job_id}).json() == []  # same clip as above, but nobody rides
    tracks = c.get(f"/jobs/{job_id}/tracks").json()
    assert "not_riding" in {t["label"] for t in tracks["tracks"].values()}
