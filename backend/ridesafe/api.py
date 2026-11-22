"""RideSafe AI HTTP API (FastAPI, ONNX Runtime, no PyTorch).

    uvicorn ridesafe.api:app --port 8000        # then open http://localhost:8000/docs

Endpoints
    GET    /health                       liveness + whether the models are loaded
    GET    /model                        classes, thresholds, input size, limits, demo mode
    GET    /metrics                      request counts and latencies, jobs, frames processed
    POST   /detect                       image -> helmet / no-helmet boxes (JSON)
    POST   /jobs/video                   upload a video -> async job (202) + owner token
    GET    /jobs                         jobs the caller may see
    GET    /jobs/{id}                    status, progress, summary
    GET    /jobs/{id}/tracks             per-frame tracked boxes + final decision per track (finished jobs)
    GET    /jobs/{id}/video              the uploaded video (supports HTTP range requests for seeking)
    DELETE /jobs/{id}                    delete a job and its results (owner)
    GET    /violations                   logged violations (filter by job and review status)
    GET    /violations/{id}              one violation
    PATCH  /violations/{id}              set review_status: pending / confirmed / rejected (owner)
    GET    /violations/{id}/snapshot     the violator's snapshot (JPEG)
    GET    /violations/{id}/plate        the plate crop (JPEG; not in demo mode)

Ownership: send a random X-Owner-Token (16-128 characters) with POST /jobs/video and with later requests;
without one, POST /jobs/video makes a token and returns it once. A browser keeps one token for all its uploads.
In demo mode (RIDESAFE_DEMO=true) visitors see and review only their own uploads plus the read-only
sample, plates are partly masked (MP04 •• 7116) and plate crops are not served. Settings: ridesafe.settings.
"""

import os
import shutil
import tempfile
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from ridesafe import __version__
from ridesafe.detector import OnnxDetector
from ridesafe.jobs import JobManager, probe_video
from ridesafe.settings import Settings
from ridesafe.thresholds import load_thresholds
from ridesafe.tracking import VIOLATION_CLASS
from ridesafe.violation_log import ViolationLog

IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
CHUNK = 1 << 20


# ----------------------------------------------------------------------------- setup

def build_components(settings):
    """Load the models and the log: (detector, violation log, thresholds, plate reader or None)."""
    detector = OnnxDetector(settings.model)
    log = ViolationLog(Path(settings.data_dir) / "violations.db")
    thresholds = load_thresholds(settings.thresholds) or {}
    plate_reader = None
    if settings.plates and os.path.exists(settings.vehicle_model):
        try:
            from ridesafe.plate_reader import PlateReader
            plate_reader = PlateReader(settings.vehicle_model)
        except ImportError:  # rapidocr_onnxruntime not installed: the API works without plates
            plate_reader = None
    return detector, log, thresholds, plate_reader


def mask_plate(plate):
    """MP04ZD7116 -> MP04 •• 7116: enough to see that a plate was read, not enough to identify anyone."""
    if not plate:
        return plate
    return f"{plate[:4]} •• {plate[-4:]}" if len(plate) >= 8 else f"{plate[:2]} ••"


class Metrics:
    """In-process request counters and latency windows for GET /metrics."""

    def __init__(self):
        self.started = time.time()
        self.requests = defaultdict(int)
        self.errors = defaultdict(int)
        self.latency = defaultdict(lambda: deque(maxlen=500))

    def record(self, route, status, ms):
        self.requests[route] += 1
        if status >= 500:
            self.errors[route] += 1
        self.latency[route].append(ms)

    def snapshot(self):
        out = {}
        for route, n in sorted(self.requests.items()):
            lat = np.array(self.latency[route]) if self.latency[route] else np.zeros(1)
            out[route] = {"requests": n, "server_errors": self.errors[route], "latency_ms_mean": round(float(lat.mean()), 1),
                          "latency_ms_p95": round(float(np.percentile(lat, 95)), 1)}
        return out


def get_manager(request: Request) -> JobManager:
    return request.app.state.jobs


def get_owner_token(x_owner_token: Annotated[str | None, Header()] = None) -> str | None:
    return x_owner_token


Manager = Annotated[JobManager, Depends(get_manager)]
Token = Annotated[str | None, Depends(get_owner_token)]  # the X-Owner-Token header


# ----------------------------------------------------------------------------- schemas

class Detection(BaseModel):
    cls: str = Field(description="With Helmet / Without Helmet")
    conf: float
    box: list[float] = Field(description="x1, y1, x2, y2 in image pixels")


class DetectResponse(BaseModel):
    width: int
    height: int
    latency_ms: float
    with_helmet: int
    without_helmet: int
    detections: list[Detection]


class Progress(BaseModel):
    frames_done: int
    frames_total: int


class JobOut(BaseModel):
    id: str
    status: Literal["queued", "running", "finished", "failed"]
    stage: str
    filename: str | None = None
    created_at: str
    fps: float
    width: int
    height: int
    duration_s: float | None
    vid_stride: int
    progress: Progress
    queue_position: int = 0
    processing_fps: float | None = None
    summary: dict | None = None
    error: str | None = None
    readonly: bool = False
    editable: bool = False
    expires_in_s: int | None = None


class JobCreated(BaseModel):
    job: JobOut
    owner_token: str = Field(description="Send as the X-Owner-Token header to review or delete this job")


class ViolationOut(BaseModel):
    id: int
    job_id: str | None
    track_id: int
    video_time_s: float | None
    first_seen_s: float | None
    last_seen_s: float | None
    votes_without_helmet: int
    votes_with_helmet: int
    max_conf: float | None
    plate: str | None
    plate_conf: float | None
    plate_frames: str | None
    review_status: Literal["pending", "confirmed", "rejected"]
    snapshot_url: str | None
    plate_image_url: str | None
    editable: bool


class ReviewIn(BaseModel):
    review_status: Literal["pending", "confirmed", "rejected"]


# ----------------------------------------------------------------------------- app

def create_app(settings=None):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
        detector, log, thresholds, plate_reader = build_components(settings)
        app.state.detector, app.state.log, app.state.thresholds = detector, log, thresholds
        app.state.plates = plate_reader is not None
        app.state.jobs = JobManager(settings, detector, log, thresholds, plate_reader)
        yield
        app.state.jobs.close()
        log.close()

    app = FastAPI(title="RideSafe AI API", version=__version__, lifespan=lifespan,
                  description="Helmet violation detection for two-wheeler riders: image detection, video jobs "
                              "with per-rider decisions, and a review queue. Source code (AGPL-3.0): "
                              "https://github.com/Md-Faiz-Alam/helmet-violation-detection")
    app.state.settings, app.state.metrics = settings, Metrics()
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["*"],
                       allow_headers=["*"], expose_headers=["Content-Range", "Accept-Ranges"])

    upload_limits = {"/detect": settings.max_image_mb, "/jobs/video": settings.max_video_mb}

    @app.middleware("http")
    async def limits_and_metrics(request: Request, call_next):
        limit = upload_limits.get(request.url.path) if request.method == "POST" else None
        length = request.headers.get("content-length")
        if limit and length and length.isdigit() and int(length) > (limit + 1) * CHUNK:  # +1 MB: form overhead
            return JSONResponse({"detail": f"File too large (limit {limit:g} MB)"}, status_code=413)
        t0 = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        name = f"{request.method} {route.path}" if route is not None else "unmatched"
        app.state.metrics.record(name, response.status_code, (time.perf_counter() - t0) * 1000)
        return response


    def job_out(job, token, manager):
        expires = None
        if not job.get("readonly") and settings.upload_ttl_s > 0:
            expires = max(0, int(job["created_ts"] + settings.upload_ttl_s - time.time()))
        return JobOut(**{k: job.get(k) for k in JobOut.model_fields if k in job},
                      queue_position=manager.queue_position(job) if job["status"] == "queued" else 0,
                      editable=manager.can_edit(job, token), expires_in_s=expires)

    def get_job(job_id, manager):
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found (it may have expired)")
        return job

    def violation_out(v, manager, token):
        job = manager.job_for_run(v["run_id"])
        plate = mask_plate(v["plate"]) if settings.demo else v["plate"]
        frames = f"{v['plate_frames_read']}/{v['plate_frames_tried']}" if v["plate_frames_read"] else None
        return ViolationOut(
            id=v["id"], job_id=job["id"] if job else None, track_id=v["track_id"], video_time_s=v["video_time_s"],
            first_seen_s=v["first_seen_s"], last_seen_s=v["last_seen_s"], votes_without_helmet=v["votes_without"],
            votes_with_helmet=v["votes_with"], max_conf=v["max_conf"], plate=plate, plate_conf=v["plate_conf"],
            plate_frames=frames, review_status=v["review_status"],
            snapshot_url=f"/violations/{v['id']}/snapshot" if v["snapshot"] else None,
            plate_image_url=f"/violations/{v['id']}/plate" if v["plate_image"] and not settings.demo else None,
            editable=manager.can_edit(job, token))

    def get_violation(violation_id, request):
        v = request.app.state.log.violation(violation_id)
        job = request.app.state.jobs.job_for_run(v["run_id"]) if v else None
        if v is None or job is None:  # rows of CLI runs are not exposed through the API
            raise HTTPException(404, "Violation not found")
        return v, job

    # ------------------------------------------------------------------ service

    @app.get("/health", tags=["service"])
    def health(request: Request):
        return {"status": "ok", "version": __version__, "model_loaded": hasattr(request.app.state, "detector")}

    @app.get("/model", tags=["service"])
    def model_info(request: Request):
        det = request.app.state.detector
        return {"classes": list(det.names.values()), "thresholds": request.app.state.thresholds,
                "input_size": det.imgsz, "dynamic_input": det.dynamic, "runtime": "onnxruntime",
                "plates_enabled": request.app.state.plates, "demo": settings.demo,
                "limits": {"max_image_mb": settings.max_image_mb, "max_video_mb": settings.max_video_mb,
                           "max_video_seconds": settings.max_video_seconds, "vid_stride": settings.vid_stride,
                           "upload_ttl_s": settings.upload_ttl_s}}

    @app.get("/metrics", tags=["service"])
    def metrics(request: Request, manager: Manager):
        m = request.app.state.metrics
        return {"uptime_s": round(time.time() - m.started), "routes": m.snapshot(), "jobs": manager.counts(),
                "video_frames_processed": manager.frames_processed}

    # ------------------------------------------------------------------ images

    @app.post("/detect", response_model=DetectResponse, tags=["images"])
    async def detect(request: Request, file: Annotated[UploadFile, File(description="JPEG, PNG, WebP or BMP image")]):
        if file.content_type not in IMAGE_TYPES:
            raise HTTPException(415, f"Unsupported image type: {file.content_type}")
        data = await file.read(int(settings.max_image_mb * CHUNK) + 1)
        if len(data) > settings.max_image_mb * CHUNK:
            raise HTTPException(413, f"File too large (limit {settings.max_image_mb:g} MB)")
        det, thr = request.app.state.detector, request.app.state.thresholds

        def decode_and_detect():  # CPU-bound: runs in a worker thread so other requests are not blocked
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return None, None, 0.0
            t0 = time.perf_counter()
            boxes = det(img, conf=min(thr.values(), default=0.25))
            return img, boxes, (time.perf_counter() - t0) * 1000

        img, boxes, latency = await run_in_threadpool(decode_and_detect)
        if img is None or boxes is None:
            raise HTTPException(422, "Could not decode the image")
        out = []
        for x1, y1, x2, y2, conf, c in boxes.tolist():
            name = det.names[int(c)]
            if conf >= thr.get(name, 0.25):
                out.append(Detection(cls=name, conf=round(conf, 4), box=[round(v, 1) for v in (x1, y1, x2, y2)]))
        n_viol = sum(d.cls == VIOLATION_CLASS for d in out)
        return DetectResponse(width=img.shape[1], height=img.shape[0], latency_ms=round(latency, 1),
                              with_helmet=len(out) - n_viol, without_helmet=n_viol, detections=out)

    # ------------------------------------------------------------------ video jobs

    @app.post("/jobs/video", response_model=JobCreated, status_code=202, tags=["video"])
    async def create_job(file: Annotated[UploadFile, File(description="MP4, MOV, AVI, MKV or WebM video")],
                         manager: Manager, token: Token):
        if token is not None and not 16 <= len(token) <= 128:
            raise HTTPException(422, "X-Owner-Token must be 16-128 characters")
        ext = Path(file.filename or "").suffix.lower()
        if ext not in VIDEO_EXTS:
            raise HTTPException(415, f"Unsupported video type: {ext or 'unknown'}")
        if manager.counts().get("queued", 0) >= settings.max_queued:
            raise HTTPException(429, "Too many videos are waiting; try again in a few minutes")
        tmp_dir = Path(tempfile.mkdtemp(dir=settings.data_dir))
        path = tmp_dir / f"upload{ext}"
        try:
            size, limit = 0, settings.max_video_mb * CHUNK
            with open(path, "wb") as f:
                while chunk := await file.read(CHUNK):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"File too large (limit {settings.max_video_mb:g} MB)")
                    f.write(chunk)
            meta = await run_in_threadpool(probe_video, path)
            if meta is None:
                raise HTTPException(422, "Could not read the video (unsupported codec or corrupt file)")
            fps, frames, _, _ = meta
            if fps and frames / fps > settings.max_video_seconds:
                raise HTTPException(422, f"Video too long ({frames / fps:.0f} s, limit {settings.max_video_seconds:g} s)")
            job, token = manager.create(path, Path(file.filename or "video").name, meta, token)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return JobCreated(job=job_out(job, token, manager), owner_token=token)

    @app.get("/jobs", response_model=list[JobOut], tags=["video"])
    def list_jobs(manager: Manager, token: Token):
        return [job_out(j, token, manager) for j in manager.visible(token)]

    @app.get("/jobs/{job_id}", response_model=JobOut, tags=["video"])
    def job_status(job_id: str, manager: Manager, token: Token):
        return job_out(get_job(job_id, manager), token, manager)

    @app.get("/jobs/{job_id}/tracks", tags=["video"])
    def job_tracks(job_id: str, manager: Manager):
        job = get_job(job_id, manager)
        path = Path(job["dir"]) / "tracks.json"
        if job["status"] != "finished" or not path.exists():
            raise HTTPException(409, f"Tracks are available when the job has finished (status: {job['status']})")
        return FileResponse(path, media_type="application/json")

    @app.get("/jobs/{job_id}/video", tags=["video"])
    def job_video(job_id: str, manager: Manager):
        job = get_job(job_id, manager)
        return FileResponse(Path(job["dir"]) / job["video"], media_type="video/mp4" if job["video"].endswith(".mp4")
                            else None, filename=job.get("filename"), content_disposition_type="inline")

    @app.delete("/jobs/{job_id}", status_code=204, tags=["video"])
    def delete_job(job_id: str, manager: Manager, token: Token):
        job = get_job(job_id, manager)
        if not manager.can_edit(job, token):
            raise HTTPException(403, "Only the uploader can delete this job")
        if job["status"] == "running":
            raise HTTPException(409, "The job is running; delete it when it has finished")
        manager.delete(job)

    # ------------------------------------------------------------------ violations

    @app.get("/violations", response_model=list[ViolationOut], tags=["review"])
    def list_violations(request: Request, manager: Manager, token: Token, job_id: str | None = None,
                        review_status: Literal["pending", "confirmed", "rejected"] | None = None,
                        limit: Annotated[int, Query(ge=1, le=1000)] = 200):
        if job_id is not None:
            run_ids = [get_job(job_id, manager).get("run_id")]
        else:
            run_ids = [j["run_id"] for j in manager.visible(token) if j.get("run_id")]
        rows = request.app.state.log.violations(run_ids=[r for r in run_ids if r], review_status=review_status,
                                                limit=limit)
        return [violation_out(v, manager, token) for v in rows]

    @app.get("/violations/{violation_id}", response_model=ViolationOut, tags=["review"])
    def one_violation(violation_id: int, request: Request, manager: Manager,
                      token: Token):
        v, _ = get_violation(violation_id, request)
        return violation_out(v, manager, token)

    @app.patch("/violations/{violation_id}", response_model=ViolationOut, tags=["review"])
    def review(violation_id: int, body: ReviewIn, request: Request, manager: Manager,
               token: Token):
        v, job = get_violation(violation_id, request)
        if not manager.can_edit(job, token):
            raise HTTPException(403, "The sample is read-only" if job.get("readonly")
                                else "Only the uploader can review this video's violations")
        request.app.state.log.set_review(violation_id, body.review_status)
        return violation_out(request.app.state.log.violation(violation_id), manager, token)

    def violation_file(violation_id, request, column):
        v, job = get_violation(violation_id, request)
        rel = v[column]
        path = (Path(job["dir"]) / rel).resolve() if rel else None
        if path is None or Path(job["dir"]).resolve() not in path.parents or not path.exists():
            raise HTTPException(404, "Image not available")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/violations/{violation_id}/snapshot", tags=["review"])
    def snapshot(violation_id: int, request: Request):
        return violation_file(violation_id, request, "snapshot")

    @app.get("/violations/{violation_id}/plate", tags=["review"])
    def plate_image(violation_id: int, request: Request):
        if settings.demo:
            raise HTTPException(404, "Plate images are not served in the public demo")
        return violation_file(violation_id, request, "plate_image")

    return app


app = create_app()
