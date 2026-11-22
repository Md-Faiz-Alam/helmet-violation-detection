"""Video jobs for the API: one worker thread analyses uploaded videos in order; results expire.

A job lives in <data_dir>/jobs/<id>/:
    input.<ext>       the uploaded video (served back to the browser for playback)
    job.json          status, progress, video metadata, summary (rewritten as the job advances)
    tracks.json       per processed frame, the tracked heads with their label at that moment, plus the
                      final decision per track: the web app draws the boxes over the original video
    violations/       one snapshot per violator (+ plate crop), as written by ridesafe.video
Violations go to the shared SQLite log (one run per job). Jobs older than the upload TTL are deleted
with their files and log rows. A pre-computed sample job (RIDESAFE_SAMPLE_DIR) is read-only and never
expires; make one with `python -m ridesafe.jobs --make-sample <video> --out <dir>`.
"""

import argparse
import hashlib
import json
import queue
import secrets
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2

from ridesafe.bytetrack import ByteTrackConfig
from ridesafe.tracking import SAFE_CLASS, UNCERTAIN, VIOLATION_CLASS
from ridesafe.video import VideoAnalyzer

LABELS = {VIOLATION_CLASS: "violation", UNCERTAIN: "uncertain", SAFE_CLASS: "helmet", None: "pending"}
ACTIVE = ("queued", "running")


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hash_token(token):
    return hashlib.sha256(token.encode()).hexdigest() if token else None


def probe_video(path):
    """(fps, frame count, width, height) of a video file, or None if OpenCV cannot read it."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        ok, _ = cap.read()
        if not ok:
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        return fps, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), \
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()


class JobManager:
    def __init__(self, settings, detector, log, thresholds, plate_reader=None, start_threads=True):
        self.s, self.detector, self.log, self.thresholds = settings, detector, log, thresholds
        self.plate_reader = plate_reader
        self.root = Path(settings.data_dir) / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs = {}
        self.lock = threading.Lock()
        self.queue = queue.Queue()
        self.frames_processed = 0  # for /metrics
        self._load_existing()
        if settings.sample_dir:
            self._load_sample(Path(settings.sample_dir))
        self._stop = threading.Event()
        if start_threads:  # off for make_sample(), which runs one job in the foreground
            threading.Thread(target=self._work, name="video-jobs", daemon=True).start()
            threading.Thread(target=self._expire_loop, name="job-expiry", daemon=True).start()

    # ------------------------------------------------------------------ state

    def _save(self, job):
        d = Path(job["dir"])
        tmp = d / "job.json.tmp"
        tmp.write_text(json.dumps({k: v for k, v in job.items() if k != "dir"}, indent=1), encoding="utf-8")
        tmp.replace(d / "job.json")

    def _update(self, job, **fields):
        with self.lock:
            job.update(fields)
        if not job.get("readonly"):
            self._save(job)

    def _load_existing(self):
        for f in sorted(self.root.glob("*/job.json")):
            try:
                job = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            job["dir"] = str(f.parent)
            if job["status"] in ACTIVE:  # the server stopped while it was waiting or running
                job.update(status="failed", stage="failed", error="Interrupted by a server restart; upload again.")
                self._save(job)
            self.jobs[job["id"]] = job

    def _load_sample(self, d):
        f = d / "job.json"
        if not f.exists():
            print(f"Sample job not found: {f}")
            return
        job = json.loads(f.read_text(encoding="utf-8"))
        job.update(dir=str(d), readonly=True, owner=None)
        source = f"sample:{job['id']}"
        run = next((r for r in self.log.runs(limit=1000) if r["source"] == source), None)
        if run is None:  # import the sample's violations into this server's log
            run_id = self.log.start_run(source, self.s.model, str(d), {"sample": True})
            for v in json.loads((d / "violations.json").read_text(encoding="utf-8")):
                vid = self.log.add_violation(run_id, v["track_id"], v["votes_without"], v["votes_with"],
                                             v["video_time_s"], v["first_seen_s"], v["last_seen_s"], v["max_conf"],
                                             v["snapshot"])
                if v.get("plate"):
                    self.log.set_plate(vid, v["plate"], v["plate_conf"], v["plate_frames_read"],
                                       v["plate_frames_tried"], v["plate_width_px"], v["plate_image"])
            self.log.finish_run(run_id, job.get("progress", {}).get("frames_done"), job.get("processing_fps"),
                                job.get("summary"))
        else:
            run_id = run["id"]
        job["run_id"] = run_id
        self.jobs[job["id"]] = job

    # ------------------------------------------------------------------ public

    def create(self, upload, filename, meta, token=None):
        """Register an uploaded video (already saved at `upload`) and queue it. Returns (job, owner token).

        `token` lets one client (a browser) own all of its uploads; a new one is made if it is not given."""
        fps, frames, width, height = meta
        job_id = uuid.uuid4().hex
        d = self.root / job_id
        d.mkdir(parents=True)
        video = d / f"input{Path(upload).suffix.lower() or '.mp4'}"
        shutil.move(str(upload), video)
        token = token or secrets.token_urlsafe(24)
        job = {"id": job_id, "created_at": utc_now(), "created_ts": time.time(), "status": "queued",
               "stage": "queued", "filename": filename, "video": video.name, "fps": fps, "width": width,
               "height": height, "frames_total": frames, "duration_s": round(frames / fps, 2) if fps else None,
               "vid_stride": self.s.vid_stride, "progress": {"frames_done": 0, "frames_total": frames},
               "processing_fps": None, "run_id": None, "summary": None, "error": None,
               "owner": hash_token(token), "readonly": False, "dir": str(d)}
        with self.lock:
            self.jobs[job_id] = job
        self._save(job)
        self.queue.put(job_id)
        return job, token

    def get(self, job_id):
        return self.jobs.get(job_id)

    def visible(self, token=None):
        """Jobs a caller may list: in demo mode the sample and their own uploads, otherwise all."""
        h = hash_token(token)
        with self.lock:
            jobs = list(self.jobs.values())
        if self.s.demo:
            jobs = [j for j in jobs if j.get("readonly") or (h and j.get("owner") == h)]
        return sorted(jobs, key=lambda j: j.get("created_at", ""), reverse=True)

    def can_edit(self, job, token):
        """May this caller review or delete the job? Never the sample; in demo mode only the uploader."""
        if job is None or job.get("readonly"):
            return False
        return not self.s.demo or (bool(token) and job.get("owner") == hash_token(token))

    def job_for_run(self, run_id):
        with self.lock:
            return next((j for j in self.jobs.values() if j.get("run_id") == run_id), None)

    def queue_position(self, job):
        with self.lock:
            queued = sorted((j for j in self.jobs.values() if j["status"] == "queued"), key=lambda j: j["created_ts"])
        return next((i + 1 for i, j in enumerate(queued) if j["id"] == job["id"]), 0)

    def delete(self, job):
        with self.lock:
            self.jobs.pop(job["id"], None)
        if job.get("run_id"):
            self.log.delete_run(job["run_id"])
        shutil.rmtree(job["dir"], ignore_errors=True)

    def counts(self):
        with self.lock:
            out = {}
            for j in self.jobs.values():
                out[j["status"]] = out.get(j["status"], 0) + 1
        return out

    def close(self):
        self._stop.set()
        self.queue.put(None)

    # ------------------------------------------------------------------ worker

    def _work(self):
        while not self._stop.is_set():
            job_id = self.queue.get()
            if job_id is None:
                return
            job = self.jobs.get(job_id)
            if job is None:  # deleted while queued
                continue
            try:
                self.run(job)
            except Exception as e:  # report it on the job, keep the worker alive
                self._update(job, status="failed", stage="failed", error=f"{type(e).__name__}: {e}")

    def run(self, job):
        """Analyse one job's video (blocking). Used by the worker and by --make-sample."""
        d = Path(job["dir"])
        settings = {"vid_stride": job["vid_stride"], "thresholds": self.thresholds, "plates": bool(self.plate_reader),
                    "runtime": "onnxruntime", "job": job["id"]}
        run_id = self.log.start_run(f"job:{job['id']}", self.s.model, str(d), settings)
        self._update(job, status="running", stage="analyzing", started_at=utc_now(), run_id=run_id)
        analyzer = VideoAnalyzer(self.detector, self.thresholds, d, log=self.log, run_id=run_id,
                                 tracker_config=ByteTrackConfig.from_yaml(self.s.tracker), fps=job["fps"],
                                 vid_stride=job["vid_stride"], plate_reader=self.plate_reader,
                                 caption_plates=not self.s.demo, on_message=None)
        cap = cv2.VideoCapture(str(d / job["video"]))
        frames, frame_idx, done = [], 0, 0
        t0 = last_save = time.perf_counter()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                if (frame_idx - 1) % job["vid_stride"]:
                    continue
                tracked = analyzer.step(frame_idx, frame)
                done += 1
                frames.append([frame_idx, [[tid, *(round(v, 1) for v in box), LABELS[analyzer.label(tid)],
                                            round(conf, 3)] for tid, _cls, conf, box in tracked]])
                if time.perf_counter() - last_save > 1.0:  # progress for polling clients, about once a second
                    last_save = time.perf_counter()
                    self._update(job, progress={"frames_done": frame_idx, "frames_total": job["frames_total"]},
                                 processing_fps=round(done / (last_save - t0), 2))
                if self._stop.is_set():
                    raise RuntimeError("server shutting down")
        finally:
            cap.release()
        self.frames_processed += done
        self._update(job, stage="reading plates" if self.plate_reader else "finishing",
                     progress={"frames_done": frame_idx, "frames_total": job["frames_total"]})
        analyzer.finish()
        elapsed = time.perf_counter() - t0

        tracker, to_time = analyzer.tracker, analyzer.to_time
        tracks = {}
        for t in tracker.all_tracks():
            entry = analyzer.logged.get((t.track_id, t.first_frame)) if t.violation else None
            tracks[str(t.track_id)] = {
                "label": LABELS[tracker.label(t)], "counted": tracker.counted(t), "violation": t.violation,
                "violation_id": entry["id"] if entry else None, "first_s": to_time(t.first_frame),
                "last_s": to_time(t.last_frame), "violation_s": to_time(t.first_violation_frame) if t.violation else None,
                "votes": {"without_helmet": t.votes.get(VIOLATION_CLASS, 0), "with_helmet": t.votes.get(SAFE_CLASS, 0)}}
        (d / "tracks.json").write_text(json.dumps({
            "fps": job["fps"], "vid_stride": job["vid_stride"], "width": job["width"], "height": job["height"],
            "box_format": ["track_id", "x1", "y1", "x2", "y2", "label", "conf"], "frames": frames, "tracks": tracks},
            separators=(",", ":")), encoding="utf-8")
        summary = {**tracker.summary(), "tracks_total": len(tracker.all_tracks()),
                   "plates_read": sum(1 for e in analyzer.logged.values() if e["plate"])}
        fps = round(done / elapsed, 2) if elapsed else None
        self.log.finish_run(run_id, done, fps, summary)
        self._update(job, status="finished", stage="finished", finished_at=utc_now(), summary=summary,
                     processing_fps=fps, progress={"frames_done": frame_idx, "frames_total": frame_idx})

    # ------------------------------------------------------------------ expiry

    def expire(self, now=None):
        """Delete finished/failed uploads older than the TTL. Returns how many were deleted."""
        if self.s.upload_ttl_s <= 0:
            return 0
        now = now or time.time()
        with self.lock:
            old = [j for j in self.jobs.values() if not j.get("readonly") and j["status"] not in ACTIVE
                   and now - j.get("created_ts", now) > self.s.upload_ttl_s]
        for j in old:
            self.delete(j)
        return len(old)

    def _expire_loop(self):
        while not self._stop.wait(60):
            self.expire()


def make_sample(video, out, settings=None):
    """Run the pipeline on one video and save it as a read-only sample job (RIDESAFE_SAMPLE_DIR)."""
    import tempfile
    from dataclasses import replace

    from ridesafe.api import build_components
    from ridesafe.settings import Settings

    meta = probe_video(video)
    if meta is None:
        raise SystemExit(f"Cannot read video: {video}")
    out = Path(out)
    with tempfile.TemporaryDirectory() as tmp:
        settings = replace(settings or Settings(), data_dir=Path(tmp), sample_dir="")
        detector, log, thresholds, plate_reader = build_components(settings)
        manager = JobManager(settings, detector, log, thresholds, plate_reader, start_threads=False)
        upload = Path(tmp) / Path(video).name
        shutil.copy(video, upload)
        job, _ = manager.create(upload, Path(video).name, meta)
        manager.run(job)
        violations = log.violations(job["run_id"])
        log.close()
        job.update(owner=None, readonly=True, run_id=None)
        manager._save(job)
        (Path(job["dir"]) / "violations.json").write_text(json.dumps(violations, indent=1), encoding="utf-8")
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(job["dir"], out)
    print(f"Sample job written to {out}: {job['summary']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Video jobs: make a read-only sample job for the API")
    ap.add_argument("--make-sample", required=True, metavar="VIDEO", help="Video to analyse")
    ap.add_argument("--out", default="samples/demo_job", help="Output folder (RIDESAFE_SAMPLE_DIR)")
    a = ap.parse_args()
    make_sample(a.make_sample, a.out)
