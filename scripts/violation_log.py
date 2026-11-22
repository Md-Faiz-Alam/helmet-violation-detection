"""SQLite violation log: every confirmed violation is written the moment it is decided.

Used by scripts/track_video.py (writer) and meant to be read by an API or dashboard while a stream
is still running (WAL mode lets readers and one writer work at the same time). Standard library only.

Tables
    runs        one row per video/stream: source, model, settings, status, speed, final summary
    violations  one row per violating rider: time, track, votes, snapshot, plate (filled in later
                by the background plate reader), review_status for a human reviewer

    python scripts/violation_log.py                    # list runs
    python scripts/violation_log.py --run 3            # list the violations of run 3
"""

import argparse
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                INTEGER PRIMARY KEY,
    started_at        TEXT NOT NULL,
    finished_at       TEXT,
    status            TEXT NOT NULL DEFAULT 'running',   -- running / finished / interrupted
    source            TEXT NOT NULL,
    weights           TEXT,
    out_dir           TEXT,
    settings          TEXT,                              -- JSON
    frames_processed  INTEGER,
    processing_fps    REAL,
    summary           TEXT                               -- JSON
);
CREATE TABLE IF NOT EXISTS violations (
    id                  INTEGER PRIMARY KEY,
    run_id              INTEGER NOT NULL REFERENCES runs(id),
    track_id            INTEGER NOT NULL,
    logged_at           TEXT NOT NULL,                   -- wall-clock time the violation was decided
    video_time_s        REAL,                            -- when no-helmet was first confirmed (file sources)
    first_seen_s        REAL,
    last_seen_s         REAL,
    votes_without       INTEGER NOT NULL,
    votes_with          INTEGER NOT NULL,
    max_conf            REAL,
    snapshot            TEXT,                            -- path relative to the run's out_dir
    plate               TEXT,                            -- NULL until read (or if unreadable)
    plate_conf          REAL,
    plate_frames_read   INTEGER,
    plate_frames_tried  INTEGER,
    plate_width_px      INTEGER,
    plate_image         TEXT,
    review_status       TEXT NOT NULL DEFAULT 'pending'  -- pending / confirmed / rejected
);
CREATE INDEX IF NOT EXISTS idx_violations_run ON violations(run_id);
CREATE INDEX IF NOT EXISTS idx_violations_logged ON violations(logged_at);
"""


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


class ViolationLog:
    """Thread-safe writer/reader for the violation database (one connection, guarded by a lock)."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.executescript(SCHEMA)
            self._db.commit()

    def _write(self, sql, params=()):
        with self._lock:
            cur = self._db.execute(sql, params)
            self._db.commit()
            return cur.lastrowid

    def start_run(self, source, weights=None, out_dir=None, settings=None):
        return self._write(
            "INSERT INTO runs (started_at, source, weights, out_dir, settings) VALUES (?, ?, ?, ?, ?)",
            (now(), str(source), weights, str(out_dir) if out_dir else None, json.dumps(settings or {})))

    def add_violation(self, run_id, track_id, votes_without, votes_with, video_time_s=None, first_seen_s=None,
                      last_seen_s=None, max_conf=None, snapshot=None):
        return self._write(
            "INSERT INTO violations (run_id, track_id, logged_at, video_time_s, first_seen_s, last_seen_s, "
            "votes_without, votes_with, max_conf, snapshot) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, track_id, now(), video_time_s, first_seen_s, last_seen_s, votes_without, votes_with,
             max_conf, snapshot))

    def set_plate(self, violation_id, plate, conf, frames_read, frames_tried, width_px, plate_image=None):
        self._write(
            "UPDATE violations SET plate=?, plate_conf=?, plate_frames_read=?, plate_frames_tried=?, "
            "plate_width_px=?, plate_image=? WHERE id=?",
            (plate, conf, frames_read, frames_tried, width_px, plate_image, violation_id))

    def set_review(self, violation_id, status):
        if status not in ("pending", "confirmed", "rejected"):
            raise ValueError(f"Unknown review status: {status}")
        self._write("UPDATE violations SET review_status=? WHERE id=?", (status, violation_id))

    def finish_run(self, run_id, frames_processed=None, processing_fps=None, summary=None, status="finished"):
        self._write(
            "UPDATE runs SET finished_at=?, status=?, frames_processed=?, processing_fps=?, summary=? WHERE id=?",
            (now(), status, frames_processed, processing_fps, json.dumps(summary or {}), run_id))

    def runs(self, limit=50):
        with self._lock:
            rows = self._db.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def violations(self, run_id=None, limit=500):
        sql, params = "SELECT * FROM violations", []
        if run_id is not None:
            sql += " WHERE run_id=?"
            params.append(run_id)
        sql += " ORDER BY id LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._db.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        with self._lock:
            self._db.close()


def main(args):
    log = ViolationLog(args.db)
    if args.run is None:
        print(f"{'run':>4}  {'status':<11} {'started':<25} {'violations':>10}  source")
        for r in log.runs():
            n = len(log.violations(r["id"]))
            print(f"{r['id']:>4}  {r['status']:<11} {r['started_at']:<25} {n:>10}  {r['source']}")
    else:
        print(f"{'id':>4} {'track':>5} {'logged_at':<25} {'video_s':>7}  {'votes':>7}  {'plate':<12} review")
        for v in log.violations(args.run):
            votes = f"{v['votes_without']}/{v['votes_without'] + v['votes_with']}"
            vt = f"{v['video_time_s']:.1f}" if v["video_time_s"] is not None else "-"
            print(f"{v['id']:>4} {v['track_id']:>5} {v['logged_at']:<25} {vt:>7}  {votes:>7}  "
                  f"{v['plate'] or '-':<12} {v['review_status']}")
    log.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Show the violation log")
    ap.add_argument("--db", default="results/violations.db")
    ap.add_argument("--run", type=int, default=None, help="Show the violations of this run")
    main(ap.parse_args())
