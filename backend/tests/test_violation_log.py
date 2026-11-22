import sqlite3
import threading

import pytest

from ridesafe.violation_log import ViolationLog


def test_run_and_violation_round_trip(tmp_path):
    log = ViolationLog(tmp_path / "v.db")
    run = log.start_run("clip.mp4", weights="models/best.pt", out_dir=tmp_path, settings={"vid_stride": 2})
    vid = log.add_violation(run, track_id=7, votes_without=12, votes_with=1, video_time_s=3.4, snapshot="violations/t7.jpg")
    log.set_plate(vid, "MP04ZD7116", 0.8, 3, 5, 140, "violations/t7_plate.jpg")
    log.finish_run(run, frames_processed=426, processing_fps=9.1, summary={"violations": 1})

    [r] = log.runs()
    assert r["status"] == "finished" and r["frames_processed"] == 426
    [v] = log.violations(run)
    assert (v["track_id"], v["plate"], v["plate_frames_read"], v["review_status"]) == (7, "MP04ZD7116", 3, "pending")
    log.close()


def test_violation_is_on_disk_before_the_run_finishes(tmp_path):
    """Crash safety: a second connection sees the row while the run is still 'running'."""
    log = ViolationLog(tmp_path / "v.db")
    run = log.start_run("rtsp://camera")
    log.add_violation(run, track_id=1, votes_without=5, votes_with=0)
    other = sqlite3.connect(tmp_path / "v.db")
    assert other.execute("SELECT status FROM runs").fetchone() == ("running",)
    assert other.execute("SELECT COUNT(*) FROM violations").fetchone() == (1,)
    other.close()
    log.close()


def test_review_status_is_validated(tmp_path):
    log = ViolationLog(tmp_path / "v.db")
    vid = log.add_violation(log.start_run("x"), track_id=1, votes_without=5, votes_with=0)
    log.set_review(vid, "confirmed")
    assert log.violations()[0]["review_status"] == "confirmed"
    with pytest.raises(ValueError):
        log.set_review(vid, "maybe")
    log.close()


def test_writes_from_two_threads(tmp_path):
    """The tracker thread adds violations while the plate thread fills in plates."""
    log = ViolationLog(tmp_path / "v.db")
    run = log.start_run("x")
    ids = [log.add_violation(run, track_id=i, votes_without=5, votes_with=0) for i in range(50)]
    t = threading.Thread(target=lambda: [log.set_plate(i, f"KA01AB{i:04d}", 0.7, 2, 5, 90) for i in ids])
    t.start()
    for i in range(50, 100):
        log.add_violation(run, track_id=i, votes_without=5, votes_with=0)
    t.join()
    rows = log.violations(run, limit=1000)
    assert len(rows) == 100 and sum(r["plate"] is not None for r in rows) == 50
    log.close()


def test_review_changes_are_audited_and_old_logs_are_upgraded(tmp_path):
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)  # a log written before the audit columns existed
    con.executescript(
        "CREATE TABLE runs (id INTEGER PRIMARY KEY, started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL "
        "DEFAULT 'running', source TEXT NOT NULL, weights TEXT, out_dir TEXT, settings TEXT, frames_processed INTEGER, "
        "processing_fps REAL, summary TEXT);"
        "CREATE TABLE violations (id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL, track_id INTEGER NOT NULL, "
        "logged_at TEXT NOT NULL, video_time_s REAL, first_seen_s REAL, last_seen_s REAL, votes_without INTEGER NOT "
        "NULL, votes_with INTEGER NOT NULL, max_conf REAL, snapshot TEXT, plate TEXT, plate_conf REAL, "
        "plate_frames_read INTEGER, plate_frames_tried INTEGER, plate_width_px INTEGER, plate_image TEXT, "
        "review_status TEXT NOT NULL DEFAULT 'pending');"
        "INSERT INTO runs (started_at, source) VALUES ('2026-01-01', 'old');"
        "INSERT INTO violations (run_id, track_id, logged_at, votes_without, votes_with) VALUES (1, 7, 'x', 5, 0);")
    con.commit()
    con.close()

    log = ViolationLog(db)
    [v] = log.violations(1)
    assert v["review_status"] == "pending" and v["reviewed_by"] is None
    log.set_review(v["id"], "confirmed", by="author")
    v = log.violation(v["id"])
    assert v is not None
    assert v["review_status"] == "confirmed" and v["reviewed_by"] == "author" and v["reviewed_at"]
    log.close()
