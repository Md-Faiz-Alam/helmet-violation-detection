"""Track riders' heads through a video with ByteTrack and flag helmet violations once per rider.

    python scripts/track_video.py --source path/to/video.mp4
    python scripts/track_video.py --source path/to/video.mp4 --show --vid-stride 2   # live window, every 2nd frame
    python scripts/track_video.py --source 0                                         # webcam
    python scripts/track_video.py --source rtsp://user:pass@camera-ip/stream         # IP camera

Each head gets a track ID. A rider is flagged as a violation only after "Without Helmet" wins
enough frames (see scripts/tracking.py), so one bad frame doesn't create a violation, and a
rider who stays in view is counted once instead of once per frame.

Every violation is written the moment the rider is decided (crash-safe, readable while a stream runs):
    results/violations.db  SQLite log (runs + violations; plates are filled in by a background thread).
                           Browse it with: python scripts/violation_log.py [--run N]

Output (results/video/<name>_<timestamp>/):
    annotated.mp4     video with track IDs, colors and live counts
    tracks.csv        one row per track: label, frames seen, votes, violation time
    violations/       one snapshot per violator (its most confident no-helmet frame) + plate crop, saved immediately
    summary.json      counts, settings and processing speed
    track_boxes.csv   every tracked box in every processed frame (written as it runs; used by eval_video.py;
                      --no-boxes turns it off for long streams)
Number plates: for each violator, the 5 frames where they are closest to the camera are kept,
their vehicle is found and its plate is read with OCR and voted across frames (scripts/plate_reader.py).
Turn it off with --no-plates. Plates are personal data: keep results private.
Stop a webcam/stream with Ctrl+C (or q in the --show window); results are still saved.
"""

import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from metrics import as_numpy, first_result
from plates import BestFrames
from tracking import SAFE_CLASS, UNCERTAIN, VIOLATION_CLASS, ViolationTracker
from utils import DEFAULT_THRESHOLDS, load_thresholds
from violation_log import ViolationLog

# BGR colors
GREEN, RED, ORANGE, GRAY = (60, 180, 60), (40, 40, 230), (0, 160, 255), (160, 160, 160)


def open_source(source):
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")
    return cap


def style(tracker, track):
    """Box color and text for a track's current (provisional) state."""
    label = tracker.label(track)
    if label == VIOLATION_CLASS:
        return RED, "NO HELMET"
    if label == UNCERTAIN:
        return ORANGE, "no helmet?"  # many no-helmet votes, but not enough (yet) for a violation
    if label == SAFE_CLASS:
        return GREEN, "helmet"
    return GRAY, "?"


def draw(frame, tracked, tracker, stats_text):
    for track_id, _, _, (x1, y1, x2, y2) in tracked:
        color, text = style(tracker, tracker.tracks[track_id])
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, color, 2)
        caption = f"#{track_id} {text}"
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), color, -1)
        cv2.putText(frame, caption, (p1[0] + 2, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Counter panel (top-left, semi-transparent)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (330, 24 + 22 * len(stats_text)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, line in enumerate(stats_text):
        cv2.putText(frame, line, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def context_crop(frame, box, pad_x=1.0, pad_up=0.6, pad_down=3.0):
    """Crop around a head box, extended downwards so the rider and vehicle are visible."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    X1, Y1 = max(0, int(x1 - pad_x * bw)), max(0, int(y1 - pad_up * bh))
    X2, Y2 = min(w, int(x2 + pad_x * bw)), min(h, int(y2 + pad_down * bh))
    crop = frame[Y1:Y2, X1:X2].copy()
    cv2.rectangle(crop, (int(x1 - X1), int(y1 - Y1)), (int(x2 - X1), int(y2 - Y1)), RED, 2)
    return crop


def train_imgsz(model, default=640):
    return int(((getattr(model, "ckpt", None) or {}).get("train_args") or {}).get("imgsz", default))


def run(args):
    model = YOLO(args.weights)
    imgsz = args.imgsz or train_imgsz(model)
    thresholds = load_thresholds(args.thresholds) or {}
    tracker = ViolationTracker(thresholds, min_frames=args.min_frames, min_ratio=args.min_ratio,
                               uncertain_ratio=args.uncertain_ratio, finalize_after=args.finalize_after)

    cap = open_source(args.source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    is_file = os.path.isfile(str(args.source))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else 0

    name = Path(str(args.source)).stem if is_file else "stream"
    out_dir = Path(args.out) / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    (out_dir / "violations").mkdir(parents=True, exist_ok=True)
    settings = {"imgsz": imgsz, "thresholds": thresholds, "tracker": args.tracker, "vid_stride": args.vid_stride,
                "min_frames": args.min_frames, "min_ratio": args.min_ratio, "finalize_after": args.finalize_after,
                "plates": not args.no_plates, "plate_format": args.plate_format, "device": args.device or "auto"}
    log = ViolationLog(args.db)
    run_id = log.start_run(args.source, args.weights, out_dir.resolve(), settings)

    print(f"Source: {args.source} ({src_fps:.1f} FPS{f', {total} frames' if total else ''}) | imgsz {imgsz} | "
          f"thresholds {thresholds} | stride {args.vid_stride} | log: {args.db} (run {run_id})")

    def to_time(processed_idx):  # processed-frame index -> seconds into the source
        return round((processed_idx - 1) * args.vid_stride / src_fps, 2) if src_fps and processed_idx > 0 else None

    plate_reader = None
    if not args.no_plates:
        try:
            from plate_reader import PlateReader
            plate_reader = PlateReader(args.vehicle_weights, fmt=args.plate_format, device=args.device)
        except ImportError:
            print("Plate reading off: pip install rapidocr_onnxruntime (or pass --no-plates)")
    # Plates are read in a background thread so a slow OCR never pauses the video (1 worker: the
    # plate models are only ever used from that thread).
    plate_pool = ThreadPoolExecutor(max_workers=1) if plate_reader else None
    plate_frames = {}   # track id -> BestFrames (closest frames of riders with no-helmet votes)
    best_crops = {}     # track id -> best no-helmet crop, until the track is decided
    logged = {}         # (track id, first frame) -> {"id", "snapshot", "crop", "plate"} for decided violations

    def read_plate(key, frames, violation_id, track_id):
        try:
            plate = plate_reader.read_track(frames, min_agree=args.plate_min_agree) if plate_reader else None
        except Exception as e:  # a failed OCR must never stop the tracker
            print(f"  ⚠️  Plate reading failed for track #{track_id}: {e}")
            return
        entry = logged[key]
        entry["plate"] = plate
        caption = f"#{track_id}  plate: {plate['plate']}" if plate else f"#{track_id}  plate: not read"
        if entry["crop"] is not None:
            cv2.imwrite(str(out_dir / entry["snapshot"]), label_snapshot(entry["crop"], caption))
            entry["crop"] = None  # done with it: free the memory
        if plate:
            plate_image = f"violations/track_{track_id:04d}_plate.jpg"
            cv2.imwrite(str(out_dir / plate_image), plate["plate_image"])
            log.set_plate(violation_id, plate["plate"], plate["conf"], plate["frames_read"], plate["frames_tried"],
                          plate["plate_width_px"], plate_image)
            print(f"  🔎 Plate for track #{track_id}: {plate['plate']} ({plate['frames_read']}/{plate['frames_tried']} "
                  f"frames agree, conf {plate['conf']:.2f}, {plate['plate_width_px']} px wide)")

    def on_violation(t):
        """A rider was just decided as a violation: write it to disk and the log NOW (not at the end)."""
        key = (t.track_id, t.first_frame)
        crop = best_crops.pop(t.track_id, None)
        frames = plate_frames.pop(t.track_id, None)
        snapshot = f"violations/track_{t.track_id:04d}.jpg" if crop is not None else None
        pending = bool(plate_pool and frames)
        if crop is not None and snapshot:
            caption = f"#{t.track_id}  plate: " + ("reading..." if pending else "off" if args.no_plates else "not read")
            cv2.imwrite(str(out_dir / snapshot), label_snapshot(crop, caption))
        violation_id = log.add_violation(
            run_id, t.track_id, t.votes.get(VIOLATION_CLASS, 0), t.votes.get(SAFE_CLASS, 0),
            video_time_s=to_time(t.first_violation_frame), first_seen_s=to_time(t.first_frame),
            last_seen_s=to_time(t.last_frame), max_conf=round(t.best_violation_conf, 3), snapshot=snapshot)
        logged[key] = {"id": violation_id, "snapshot": snapshot, "crop": crop if pending else None, "plate": None}
        when = to_time(t.first_violation_frame)
        print(f"  🚨 Violation logged: track #{t.track_id}" + (f" (no helmet from {when:.1f}s)" if when is not None else "")
              + f" -> log id {violation_id}")
        if plate_pool and frames:
            plate_pool.submit(read_plate, key, frames.payloads(), violation_id, t.track_id)

    writer = None
    boxes_file = open(out_dir / "track_boxes.csv", "w", newline="", encoding="utf-8")
    boxes_csv = csv.writer(boxes_file)
    boxes_csv.writerow(["frame", "track_id", "class", "conf", "x1", "y1", "x2", "y2"])
    frame_idx = processed = 0
    status = "finished"
    t_start = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if (frame_idx - 1) % args.vid_stride:
                continue
            if args.max_frames and processed >= args.max_frames:
                break
            processed += 1

            # Low conf so ByteTrack's second stage can keep weak detections on existing tracks;
            # per-class thresholds are applied afterwards, when tracks vote.
            # agnostic_nms: the detector sometimes puts a "With" AND a "Without Helmet" box on the
            # same head; without it ByteTrack sees two riders and the violator gets logged twice.
            r = first_result(model.track(frame, persist=True, tracker=args.tracker, conf=0.1, imgsz=imgsz,
                                         agnostic_nms=True, device=args.device, verbose=False))
            tracked = []
            if r.boxes is not None and r.boxes.is_track:  # rows: x1, y1, x2, y2, track id, conf, cls
                for x1, y1, x2, y2, tid, conf, c in as_numpy(r.boxes.data).tolist():
                    tracked.append((int(tid), r.names[int(c)], conf, [x1, y1, x2, y2]))

            if not args.no_boxes:
                for tid, cls_name, conf, (x1, y1, x2, y2) in tracked:  # frame = 1-based index in the source video
                    boxes_csv.writerow([frame_idx, tid, cls_name, round(conf, 3), round(x1), round(y1), round(x2), round(y2)])

            for t in tracker.update(processed, tracked, frame_size=(frame.shape[1], frame.shape[0])):
                on_violation(t)

            # Plate candidates: the closest frames of every rider with at least one no-helmet vote
            if plate_reader:
                for tid, _, _, box in tracked:
                    if tracker.tracks[tid].votes.get(VIOLATION_CLASS, 0):
                        area = (box[2] - box[0]) * (box[3] - box[1])
                        plate_frames.setdefault(tid, BestFrames(k=args.plate_frames)).offer(
                            processed, area, lambda f=frame, b=box: plate_reader.rider_region(f, b))

            # Keep the most confident no-helmet crop of every open track for its snapshot
            for tid, cls_name, _conf, box in tracked:
                if cls_name == VIOLATION_CLASS and tracker.tracks[tid].best_violation_frame == processed:
                    best_crops[tid] = context_crop(frame, box)

            # Free what decided (non-violation) riders no longer need: memory stays flat on long streams
            for store in (plate_frames, best_crops):
                for tid in [tid for tid in store if tid not in tracker.open]:
                    del store[tid]

            elapsed = time.perf_counter() - t_start
            s = tracker.summary()
            stats = [f"Riders: {s['riders']}   Helmet: {s['with_helmet']}   Violations: {s['violations']}",
                     f"Frame {frame_idx}{f'/{total}' if total else ''}   {processed / elapsed:.1f} FPS (CPU)"]
            frame = draw(frame, tracked, tracker, stats)

            if not args.no_video:
                if writer is None:
                    out_fps = (src_fps / args.vid_stride) if src_fps else 10
                    writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter.fourcc(*"mp4v"),
                                             out_fps, (frame.shape[1], frame.shape[0]))
                writer.write(frame)
            if args.show:
                cv2.imshow("Helmet tracking (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if processed % 50 == 0:
                print(f"  frame {frame_idx}{f'/{total}' if total else ''} | {processed / elapsed:.1f} FPS | {s}")
    except KeyboardInterrupt:
        status = "interrupted"
        print("Stopped by user, saving results...")
    finally:
        boxes_file.close()
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    for t in tracker.finish():
        on_violation(t)
    if plate_pool:
        print("Waiting for plate reading to finish...")
        plate_pool.shutdown(wait=True)
    elapsed = time.perf_counter() - t_start
    out_dir, summary = save_results(out_dir, args, tracker, logged, to_time, frame_idx, processed, elapsed, settings)
    log.finish_run(run_id, processed, summary["processing_fps"], summary, status)
    log.close()
    return out_dir, summary


def label_snapshot(img, text):
    """Add a caption bar under a snapshot."""
    bar = np.full((28, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([img, bar])


def save_results(out_dir, args, tracker, logged, to_time, frame_idx, processed, elapsed, settings):
    """End of run: the per-track table and summary (violations themselves were logged as they happened)."""
    rows, violations = [], []
    for t in sorted(tracker.all_tracks(), key=lambda t: (t.track_id, t.first_frame)):
        entry = logged.get((t.track_id, t.first_frame)) if t.violation else None
        plate = entry["plate"] if entry else None
        if entry:
            violations.append({"track_id": t.track_id, "log_id": entry["id"], "time_s": to_time(t.first_violation_frame),
                               "plate": plate["plate"] if plate else None, "snapshot": entry["snapshot"] or ""})
        rows.append({
            "track_id": t.track_id,
            "label": tracker.label(t) or "",
            "counted": tracker.counted(t),
            "violation": t.violation,
            "first_time_s": to_time(t.first_frame),
            "last_time_s": to_time(t.last_frame),
            "violation_time_s": to_time(t.first_violation_frame) if t.violation else None,
            "frames_seen": t.frames_seen,
            "votes_with_helmet": t.votes.get(SAFE_CLASS, 0),
            "votes_without_helmet": t.votes.get(VIOLATION_CLASS, 0),
            "max_no_helmet_conf": round(t.best_violation_conf, 3),
            "plate": plate["plate"] if plate else "",
            "plate_conf": plate["conf"] if plate else "",
            "plate_frames": f"{plate['frames_read']}/{plate['frames_tried']}" if plate else "",
            "plate_width_px": plate["plate_width_px"] if plate else "",
            "snapshot": (entry["snapshot"] or "") if entry else "",
        })
    with open(out_dir / "tracks.csv", "w", newline="", encoding="utf-8") as f:
        fields = list(rows[0].keys()) if rows else ["track_id"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "source": str(args.source),
        "weights": args.weights,
        **tracker.summary(),
        "tracks_total": len(tracker.all_tracks()),
        "plates_read": sum(v["plate"] is not None for v in violations),
        "violation_log": violations,
        "log_db": str(args.db),
        "frames_read": frame_idx,
        "frames_processed": processed,
        "processing_fps": round(processed / elapsed, 2) if elapsed else None,
        "settings": settings,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ {processed} frames in {elapsed:.1f}s ({summary['processing_fps']} FPS). Results: {os.path.normpath(out_dir)}")
    print(f"   Riders: {summary['riders']} | With helmet: {summary['with_helmet']} | "
          f"Violations: {summary['violations']} (plates read: {summary['plates_read']}) | "
          f"Uncertain: {summary['uncertain']} | Raw tracks: {summary['tracks_total']}")
    return out_dir, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Helmet violation tracking on video (ByteTrack)")
    ap.add_argument("--source", required=True, help="Video file, webcam index (0) or stream URL (rtsp://...)")
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--thresholds", default=DEFAULT_THRESHOLDS, help="Per-class thresholds YAML")
    ap.add_argument("--tracker", default="configs/bytetrack.yaml")
    ap.add_argument("--imgsz", type=int, default=None, help="Inference size (default: the model's training size)")
    ap.add_argument("--min-frames", type=int, default=5, help="No-helmet votes needed to confirm a violation")
    ap.add_argument("--min-ratio", type=float, default=0.6, help="Share of a track's votes that must be no-helmet")
    ap.add_argument("--uncertain-ratio", type=float, default=0.4,
                    help="Not a violation, but this share of no-helmet votes marks the rider Uncertain")
    ap.add_argument("--finalize-after", type=int, default=30,
                    help="Decide a track after it is unseen for this many processed frames (match track_buffer)")
    ap.add_argument("--vid-stride", type=int, default=1, help="Process every Nth frame (2-3 speeds up CPU runs)")
    ap.add_argument("--max-frames", type=int, default=0, help="Stop after this many processed frames (0 = all)")
    ap.add_argument("--device", default=None, help="cpu, 0, ... (default: auto)")
    ap.add_argument("--show", action="store_true", help="Show a live window")
    ap.add_argument("--no-video", action="store_true", help="Don't write annotated.mp4")
    ap.add_argument("--no-plates", action="store_true", help="Don't read number plates")
    ap.add_argument("--plate-format", default="india", choices=["india", "any"], help="Plate text validation")
    ap.add_argument("--plate-frames", type=int, default=5, help="Frames per violator to read the plate from")
    ap.add_argument("--plate-min-agree", type=int, default=2, help="Frames that must read the same plate text")
    ap.add_argument("--vehicle-weights", default="yolo26s.pt", help="COCO detector used to find the rider's vehicle")
    ap.add_argument("--db", default="results/violations.db", help="SQLite violation log (created if missing)")
    ap.add_argument("--no-boxes", action="store_true", help="Don't write track_boxes.csv (long streams)")
    ap.add_argument("--out", default="results/video")
    run(ap.parse_args())
