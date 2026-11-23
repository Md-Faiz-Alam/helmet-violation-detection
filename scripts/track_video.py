"""Track riders' heads through a video with ByteTrack and flag helmet violations once per rider.

    python scripts/track_video.py --source path/to/video.mp4
    python scripts/track_video.py --source path/to/video.mp4 --show --vid-stride 2   # live window, every 2nd frame
    python scripts/track_video.py --source 0                                         # webcam
    python scripts/track_video.py --source rtsp://user:pass@camera-ip/stream         # IP camera

Runs on ONNX Runtime without PyTorch (export the models first: python scripts/export_onnx.py --vehicles).
The pipeline itself is ridesafe.video, the same code the API runs: each head gets a track ID, and a
rider is flagged as a violation only after "Without Helmet" wins enough frames (see
backend/ridesafe/tracking.py), so one bad frame doesn't create a violation, and a rider who stays in
view is counted once instead of once per frame.

Every violation is written the moment the rider is decided (crash-safe, readable while a stream runs):
    results/violations.db  SQLite log (runs + violations; plates are filled in by a background thread).
                           Browse it with: python -m ridesafe.violation_log [--run N]

Output (results/video/<name>_<timestamp>/):
    annotated.mp4     video with track IDs, colors and live counts
    tracks.csv        one row per track: label, frames seen, votes, violation time
    violations/       one snapshot per violator (its most confident no-helmet frame) + plate crop, saved immediately
    summary.json      counts, settings and processing speed
    track_boxes.csv   every tracked box in every processed frame (written as it runs; used by eval_video.py;
                      --no-boxes turns it off for long streams)
Number plates: for each violator, the 5 frames where they are closest to the camera are kept,
their vehicle is found and its plate is read with OCR and voted across frames (backend/ridesafe/plate_reader.py).
Turn it off with --no-plates. Plates are personal data: keep results private.
Stop a webcam/stream with Ctrl+C (or q in the --show window); results are still saved.
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import cv2

from ridesafe.bytetrack import ByteTrackConfig
from ridesafe.detector import OnnxDetector
from ridesafe.thresholds import DEFAULT_THRESHOLDS, load_thresholds
from ridesafe.tracking import SAFE_CLASS, UNCERTAIN, VIOLATION_CLASS
from ridesafe.video import VideoAnalyzer
from ridesafe.violation_log import ViolationLog

# BGR colors
GREEN, RED, ORANGE, GRAY = (60, 180, 60), (40, 40, 230), (0, 160, 255), (160, 160, 160)


def open_source(source):
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")
    return cap


def style(label):
    """Box color and text for a track's current (provisional) label."""
    if label == VIOLATION_CLASS:
        return RED, "NO HELMET"
    if label == UNCERTAIN:
        return ORANGE, "no helmet?"  # many no-helmet votes, but not enough (yet) for a violation
    if label == SAFE_CLASS:
        return GREEN, "helmet"
    return GRAY, "?"


def draw(frame, tracked, analyzer, stats_text):
    for track_id, _, _, (x1, y1, x2, y2) in tracked:
        color, text = style(analyzer.label(track_id))
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


def load_detector(path, imgsz=None):
    if not os.path.exists(path):
        raise SystemExit(f"Model not found: {path}. Export it first: python scripts/export_onnx.py --vehicles")
    return OnnxDetector(path, imgsz=imgsz)


def run(args):
    detector = load_detector(args.weights, args.imgsz)
    thresholds = load_thresholds(args.thresholds) or {}

    cap = open_source(args.source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    is_file = os.path.isfile(str(args.source))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else 0

    name = Path(str(args.source)).stem if is_file else "stream"
    out_dir = Path(args.out) / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = {"imgsz": detector.imgsz, "thresholds": thresholds, "tracker": args.tracker,
                "vid_stride": args.vid_stride, "min_frames": args.min_frames, "min_ratio": args.min_ratio,
                "finalize_after": args.finalize_after, "plates": not args.no_plates, "plate_format": args.plate_format,
                "riding_check": args.min_ride_ratio if args.riding_check else None,
                "dedupe_gap": args.dedupe_gap or None, "runtime": "onnxruntime"}
    log = ViolationLog(args.db)
    run_id = log.start_run(args.source, args.weights, out_dir.resolve(), settings)

    plate_reader = None
    if not args.no_plates:
        try:
            from ridesafe.plate_reader import PlateReader
            if not os.path.exists(args.vehicle_model):
                raise SystemExit(f"Vehicle model not found: {args.vehicle_model}. "
                                 "Export it: python scripts/export_onnx.py --vehicles (or pass --no-plates)")
            plate_reader = PlateReader(args.vehicle_model, fmt=args.plate_format)
        except ImportError:
            print("Plate reading off: pip install rapidocr_onnxruntime (or pass --no-plates)")

    vehicle_detector = None
    if args.riding_check:  # same COCO model as plate reading, run separately on the main loop
        if not os.path.exists(args.vehicle_model):
            raise SystemExit(f"Vehicle model not found: {args.vehicle_model}. Export it: python scripts/export_onnx.py --vehicles")
        vehicle_detector = OnnxDetector(args.vehicle_model)

    analyzer = VideoAnalyzer(detector, thresholds, out_dir, log=log, run_id=run_id,
                             tracker_config=ByteTrackConfig.from_yaml(args.tracker), fps=src_fps,
                             vid_stride=args.vid_stride, min_frames=args.min_frames, min_ratio=args.min_ratio,
                             uncertain_ratio=args.uncertain_ratio, finalize_after=args.finalize_after,
                             plate_reader=plate_reader, plate_frames=args.plate_frames,
                             plate_min_agree=args.plate_min_agree, vehicle_detector=vehicle_detector,
                             riding_every=args.riding_every, min_ride_ratio=args.min_ride_ratio,
                             dedupe_gap=args.dedupe_gap or None)

    print(f"Source: {args.source} ({src_fps:.1f} FPS{f', {total} frames' if total else ''}) | imgsz {detector.imgsz} | "
          f"thresholds {thresholds} | stride {args.vid_stride} | log: {args.db} (run {run_id})")

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
            tracked = analyzer.step(frame_idx, frame)

            if not args.no_boxes:
                for tid, cls_name, conf, (x1, y1, x2, y2) in tracked:  # frame = 1-based index in the source video
                    boxes_csv.writerow([frame_idx, tid, cls_name, round(conf, 3), round(x1), round(y1), round(x2), round(y2)])

            elapsed = time.perf_counter() - t_start
            s = analyzer.tracker.summary()
            stats = [f"Riders: {s['riders']}   Helmet: {s['with_helmet']}   Violations: {s['violations']}",
                     f"Frame {frame_idx}{f'/{total}' if total else ''}   {processed / elapsed:.1f} FPS (CPU)"]
            if not args.no_video or args.show:
                frame = draw(frame, tracked, analyzer, stats)
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

    analyzer.finish()
    elapsed = time.perf_counter() - t_start
    summary = save_results(out_dir, args, analyzer, frame_idx, processed, elapsed, settings)
    log.finish_run(run_id, processed, summary["processing_fps"], summary, status)
    log.close()
    return out_dir, summary


def save_results(out_dir, args, analyzer, frame_idx, processed, elapsed, settings):
    """End of run: the per-track table and summary (violations themselves were logged as they happened)."""
    tracker, to_time = analyzer.tracker, analyzer.to_time
    rows, violations = [], []
    for t in sorted(tracker.all_tracks(), key=lambda t: (t.track_id, t.first_frame)):
        entry = analyzer.logged.get((t.track_id, t.first_frame)) if t.violation else None
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
            "duplicate_of": "" if t.duplicate_of is None else t.duplicate_of,
            "ride_checks": t.ride_checks,
            "on_motorcycle": t.on_motorcycle,
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

    print(f"\nDone: {processed} frames in {elapsed:.1f}s ({summary['processing_fps']} FPS). Results: {os.path.normpath(out_dir)}")
    print(f"   Riders: {summary['riders']} | With helmet: {summary['with_helmet']} | "
          f"Violations: {summary['violations']} (plates read: {summary['plates_read']}) | "
          f"Uncertain: {summary['uncertain']} | Raw tracks: {summary['tracks_total']}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Helmet violation tracking on video (ByteTrack, ONNX Runtime)")
    ap.add_argument("--source", required=True, help="Video file, webcam index (0) or stream URL (rtsp://...)")
    ap.add_argument("--weights", default="models/best.onnx", help="Helmet model (ONNX, from scripts/export_onnx.py)")
    ap.add_argument("--thresholds", default=DEFAULT_THRESHOLDS, help="Per-class thresholds YAML")
    ap.add_argument("--tracker", default="configs/bytetrack.yaml")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="Detector input size (default: the export size, 640); larger finds smaller, far-away heads")
    ap.add_argument("--min-frames", type=int, default=5, help="No-helmet votes needed to confirm a violation")
    ap.add_argument("--min-ratio", type=float, default=0.6, help="Share of a track's votes that must be no-helmet")
    ap.add_argument("--uncertain-ratio", type=float, default=0.4,
                    help="Not a violation, but this share of no-helmet votes marks the rider Uncertain")
    ap.add_argument("--finalize-after", type=int, default=30,
                    help="Decide a track after it is unseen for this many processed frames (match track_buffer)")
    ap.add_argument("--vid-stride", type=int, default=1, help="Process every Nth frame (2-3 speeds up CPU runs)")
    ap.add_argument("--max-frames", type=int, default=0, help="Stop after this many processed frames (0 = all)")
    ap.add_argument("--show", action="store_true", help="Show a live window")
    ap.add_argument("--no-video", action="store_true", help="Don't write annotated.mp4")
    ap.add_argument("--no-plates", action="store_true", help="Don't read number plates")
    ap.add_argument("--plate-format", default="india", choices=["india", "any"], help="Plate text validation")
    ap.add_argument("--plate-frames", type=int, default=5, help="Frames per violator to read the plate from")
    ap.add_argument("--plate-min-agree", type=int, default=2, help="Frames that must read the same plate text")
    ap.add_argument("--vehicle-model", default="models/vehicles.onnx",
                    help="COCO detector (ONNX) used to find the rider's vehicle")
    ap.add_argument("--db", default="results/violations.db", help="SQLite violation log (created if missing)")
    ap.add_argument("--riding-check", action="store_true",
                    help="Only log violations for heads on a motorcycle (not pedestrians, not cyclists)")
    ap.add_argument("--riding-every", type=int, default=2, help="Riding check every Nth processed frame")
    ap.add_argument("--min-ride-ratio", type=float, default=0.5,
                    help="Share of a track's riding checks that must find it on a motorcycle")
    ap.add_argument("--dedupe-gap", type=int, default=0,
                    help="Link a new no-helmet track to a violator logged up to this many processed frames earlier "
                         "at the same place, instead of logging it again (0 = off)")
    ap.add_argument("--no-boxes", action="store_true", help="Don't write track_boxes.csv (long streams)")
    ap.add_argument("--out", default="results/video")
    run(ap.parse_args())
