"""The video pipeline shared by scripts/track_video.py and the API: detect -> track -> decide -> log.

    analyzer = VideoAnalyzer(OnnxDetector("models/best.onnx"), thresholds, out_dir, log=log, run_id=run_id,
                             fps=25, vid_stride=2)
    for frame_idx, frame in frames:                 # frame_idx: 1-based index in the source video
        tracked = analyzer.step(frame_idx, frame)   # [(track id, class, conf, [x1, y1, x2, y2]), ...]
    analyzer.finish()                               # decide the open tracks, wait for plate reading

Per frame: the helmet detector runs at a low confidence with class-agnostic NMS (so one head never
becomes two riders), ByteTrack (ridesafe.bytetrack) links the heads across frames, and
ridesafe.tracking turns each track's votes into one decision per rider. A violation is written to the
SQLite log and a snapshot is saved the moment it is decided; its plate is read in a background thread
from the frames where the rider was closest to the camera, so OCR never pauses the video.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from ridesafe.association import AssociationRule, associate
from ridesafe.bytetrack import ByteTrackConfig, ByteTracker
from ridesafe.plates import BestFrames
from ridesafe.tracking import SAFE_CLASS, VIOLATION_CLASS, ViolationTracker

logger = logging.getLogger(__name__)

def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union > 0 else 0.0


RIDING_RULE = AssociationRule(vehicle_conf=0.2)  # riding check: head over a motorcycle/bicycle (association.py)
DETECT_CONF = 0.1  # low, so ByteTrack's second stage can keep weak detections on existing tracks
RED = (40, 40, 230)


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


def label_snapshot(img, text):
    """Add a caption bar under a snapshot."""
    bar = np.full((28, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([img, bar])


class VideoAnalyzer:
    def __init__(self, detector, thresholds, out_dir, *, log=None, run_id=None, tracker_config=None, fps=0.0,
                 vid_stride=1, min_frames=5, min_ratio=0.6, uncertain_ratio=0.4, finalize_after=30,
                 plate_reader=None, plate_frames=5, plate_min_agree=2, caption_plates=True, on_message=print,
                 vehicle_detector=None, riding_every=2, min_ride_ratio=0.5, min_ride_checks=2, dedupe_gap=None):
        """vehicle_detector: a COCO OnnxDetector; when given, the riding check is on (ridesafe.tracking): every
        `riding_every` processed frames, heads are matched to the motorcycles and bicycles in the frame, and only
        tracks on a motorcycle in at least `min_ride_ratio` of their checks can be violations."""
        self.detector = detector
        self.names = detector.names
        self.bytetrack = ByteTracker(tracker_config or ByteTrackConfig())
        self.tracker = ViolationTracker(thresholds, min_frames=min_frames, min_ratio=min_ratio,
                                        uncertain_ratio=uncertain_ratio, finalize_after=finalize_after,
                                        min_ride_ratio=min_ride_ratio if vehicle_detector else None,
                                        min_ride_checks=min_ride_checks, dedupe_gap=dedupe_gap)
        self.vehicles, self.riding_every = vehicle_detector, riding_every
        if vehicle_detector:
            self.two_wheelers = {i: n for i, n in vehicle_detector.names.items() if n in ("motorcycle", "bicycle")}
        self.out_dir = Path(out_dir)
        (self.out_dir / "violations").mkdir(parents=True, exist_ok=True)
        self.log, self.run_id = log, run_id
        self.fps, self.vid_stride = fps, vid_stride
        self.plate_reader, self.plate_k, self.plate_min_agree = plate_reader, plate_frames, plate_min_agree
        self.caption_plates = caption_plates  # False: snapshots never show the plate text (public demo)
        # 1 worker: the plate models are only ever used from that thread
        self.plate_pool = ThreadPoolExecutor(max_workers=1) if plate_reader else None
        self.say = on_message or (lambda _msg: None)
        self.processed = 0      # processed-frame counter (the tracker's clock)
        self.plate_frames = {}  # track id -> BestFrames (closest frames of riders with no-helmet votes)
        self.best_crops = {}    # track id -> best no-helmet crop, until the track is decided
        self.logged = {}        # (track id, first frame) -> {"id", "snapshot", "crop", "plate"} for violations

    def to_time(self, processed_idx):
        """Processed-frame index -> seconds into the source video (None if the FPS is unknown)."""
        if not self.fps or processed_idx <= 0:
            return None
        return round((processed_idx - 1) * self.vid_stride / self.fps, 2)

    def step(self, frame_idx, frame):
        """Process one frame. Returns the confirmed tracks in it: [(track id, class name, conf, box)]."""
        self.processed += 1
        h, w = frame.shape[:2]
        dets = self.detector(frame, conf=DETECT_CONF, agnostic=True)
        tracks = self.bytetrack.update(dets)
        tracks[:, [0, 2]] = tracks[:, [0, 2]].clip(0, w)  # Kalman boxes can drift past the border
        tracks[:, [1, 3]] = tracks[:, [1, 3]].clip(0, h)
        tracked = [(int(tid), self.names[int(c)], float(conf), [x1, y1, x2, y2])
                   for x1, y1, x2, y2, tid, conf, c in tracks.tolist()]

        if self.vehicles and self.processed % self.riding_every == 0:
            self._check_riding(frame, tracked)
        for t in self.tracker.update(self.processed, tracked, frame_size=(frame.shape[1], frame.shape[0])):
            self._on_violation(t)

        reader = self.plate_reader
        if reader:  # plate candidates: the closest frames of every rider with a no-helmet vote
            for tid, _, _, box in tracked:
                if self.tracker.tracks[tid].votes.get(VIOLATION_CLASS, 0):
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    self.plate_frames.setdefault(tid, BestFrames(k=self.plate_k)).offer(
                        self.processed, area, lambda f=frame, b=box: reader.rider_region(f, b))

        # Keep the most confident no-helmet crop of every open track for its snapshot
        for tid, cls_name, _conf, box in tracked:
            if cls_name == VIOLATION_CLASS and self.tracker.tracks[tid].best_violation_frame == self.processed:
                self.best_crops[tid] = context_crop(frame, box)

        # Free what decided (non-violation) riders no longer need: memory stays flat on long streams
        for store in (self.plate_frames, self.best_crops):
            for tid in [tid for tid in store if tid not in self.tracker.open]:
                del store[tid]
        return tracked

    def _check_riding(self, frame, tracked):
        """Riding check: which two-wheeler (if any) is each head on? Only run while some open track has a
        no-helmet vote, since only those can become violations (helmeted traffic costs nothing extra)."""
        if self.vehicles is None or not any(t.votes.get(VIOLATION_CLASS, 0) for t in self.tracker.open.values()):
            return
        suspects = [(tid, box) for tid, _, _, box in tracked]
        pred = self.vehicles(frame, conf=RIDING_RULE.vehicle_conf)
        found = [(tuple(p[:4]), float(p[4]), self.two_wheelers[int(p[5])]) for p in pred
                 if int(p[5]) in self.two_wheelers]
        # The detector can box one two-wheeler twice, as bicycle and as motorcycle: keep the more confident
        found.sort(key=lambda v: -v[1])
        kept = []
        for v in found:
            if all(box_iou(v[0], k[0]) < 0.7 for k in kept):
                kept.append(v)
        found = kept
        match = associate([box for _, box in suspects], [(b, c) for b, c, _ in found], RIDING_RULE)
        for (tid, _), j in zip(suspects, match, strict=True):
            self.tracker.observe_riding(tid, found[j][2] if j >= 0 else None)

    def label(self, track_id):
        """Current (provisional) label of a track: Without Helmet / Uncertain / With Helmet / None."""
        return self.tracker.label(self.tracker.tracks[track_id])

    def finish(self):
        """End of the video: decide the open tracks and wait for plate reading to finish."""
        for t in self.tracker.finish():
            self._on_violation(t)
        if self.plate_pool:
            self.say("Waiting for plate reading to finish...")
            self.plate_pool.shutdown(wait=True)

    def _on_violation(self, t):
        """A rider was just decided as a violation: write it to disk and the log NOW (not at the end)."""
        key = (t.track_id, t.first_frame)
        crop = self.best_crops.pop(t.track_id, None)
        frames = self.plate_frames.pop(t.track_id, None)
        snapshot = f"violations/track_{t.track_id:04d}.jpg" if crop is not None else None
        pending = bool(self.plate_pool and frames)
        if crop is not None and snapshot:
            caption = f"#{t.track_id}  plate: " + ("reading..." if pending else "off" if not self.plate_reader
                                                    else "not read")
            cv2.imwrite(str(self.out_dir / snapshot), label_snapshot(crop, caption))
        violation_id = None
        if self.log is not None:
            violation_id = self.log.add_violation(
                self.run_id, t.track_id, t.votes.get(VIOLATION_CLASS, 0), t.votes.get(SAFE_CLASS, 0),
                video_time_s=self.to_time(t.first_violation_frame), first_seen_s=self.to_time(t.first_frame),
                last_seen_s=self.to_time(t.last_frame), max_conf=round(t.best_violation_conf, 3), snapshot=snapshot)
        self.logged[key] = {"id": violation_id, "snapshot": snapshot, "crop": crop if pending else None, "plate": None}
        when = self.to_time(t.first_violation_frame)
        self.say(f"  Violation logged: track #{t.track_id}" + (f" (no helmet from {when:.1f}s)" if when is not None
                                                               else "") + f" -> log id {violation_id}")
        if self.plate_pool and frames:
            self.plate_pool.submit(self._read_plate, key, frames.payloads(), violation_id, t.track_id)

    def _read_plate(self, key, frames, violation_id, track_id):
        try:
            plate = self.plate_reader.read_track(frames, min_agree=self.plate_min_agree) if self.plate_reader else None
        except Exception as e:  # a failed OCR must never stop the tracker
            logger.warning("plate reading failed for track %d (violation %s)", track_id, violation_id, exc_info=True)
            self.say(f"  Plate reading failed for track #{track_id}: {e}")
            return
        entry = self.logged[key]
        entry["plate"] = plate
        shown = (plate["plate"] if self.caption_plates else "read") if plate else "not read"
        caption = f"#{track_id}  plate: {shown}"
        if entry["crop"] is not None:
            cv2.imwrite(str(self.out_dir / entry["snapshot"]), label_snapshot(entry["crop"], caption))
            entry["crop"] = None  # done with it: free the memory
        if plate:
            plate_image = f"violations/track_{track_id:04d}_plate.jpg"
            cv2.imwrite(str(self.out_dir / plate_image), plate["plate_image"])
            if self.log is not None and violation_id is not None:
                self.log.set_plate(violation_id, plate["plate"], plate["conf"], plate["frames_read"],
                                   plate["frames_tried"], plate["plate_width_px"], plate_image)
            self.say(f"  Plate for track #{track_id}: {plate['plate']} ({plate['frames_read']}/"
                     f"{plate['frames_tried']} frames agree, conf {plate['conf']:.2f}, {plate['plate_width_px']} px wide)")
