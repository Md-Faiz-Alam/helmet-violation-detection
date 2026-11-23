"""Turn per-frame tracked detections into per-rider decisions (pure Python, no model code).

ByteTrack gives every head a track ID that persists across frames. The detector's class for
that head can flicker between frames (a rear view, motion blur, an ID switch between two
overlapping heads), so no single frame decides anything. Each track collects votes instead:

  * A frame votes for a class only if the detection's confidence passes that class's tuned
    threshold (configs/thresholds.yaml), and the box is not cut off by the frame border
    (a half-visible head often looks helmetless).
  * A track **is a violation** when it has at least `min_frames` "Without Helmet" votes AND
    they make up at least `min_ratio` of all its votes, counted over the WHOLE track.
  * The decision is **final when the track ends**: not seen for `finalize_after` frames (the
    rider left), or the video ended. Only then is a violation logged, once per track. While a
    track is alive, `is_violating()` gives the provisional state for display.
  * A track that is not a violation but has at least `uncertain_ratio` no-helmet votes (e.g. 16 of
    33) is **Uncertain**, not "With Helmet": the evidence is split and a person should look at it.
  * Tracks seen in fewer than `min_frames` frames are ignored in the counts (one-off false alarms).
  * Optional riding check (`min_ride_ratio`): the helmet model cannot tell a rider from a pedestrian, so
    the video pipeline can record, every few frames, whether each head sits on a motorcycle or a bicycle
    (ridesafe.association). A track is then only a violation if it was on a motorcycle in at least
    `min_ride_ratio` of at least `min_ride_checks` checks; mostly on a bicycle is a **Cyclist** (no helmet
    rule for bicycles), too little evidence of riding is **Not riding**. Off (None) by default.
  * Optional duplicate check (`dedupe_gap`): when a rider is hidden for a moment (behind a bus, a pillion,
    motion blur) ByteTrack gives them a new ID, and the same violator would be logged twice. A track about to
    be logged is instead linked to an earlier violator (`duplicate_of`, not logged again) when it starts at
    most `dedupe_gap` frames after that violator's track ended, within `dedupe_dist` head heights of where
    it ended or of where its last movement was heading, with a head size within `dedupe_scale` x of its last
    one. A driver and a pillion sit about 1.4 head heights apart, so they stay separate. Off (None) by default.
"""

from collections import deque
from dataclasses import dataclass, field

VIOLATION_CLASS = "Without Helmet"
SAFE_CLASS = "With Helmet"
UNCERTAIN = "Uncertain"  # many no-helmet votes, but not enough to call it a violation
NOT_RIDING = "Not riding"  # riding check on: the head was rarely on a two-wheeler (a pedestrian)
CYCLIST = "Cyclist"  # riding check on: mostly on a bicycle, where no helmet is required


@dataclass
class TrackState:
    track_id: int
    first_frame: int
    last_frame: int
    frames_seen: int = 0
    votes: dict = field(default_factory=dict)        # class name -> frames above threshold
    conf_sum: dict = field(default_factory=dict)     # class name -> summed confidence of those frames
    best_violation_conf: float = 0.0                 # most confident no-helmet frame (for the snapshot)
    best_violation_frame: int = -1
    first_violation_frame: int = -1                  # first frame where the rule held (event timestamp)
    finalized: bool = False
    violation: bool = False                          # final decision, set when finalized
    ride_checks: int = 0                             # riding check: frames where it was looked for
    on_motorcycle: int = 0                           # ... and found the head on a motorcycle
    on_bicycle: int = 0                              # ... or on a bicycle
    first_box: tuple | None = None                   # duplicate check: where the track started
    recent: deque = field(default_factory=lambda: deque(maxlen=6))  # ... and its last (frame, box) positions
    duplicate_of: int | None = None                  # ... the logged violator this track continues

    @property
    def total_votes(self):
        return sum(self.votes.values())

    def majority(self):
        """Class with the most votes (ties: higher mean confidence), or None without votes."""
        if not self.votes:
            return None
        return max(self.votes, key=lambda c: (self.votes[c], self.conf_sum[c] / self.votes[c]))


class ViolationTracker:
    """Accumulates tracked detections frame by frame and decides, per track, who is a violator."""

    def __init__(self, thresholds, min_frames=5, min_ratio=0.6, uncertain_ratio=0.4, finalize_after=30,
                 edge_margin=2, default_threshold=0.25, min_ride_ratio=None, min_ride_checks=2,
                 dedupe_gap=None, dedupe_dist=1.0, dedupe_scale=1.4):
        self.thresholds = dict(thresholds or {})
        self.default_threshold = default_threshold
        self.min_frames = min_frames
        self.min_ratio = min_ratio
        self.uncertain_ratio = uncertain_ratio
        self.finalize_after = finalize_after
        self.edge_margin = edge_margin
        self.min_ride_ratio, self.min_ride_checks = min_ride_ratio, min_ride_checks
        self.dedupe_gap, self.dedupe_dist, self.dedupe_scale = dedupe_gap, dedupe_dist, dedupe_scale
        self.violators = deque(maxlen=50)  # logged violators and their duplicates, for the duplicate check
        self.tracks = {}   # open (and recently finalized) tracks by ID
        self.open = {}     # only the undecided tracks: the per-frame work does not grow with run length
        self.archive = []  # finalized tracks whose ID was later reused
        self.events = []  # finalized violations in the order they were decided

    # ------------------------------------------------------------------ rules

    def threshold(self, cls_name):
        return self.thresholds.get(cls_name, self.default_threshold)

    def riding(self, t):
        """None if the riding check is off; else 'motorcycle', 'bicycle' or None (not riding / no evidence)."""
        if self.min_ride_ratio is None:
            return None
        if t.ride_checks < self.min_ride_checks:
            return None
        if t.on_bicycle > t.on_motorcycle and t.on_bicycle / t.ride_checks >= self.min_ride_ratio:
            return "bicycle"
        return "motorcycle" if t.on_motorcycle / t.ride_checks >= self.min_ride_ratio else None

    def may_violate(self, t):
        """Riding check off, or the track rides a motorcycle (helmets are required there, not on bicycles)."""
        return self.min_ride_ratio is None or self.riding(t) == "motorcycle"

    def is_violating(self, t):
        """The violation rule on the votes so far (provisional while the track is alive)."""
        v = t.votes.get(VIOLATION_CLASS, 0)
        return v >= self.min_frames and v / t.total_votes >= self.min_ratio and self.may_violate(t)

    def label(self, t):
        """'Without Helmet' only if the violation rule holds; 'Uncertain' if at least `uncertain_ratio`
        of the votes are no-helmet without meeting the rule; otherwise the majority class (None without votes)."""
        if (t.finalized and (t.violation or t.duplicate_of is not None)) or (not t.finalized and self.is_violating(t)):
            return VIOLATION_CLASS
        if self.min_ride_ratio is not None and t.votes.get(VIOLATION_CLASS, 0):
            ride = self.riding(t)
            if ride == "bicycle":
                return CYCLIST
            if ride is None:
                return NOT_RIDING
        if t.total_votes and t.votes.get(VIOLATION_CLASS, 0) / t.total_votes >= self.uncertain_ratio:
            return UNCERTAIN
        return t.majority()

    def observe_riding(self, track_id, vehicle):
        """Riding check for one frame: vehicle = 'motorcycle', 'bicycle' or None (on no two-wheeler)."""
        t = self.tracks.get(track_id)
        if t is None or t.finalized:
            return
        t.ride_checks += 1
        t.on_motorcycle += vehicle == "motorcycle"
        t.on_bicycle += vehicle == "bicycle"

    def counted(self, t):
        """A real rider: seen long enough, with votes, and not a duplicate of a rider already counted."""
        return t.frames_seen >= self.min_frames and t.majority() is not None and t.duplicate_of is None

    def continues(self, t):
        """The logged violator this track continues (duplicate check), or None."""
        if self.dedupe_gap is None or t.first_box is None:
            return None
        x1, y1, x2, y2 = t.first_box
        cx, cy, h = (x1 + x2) / 2, (y1 + y2) / 2, y2 - y1
        for v in reversed(self.violators):
            gap = t.first_frame - v.last_frame
            if not 0 < gap <= self.dedupe_gap or not v.recent:
                continue
            (f0, b0), (f1, b1) = v.recent[0], v.recent[-1]
            vh = b1[3] - b1[1]
            if vh <= 0 or not 1 / self.dedupe_scale <= h / vh <= self.dedupe_scale:
                continue
            ex, ey = (b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2  # where it was last seen
            n = max(f1 - f0, 1)  # ... and where its last movement was heading
            px = ex + ((b1[0] + b1[2]) - (b0[0] + b0[2])) / 2 / n * gap
            py = ey + ((b1[1] + b1[3]) - (b0[1] + b0[3])) / 2 / n * gap
            dist = min(((cx - ex) ** 2 + (cy - ey) ** 2) ** 0.5, ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5)
            if dist <= self.dedupe_dist * vh:
                return v
        return None

    def _at_edge(self, box, frame_size):
        if not frame_size or not self.edge_margin:
            return False
        w, h = frame_size
        x1, y1, x2, y2 = box
        m = self.edge_margin
        return x1 <= m or y1 <= m or x2 >= w - m or y2 >= h - m

    # ------------------------------------------------------------------ updates

    def update(self, frame_idx, detections, frame_size=None):
        """Add one frame of tracked detections.

        detections: iterable of (track_id, class_name, confidence, (x1, y1, x2, y2)).
        frame_size: (width, height), used to ignore votes from boxes cut off by the border.
        Returns the tracks finalized as violations in this call.
        """
        for track_id, cls_name, conf, box in detections:
            t = self.tracks.get(track_id)
            if t is not None and t.finalized:  # a finalized ID that comes back starts a fresh track
                self.archive.append(t)
                t = None
            if t is None:
                t = self.tracks[track_id] = TrackState(track_id, first_frame=frame_idx, last_frame=frame_idx)
                self.open[track_id] = t
            t.last_frame = frame_idx
            t.frames_seen += 1
            if t.first_box is None:
                t.first_box = tuple(box)
            t.recent.append((frame_idx, tuple(box)))

            if conf < self.threshold(cls_name) or self._at_edge(box, frame_size):
                continue  # tracked, but this frame doesn't vote
            t.votes[cls_name] = t.votes.get(cls_name, 0) + 1
            t.conf_sum[cls_name] = t.conf_sum.get(cls_name, 0.0) + conf
            if cls_name == VIOLATION_CLASS and conf > t.best_violation_conf:
                t.best_violation_conf, t.best_violation_frame = conf, frame_idx
            if t.first_violation_frame < 0 and self.is_violating(t):
                t.first_violation_frame = frame_idx

        stale = [t for t in self.open.values() if frame_idx - t.last_frame > self.finalize_after]
        return self._finalize(stale)

    def finish(self):
        """End of video/stream: finalize every track still open."""
        return self._finalize(list(self.open.values()))

    def _finalize(self, tracks):
        logged = []
        for t in sorted(tracks, key=lambda t: t.last_frame):  # earlier tracks first: the duplicate check needs them
            t.finalized = True
            self.open.pop(t.track_id, None)
            t.violation = self.counted(t) and self.is_violating(t)
            if t.violation and (earlier := self.continues(t)) is not None:
                t.violation = False  # the same violator again: linked, not logged twice
                t.duplicate_of = earlier.duplicate_of if earlier.duplicate_of is not None else earlier.track_id
            if t.violation:
                self.events.append(t)
                logged.append(t)
            if t.violation or t.duplicate_of is not None:
                self.violators.append(t)
        return logged

    # ------------------------------------------------------------------ counts

    def all_tracks(self):
        return self.archive + list(self.tracks.values())

    def summary(self):
        """Counts over tracks long enough to be real riders (provisional for tracks still open)."""
        tracks = [t for t in self.all_tracks() if self.counted(t)]
        labels = [self.label(t) for t in tracks]
        violations = labels.count(VIOLATION_CLASS)
        with_helmet = labels.count(SAFE_CLASS)
        return {
            "riders": len(tracks),
            "with_helmet": with_helmet,
            "violations": violations,
            "uncertain": labels.count(UNCERTAIN),
        }
