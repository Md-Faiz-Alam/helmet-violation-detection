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
"""

from dataclasses import dataclass, field

VIOLATION_CLASS = "Without Helmet"
SAFE_CLASS = "With Helmet"
UNCERTAIN = "Uncertain"  # many no-helmet votes, but not enough to call it a violation


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
    best_violation_box: tuple = ()
    first_violation_frame: int = -1                  # first frame where the rule held (event timestamp)
    finalized: bool = False
    violation: bool = False                          # final decision, set when finalized

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
                 edge_margin=2, default_threshold=0.25):
        self.thresholds = dict(thresholds or {})
        self.default_threshold = default_threshold
        self.min_frames = min_frames
        self.min_ratio = min_ratio
        self.uncertain_ratio = uncertain_ratio
        self.finalize_after = finalize_after
        self.edge_margin = edge_margin
        self.tracks = {}   # open (and recently finalized) tracks by ID
        self.archive = []  # finalized tracks whose ID was later reused
        self.events = []  # finalized violations in the order they were decided

    # ------------------------------------------------------------------ rules

    def threshold(self, cls_name):
        return self.thresholds.get(cls_name, self.default_threshold)

    def is_violating(self, t):
        """The violation rule on the votes so far (provisional while the track is alive)."""
        v = t.votes.get(VIOLATION_CLASS, 0)
        return v >= self.min_frames and v / t.total_votes >= self.min_ratio

    def label(self, t):
        """'Without Helmet' only if the violation rule holds; 'Uncertain' if at least `uncertain_ratio`
        of the votes are no-helmet without meeting the rule; otherwise the majority class (None without votes)."""
        if (t.finalized and t.violation) or (not t.finalized and self.is_violating(t)):
            return VIOLATION_CLASS
        if t.total_votes and t.votes.get(VIOLATION_CLASS, 0) / t.total_votes >= self.uncertain_ratio:
            return UNCERTAIN
        return t.majority()

    def counted(self, t):
        return t.frames_seen >= self.min_frames and t.majority() is not None

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
            t.last_frame = frame_idx
            t.frames_seen += 1

            if conf < self.threshold(cls_name) or self._at_edge(box, frame_size):
                continue  # tracked, but this frame doesn't vote
            t.votes[cls_name] = t.votes.get(cls_name, 0) + 1
            t.conf_sum[cls_name] = t.conf_sum.get(cls_name, 0.0) + conf
            if cls_name == VIOLATION_CLASS and conf > t.best_violation_conf:
                t.best_violation_conf, t.best_violation_frame, t.best_violation_box = conf, frame_idx, tuple(box)
            if t.first_violation_frame < 0 and self.is_violating(t):
                t.first_violation_frame = frame_idx

        stale = [t for t in self.tracks.values() if not t.finalized and frame_idx - t.last_frame > self.finalize_after]
        return self._finalize(stale)

    def finish(self):
        """End of video/stream: finalize every track still open."""
        return self._finalize([t for t in self.tracks.values() if not t.finalized])

    def _finalize(self, tracks):
        logged = []
        for t in tracks:
            t.finalized = True
            t.violation = self.counted(t) and self.is_violating(t)
            if t.violation:
                self.events.append(t)
                logged.append(t)
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
