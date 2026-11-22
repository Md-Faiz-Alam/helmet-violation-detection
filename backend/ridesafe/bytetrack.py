"""ByteTrack multi-object tracker in numpy (no PyTorch): Kalman filter + two-stage IoU matching.

    tracker = ByteTracker(ByteTrackConfig.from_yaml("configs/bytetrack.yaml"))
    tracks = tracker.update(dets)   # dets (N, 6) [x1, y1, x2, y2, conf, cls] -> (M, 7) [x1, y1, x2, y2, id, conf, cls]

A port of the ByteTrack tracker in Ultralytics 8.4 (ultralytics/trackers/byte_tracker.py, AGPL-3.0, itself
based on ByteTrack by Zhang et al., MIT), restricted to axis-aligned boxes and without camera-motion
compensation, so that the API can track without PyTorch and give the same tracks as scripts/track_video.py did:
  1. detections >= track_high_thresh are matched to tracked + lost tracks (IoU fused with the score);
  2. the remaining tracked tracks are matched to the weak detections (track_low_thresh .. track_high_thresh);
  3. tracks born last frame (unconfirmed) are matched to what is left, or removed;
  4. unmatched strong detections >= new_track_thresh start new tracks (confirmed when matched again);
  5. lost tracks are dropped after track_buffer frames.
Returned boxes are the Kalman-filtered track boxes, as in Ultralytics.
"""

from dataclasses import dataclass

import lap
import numpy as np
import yaml

NEW, TRACKED, LOST, REMOVED = 0, 1, 2, 3


@dataclass(frozen=True)
class ByteTrackConfig:
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    fuse_score: bool = True

    @classmethod
    def from_yaml(cls, path):
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(**{k: cfg[k] for k in cls.__dataclass_fields__ if k in cfg})


class KalmanFilterXYAH:
    """Constant-velocity Kalman filter on (centre x, centre y, aspect ratio w/h, height)."""

    def __init__(self):
        self.motion = np.eye(8)
        self.motion[:4, 4:] = np.eye(4)
        self.update_mat = np.eye(4, 8)
        self.w_pos, self.w_vel = 1.0 / 20, 1.0 / 160

    def initiate(self, z):
        z = np.asarray(z, dtype=np.float64)
        h = z[3]
        std = [2 * self.w_pos * h, 2 * self.w_pos * h, 1e-2, 2 * self.w_pos * h,
               10 * self.w_vel * h, 10 * self.w_vel * h, 1e-5, 10 * self.w_vel * h]
        return np.r_[z, np.zeros(4)], np.diag(np.square(std))

    def multi_predict(self, mean, cov):
        h = mean[:, 3]
        one = np.ones_like(h)
        std = np.stack([self.w_pos * h, self.w_pos * h, 1e-2 * one, self.w_pos * h,
                        self.w_vel * h, self.w_vel * h, 1e-5 * one, self.w_vel * h], axis=1)
        motion_cov = np.zeros((len(mean), 8, 8))
        motion_cov[:, range(8), range(8)] = np.square(std)
        mean = mean @ self.motion.T
        cov = np.dot(np.dot(self.motion, cov).transpose((1, 0, 2)), self.motion.T) + motion_cov
        return mean, cov

    def update(self, mean, cov, z):
        h = mean[3]
        innovation_cov = np.diag(np.square([self.w_pos * h, self.w_pos * h, 1e-1, self.w_pos * h]))
        proj_mean, proj_cov = mean[:4], cov[:4, :4] + innovation_cov
        gain = np.linalg.solve(proj_cov, cov[:, :4].T).T
        new_mean = mean + (z - proj_mean) @ gain.T
        new_cov = cov - np.linalg.multi_dot((gain, proj_cov, gain.T))
        return new_mean, new_cov


def xyxy_to_xyah(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / h, h])


class Track:
    def __init__(self, xyxy, score, cls, idx):
        self.det_xyxy = np.asarray(xyxy, dtype=np.float32)
        self.score, self.cls, self.idx = float(score), cls, idx
        self.mean = self.cov = None
        self.state, self.is_activated = NEW, False
        self.track_id = self.start_frame = self.frame_id = 0

    @property
    def xyxy(self):
        if self.mean is None:
            return self.det_xyxy.copy()
        cx, cy, a, h = self.mean[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def activate(self, kf, frame_id, track_id):
        self.track_id = track_id
        self.mean, self.cov = kf.initiate(xyxy_to_xyah(self.det_xyxy))
        self.state = TRACKED
        self.is_activated = frame_id == 1
        self.frame_id = self.start_frame = frame_id

    def update(self, kf, det, frame_id):
        """Matched to a detection (also used to re-find a lost track; the ID is kept)."""
        self.mean, self.cov = kf.update(self.mean, self.cov, xyxy_to_xyah(det.det_xyxy))
        self.state, self.is_activated, self.frame_id = TRACKED, True, frame_id
        self.score, self.cls, self.idx = det.score, det.cls, det.idx


def iou_matrix(a, b):
    """1 - IoU between two lists of tracks (cost matrix, float32)."""
    if not a or not b:
        return np.ones((len(a), len(b)), dtype=np.float32)
    A = np.asarray([t.xyxy for t in a], dtype=np.float32)
    B = np.asarray([t.xyxy for t in b], dtype=np.float32)
    inter = (np.minimum(A[:, None, 2], B[:, 2]) - np.maximum(A[:, None, 0], B[:, 0])).clip(0) * \
            (np.minimum(A[:, None, 3], B[:, 3]) - np.maximum(A[:, None, 1], B[:, 1])).clip(0)
    area_a = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    area_b = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return 1 - inter / (area_b + area_a[:, None] - inter + 1e-7)


def linear_assignment(cost, thresh):
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    x, y = lap.lapjv(cost, extend_cost=True, cost_limit=thresh, return_cost=False)
    return [(i, int(j)) for i, j in enumerate(x) if j >= 0], list(np.where(x < 0)[0]), list(np.where(y < 0)[0])


def joint(a, b):
    ids = {t.track_id for t in a}
    return a + [t for t in b if t.track_id not in ids]


def sub(a, b):
    ids = {t.track_id for t in b}
    return [t for t in a if t.track_id not in ids]


def remove_duplicates(a, b, thresh=0.15):
    """A tracked and a lost track on the same box: keep the older one."""
    pairs = np.where(iou_matrix(a, b) < thresh)
    drop_a, drop_b = set(), set()
    for p, q in zip(*pairs, strict=True):
        if a[p].frame_id - a[p].start_frame > b[q].frame_id - b[q].start_frame:
            drop_b.add(q)
        else:
            drop_a.add(p)
    return [t for i, t in enumerate(a) if i not in drop_a], [t for i, t in enumerate(b) if i not in drop_b]


class ByteTracker:
    def __init__(self, cfg=ByteTrackConfig()):
        self.cfg = cfg
        self.kf = KalmanFilterXYAH()
        self.tracked, self.lost, self.removed = [], [], []
        self.frame_id = 0
        self._next_id = 0

    def new_id(self):
        self._next_id += 1
        return self._next_id

    def _fused_cost(self, tracks, dets):
        cost = iou_matrix(tracks, dets)
        if self.cfg.fuse_score and cost.size:
            cost = 1 - (1 - cost) * np.array([d.score for d in dets])[None]
        return cost

    def update(self, dets):
        """One frame. dets: (N, 6) [x1, y1, x2, y2, conf, cls] -> (M, 7) [x1, y1, x2, y2, track id, conf, cls]."""
        cfg, kf = self.cfg, self.kf
        self.frame_id += 1
        dets = np.asarray(dets, dtype=np.float32).reshape(-1, 6)
        valid = (dets[:, 2] > dets[:, 0]) & (dets[:, 3] > dets[:, 1])
        scores = dets[:, 4]
        high = np.flatnonzero(valid & (scores >= cfg.track_high_thresh))
        low = np.flatnonzero(valid & (scores > cfg.track_low_thresh) & (scores < cfg.track_high_thresh))
        detections = [Track(dets[i, :4], dets[i, 4], dets[i, 5], int(i)) for i in high]
        detections_low = [Track(dets[i, :4], dets[i, 4], dets[i, 5], int(i)) for i in low]

        activated, refind, lost, removed = [], [], [], []
        unconfirmed = [t for t in self.tracked if not t.is_activated]
        confirmed = [t for t in self.tracked if t.is_activated]
        pool = joint(confirmed, self.lost)
        if pool:
            mean = np.asarray([t.mean for t in pool])
            mean[[i for i, t in enumerate(pool) if t.state != TRACKED], 7] = 0  # lost tracks don't grow
            mean, cov = kf.multi_predict(mean, np.asarray([t.cov for t in pool]))
            for t, m, c in zip(pool, mean, cov, strict=True):
                t.mean, t.cov = m, c

        # 1. strong detections <-> tracked + lost tracks
        matches, u_track, u_det = linear_assignment(self._fused_cost(pool, detections), cfg.match_thresh)
        for it, idet in matches:
            t = pool[it]
            (activated if t.state == TRACKED else refind).append(t)
            t.update(kf, detections[idet], self.frame_id)

        # 2. weak detections <-> tracked tracks that are still unmatched
        remaining = [pool[i] for i in u_track if pool[i].state == TRACKED]
        if remaining and detections_low:
            matches, u_rem, _ = linear_assignment(iou_matrix(remaining, detections_low), 0.5)
            for it, idet in matches:
                remaining[it].update(kf, detections_low[idet], self.frame_id)
                activated.append(remaining[it])
        else:
            u_rem = list(range(len(remaining)))
        for i in u_rem:
            remaining[i].state = LOST
            lost.append(remaining[i])

        # 3. unconfirmed tracks (born last frame) <-> leftover strong detections
        detections = [detections[i] for i in u_det]
        if unconfirmed:
            matches, u_unconf, u_det = linear_assignment(self._fused_cost(unconfirmed, detections), 0.7)
            for it, idet in matches:
                unconfirmed[it].update(kf, detections[idet], self.frame_id)
                activated.append(unconfirmed[it])
            for i in u_unconf:
                unconfirmed[i].state = REMOVED
                removed.append(unconfirmed[i])
        else:
            u_det = list(range(len(detections)))

        # 4. new tracks
        for i in u_det:
            if detections[i].score >= cfg.new_track_thresh:
                detections[i].activate(kf, self.frame_id, self.new_id())
                activated.append(detections[i])

        # 5. drop tracks lost for too long
        for t in self.lost:
            if self.frame_id - t.frame_id > cfg.track_buffer:
                t.state = REMOVED
                removed.append(t)

        self.tracked = joint(joint([t for t in self.tracked if t.state == TRACKED], activated), refind)
        self.lost = sub(sub(self.lost, self.tracked) + lost, self.removed)
        self.tracked, self.lost = remove_duplicates(self.tracked, self.lost)
        self.removed = (self.removed + removed)[-1000:]

        out = [[*t.xyxy.tolist(), t.track_id, t.score, t.cls] for t in self.tracked if t.is_activated]
        return np.asarray(out, dtype=np.float32).reshape(-1, 7)
