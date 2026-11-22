"""The numpy ByteTrack port (ridesafe.bytetrack) gives the same tracks as Ultralytics' BYTETracker.

Both trackers get the same synthetic detections: several heads moving at different speeds, with
noisy boxes, weak frames (second-stage matching), missed frames (lost / re-found tracks), heads
that leave and arrive, and one-frame false alarms. Runs with the smoke tests (needs Ultralytics).
"""

import numpy as np
import pytest

pytest.importorskip("ultralytics")

from ridesafe.bytetrack import ByteTrackConfig, ByteTracker  # noqa: E402

pytestmark = pytest.mark.smoke


def synthetic_detections(seed, frames=150):
    rng = np.random.default_rng(seed)
    heads = []
    for _ in range(8):
        start = int(rng.integers(0, frames - 30))
        heads.append({"start": start, "end": start + int(rng.integers(20, 120)),
                      "xy": rng.uniform(50, 1100, 2), "v": rng.uniform(-6, 6, 2),
                      "wh": rng.uniform(25, 90, 2), "cls": int(rng.integers(0, 2))})
    seq = []
    for f in range(frames):
        dets = []
        for h in heads:
            if not h["start"] <= f < h["end"] or rng.random() < 0.08:  # outside its lifetime, or missed
                continue
            c = h["xy"] + h["v"] * (f - h["start"]) + rng.normal(0, 1.5, 2)
            w, ht = h["wh"] * rng.uniform(0.95, 1.05, 2)
            conf = rng.uniform(0.11, 0.2) if rng.random() < 0.15 else rng.uniform(0.3, 0.95)
            dets.append([c[0], c[1], c[0] + w, c[1] + ht, conf, h["cls"]])
        if rng.random() < 0.1:  # a one-frame false alarm
            x, y = rng.uniform(0, 1200, 2)
            dets.append([x, y, x + 40, y + 50, rng.uniform(0.2, 0.5), 1])
        seq.append(np.asarray(dets, dtype=np.float32).reshape(-1, 6))
    return seq


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_same_tracks_as_ultralytics(seed):
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace, YAML

    cfg_path = "configs/bytetrack.yaml"
    ours = ByteTracker(ByteTrackConfig.from_yaml(cfg_path))
    theirs = BYTETracker(IterableSimpleNamespace(**YAML.load(cfg_path)))
    for f, dets in enumerate(synthetic_detections(seed)):
        a = ours.update(dets)
        b = theirs.update(Boxes(dets, (1280, 1280)).numpy())  # rows: x1, y1, x2, y2, id, conf, cls, idx
        b = np.asarray(b, dtype=np.float32).reshape(-1, 8)[:, :7]
        a, b = a[np.argsort(a[:, 4])], b[np.argsort(b[:, 4])]
        assert a[:, 4].tolist() == b[:, 4].tolist(), f"frame {f + 1}: track IDs differ"
        np.testing.assert_allclose(a, b, atol=1e-2, err_msg=f"frame {f + 1}")
