"""Unit tests for the numpy ByteTrack port (no model, no PyTorch)."""

import numpy as np
import pytest

pytest.importorskip("lap")

from ridesafe.bytetrack import ByteTrackConfig, ByteTracker  # noqa: E402

CFG = ByteTrackConfig(track_high_thresh=0.16, track_low_thresh=0.1, new_track_thresh=0.2, track_buffer=30)


def box(x, y, conf=0.9, cls=0, w=40, h=50):
    return [x, y, x + w, y + h, conf, cls]


def test_new_track_is_confirmed_on_its_second_frame_and_keeps_its_id():
    t = ByteTracker(CFG)
    t.update([box(100, 100)])              # frame 1: tracks born in the first frame are confirmed at once
    t.update([box(104, 100), box(400, 300)])  # a new head appears: not reported yet
    out = t.update([box(108, 100), box(404, 300)])
    ids = sorted(out[:, 4].tolist())
    assert ids == [1, 2]
    assert len(t.update([box(112, 100), box(408, 300)])) == 2


def test_weak_detection_keeps_an_existing_track_but_cannot_start_one():
    t = ByteTracker(CFG)
    t.update([box(100, 100)])
    out = t.update([box(103, 100, conf=0.12), box(500, 500, conf=0.12)])  # weak: below track_high_thresh
    assert out[:, 4].tolist() == [1]


def test_lost_track_is_refound_with_the_same_id_within_the_buffer():
    t = ByteTracker(CFG)
    for i in range(5):
        t.update([box(100 + 2 * i, 100)])
    for _ in range(10):                       # occluded for 10 frames
        assert len(t.update([])) == 0
    out = t.update([box(110, 100)])
    assert out[:, 4].tolist() == [1]


def test_track_is_dropped_after_the_buffer():
    t = ByteTracker(ByteTrackConfig(track_buffer=3))
    t.update([box(100, 100)])
    for _ in range(5):
        t.update([])
    t.update([box(100, 100)])                 # same place, but the old track is gone: a new, unconfirmed one
    out = t.update([box(100, 100)])
    assert out[:, 4].tolist() == [2]


def test_output_columns():
    out = ByteTracker(CFG).update([box(10, 20, conf=0.8, cls=1)])
    x1, y1, x2, y2, tid, conf, cls = out[0]
    assert (tid, cls) == (1, 1) and conf == pytest.approx(0.8)
    np.testing.assert_allclose([x1, y1, x2, y2], [10, 20, 50, 70], atol=1e-3)
    assert ByteTracker(CFG).update(np.zeros((0, 6))).shape == (0, 7)
