"""Unit tests for the torch-free ONNX pre/post-processing (no model needed)."""

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("onnxruntime")

from ridesafe.detector import OnnxDetector, letterbox, nms  # noqa: E402


def test_letterbox_keeps_aspect_and_centres():
    img = np.zeros((480, 640, 3), np.uint8)
    out, r, (px, py) = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    assert r == 1.0 and px == 0 and py == 80
    assert out[0, 0, 0] == 114 and out[320, 320, 0] == 0


def test_nms_removes_overlaps_keeps_separate_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], np.float32)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7]), 0.7)
    assert keep.tolist() == [0, 2]


def make_detector():
    det = OnnxDetector.__new__(OnnxDetector)  # skip loading a model: test postprocess only
    det.conf, det.iou, det.max_det = 0.25, 0.7, 300
    return det


def test_postprocess_maps_back_to_image_pixels_and_is_class_aware():
    # 640x480 image letterboxed to 640x640 -> ratio 1, pad (0, 80)
    raw = np.array([
        # cx, cy, w, h, score helmet, score no-helmet
        [100, 180, 20, 40, 0.9, 0.1],   # helmet at (90, 80)-(110, 120) in image pixels
        [101, 181, 20, 40, 0.1, 0.8],   # same place, other class: kept (class-aware NMS)
        [102, 182, 20, 40, 0.85, 0.0],  # duplicate helmet: suppressed
        [300, 300, 10, 10, 0.1, 0.2],   # below conf: dropped
    ], np.float32).T[None]
    out = make_detector().postprocess(raw, 1.0, (0, 80), (480, 640, 3), 0.25)
    assert len(out) == 2
    np.testing.assert_allclose(out[0], [90, 80, 110, 120, 0.9, 0], atol=1e-4)
    assert out[1, 5] == 1


def test_postprocess_empty():
    raw = np.zeros((1, 6, 10), np.float32)
    assert make_detector().postprocess(raw, 1.0, (0, 0), (640, 640, 3), 0.25).shape == (0, 6)
