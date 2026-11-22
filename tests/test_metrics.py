"""Unit tests for the shared mAP50 function used by scripts/benchmark_onnx.py."""

import numpy as np
import pytest

pytest.importorskip("cv2")

from metrics import average_precision  # noqa: E402


def test_average_precision_perfect_and_half():
    gt = np.array([[0, 0, 0, 10, 10], [0, 20, 20, 30, 30]], float)
    perfect = {"gt": gt, "pred": np.array([[0, 0, 10, 10, 0.9, 0], [20, 20, 30, 30, 0.8, 0]], float)}
    # 0.995, not 1.0: the 101-point interpolation ends at precision 0, exactly like Ultralytics' mAP
    assert average_precision([perfect], 1)[0] == pytest.approx(0.995, abs=1e-6)
    half = {"gt": gt, "pred": np.array([[0, 0, 10, 10, 0.9, 0]], float)}
    # recall 0.5 at precision 1; like Ultralytics, precision is interpolated linearly to 0 at recall 1
    assert average_precision([half], 1)[0] == pytest.approx(0.75, abs=0.01)
