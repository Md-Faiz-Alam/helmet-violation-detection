"""RideSafe AI runtime package: helmet detection on ONNX Runtime, per-rider violation decisions,
number-plate rules and the SQLite violation log. Needs no PyTorch.

    detector       OnnxDetector: letterbox, ONNX Runtime inference, NMS (numpy + OpenCV)
    tracking       ViolationTracker: per-track voting and the violation rule
    association    which bicycle/motorcycle a head is on
    plates         plate text cleaning, Indian format rules, voting across frames
    violation_log  SQLite log of runs and violations (with review status)
    thresholds     per-class confidence thresholds (configs/thresholds.yaml)

Training, evaluation and data tooling live in scripts/ and use PyTorch through Ultralytics.
"""

__version__ = "0.1.0"
