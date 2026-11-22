"""RideSafe AI runtime package: helmet detection on ONNX Runtime, per-rider violation decisions,
number-plate rules and the SQLite violation log. Needs no PyTorch.

    api            FastAPI app: image detection, video jobs, review queue (uvicorn ridesafe.api:app)
    jobs           async video jobs: worker thread, progress, expiry, read-only sample job
    settings       API settings from RIDESAFE_* environment variables
    video          the video pipeline: detect -> track -> decide -> log (used by the API and the CLI)
    detector       OnnxDetector: letterbox, ONNX Runtime inference, NMS (numpy + OpenCV)
    bytetrack      ByteTrack multi-object tracker in numpy
    tracking       ViolationTracker: per-track voting and the violation rule
    association    which bicycle/motorcycle a head is on
    plates         plate text cleaning, Indian format rules, voting across frames
    plate_reader   vehicle detector (ONNX) + RapidOCR: a violator's plate
    violation_log  SQLite log of runs and violations (with review status)
    thresholds     per-class confidence thresholds (configs/thresholds.yaml)

Training, evaluation and data tooling live in scripts/ and use PyTorch through Ultralytics.
"""

__version__ = "0.1.0"
