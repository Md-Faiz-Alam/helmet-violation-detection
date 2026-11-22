# Torch-free video pipeline vs the PyTorch pipeline

_Measured on 2026-11-22 on an Intel Core i7-7700K (CPU only), on the 4 test clips of the video benchmark._

**Why:** the API's Docker image runs without PyTorch: ONNX Runtime for the detector
(`backend/ridesafe/detector.py`), a numpy port of ByteTrack (`backend/ridesafe/bytetrack.py`) and the
shared pipeline (`backend/ridesafe/video.py`). Before replacing the Ultralytics-based pipeline it had to be
shown that the decisions do not change.

**How:** both pipelines ran on the same clips with the same settings (`--vid-stride 2` for the highway and
Pune clips, 1 for the others, as in [video_eval.md](video_eval.md); `--no-plates`), and both runs were scored by
`scripts/eval_video.py` against the hand-checked ground truth. The per-frame track boxes
(`track_boxes.csv`: frame, track ID, class, confidence, box) were also compared line by line.

## Result

| | PyTorch (Ultralytics `model.track`) | ONNX Runtime + numpy ByteTrack |
|---|---|---|
| Violation recall | 45% (18/40) | 45% (18/40) |
| Violation precision | 75% (18/24) | 75% (18/24) |
| Riders found | 56% (30/54) | 56% (30/54) |
| Duplicate counts / false riders | 12 / 12 | 12 / 12 |
| Duplicate violation logs | 7 | 7 |
| `track_boxes.csv` | | identical on 3 clips; on the night clip 1 of 757 rows differs (confidence 0.223 vs 0.224) |
| Processing speed (CPU, no plates) | 6.2-8.5 FPS | 15.5-17.1 FPS |

The torch-free pipeline makes the same decisions about every rider and is about **2x faster**.

## What it took to match exactly

1. **Dynamic input size.** Ultralytics letterboxes a PyTorch model's input to the smallest multiple of 32
   (a 1280x720 frame becomes 640x384). A fixed 640x640 ONNX export pads more, which shifts weak detections:
   with it, recall was 19/40 and precision 19/26 (within noise, but not the same). The model is now exported
   with a dynamic input size and letterboxed the same way: identical detections on 109 of 109 sampled
   video frames, and 52 ms instead of 79 ms (PyTorch) or 99 ms (fixed 640x640) per frame.
2. **NMS IoU 0.5.** Ultralytics 8.4 uses IoU 0.5 for prediction (0.7 only for validation); the ONNX
   detector first used 0.7 and kept a few extra weak boxes.
3. **Track boxes clipped to the frame.** Kalman-predicted boxes can drift past the image border;
   Ultralytics clips them, so the pipeline does too.

The ByteTrack port itself is checked against Ultralytics' `BYTETracker` in `tests/test_bytetrack_parity.py`
(same IDs and boxes on every frame of 4 synthetic sequences with occlusions, weak detections, arrivals and
departures).

Reproduce: run `scripts/track_video.py` (now ONNX) on the 4 clips with the settings above, score with
`scripts/eval_video.py`, and compare with the numbers in [video_eval.md](video_eval.md), which were measured
with the PyTorch pipeline.
