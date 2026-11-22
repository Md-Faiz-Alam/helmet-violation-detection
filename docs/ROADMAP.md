# Roadmap

What has been built, the decisions behind it, and what comes next. Numbers are explained in
[EVALUATION.md](EVALUATION.md).

## Where it stands (November 2026)

| | Status |
|---|---|
| Model | YOLO26s, picked from 9 experiments on validation data; extra headwear training data |
| Image results | test mAP50 0.937; held-out headwear test mAP50 0.32 → 0.81 |
| Video results | 4 hand-labelled test clips: violation recall 18/40 (45%), precision 18/24 (75%), no helmeted rider flagged |
| Product | FastAPI + ONNX Runtime API (no PyTorch), React web app with a review console |
| Deployment | live on Azure, redeployed automatically from GitHub after all tests pass |

## Done

**1. Make the results trustworthy**
- Fixed the original training and detection scripts; per-class evaluation with the violation class reported separately.
- Per-class threshold tuning on validation (F2, recall weighted 2×), error analysis with an image gallery, and a
  dataset audit (train is 79% motorcycles, val/test about 45% bicycles).
- Config-driven training, 9 experiments on Kaggle (YOLOv8, YOLO11, YOLO26; image size; oversampling), chosen on
  validation only.

**2. Measure what matters: video, per rider**
- ByteTrack tracking with one decision per rider (votes over the whole track, decided when the rider leaves).
- A hand-labelled video benchmark: 4 test clips (54 riders, 40 violators) plus 4 validation clips, scored per rider.
- It found the main failure: caps, skullcaps and head cloths read as helmets.
- Extra training data for that failure, leak-checked against the project's own test images (231 duplicates removed):
  headwear test mAP50 0.32 → 0.81, highway clip 5/12 → 8/12 violators caught.
- Number-plate reading for violators (best effort), and a SQLite violation log written as violations happen.
- Tested and **rejected**: a "head must be on a vehicle" filter (removed 3 false alarms but lost real violators), and
  INT8 quantization (1.8× faster, but no-helmet recall 0.85 → 0.54).

**3. Ship it as a product**
- ONNX Runtime inference with a numpy port of ByteTrack: identical decisions to the PyTorch pipeline, about 2× faster
  on CPU, in a 620 MB Docker image (2.27 GB with PyTorch).
- FastAPI service with async video jobs, a review queue, author/visitor access rules, rate limits, signed image links
  and an audit trail.
- React web app: photo and video analysis with tracked boxes drawn in sync with playback, a review console with
  keyboard shortcuts, and a model card.
- Continuous deployment to Azure (Container Apps + Static Web Apps) with OIDC sign-in, path filters and PR previews.
- A public sample with faces and number plates blurred automatically.

## Next

In order of expected impact:

1. **Video recall.** More than half of violators are still missed, mostly far-away riders (heads under ~36 px) and
   headwear still read as a helmet (headscarves, face scarves, woolly caps). Label the 4 validation clips first, then
   try a larger input size or tiled inference, retune the tracker thresholds, and add more headwear data.
2. **"Is this person riding?"** All 6 false violations on the test clips are pedestrians or other non-riders. Measure a
   temporal head-on-vehicle check on the validation clips.
3. **Plate reading.** A plate-specific recognizer (for example fast-plate-ocr), measured on the saved plate crops;
   stricter vehicle matching so a rider behind another bike does not get its plate.
4. **Retire the Streamlit demo** (`app.py`, `packages.txt`, its dependency and smoke test).
5. **Production hardening.** Error monitoring and alerts, model versioning, and a job queue that survives restarts if
   the demo ever needs more than one replica.
6. **Optional:** a vision-language model as a second opinion on low-confidence crops, measured for false-alarm
   reduction and cost per call.

Policy note: turbans are labelled "without helmet", which is visually true; Indian law exempts turban-wearing Sikhs,
so a deployed system needs a policy for turbans rather than a blanket "headwear = violation".
