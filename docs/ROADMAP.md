# Roadmap

What has been built, why, and what comes next. Every number is explained in [EVALUATION.md](EVALUATION.md).

## Where it stands (November 2026)

| Area | Status |
|---|---|
| Model | YOLO26s, chosen from 9 trained experiments on validation data, with extra headwear training data |
| Photos | test mAP50 0.937; held-out headwear test mAP50 0.32 → 0.81 |
| Video (4 test clips, 40 violators) | 17 of 40 violators caught (42%); 17 of 18 logs correct (94%); no helmeted rider flagged |
| Product | API on ONNX Runtime (no PyTorch) and a React web app with a review console |
| Delivery | live on Azure, redeployed from GitHub only after every test passes |

**The engineering is complete for a public demo; the model's video recall is the open problem.**

## Done

**1. Trustworthy results**
- Fixed the original training and detection scripts; per-class metrics, with the violation class reported on its own.
- Threshold tuned on validation (F2: recall weighted 2×), an error analysis with an image gallery and a dataset
  audit (train is 79% motorcycles, validation and test about 45% bicycles).
- Config-driven training and 9 experiments on Kaggle (YOLOv8, YOLO11, YOLO26; input size; oversampling), chosen on
  validation only.

**2. Measuring what matters: video, per rider**
- ByteTrack tracking with one decision per rider (votes over the whole track, decided when the rider leaves).
- A hand-labelled video benchmark: 4 test clips (54 riders, 40 violators) and 5 validation clips (61 riders,
  34 violators), scored per rider.
- It found the main failure (caps, skullcaps and head cloths read as helmets), fixed with 2,537 extra training images
  leak-checked against the project's own test data: headwear test mAP50 0.32 → 0.81.
- Number-plate reading for violators (best effort) and a SQLite violation log written as violations happen.

**3. A product**
- ONNX Runtime and a numpy port of ByteTrack: the same decisions as the PyTorch pipeline, about 2× faster on a CPU,
  in a 620 MB Docker image instead of 2.27 GB.
- An API with background video jobs, a review queue, author and visitor access rules, rate limits, signed image links,
  an audit trail and logging.
- A React web app: photo and video analysis with tracked boxes drawn in sync with playback, a review console with
  keyboard shortcuts, and a model card.
- Continuous deployment to Azure with OIDC sign-in (no stored passwords), a strict Content-Security-Policy, pinned
  dependencies, and browser end-to-end tests against the real container.
- A public sample with faces and number plates blurred automatically.

**4. Fewer false and duplicate logs**
- **Riding check:** only heads on a motorcycle can be violators, so pedestrians and cyclists are not logged.
- **Duplicate check:** a violator hidden for a moment is linked to their first log instead of being logged again.
- Together: test precision 78% → 94% (riding check 78% → 86%, duplicate check 86% → 94%), with no real violator
  lost.

**Tried and rejected** (evidence in the reports): INT8 quantization (1.8× faster, but no-helmet recall 0.85 → 0.54);
a head-on-a-vehicle filter for photos (lost real violators); a larger 960 px input (helped on validation, not on
test); a check that the person moves with the motorcycle (box jitter is as large as the motion).

## Next

In order of expected impact:

1. **Video recall.** More than half of the violators are missed: far-away riders (heads under about 36 px are
   rarely tracked) and headwear still read as a helmet (headscarves, face scarves, woolly caps). Next: more headwear and
   rear-view training data, and a second stage that classifies a zoomed-in crop of each tracked head.
2. **People standing in traffic.** A pedestrian standing among moving motorcycles can still pass the riding check (the
   remaining false log on the test clips). Tracking the motorcycles themselves across frames would show who moves
   with them.
3. **A larger, fresh test set.** The 4 test clips have been used to check several decisions. New clips nobody has
   tuned on, ideally with 100+ violators, would give final numbers with tighter error bars.
4. **Plate reading.** A plate-specific recognizer (for example fast-plate-ocr), measured on the saved plate crops, and
   stricter matching so a rider behind another bike does not get its plate.
5. **Operations.** Alerts on failed jobs, model versioning, and a job queue that survives restarts if the demo ever
   needs more than one replica.

**Policy note:** turbans are labelled "without helmet", which is visually true; Indian law exempts turban-wearing
Sikhs, so a deployed system needs a policy for turbans rather than a blanket "headwear = violation".
