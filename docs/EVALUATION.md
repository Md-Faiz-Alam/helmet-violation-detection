# Evaluation

How the model and the video system were chosen and how well they work. Every number comes from a script in
[`scripts/`](../scripts/) and a generated report in [`reports/`](../reports/README.md).

**One rule throughout:** validation data is used for every choice and every tuned setting; the test data only
reports results. Where the test data was used to check a decision, that is said.

**Contents:** [Summary](#summary) · [Data](#data) · [The photo model](#the-photo-model) ·
[The video benchmark](#the-video-benchmark) · [Video pipeline decisions](#video-pipeline-decisions) ·
[Number plates](#number-plates) · [Limitations](#limitations) · [Reproduce](#reproduce-the-numbers)

---

## Summary

| What is measured | Data | Result |
|---|---|---|
| **Video, per rider** (the live system) | 4 test clips: 54 riders, 40 violators | **17/18 logs correct (94%)**, **17/40 violators caught (42%)**, no helmeted rider flagged |
| Video, per rider | 5 validation clips: 61 riders, 34 violators | 16/18 logs correct (89%), 16/34 caught (47%) |
| Photos | original test set, 63 images | mAP50 **0.937**; no-helmet recall 0.85 |
| Photos | headwear test set, 635 images | mAP50 **0.81** (0.32 before the headwear data) |

**In plain words:** the system is precise (it rarely logs someone who did nothing wrong) but misses more than half
of the violators. The misses are mostly riders far from the camera and headwear still read as a helmet.

**Why video precision matters most:** every log goes to a reviewer and could become a fine, so a false log costs
more than a miss. The thresholds and checks were chosen with that in mind.

**How certain are these numbers?** Not very: with 40 test violators, 17/40 means a true recall somewhere between
29% and 58% (95% interval). The benchmark covers four different scenes, but it is small.

## Data

### Photos

| Split | Images | With Helmet boxes | Without Helmet boxes | Used for |
|---|---|---|---|---|
| train | 3,546 | 10,091 | 4,724 | training |
| valid | 126 | 184 | 115 | choosing models and thresholds |
| test | 63 | 123 | 61 | reporting only |
| headwear test | 635 | | | reporting only (caps, turbans, hijabs, masks, rear views) |

The main dataset is [Bike Helmet Detection v2](https://universe.roboflow.com/bike-helmets/bike-helmet-detection-2vdjo/dataset/2)
(Public Domain). Its training images are augmented copies of far fewer originals, and train is 79% motorcycle images
while validation and test are about 45% bicycles ([dataset audit](../reports/dataset_audit.md)). Training also uses
2,537 headwear images from a second dataset, explained [below](#extra-data-for-the-headwear-failure).

### Video benchmark

Nine Pexels clips of Indian traffic, listed in [`eval/video_gt/videos.yaml`](../eval/video_gt/videos.yaml) and
downloaded with `python scripts/download_eval_videos.py`:

- **Test (4 clips):** a highway with riders facing the camera, a crowded Pune junction, a high CCTV-like angle, and a
  night scene in fog. 54 riders, 40 without a helmet.
- **Validation (5 clips):** a Bangalore junction, two city roads seen from behind, a railway crossing at dusk, and a
  crowded Lucknow market street with pedestrians among the riders. 61 riders, 34 without a helmet.

Every rider has a label in [`eval/video_gt/`](../eval/video_gt/): helmet or not, the vehicle (bicycle riders need
no helmet), a readable plate if any, and head boxes at keyframes. Labels were drafted from tracker boxes and from an
independent "person on a two-wheeler" detector (so riders the helmet model never sees are still labelled), then
checked by eye on full-resolution crops and reviewed by the author. Later fixes, such as riders missing from the
first pass, are recorded in each file's `annotation.corrections`.

## The photo model

### Model selection: 9 experiments

All candidates were trained with the same recipe on a Kaggle T4 GPU and compared with
[`scripts/benchmark.py`](../scripts/benchmark.py). They were chosen on the **validation** columns; the test columns
are reported once. CPU latency is one image at a time on an Intel i7-7700K, in PyTorch.

| Model | Params (M) | CPU ms/image | val mAP50 | val no-helmet recall | test mAP50 | test no-helmet recall |
|---|---|---|---|---|---|---|
| **YOLO26s + oversampling + headwear data** ✅ | 9.9 | 108 | 0.918 | **0.887** | **0.937** | 0.852 |
| YOLO26s + oversampling + headwear data, longer schedule | 9.9 | 108 | 0.916 | 0.896 | 0.933 | 0.934 |
| YOLO26s + oversampling | 9.9 | 108 | **0.925** | 0.843 | 0.917 | 0.869 |
| YOLO26s | 9.9 | 106 | 0.893 | 0.809 | 0.915 | 0.885 |
| YOLO11s | 9.4 | 103 | 0.881 | 0.774 | 0.914 | 0.869 |
| YOLOv8s @ 416 px | 11.1 | 55 | 0.859 | 0.730 | 0.893 | 0.770 |
| YOLOv8n | 3.1 | 49 | 0.854 | 0.774 | 0.871 | 0.803 |
| YOLO26n | 2.5 | 51 | 0.824 | 0.713 | 0.909 | 0.787 |
| YOLOv8s (first baseline) | 11.1 | 110 | 0.818 | 0.670 | 0.869 | 0.770 |

No-helmet recall at confidence 0.25. Full tables: [model comparison](../reports/model_comparison.md),
[headwear test](../reports/model_comparison_headwear.md).

- **Architecture mattered most:** YOLO26s beat YOLOv8s by +0.08 validation mAP50 at the same speed; repeating the
  violator images (oversampling, against the 2:1 class imbalance) added +0.03.
- **The headwear data was kept on evidence, not on validation mAP50:** the three YOLO26s + oversampling variants are
  within noise on validation mAP50 (0.916-0.925), but validation no-helmet recall rose 0.84 → 0.89 and the held-out
  headwear test went 0.32 → 0.81. The two headwear runs tie on validation, so the shorter one was kept.

The chosen model: YOLO26s, 9.9M parameters, AdamW (lr 0.001), 640 px input, batch 16, best epoch 28.

### Extra data for the headwear failure

The video benchmark showed the first model reading **skullcaps, caps and head cloths as helmets**.
[`scripts/mine_headwear.py`](../scripts/mine_headwear.py) built training data for exactly that from a public second
dataset ([Helmet, Rider, ANPR Detection v12](https://universe.roboflow.com/onlytusik/helmet-rider-anpr-detection/dataset/12),
CC BY 4.0, 14.5k images):

- **No leakage:** that dataset contains some of this project's own source images. 231 images matching ours by name or
  by perceptual hash were removed, so no test image can be learned.
- **Cleaning:** one copy per original photo, polygons converted to boxes, close-ups and riders without a labelled
  head dropped.
- **Mining:** the first model's mistakes showed what to add: caps, turbans, hijabs, hoods, medical masks and many
  rear views.
- **Split by source** (all frames of one video on the same side): 2,537 images for training, 635 held out as the
  headwear test set.

Result on the headwear test: mAP50 0.32 → 0.81, no-helmet recall 0.28 → 0.76; on the highway video, violators caught
went from 5 to 8 of 12.

### Operating threshold

The no-helmet threshold was tuned **on validation** to maximise F2 (recall weighted 2×), then checked once on test
([threshold report](../reports/thresholds.md)):

| Without Helmet, original test set | Threshold | Precision | Recall | Missed | False alarms |
|---|---|---|---|---|---|
| First baseline (YOLOv8s) | 0.06 | 0.634 | 0.852 | 9 of 61 | 30 |
| YOLO26s + oversampling | 0.16 | 0.855 | 0.869 | 8 of 61 | 9 |
| **Current model** | **0.31** | 0.722 | 0.852 | 9 of 61 | 20 |

The current model's extra "false alarms" on photos are mostly correct calls the dataset never labelled: people in
caps or bandanas, and riders in crowds whose heads were not boxed (the dataset labels only some heads per image). A
few are pedestrians, which the video pipeline's riding check now handles ([error analysis](../reports/error_analysis.md)).

### What the photo analysis found

- **Tiny heads are hard for every model:** below 24 px tall, no-helmet recall is under 0.10 on the headwear test.
- **Helmeted riders are rarely called violators:** with the headwear data, the model misses 8 helmeted heads on
  validation (22 before), and small helmeted heads went from 0.69 to 0.80 recall.
- **Vehicle type does not change recall much**, despite the motorcycle/bicycle shift between splits.

## The video benchmark

### How it is scored

[`scripts/eval_video.py`](../scripts/eval_video.py) scores each **rider**, because a fine is issued per rider, not per
frame. A track belongs to the labelled rider whose head boxes it overlaps (IoU ≥ 0.3 within 8 frames of a keyframe).
Then:

- **Violation recall:** labelled violators with at least one logged violation.
- **Violation precision:** logged violations that belong to a real violator. A log on a helmeted rider, on a cyclist
  or on nobody is a false log; a second log for the same violator is a duplicate.
- Riders whose head is never at least 24 px tall are left out (no system can judge them).

### Results with the live settings

The live pipeline: 640 px input, every 2nd frame, the vote rule, the riding check and the duplicate check
([full report](../reports/video_eval.md)).

| Test clip | Scene | Riders (without helmet) | Violators caught | Logs correct |
|---|---|---|---|---|
| Highway | daytime, riders facing the camera | 16 (12) | 8/12 | 8/8 |
| Pune junction (20 s) | crowded signal, many pedestrians | 30 (26) | 8/26 | 8/9 |
| High-angle junction | CCTV-like view, almost all heads under 24 px | 3 (0) | n/a | no logs |
| Night and fog | riders seen from behind | 5 (2) | 1/2 | 1/1 |
| **All test clips** | | **54 (40)** | **17/40 (42%)** | **17/18 (94%)** |

The Pune section was picked by a general-purpose detector (the 20 s with the most two-wheelers), not by the helmet
model, so the test does not favour scenes the model already handles.

**What limits the result:**
- **The scene decides recall:** 67% on clean, front-facing traffic, 31% at the crowded junction, where many riders
  are never detected at all.
- **Far riders** (heads under about 36 px) are rarely tracked; in the high-angle clip none of the many small riders
  is found.
- **Headwear is still the main miss:** headscarves, scarves over the face, a woolly cap and a cloth cap are read as
  helmets. Skullcaps, the original failure, are now caught.
- **The one false log** is a man standing on the pavement at the Pune junction while motorcycles pass in front of
  him.

## Video pipeline decisions

Each change was chosen on the validation clips, then checked on the test clips. Precision and recall are over all
clips of the set.

| Change | Validation | Test | Decision |
|---|---|---|---|
| One box per head (class-agnostic NMS), *uncertain* label for split votes | highway clip: 8 logs for 5 people → 1 each | | adopted |
| **Riding check:** only a head on a motorcycle can be a violator | precision 58% → 67% | precision 78% → 86%, recall unchanged | **adopted** ([report](../reports/video_riding_check.md)) |
| **Duplicate check:** link a violator who reappears to the first log | duplicate logs 10 → 3, precision 84% → 89%, recall unchanged | precision 86% → 94% | **adopted** ([report](../reports/video_duplicate_logs.md)) |
| Larger input, 960 px | recall 52% → 70% | precision 78% → 40% (pedestrians in crowds) | rejected ([report](../reports/video_input_size.md)) |
| "Moves with the motorcycle" check | riders and the pedestrian score the same: box jitter ≥ the motion | not run | rejected ([report](../reports/video_riding_check.md#tried-and-rejected-does-the-person-move-with-the-motorcycle)) |
| ONNX Runtime + numpy ByteTrack instead of PyTorch | | identical decisions on all 4 clips, about 2× faster | adopted ([report](../reports/runtime_parity.md)) |
| INT8 quantization | | 1.8× faster, but photo no-helmet recall 0.85 → 0.54 | rejected ([report](../reports/onnx_benchmark.md)) |

Validation numbers are from the labels as they were at the time of each decision; the validation labels were later
corrected (2 riders added, 2 keyframes moved and 3 added), which is why the reports show slightly different baselines.

**Note on the duplicate check's test result:** the scored recall went from 18 to 17 of 40. Every link was checked by
eye: the three logs it merged were all of the standing pedestrian, and those logs had been credited to riders who
passed in front of him. No real violator was lost ([details](../reports/video_duplicate_logs.md#test-4-clips-54-riders-40-violators-checked-once-after-the-choice)).

## Number plates

For each violator, the 5 frames where they are closest to the camera are kept; a general-purpose detector finds the
vehicle they ride ([association.py](../backend/ridesafe/association.py)), RapidOCR reads it, and
[plates.py](../backend/ridesafe/plates.py) checks the Indian format (state, district, series, number), fixes
look-alike characters by position and votes across frames. A plate is reported only if **at least 2 frames agree**.

On the highway clip, 3 violators have a plate a person can read:

| Readable plates (3) | 720p video | 1080p video |
|---|---|---|
| Correct | 0 | 1 (`MP04MF3217`, 97 px wide) |
| Wrong | 1 | 1 |
| Not read | 2 | 1 |

Resolution matters, but it is not the only limit: one plate is misread as `MP02ZD7116` (instead of `MP04ZD7116`) even
at 286 px wide, because the general-purpose OCR misreads its open-top 4, and a rider behind another bike can be
given the closer bike's plate. **Plate text is evidence for a reviewer, not proof.** A plate-specific recognizer is
the next step.

## Limitations

- **Small sets:** 63 test photos and 40 test violators, so differences of a few points are within noise.
- **The test clips have been used to check several decisions** (input size, riding check, duplicate check). Each
  decision was made on validation first, but a fresh test set would give cleaner final numbers.
- **Recall is low on crowded and distant scenes**, and night, rain and CCTV angles are thinly covered.
- **People standing in traffic** can still pass the riding check.
- **Turbans** are labelled *no helmet*, which is visually true; Indian law exempts turban-wearing Sikhs, so a
  deployment needs a policy for turbans, not a blanket rule.

## Reproduce the numbers

```bash
python scripts/evaluate.py --split test           # photo metrics on the test set
python scripts/benchmark.py                       # -> reports/model_comparison.md
python scripts/tune_thresholds.py                 # -> configs/thresholds.yaml + reports/thresholds.md
python scripts/download_eval_videos.py            # the 9 benchmark clips -> data/videos/
python scripts/track_video.py --source data/videos/indian_highway_riders.mp4 --no-video   # live settings
python scripts/eval_video.py --gt eval/video_gt/indian_highway_riders.yaml --run results/video/<that run>
```
`eval_video.py` takes several `--gt` / `--run` pairs (in the same order) and writes `reports/video_eval.md`. To label
a new clip: draft labels with `scripts/video_gt_draft.py`, render review sheets with `scripts/review_gt.py`, correct
them by hand, then score. More in [DEVELOPMENT.md](DEVELOPMENT.md#evaluate-and-analyse).
