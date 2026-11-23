# Evaluation

How the model and the video system were chosen and measured. Every number here comes from a script in
[`scripts/`](../scripts/) and a generated report in [`reports/`](../reports/README.md). Validation data is used for
choosing and tuning; test data only for the final numbers.

**Contents:** [Image results](#image-results) · [Model selection](#model-selection-9-experiments) ·
[Headwear data](#new-data-for-the-headwear-failure) · [Operating threshold](#operating-threshold) ·
[Findings](#what-the-analysis-found) · [Limitations](#known-limitations) · [Video benchmark](#video-benchmark-per-rider) ·
[Number plates](#number-plates-best-effort)

## Image results

Model: **YOLO26s** (9.9M params), fine-tuned (AdamW, lr0=1e-3, imgsz=640, batch 16, best epoch 28) on a Tesla T4. Training data: the original dataset with violator images **repeated 2×** (class imbalance), plus **2,537 images from a second dataset** chosen to cover what the first model got wrong: non-helmet headwear and rear views ([how it was built](#new-data-for-the-headwear-failure)).

| Test set (never used for training or selection) | Images | mAP50 | Without Helmet AP50 | Without Helmet recall* | Previous model |
|---|---|---|---|---|---|
| Original test set | 63 | **0.937** | 0.889 | 0.852 | 0.917 / 0.881 / 0.869 |
| **Headwear test set** (caps, turbans, hijabs, masks, rear views) | 635 | **0.810** | **0.781** | **0.763** | 0.320 / 0.310 / 0.285 |
| **Video, per rider** (4 hand-labelled test clips, 40 violators) | 54 riders | n/a | n/a | **18/40** at 75% precision (no helmeted rider flagged) | highway clip only: 5/12 → now 8/12 |

\*At the operating threshold. "Previous model" = the YOLO26s + oversampling model before the headwear data.

## Model selection: 9 experiments

All candidates were trained with the same recipe on Kaggle (T4) and compared with [`scripts/benchmark.py`](../scripts/benchmark.py). Models were picked on the **validation** columns only; test is reported as the unbiased estimate. CPU latency is batch 1 on an Intel i7-7700K.

| Model | Params (M) | CPU ms/img | val mAP50 | val no-helmet recall | test mAP50 | test no-helmet recall |
|---|---|---|---|---|---|---|
| **YOLO26s + oversampling + headwear data** ✅ | 9.9 | 108 | 0.918 | **0.887** | **0.937** | 0.852 |
| YOLO26s + oversampling + headwear data, longer schedule | 9.9 | 108 | 0.916 | 0.896 | 0.933 | 0.934 |
| YOLO26s + oversampling (previous model) | 9.9 | 108 | **0.925** | 0.843 | 0.917 | 0.869 |
| YOLO26s | 9.9 | 106 | 0.893 | 0.809 | 0.915 | 0.885 |
| YOLO11s | 9.4 | 103 | 0.881 | 0.774 | 0.914 | 0.869 |
| YOLOv8s @ 416 px | 11.1 | 55 | 0.859 | 0.730 | 0.893 | 0.770 |
| YOLOv8n | 3.1 | 49 | 0.854 | 0.774 | 0.871 | 0.803 |
| YOLO26n | 2.5 | 51 | 0.824 | 0.713 | 0.909 | 0.787 |
| YOLOv8s (baseline recipe) | 11.1 | 110 | 0.818 | 0.670 | 0.869 | 0.770 |

No-helmet recall is at confidence 0.25. Full tables: [reports/model_comparison.md](../reports/model_comparison.md), [headwear test](../reports/model_comparison_headwear.md).

- **Architecture mattered most** among the first 7: YOLO26s beat YOLOv8s by +0.08 val mAP50 at the same speed; oversampling violators added +0.03.
- **The headwear data was chosen on evidence, not by val mAP50:** on the original val set the three YOLO26s+oversampling variants are within noise (0.916-0.925 mAP50), while no-helmet recall on val rose 0.84 → 0.89 and the held-out headwear test went 0.32 → 0.81 mAP50. The two headwear runs tie on val, so the shorter one was kept.

## New data for the headwear failure

The [video evaluation](#video-benchmark-per-rider) showed the model reading **skullcaps, caps and head cloths as helmets**. [`scripts/mine_headwear.py`](../scripts/mine_headwear.py) turned a public second dataset ([Helmet, Rider, ANPR Detection v12](https://universe.roboflow.com/onlytusik/helmet-rider-anpr-detection/dataset/12), CC BY 4.0, 14.5k images) into clean training data:
- **Leakage first:** that dataset is a mix of other public datasets and **contains our own source images**. 231 images that match ours by name or by perceptual hash were removed (the hash distances show a clean gap between duplicates and the rest), so no test image can be learned.
- **Cleaning:** one copy per original photo (its augmented copies dropped), polygons converted to boxes, product photos/close-ups and riders without a labelled head dropped.
- **Mining with the old model** showed what it got wrong: caps, turbans, hijabs, hoods, **medical face masks** and many **rear views**.
- **Split by source** (all frames of a video on one side): 2,537 images for training, **635 held out as the headwear test set**.

Turbans are labelled "without helmet", which is visually true; Indian law exempts turban-wearing Sikhs, so that is a deployment policy, not a model decision.

## Operating threshold

The violation-class threshold is tuned **on the validation set** to maximize F2 (recall weighted 2×), then checked once on test:

| Without Helmet (original test set) | Threshold | Precision | Recall | Missed violators | False alarms |
|---|---|---|---|---|---|
| First baseline (YOLOv8s), tuned | 0.06 | 0.634 | 0.852 | 9 of 61 | 30 |
| YOLO26s + oversampling, tuned | 0.16 | 0.855 | 0.869 | 8 of 61 | 9 |
| **Current (+ headwear data), tuned** | **0.31** | 0.722 | 0.852 | 9 of 61 | 20 |

**Why false alarms went up on this test set:** the current model now (correctly) calls caps and bandanas "no helmet". Most of its extra "false alarms" are people in caps or bandanas, and riders at motorbike rallies, whose heads were never labelled in the original dataset, which only labels some heads per image. A few (about 6-8 on val) are real errors: **pedestrians** without helmets, since the model has no notion of who is riding. On the highway test clip every logged violation was correct (8/8); across all 4 test clips 18 of 24 logs were real, and all 6 false logs were people who were not riding. Details: [reports/thresholds.md](../reports/thresholds.md), [error analysis](../reports/error_analysis.md).

## What the analysis found

- **Headwear was the main gap** and the extra data closed most of it on the held-out headwear test (no-helmet recall 0.28 → 0.76) and on the highway clip (5 → 8 of 12 violators caught). Headscarves, face scarves and woolly caps are still missed on video (see the video benchmark below).
- **With-helmet recall recovered:** oversampling alone had made the model miss more helmeted heads (22 missed on val); with the extra data it misses 8, and small helmeted heads went from 0.69 to 0.80 recall.
- **Tiny heads remain hard for every model:** below 24 px tall, no-helmet recall is under 0.10 on the headwear test; far-away riders in video are the main remaining misses.
- **A "head must be on a vehicle" filter was tested and rejected.** A COCO detector finds bicycles/motorcycles, and a head counts only if it sits where a rider's head would be (geometry measured on train, limits tuned on val). On val, 16 of the 19 no-helmet false alarms of the previous model already sat on or right next to a bike, so the filter removed 3 false alarms but also dropped real violators. The same association is reused to find a violator's vehicle for plate reading. [Association report](../reports/association.md)
- **Distribution shift between splits:** train is 79% motorcycle images, but val and test are about 45% bicycle. Recall is similar on both vehicle types. [Dataset audit](../reports/dataset_audit.md)

## Known limitations
- The original test set is small (63 images, 61 violators), and the video benchmark is 4 short clips (40 violators), so differences of a few points are within noise.
- On video, people who are not riding (pedestrians standing among riders) can be logged as violators: all 6 false violations on the test clips are of this kind.
- Night, rain and CCTV angles are still thinly covered.
- Pedestrians without helmets can be flagged; there is no reliable "is riding" signal yet.
- CPU inference takes about 97 ms per 640x640 image with ONNX Runtime (121 ms with PyTorch, Intel i7-7700K) and 52 ms per video frame (79 ms with PyTorch), with identical results; see [reports/onnx_benchmark.md](../reports/onnx_benchmark.md). INT8 quantization was tried and rejected (1.8x faster, but no-helmet recall 0.85 → 0.54).

## Video benchmark, per rider

The video benchmark has 4 **test** clips (reported only, never used for tuning) and 4 **validation** clips (for tuning video settings), all from Pexels ([manifest](../eval/video_gt/videos.yaml); `python scripts/download_eval_videos.py` fetches them). Every rider in the test clips was labelled in [`eval/video_gt/`](../eval/video_gt/) (helmet yes/no, readable plate, head boxes): drafted from the tracker's boxes, checked by hand on 1080p crops and reviewed by the author, who corrected entries that were not riders. [scripts/eval_video.py](../scripts/eval_video.py) scores each **rider**, not each frame.

The first 4 validation clips (43 riders, 23 violators) were labelled the same way, with two additions, because a draft built
only from the helmet model's tracks cannot contain the riders that model misses: the drafts came from a high-recall
run (1280 px input, low tracker thresholds), and an independent COCO check (a person on a motorcycle or bicycle with no
tracked head) proposed the riders the model never boxed, including 5 of the 9 riders at the railway crossing. Every
entry was checked by eye and reviewed by the author. **Bicycle riders** are labelled (`vehicle: bicycle`) but need no
helmet: they are never violators, and a violation logged on one counts as a false log. Baseline on validation with the
production settings: 12 of 23 violators caught (52%), 12 of 16 logs correct (75%), 8 duplicate logs for one rider.

**Larger input size (tried, not adopted).** On validation, a 960 px input raised violation recall from 52% to 70% and
precision from 75% to 89%. On the test clips it did not hold: recall 45% → 42% and precision 78% → 40%, because the
larger input also finds more pedestrians' heads in crowded scenes, and the system has no signal for who is riding.
Production stays at 640; a riding check comes first. Details: [reports/video_input_size.md](../reports/video_input_size.md).

**Riding check.** Only a head that sits on a motorcycle (COCO model, checked every 2nd frame while someone has a
no-helmet vote) can be a violator; mostly on a bicycle is a cyclist. On a fifth validation clip, a crowded market
street, and then on the test clips it kept every violator and raised precision at 640 from 78% to 86% on test. It does
not rescue 960: pedestrians standing among the bikes at a crowded junction still pass a per-frame geometric check.
Details: [reports/video_riding_check.md](../reports/video_riding_check.md).

**Validation label fixes (2026-11-23).** Looking at the validation false logs showed 2 riders missing from the labels
(a scooter driver hidden behind his pillion, a rider late in the market clip), 2 keyframes on a pillion's raised hands
instead of her head, and a gap in one rider's keyframes; each fix is noted in that clip's `annotation.corrections`.
Validation is now 5 clips, 61 scored riders, 34 violators. Production (640 + riding check) on it: 16 of 34 violators
caught (47%), 16 of 19 logs correct (84%), 10 duplicate logs; the 2 logs on non-riders are a cycle-rickshaw
passenger and a pedestrian walking past parked bikes. Earlier reports keep the numbers from the labels at the time.

| Test clip | Scene | Riders (no helmet) | Violation recall | Violation precision |
|---|---|---|---|---|
| Highway | daytime, riders facing the camera | 16 (12) | 8/12 | 8/8 |
| Pune junction (20 s) | crowded signal, many pedestrians | 30 (26) | 9/26 | 9/13 |
| High-angle junction | CCTV-like view, almost all heads under 24 px | 3 (0) | n/a | 0/2 |
| Night / fog | riders seen from behind | 5 (2) | 1/2 | 1/1 |
| **All test clips** | | **54 (40)** | **18/40 (45%)** | **18/24 (75%)** |

The Pune section was chosen with the COCO detector (the 20 s with the most bicycles/motorcycles), not with the helmet model, so the test does not favour scenes the model already handles. Full report: [reports/video_eval.md](../reports/video_eval.md).

**What this shows:**
- **No helmeted rider was ever flagged** (0 of 6 false violations); every false violation is on someone who is **not riding**: a pedestrian standing at the Pune crossing (logged twice), people in the crowd, a tempo. The model has no notion of who is riding.
- **Recall depends heavily on the scene:** 67% on clean, front-facing traffic, 35% at a crowded junction, where many riders are never detected at all and headwear is common.
- **Headwear is still the main miss** even after the extra training data: headscarves, scarves wrapped over the face, a red woolly cap and a cloth cap are read as helmets. Skullcaps, the original failure, are now caught.
- **Far riders** (heads under about 36 px) are rarely tracked; in the high-angle clip the model detects none of the dozens of small riders.

With 40 violators the result is still uncertain (95% range for recall roughly 31-60%), but it rests on four different scenes instead of one clip. (Indian law exempts turban-wearing Sikhs from the helmet rule, so a deployed system needs a policy for turbans rather than a blanket "headwear = violation".)

Earlier fixes found on the highway clip: class-agnostic NMS removed duplicate logs (8 logs for 5 people before), and split votes (≥ 40% no-helmet) are shown as **Uncertain** (orange) for a reviewer instead of "With Helmet". To evaluate your own clip (see [DEVELOPMENT.md](DEVELOPMENT.md#video-from-the-command-line)): run `track_video.py`, draft labels with `scripts/video_gt_draft.py`, correct them by hand, then run `eval_video.py` (put videos in `data/videos/`, which git ignores).

## Number plates (best effort)
For each violator, the 5 frames where they are closest to the camera are kept; a COCO detector finds the vehicle the rider is on ([association.py](../backend/ridesafe/association.py)), RapidOCR reads it, and [plates.py](../backend/ridesafe/plates.py) validates the Indian format (state code, district, series, number), fixes look-alike characters by position (series letters never use I/O, so a 0 there is a D) and votes across frames. A plate is only reported if **at least 2 frames agree**; the output shows how many agreed and the plate's width in pixels. `--no-plates` turns it off.

Same clip at two resolutions, scored by `eval_video.py` on the 3 violators whose plate a person can read:

| Readable plates (3) | 720p | 1080p |
|---|---|---|
| Correct | 0 | **1** (`MP04MF3217`, 97 px wide; unreadable at 720p) |
| Wrong | 1 | 1 |
| Not read | 2 | 1 |

**What this shows:** resolution matters (a plate became readable at 1080p), but it is not the only limit. The one wrong plate is `MP04ZD7116`, read as `MP02ZD7116` in 4 of 5 frames **even at 286 px wide and sharp**: the general-purpose OCR model misreads this plate's open-top "4". A second limit: when a rider's own bike is hidden behind a closer one, the reader can attach the closer bike's plate (seen on this clip for a pillion-carrying rider behind another). A plate-specific recognizer (for example fast-plate-ocr) and stricter vehicle matching are the next steps; until then plate text is evidence for a human reviewer, not an automatic fine. Plates are personal data: keep outputs private and blur them in public demos.
