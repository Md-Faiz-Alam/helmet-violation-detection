# 🪖 RideSafe AI: Helmet Violation Detection

[![CI](https://github.com/Md-Faiz-Alam/helmet-violation-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Md-Faiz-Alam/helmet-violation-detection/actions/workflows/ci.yml)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)

Detects riders (motorcycles and bicycles) **with** and **without** helmets in images and videos using a fine-tuned **YOLO26s** model, chosen from 9 trained candidates and retrained with extra data for non-helmet headwear (caps, skullcaps, turbans, hijabs, face masks). It includes evaluation on images and on hand-labelled video, threshold tuning, error analysis and experiment tooling, ByteTrack video tracking with per-rider violation decisions, plus a **Streamlit** web app.

> The class that matters most for road safety is **Without Helmet** (the violation). Missing a violator costs more than a false alarm, so this project optimizes and reports that class separately.

---

## 📊 Results

Model: **YOLO26s** (9.9M params), fine-tuned (AdamW, lr0=1e-3, imgsz=640, batch 16, best epoch 28) on a Tesla T4. Training data: the original dataset with violator images **repeated 2×** (class imbalance), plus **2,537 images from a second dataset** chosen to cover what the first model got wrong: non-helmet headwear and rear views ([how it was built](#new-data-for-the-headwear-failure)).

| Test set (never used for training or selection) | Images | mAP50 | Without Helmet AP50 | Without Helmet recall* | Previous model |
|---|---|---|---|---|---|
| Original test set | 63 | **0.937** | 0.889 | 0.852 | 0.917 / 0.881 / 0.869 |
| **Headwear test set** (caps, turbans, hijabs, masks, rear views) | 635 | **0.810** | **0.781** | **0.763** | 0.320 / 0.310 / 0.285 |
| **Video, per rider** (hand-labelled, 12 violators) | 14 s clip | n/a | n/a | **8/12** at 100% precision | 5/12 at 100% precision |

\*At the operating threshold. "Previous model" = the YOLO26s + oversampling model before the headwear data.

### Model selection: 9 experiments

All candidates were trained with the same recipe on Kaggle (T4) and compared with [`scripts/benchmark.py`](scripts/benchmark.py). Models were picked on the **validation** columns only; test is reported as the unbiased estimate. CPU latency is batch 1 on an Intel i7-7700K.

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

No-helmet recall is at confidence 0.25. Full tables: [reports/model_comparison.md](reports/model_comparison.md), [headwear test](reports/model_comparison_headwear.md).

- **Architecture mattered most** among the first 7: YOLO26s beat YOLOv8s by +0.08 val mAP50 at the same speed; oversampling violators added +0.03.
- **The headwear data was chosen on evidence, not by val mAP50:** on the original val set the three YOLO26s+oversampling variants are within noise (0.916-0.925 mAP50), while no-helmet recall on val rose 0.84 → 0.89 and the held-out headwear test went 0.32 → 0.81 mAP50. The two headwear runs tie on val, so the shorter one was kept.

### New data for the headwear failure

The [video evaluation](#video-track-riders-and-flag-violations) showed the model reading **skullcaps, caps and head cloths as helmets**. [`scripts/mine_headwear.py`](scripts/mine_headwear.py) turned a public second dataset ([Helmet, Rider, ANPR Detection v12](https://universe.roboflow.com/onlytusik/helmet-rider-anpr-detection/dataset/12), CC BY 4.0, 14.5k images) into clean training data:
- **Leakage first:** that dataset is a mix of other public datasets and **contains our own source images**. 231 images that match ours by name or by perceptual hash were removed (the hash distances show a clean gap between duplicates and the rest), so no test image can be learned.
- **Cleaning:** one copy per original photo (its augmented copies dropped), polygons converted to boxes, product photos/close-ups and riders without a labelled head dropped.
- **Mining with the old model** showed what it got wrong: caps, turbans, hijabs, hoods, **medical face masks** and many **rear views**.
- **Split by source** (all frames of a video on one side): 2,537 images for training, **635 held out as the headwear test set**.

Turbans are labelled "without helmet", which is visually true; Indian law exempts turban-wearing Sikhs, so that is a deployment policy, not a model decision.

### Operating threshold

The violation-class threshold is tuned **on the validation set** to maximize F2 (recall weighted 2×), then checked once on test:

| Without Helmet (original test set) | Threshold | Precision | Recall | Missed violators | False alarms |
|---|---|---|---|---|---|
| First baseline (YOLOv8s), tuned | 0.06 | 0.634 | 0.852 | 9 of 61 | 30 |
| YOLO26s + oversampling, tuned | 0.16 | 0.855 | 0.869 | 8 of 61 | 9 |
| **Current (+ headwear data), tuned** | **0.31** | 0.722 | 0.852 | 9 of 61 | 20 |

**Why false alarms went up on this test set:** the current model now (correctly) calls caps and bandanas "no helmet". Most of its extra "false alarms" are people in caps or bandanas, and riders at motorbike rallies, whose heads were never labelled in the original dataset, which only labels some heads per image. A few (about 6-8 on val) are real errors: **pedestrians** without helmets, since the model has no notion of who is riding. On real traffic video, every logged violation was correct (8/8). Details: [reports/thresholds.md](reports/thresholds.md), [error analysis](reports/error_analysis.md).

### What the analysis found

- **Headwear was the main gap** and the extra data closed most of it on the held-out headwear test (no-helmet recall 0.28 → 0.76) and on video (5 → 8 of 12 violators caught, still no false violations).
- **With-helmet recall recovered:** oversampling alone had made the model miss more helmeted heads (22 missed on val); with the extra data it misses 8, and small helmeted heads went from 0.69 to 0.80 recall.
- **Tiny heads remain hard for every model:** below 24 px tall, no-helmet recall is under 0.10 on the headwear test; far-away riders in video are the main remaining misses.
- **A "head must be on a vehicle" filter was tested and rejected.** A COCO detector finds bicycles/motorcycles, and a head counts only if it sits where a rider's head would be (geometry measured on train, limits tuned on val). On val, 16 of the 19 no-helmet false alarms of the previous model already sat on or right next to a bike, so the filter removed 3 false alarms but also dropped real violators. The same association is reused to find a violator's vehicle for plate reading. [Association report](reports/association.md)
- **Distribution shift between splits:** train is 79% motorcycle images, but val and test are about 45% bicycle. Recall is similar on both vehicle types. [Dataset audit](reports/dataset_audit.md)

**Known limitations**
- The original test set is small (63 images, 61 violators), and the video benchmark is one 14 s clip (12 violators), so differences of a few points are within noise.
- Night, rain and CCTV angles are still thinly covered.
- Pedestrians without helmets can be flagged; there is no reliable "is riding" signal yet.
- CPU inference takes about 110 ms per image at 640 px (Intel i7-7700K, PyTorch). An ONNX export is planned.

---

## 📂 Project Structure
```
helmet-violation-detection/
├── app.py                          # Streamlit web app (upload / webcam / samples)
├── configs/
│   ├── base.yaml                   # shared training recipe
│   ├── experiments/*.yaml          # one file per experiment (YOLOv8 / YOLO11 / YOLO26, imgsz, oversampling)
│   ├── thresholds.yaml             # tuned per-class confidence thresholds
│   └── bytetrack.yaml              # tracker settings, aligned with the tuned thresholds
├── scripts/
│   ├── train.py                    # config-driven training
│   ├── evaluate.py                 # per-class metrics + confusion matrix / PR curves
│   ├── tune_thresholds.py          # per-class threshold tuning (val) + test report
│   ├── error_analysis.py           # error breakdown, recall by object size, image gallery
│   ├── association.py              # head-to-vehicle geometry (which bike is this head on?)
│   ├── plate_reader.py             # violator's vehicle -> OCR -> plate text (RapidOCR, CPU)
│   ├── plates.py                   # plate format rules, look-alike fixes, voting (unit-tested)
│   ├── eval_association.py         # tests a head-on-vehicle filter on val/test -> reports/association.md
│   ├── dataset_audit.py            # split composition (motorcycle/bicycle) + recall per vehicle type
│   ├── benchmark.py                # model comparison: accuracy, size, CPU latency
│   ├── make_oversampled_split.py   # oversample violation images (class imbalance)
│   ├── detect.py                   # inference on an image, folder, or video (frame by frame)
│   ├── track_video.py              # video/webcam/RTSP: ByteTrack tracking + one decision per rider
│   ├── eval_video.py               # scores a video run per rider against hand-labelled ground truth
│   ├── video_gt_draft.py           # drafts video ground truth from a run, for a human to correct
│   ├── review_gt.py                # renders ground truth as review sheets (one row per rider)
│   ├── tracking.py                 # per-track voting and violation rule (pure Python, unit-tested)
│   ├── violation_log.py            # SQLite violation log (written as violations happen) + viewer
│   ├── metrics.py                  # shared prediction-to-ground-truth matching
│   └── utils.py                    # thresholds + image listing helpers
├── notebook/
│   └── experiments_kaggle.ipynb    # Kaggle GPU: downloads the data, trains and compares experiment configs
├── reports/                        # generated analysis reports + error gallery
├── models/best.pt                  # trained weights, YOLO26s (20 MB)
├── data/data.yaml                  # dataset config (images are downloaded separately)
├── eval/video_gt/                  # hand-checked video ground truth (YAML; videos are downloaded separately)
├── tests/                          # pytest unit tests
├── samples/                        # demo images for the app
├── docs/PROJECT_REVIEW_AND_PLAN.md # roadmap
├── requirements.txt
├── requirements-dev.txt            # + pytest, ruff
├── ruff.toml / pytest.ini          # lint and test settings
├── .github/workflows/ci.yml        # CI: lint, unit tests, smoke tests
├── LICENSE                         # AGPL-3.0
├── packages.txt                    # system packages for Streamlit Cloud
└── .env.example                    # environment variables
```

---

## 📦 Dataset

[Bike Helmet Detection v2](https://universe.roboflow.com/bike-helmets/bike-helmet-detection-2vdjo/dataset/2) from Roboflow Universe (Public Domain, curated by Syed Salman Reza).

| Split | Images | With Helmet boxes | Without Helmet boxes |
|---|---|---|---|
| train | 3,546 | 10,091 | 4,724 |
| valid | 126 | 184 | 115 |
| test | 63 | 123 | 61 |

Images are 416×416. The classes are imbalanced about 2:1 in favour of *With Helmet*.

**Download** (needs a free Roboflow API key):
```bash
pip install roboflow
export ROBOFLOW_API_KEY=your_key        # Windows PowerShell: $env:ROBOFLOW_API_KEY="your_key"
python -c "import os; from roboflow import Roboflow; Roboflow(api_key=os.environ['ROBOFLOW_API_KEY']).workspace('bike-helmets').project('bike-helmet-detection-2vdjo').version(2).download('yolov8')"
```
This creates a `Bike-Helmet-Detection-2/` folder. Move its `train/`, `valid/` and `test/` folders into `data/`. Keep the `data/data.yaml` from this repo; it already points to them.

---

## ⚙️ Setup

```bash
git clone https://github.com/Md-Faiz-Alam/helmet-violation-detection.git
cd helmet-violation-detection
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac/Linux
pip install -r requirements.txt
```
Tested with Python 3.13 (3.11+ should work). Run all commands from the project root.

---

## 🚀 Usage

### Web app
```bash
streamlit run app.py
```
- Upload an image, take a webcam photo, or pick a sample image
- Tuned per-class thresholds (default) or one adjustable threshold
- Counts riders with/without helmets and flags violations

### Inference on images
```bash
python scripts/detect.py --source data/test/images --n_samples 5   # tuned thresholds from configs/thresholds.yaml
python scripts/detect.py --source samples/image1.jpg --conf 0.4    # one threshold for all classes
```
Annotated images go to `results/detect/run_<timestamp>/`. For video, webcam or a camera stream use `track_video.py` (below).

### Video: track riders and flag violations
`detect.py` labels every frame on its own, so one rider becomes hundreds of detections. `track_video.py` follows each head with **ByteTrack** and makes **one decision per rider**:
```bash
python scripts/track_video.py --source path/to/video.mp4
python scripts/track_video.py --source path/to/video.mp4 --show --vid-stride 2   # live window, every 2nd frame (faster on CPU)
python scripts/track_video.py --source 0                                         # webcam
python scripts/track_video.py --source rtsp://<camera-address>/stream            # IP camera
```
How a violation is decided ([scripts/tracking.py](scripts/tracking.py)):
- One box per head: class-agnostic NMS removes the occasional second box of the other class on the same head (otherwise ByteTrack sees two riders and logs the violator twice).
- Every frame where a head passes its class's tuned threshold is a **vote** for that class. Heads cut off by the frame border don't vote, because a half-visible head often looks helmetless.
- A track is a **violation** if it has at least 5 no-helmet votes **and** they are at least 60% of its votes, counted over the whole track (`--min-frames`, `--min-ratio`).
- The decision is made **when the rider leaves** (track lost for 30 frames, or the video ends), so early frames or an ID switch between two overlapping heads can't lock in a wrong verdict. Each violation is logged once.
- On screen: green = helmet, red = violation, orange = "no helmet?" (more no-helmet votes, not enough to decide).

**Measured on real footage** ([Pexels clip](https://www.pexels.com/video/34395449/) by Aamir Somewhere, 14 s, riders on an Indian highway facing the camera). Every rider was labelled in [eval/video_gt/](eval/video_gt/indian_highway_riders.yaml) (helmet yes/no, readable plate, head boxes; drafted with model help, then reviewed by a person, who added 2 missed riders), and [scripts/eval_video.py](scripts/eval_video.py) scores each rider, not each frame:

| Per rider (16 riders, 12 without a helmet) | Previous model, 720p | **Current, 720p** | **Current, 1080p** |
|---|---|---|---|
| **Violation precision** (logged violations that are real) | 5 / 5 (100%) | **8 / 8 (100%)** | **8 / 8 (100%)** |
| **Violation recall** (violators that were logged) | 5 / 12 (42%) | **8 / 12 (67%)** | **8 / 12 (67%)** |
| Duplicate logs (same violator logged twice) | 0 | 2 | 2 |
| Riders found | 12 / 16 | 9 / 16 | 9 / 16 |

Reports: [video_eval.md](reports/video_eval.md), [video_eval_1080p.md](reports/video_eval_1080p.md).

**What this shows:** the first video evaluation found that 4 of the previous model's 7 misses wore something on their head (two white skullcaps, a cloth cap, a cloth tied on the head) and were read as "With Helmet". After adding headwear data, **both skullcap riders and the skullcap pillion are caught**, and every logged violation is still real. What remains:
- **far-away riders** (heads about 27-36 px tall) are no longer tracked at all with the current model, so "riders found" dropped; they need a higher input resolution, tiled inference or retuned tracker thresholds;
- **2 duplicate logs**: the bearded driver and his pillion are hidden behind a closer rider for a moment and get a second track;
- one rider's head is cut off by the top of the frame.

(Indian law exempts turban-wearing Sikhs from the helmet rule, so a deployed system needs a policy for turbans rather than a blanket "headwear = violation".)

Earlier fixes found on this clip: class-agnostic NMS removed duplicate logs (8 logs for 5 people before), and split votes (≥ 40% no-helmet) are shown as **Uncertain** (orange) for a reviewer instead of "With Helmet". To evaluate your own clip: run `track_video.py`, draft labels with `scripts/video_gt_draft.py`, correct them by hand, then run `eval_video.py` (put videos in `data/videos/`, which git ignores).

#### Number plates (best effort)
For each violator, the 5 frames where they are closest to the camera are kept; a COCO detector finds the vehicle the rider is on ([association.py](scripts/association.py)), RapidOCR reads it, and [plates.py](scripts/plates.py) validates the Indian format (state code, district, series, number), fixes look-alike characters by position (series letters never use I/O, so a 0 there is a D) and votes across frames. A plate is only reported if **at least 2 frames agree**; the output shows how many agreed and the plate's width in pixels. `--no-plates` turns it off.

Same clip at two resolutions, scored by `eval_video.py` on the 3 violators whose plate a person can read:

| Readable plates (3) | 720p | 1080p |
|---|---|---|
| Correct | 0 | **1** (`MP04MF3217`, 97 px wide; unreadable at 720p) |
| Wrong | 1 | 1 |
| Not read | 2 | 1 |

**What this shows:** resolution matters (a plate became readable at 1080p), but it is not the only limit. The one wrong plate is `MP04ZD7116`, read as `MP02ZD7116` in 4 of 5 frames **even at 286 px wide and sharp**: the general-purpose OCR model misreads this plate's open-top "4". A second limit: when a rider's own bike is hidden behind a closer one, the reader can attach the closer bike's plate (seen on this clip for a pillion-carrying rider behind another). A plate-specific recognizer (for example fast-plate-ocr) and stricter vehicle matching are the next steps; until then plate text is evidence for a human reviewer, not an automatic fine. Plates are personal data: keep outputs private and blur them in public demos.

#### Violation log (SQLite) and long-running streams
Every violation is written to **`results/violations.db`** the moment the rider is decided, not at the end of the run, so a crash or Ctrl+C loses nothing and another program (a dashboard, the planned API) can read the log while a camera stream is still running (SQLite WAL mode). Each run is a row in `runs` (source, model, settings, status, speed); each violator a row in `violations` (time, track, votes, snapshot, plate, `review_status` = pending / confirmed / rejected for a human reviewer).
```bash
python scripts/violation_log.py            # list runs
python scripts/violation_log.py --run 1    # list one run's violations
```
For long streams, memory stays flat: data for riders who have been decided is freed, the tracker only checks undecided riders each frame, and plates are read in a **background thread**, so OCR never pauses the video (`--no-boxes` also skips the per-frame box file).

Outputs in `results/video/<name>_<timestamp>/`: `annotated.mp4`, `tracks.csv` (one row per track, with plate), `violations/` (one snapshot per violator, captioned with its plate, plus the plate crop) and `summary.json` (counts + violation log). Speed on CPU (i7-7700K, `--vid-stride 2`): about **9 FPS** for tracking at 640 px, about 5.5 FPS with plate reading on the test clip (plates are read only for violators, about 2-3 s each, once they leave the frame); use `--vid-stride 2` or `3` for real-time on longer videos.

### Tests and lint
```bash
pip install -r requirements-dev.txt
ruff check .          # lint (rules in ruff.toml)
pyright               # type check (settings in pyrightconfig.json; the same engine as VS Code's Pylance)
pytest                # unit tests: tracking votes, plate rules, association, video scoring (~3 s, no model)
pytest -m smoke       # end-to-end: the real model through detect.py, track_video.py and the app (~20 s on CPU)
```
GitHub Actions ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs lint + unit tests on every push and pull request, then type checking and the smoke tests with CPU-only PyTorch.

### Evaluate and analyse
```bash
python scripts/evaluate.py --split test        # mAP, per-class metrics, confusion matrix
python scripts/tune_thresholds.py              # -> configs/thresholds.yaml + reports/thresholds.md
python scripts/error_analysis.py               # -> reports/error_analysis.md + gallery (val split)
python scripts/dataset_audit.py                # -> reports/dataset_audit.md
python scripts/eval_association.py             # -> reports/association.md (head-on-vehicle filter: val tuning, test report)
python scripts/eval_video.py --gt eval/video_gt/indian_highway_riders.yaml --run results/video/<run>   # -> reports/video_eval.md
```

### Train and compare models
Every experiment is a small YAML file in `configs/experiments/` that inherits `configs/base.yaml`:
```bash
python scripts/make_oversampled_split.py       # only needed for the *_oversample experiment
python scripts/train.py --config configs/experiments/yolo26s_640_oversample.yaml --device 0
python scripts/benchmark.py                    # -> reports/model_comparison.md
```
- Runs are saved to `results/train/<experiment>/`. Add `--promote` to copy the best weights to `models/best.pt`, which the app uses.
- No GPU? Run [notebook/experiments_kaggle.ipynb](notebook/experiments_kaggle.ipynb) on Kaggle (free GPU, 30 h/week). It trains all experiments in one background run and saves the results as the notebook's Output.
- Experiment tracking: `pip install mlflow`, then set `MLFLOW_TRACKING_URI=sqlite:///mlflow.db`. Ultralytics logs every run to MLflow automatically; browse them with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

---

## 🗺️ Roadmap

See [docs/PROJECT_REVIEW_AND_PLAN.md](docs/PROJECT_REVIEW_AND_PLAN.md).
1. **ML rigor** ✅: threshold tuning, error analysis, dataset audit, model comparison (YOLOv8 / YOLO11 / YOLO26) and class rebalancing
2. **Video system** ✅: rider tracking with ByteTrack ✅, per-rider violation logic ✅, head-to-vehicle association ✅ (evaluated; not worth it as a false-alarm filter on this data), number-plate OCR ✅ (best effort), video evaluation set ✅, headwear data ✅, SQLite violation log ✅
3. **Deployment:** FastAPI + Docker, ONNX optimization with benchmarks, CI, live demo

---

## 🙌 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) (YOLOv8 / YOLO11 / YOLO26)
- [Roboflow Universe](https://universe.roboflow.com/), the Bike Helmet Detection dataset by Syed Salman Reza (Public Domain)
- [Helmet, Rider, ANPR Detection v12](https://universe.roboflow.com/onlytusik/helmet-rider-anpr-detection/dataset/12) by OnlyTusik (CC BY 4.0), extra training data for non-helmet headwear (cleaned and de-duplicated against our data by `scripts/mine_headwear.py`)
- [Pexels](https://www.pexels.com/video/34395449/), test video by Aamir Somewhere (Pexels license; not redistributed here, downloaded by URL)
- [RapidOCR](https://github.com/RapidAI/RapidOCR) (PaddleOCR models on ONNX Runtime) and [ByteTrack](https://github.com/ifzhang/ByteTrack) via Ultralytics

---

## 📄 License

[AGPL-3.0](LICENSE). This project is built on [Ultralytics](https://github.com/ultralytics/ultralytics), which is AGPL-3.0, so the code here (and any service that runs it for other people, such as a hosted demo or API) is shared under the same license. The dataset is Public Domain; the test video keeps its own Pexels license.

**Responsible use:** number plates and faces are personal data. Plate reading is best effort (see the video results above) and meant as evidence for a human reviewer, not for automatic fines. Keep outputs private and blur faces and plates in public demos.
