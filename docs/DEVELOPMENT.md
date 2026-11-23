# Development guide

Everything needed to work on RideSafe AI locally: setup, data, the command-line tools, training, evaluation and
tests. For the HTTP API see [API.md](API.md); for Docker and Azure see [DEPLOYMENT.md](DEPLOYMENT.md).

**Contents:** [Setup](#setup) · [Project structure](#project-structure) · [Dataset](#dataset) ·
[Web app and API locally](#web-app-and-api-locally) · [Images from the command line](#images-from-the-command-line) ·
[Video from the command line](#video-from-the-command-line) · [Evaluate and analyse](#evaluate-and-analyse) ·
[Train and compare models](#train-and-compare-models) · [ONNX export](#onnx-export) · [Tests and lint](#tests-and-lint) ·
[Rebuild the public sample](#rebuild-the-public-sample)

---

## Setup

```bash
git clone https://github.com/Md-Faiz-Alam/helmet-violation-detection.git
cd helmet-violation-detection
python -m venv .venv
.venv\Scripts\activate             # Windows
source .venv/bin/activate          # Mac/Linux
pip install -r requirements-dev.txt
python scripts/export_onnx.py --vehicles   # once: models/best.onnx + models/vehicles.onnx
```

- Tested with Python 3.13 (3.11+ should work). Run every command from the project root.
- `requirements.txt` installs the ML tooling (Ultralytics, PyTorch) and the `ridesafe` package from `backend/` in
  editable mode, so scripts use the working copy. `requirements-dev.txt` adds pytest, ruff, Pyright, ONNX export
  tools and `imageio-ffmpeg` (only for rebuilding the public sample video).
- The web app needs Node.js 24: `cd frontend && npm install`.
- Settings are environment variables; [`.env.example`](../.env.example) lists them. Real values go in `.env`, which
  git ignores.

## Project structure

```
helmet-violation-detection/
├── backend/                     # torch-free runtime + HTTP API (ONNX Runtime)
│   ├── ridesafe/
│   │   ├── api.py               # FastAPI app: detection, video jobs, review queue, access rules
│   │   ├── jobs.py              # async video jobs (worker thread, progress, expiry, read-only sample)
│   │   ├── settings.py          # RIDESAFE_* environment variables
│   │   ├── video.py             # video pipeline: detect -> track -> decide -> log
│   │   ├── detector.py          # OnnxDetector: letterbox, ONNX Runtime, NMS
│   │   ├── bytetrack.py         # ByteTrack in numpy (parity-tested against Ultralytics)
│   │   ├── tracking.py          # per-rider voting and the violation rule
│   │   ├── plate_reader.py      # violator's vehicle -> OCR -> plate text (RapidOCR)
│   │   ├── plates.py            # Indian plate format rules, look-alike fixes, voting
│   │   ├── association.py       # which vehicle is this head on?
│   │   ├── privacy.py           # blurs faces and plates for the public sample
│   │   ├── violation_log.py     # SQLite violation log
│   │   └── thresholds.py        # per-class thresholds from configs/thresholds.yaml
│   ├── tests/                   # unit tests (no model, no PyTorch)
│   ├── Dockerfile               # API image: ONNX export stage -> torch-free runtime
│   └── requirements-api.txt     # pinned runtime dependencies of the image
├── frontend/                    # React + TypeScript web app (Vite, Tailwind, TanStack Query, Recharts)
│   ├── src/pages/               # Overview, Analyze, JobPage, Review, ModelCard
│   └── public/staticwebapp.config.json   # routing, caching and security headers on Azure
├── scripts/                     # ML tooling (PyTorch via Ultralytics): train, evaluate, data, video runs
├── configs/                     # training recipe, experiments/*.yaml, thresholds.yaml, bytetrack.yaml
├── models/best.pt               # trained weights, YOLO26s (20 MB); .onnx files are generated
├── eval/video_gt/               # hand-checked video ground truth + clip manifest (videos downloaded separately)
├── reports/                     # generated evaluation reports (index: reports/README.md)
├── samples/                     # demo images + demo_job/ (the anonymised public sample)
├── notebook/experiments_kaggle.ipynb   # trains experiments on Kaggle's free GPU
├── tests/                       # tests of the scripts + end-to-end smoke tests
├── deploy/azure_setup.sh        # one-time Azure setup (Cloud Shell)
├── docs/                        # these guides
├── .github/workflows/           # ci.yml (tests + deploys), preview-close.yml
├── docker-compose.yml           # local stack: API + web app
└── LICENSE                      # AGPL-3.0
```

## Dataset

[Bike Helmet Detection v2](https://universe.roboflow.com/bike-helmets/bike-helmet-detection-2vdjo/dataset/2) from
Roboflow Universe (Public Domain, curated by Syed Salman Reza).

| Split | Images | With Helmet boxes | Without Helmet boxes |
|---|---|---|---|
| train | 3,546 | 10,091 | 4,724 |
| valid | 126 | 184 | 115 |
| test | 63 | 123 | 61 |

Images are 416×416; the classes are imbalanced about 2:1 in favour of *With Helmet*. The training images are
Roboflow augmentations of far fewer originals, and valid/test are small, so metrics are noisy.

**Download** (needs a free Roboflow API key, read from the environment only):
```bash
$env:ROBOFLOW_API_KEY="your_key"          # PowerShell; Mac/Linux: export ROBOFLOW_API_KEY=your_key
python scripts/download_roboflow.py --workspace bike-helmets --project bike-helmet-detection-2vdjo --version 2
```
Move the downloaded `train/`, `valid/` and `test/` folders into `data/`; `data/data.yaml` already points to them.
The extra headwear data comes from the script's default dataset (`python scripts/download_roboflow.py`), cleaned
and leak-checked by `python scripts/mine_headwear.py` (see [EVALUATION.md](EVALUATION.md#new-data-for-the-headwear-failure)).
The video clips are listed in [`eval/video_gt/videos.yaml`](../eval/video_gt/videos.yaml) and downloaded with
`python scripts/download_eval_videos.py`.

## Web app and API locally

```bash
uvicorn ridesafe.api:app --port 8000                  # API, docs at http://localhost:8000/docs
cd frontend && npm run dev                            # web app at http://localhost:5173, /api proxied to :8000
```
`RIDESAFE_API_PROXY=http://localhost:8001 npm run dev` points the web app at an API on another port. Or run both in
Docker with `docker compose up --build` (see [DEPLOYMENT.md](DEPLOYMENT.md#docker)).

## Images from the command line

```bash
python scripts/detect.py --source data/test/images --n_samples 5   # tuned thresholds from configs/thresholds.yaml
python scripts/detect.py --source samples/image1.jpg --conf 0.4    # one threshold for all classes
```
Annotated images go to `results/detect/run_<timestamp>/`.

## Video from the command line

`detect.py` labels every frame on its own, so one rider becomes hundreds of detections. `track_video.py` follows
each head with ByteTrack and makes **one decision per rider**, using the same torch-free pipeline as the API
([`backend/ridesafe/video.py`](../backend/ridesafe/video.py)):
```bash
python scripts/track_video.py --source path/to/video.mp4
python scripts/track_video.py --source path/to/video.mp4 --show --vid-stride 2   # live window, every 2nd frame
python scripts/track_video.py --source 0                                         # webcam
python scripts/track_video.py --source rtsp://<camera-address>/stream            # IP camera
python scripts/track_video.py --source path/to/video.mp4 --imgsz 1280             # larger input: finds smaller, far-away heads (slower)
```

**How a violation is decided** ([`tracking.py`](../backend/ridesafe/tracking.py)):
- One box per head: class-agnostic NMS removes a second box of the other class on the same head (otherwise the
  tracker sees two riders and logs the violator twice).
- Every frame where a head passes its class's tuned threshold is a **vote**. Heads cut off by the frame border
  don't vote, because a half-visible head often looks helmetless.
- A track is a **violation** if it has at least 5 no-helmet votes **and** they are at least 60% of its votes
  (`--min-frames`, `--min-ratio`). Split votes (≥ 40% no-helmet) are **Uncertain**, for a reviewer.
- The decision is made **when the rider leaves** (lost for 30 frames, or the video ends), so early frames or an ID
  switch can't lock in a wrong verdict. Each violation is logged once.
- On screen: green = helmet, red = violation, orange = uncertain.

**Number plates (best effort):** for each violator the 5 frames where they are closest to the camera are kept, a
COCO detector finds their vehicle, RapidOCR reads it, and [`plates.py`](../backend/ridesafe/plates.py) validates the
Indian format, fixes look-alike characters and votes; a plate is reported only if at least 2 frames agree.
`--no-plates` turns it off. Accuracy: [EVALUATION.md](EVALUATION.md#number-plates-best-effort).

**Outputs** in `results/video/<name>_<timestamp>/`: `annotated.mp4`, `tracks.csv` (one row per track),
`violations/` (a snapshot per violator, plus the plate crop) and `summary.json`.

**Violation log:** every violation is written to `results/violations.db` (SQLite, WAL mode) the moment the rider is
decided, so a crash loses nothing and other programs can read it while a stream runs:
```bash
python -m ridesafe.violation_log            # list runs
python -m ridesafe.violation_log --run 1    # one run's violations
```
For long streams memory stays flat: decided riders are freed, and plates are read in a background thread so OCR
never pauses the video (`--no-boxes` also skips the per-frame box file).

**Speed** on CPU (i7-7700K, `--vid-stride 2`): about 16 FPS for tracking at 640 px, about 9 FPS with plate reading
([runtime parity report](../reports/runtime_parity.md)).

## Evaluate and analyse

```bash
python scripts/evaluate.py --split test        # mAP, per-class metrics, confusion matrix
python scripts/tune_thresholds.py              # -> configs/thresholds.yaml + reports/thresholds.md
python scripts/error_analysis.py               # -> reports/error_analysis.md + gallery (val split)
python scripts/dataset_audit.py                # -> reports/dataset_audit.md
python scripts/eval_association.py             # -> reports/association.md
python scripts/eval_video.py --gt eval/video_gt/indian_highway_riders.yaml --run results/video/<run>
```
To label a new clip: run `track_video.py`, draft labels with `scripts/video_gt_draft.py`, render review sheets with
`scripts/review_gt.py`, correct by hand, then score with `eval_video.py`. Put videos in `data/videos/` (git ignores
it).

## Train and compare models

Every experiment is a small YAML file in `configs/experiments/` that inherits `configs/base.yaml`:
```bash
python scripts/make_oversampled_split.py       # only for the *_oversample experiments
python scripts/train.py --config configs/experiments/yolo26s_640_headwear.yaml --device 0
python scripts/benchmark.py                    # -> reports/model_comparison.md
```
- Runs are saved to `results/train/<experiment>/`; `--promote` copies the best weights to `models/best.pt`.
- No GPU? [`notebook/experiments_kaggle.ipynb`](../notebook/experiments_kaggle.ipynb) trains the experiments on
  Kaggle's free GPU in one background run and saves the results as the notebook's output.
- Experiment tracking: `pip install mlflow`, set `MLFLOW_TRACKING_URI=sqlite:///mlflow.db`; Ultralytics logs every
  run, and `mlflow ui --backend-store-uri sqlite:///mlflow.db` browses them.

## ONNX export

```bash
python scripts/export_onnx.py --vehicles       # -> models/best.onnx + models/vehicles.onnx (add --int8 to try INT8)
python scripts/benchmark_onnx.py               # -> reports/onnx_benchmark.md (PyTorch vs ONNX FP32 / INT8)
```
[`detector.py`](../backend/ridesafe/detector.py) runs the exported model with only numpy, OpenCV and ONNX Runtime:
the same boxes as PyTorch, 1.25× faster on 640×640 images and about 1.5× on video frames. INT8 was rejected
(no-helmet recall 0.85 → 0.54).

## Tests and lint

```bash
ruff check .          # lint (ruff.toml)
pyright               # type check (pyrightconfig.json)
pytest                # 88 unit tests: tracking, ByteTrack, plates, association, detector, API rules, privacy (~15 s, no model)
pytest -m smoke       # 8 end-to-end tests with the real models (~40 s on CPU; export the ONNX models first)
cd frontend && npm run lint && npm run typecheck && npm test && npm run build
```
Browser tests ([`frontend/e2e/`](../frontend/e2e/), Playwright): the production build, served with the deployed
Content-Security-Policy, against a running API with the sample job (photo check, sample video, violation snapshots);
they fail on anything the policy blocks.
```bash
RIDESAFE_SAMPLE_DIR=samples/demo_job uvicorn ridesafe.api:app --port 8000     # or the Docker image
cd frontend && npx playwright install chromium && npm run e2e                 # E2E_API_URL=... for another port
```
CI ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)) runs all of these on every push and pull request,
plus a Docker image build that checks PyTorch is absent, calls `/detect` on the running container and runs the
browser tests against it.

## Rebuild the public sample

The live site's sample is the highway test clip, analysed on the original video and published with faces and
plates blurred ([`privacy.py`](../backend/ridesafe/privacy.py), details in
[DEPLOYMENT.md](DEPLOYMENT.md#the-public-sample)). After changing the model or the video settings:
```bash
python -m ridesafe.jobs --make-sample data/videos/indian_highway_riders.mp4 --out samples/demo_job --anonymize \
  --credit "Video: Aamir Somewhere on Pexels (pexels.com/video/34395449). Faces and number plates blurred."
```
It takes about 9 minutes on a 4-core CPU. Look through the result before committing it: the blurring is automatic
and should be checked by eye.
