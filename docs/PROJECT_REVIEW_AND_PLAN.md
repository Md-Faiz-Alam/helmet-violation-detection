# Helmet Detection: Roadmap

_Last updated: 2026-11-20_

**Goal:** turn this into a portfolio-grade **computer-vision / ML engineering** project for AI/ML roles.
GenAI is covered by separate projects (RAG, VeiSight), so this one focuses on detection, evaluation, video and deployment.

**Working title after Phase 2:** *RideSafe AI: Helmet Violation Detection System* (repo: `helmet-violation-detection`).
Rename only once the project actually detects violations in video.

---

## Where we are

**Baseline model:** YOLOv8s, 30 epochs, AdamW, imgsz 640 (trained on a Tesla T4 GPU).

**Held-out test set** (63 images, 184 boxes), measured with `scripts/evaluate.py`:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| With Helmet | 0.878 | 0.935 | 0.947 | 0.520 |
| **Without Helmet** | 0.833 | **0.721** | 0.799 | 0.400 |
| All | 0.855 | 0.828 | 0.873 | 0.460 |

**Main weakness:** about 28% of riders without a helmet are missed. The training data has about 2× more *With Helmet* boxes (10,091 vs 4,724).

**Dataset caveats:** the 3,546 training images are Roboflow augmentations of far fewer originals, and the valid/test sets are small (126/63 images), so metrics are noisy.

---

## Phase 0: Fix & clean ✅ done

- Fixed the `train.py` crash (`--weights` never passed) and the `detect.py` `max_det` misuse
- Added `scripts/evaluate.py` (per-class test metrics, confusion matrix, PR curves)
- App: model caching, a confidence slider, violation counts, a correct "no detections" message, EXIF rotation for phone photos
- Notebook: removed the hardcoded API key and the filler/duplicate cells, fixed incorrect explanations, added test-set evaluation
- Pinned dependencies (tested on Python 3.13), fixed `packages.txt` and `.gitignore`, added `.env.example`
- README rewritten to describe only what exists
- **Still to do by you:** revoke the old Roboflow API key (it remains in the git history)

---

## Phase 1: ML rigor ⭐ in progress

Done (on CPU, with the existing model):
- [x] Per-class threshold tuning on val (F2 for the violation class): **test Without Helmet recall 0.738 → 0.852** (missed 16 → 9), precision 0.833 → 0.634. Tuning *With Helmet* did not transfer to test, so it stays at 0.25. → `reports/thresholds.md`
- [x] Error analysis with gallery: many false alarms are unlabeled pedestrians/bystanders (the model doesn't know who is *riding*); misses are mostly crowds, rear views, small heads; goggles get confused with helmets → `reports/error_analysis.md`
- [x] Dataset audit: train is 79% motorcycle but val/test are ~45% bicycle; recall is similar on both vehicle types → `reports/dataset_audit.md`
- [x] Config-driven training (`configs/base.yaml` + `configs/experiments/`), MLflow auto-logging, `--promote` instead of silently overwriting `models/best.pt`
- [x] Tools ready: `benchmark.py` (comparison table), `make_oversampled_split.py` (violation images ×2), Kaggle notebook `notebook/experiments_kaggle.ipynb`
- [x] App and `detect.py` use the tuned thresholds

Next (needs a GPU, so run it on Kaggle):
- [ ] Run `notebook/experiments_kaggle.ipynb` (Save & Run All): YOLOv8n/s, YOLOv8s@416, YOLO11s, YOLO26n/s, YOLO26s + oversampling
- [ ] Download the results and run `python scripts/benchmark.py` locally (fair CPU latency) → `reports/model_comparison.md`
- [ ] Pick the winner on **val** columns, `--promote` it, then re-run `tune_thresholds.py` and `error_analysis.py`
- [ ] Optional: merge a second helmet dataset with more violators and night/CCTV images
- [ ] Re-run the training notebook on Kaggle so its Step 6 test-set outputs are saved

## Phase 2: Video system ⭐

- [ ] Video/RTSP processing with ByteTrack, so each rider is counted once
- [ ] Head-to-vehicle association: count a head only if it sits above a detected bicycle/motorcycle (fixes the pedestrian false alarms found in Phase 1)
- [ ] Violation logic: flag "no helmet" only if it persists for N frames
- [ ] Number-plate detection + OCR (PaddleOCR/EasyOCR) for violators
- [ ] Violation log (SQLite): timestamp, track ID, snapshot, plate text, confidence

## Phase 3: Engineering & deployment ⭐

- [ ] Package layout: `src/helmet_detection/`, `api/`, `app/`, `configs/`, `tests/`
- [ ] FastAPI service: `/detect`, `/detect/video`, `/violations`, `/health`
- [ ] ONNX (+ INT8) export with a speed benchmark table (PyTorch vs ONNX, CPU FPS)
- [ ] Docker + docker-compose; pytest; GitHub Actions CI (ruff + tests)
- [ ] Weights on the Hugging Face Hub instead of git; live demo on Hugging Face Spaces

## Phase 4: Small GenAI touch (optional)

- [ ] VLM second opinion for low-confidence crops (0.3–0.6): measure the false-alarm reduction and cost per call

## Phase 5: Presentation

- [ ] README hero: demo GIF, live-demo badge, architecture diagram, results and benchmark tables
- [ ] Limitations & ethics section (privacy of plate data, bias across lighting/regions)
- [ ] Short LinkedIn/Medium write-up: "How I improved recall on the violation class"

---

## Resume line (fill in real numbers as you go)

> Built a helmet-violation detection system (YOLO + ByteTrack + plate OCR). I raised no-helmet recall from 74% to 85% on a held-out test set with validation-tuned per-class thresholds (and to XX% with class rebalancing / model selection), and exported to ONNX for X× faster CPU inference (XX FPS). Deployed with FastAPI, Docker and CI, with a live demo on Hugging Face Spaces.
