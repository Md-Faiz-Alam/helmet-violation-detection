# Helmet Detection: Roadmap

_Last updated: 2026-11-21_

**Goal:** turn this into a portfolio-grade **computer-vision / ML engineering** project for AI/ML roles.
GenAI is covered by separate projects (RAG, VeiSight), so this one focuses on detection, evaluation, video and deployment.

**Working title after Phase 2:** *RideSafe AI: Helmet Violation Detection System* (repo: `helmet-violation-detection`).
Rename only once the project actually detects violations in video.

---

## Where we are

**Current model:** YOLO26s + 2× oversampling of violator images, 50 epochs, AdamW, imgsz 640 (Kaggle T4). Picked on val from 7 experiments (`reports/model_comparison.md`).

**Held-out test set** (63 images, 184 boxes), measured with `scripts/evaluate.py`:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| With Helmet | 0.949 | 0.927 | 0.952 | 0.517 |
| **Without Helmet** | 0.857 | **0.885** | 0.881 | 0.435 |
| All | 0.903 | 0.906 | 0.917 | 0.476 |

First baseline (YOLOv8s, 30 epochs) for comparison: Without Helmet mAP50 0.799, recall 0.721, all mAP50 0.873.

**Main remaining weakness:** false alarms on unlabeled pedestrians/bystanders (the model doesn't know who is riding), and small *With Helmet* heads (val recall 0.69 below 32 px).

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

## Phase 1: ML rigor ✅ done

Done (on CPU, with the existing model):
- [x] Per-class threshold tuning on val (F2 for the violation class): **test Without Helmet recall 0.738 → 0.852** (missed 16 → 9), precision 0.833 → 0.634. Tuning *With Helmet* did not transfer to test, so it stays at 0.25. → `reports/thresholds.md`
- [x] Error analysis with gallery: many false alarms are unlabeled pedestrians/bystanders (the model doesn't know who is *riding*); misses are mostly crowds, rear views, small heads; goggles get confused with helmets → `reports/error_analysis.md`
- [x] Dataset audit: train is 79% motorcycle but val/test are ~45% bicycle; recall is similar on both vehicle types → `reports/dataset_audit.md`
- [x] Config-driven training (`configs/base.yaml` + `configs/experiments/`), MLflow auto-logging, `--promote` instead of silently overwriting `models/best.pt`
- [x] Tools ready: `benchmark.py` (comparison table), `make_oversampled_split.py` (violation images ×2), Kaggle notebook `notebook/experiments_kaggle.ipynb`
- [x] App and `detect.py` use the tuned thresholds

Model selection (Kaggle T4, 7 experiments):
- [x] Trained YOLOv8n/s, YOLOv8s@416, YOLO11s, YOLO26n/s, YOLO26s + oversampling with `notebook/experiments_kaggle.ipynb`
- [x] `benchmark.py` locally (CPU latency on i7-7700K) → `reports/model_comparison.md`
- [x] Winner on val: **YOLO26s + oversampling** (val mAP50 0.925, val no-helmet recall 0.843), promoted to `models/best.pt`
- [x] Re-tuned thresholds (Without Helmet 0.16): test no-helmet recall 0.869, precision 0.855 (vs 0.852 / 0.634 with the old model at 0.06)
- [x] Re-ran error analysis and dataset audit: no-helmet background false alarms on val 66 → 18; small-head no-helmet recall 0.75 → 0.90

Optional follow-ups:
- [ ] Train the winner for 100 epochs (every run's best epoch was 41–48 of 50)
- [ ] Merge a second helmet dataset with more violators and night/CCTV images
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

> Built a helmet-violation detection system (YOLO + ByteTrack + plate OCR). I raised no-helmet recall on a held-out test set from 74% to 87% while cutting false alarms 3× (model selection across 7 YOLOv8/YOLO11/YOLO26 experiments, class rebalancing and validation-tuned per-class thresholds), and exported to ONNX for X× faster CPU inference (XX FPS). Deployed with FastAPI, Docker and CI, with a live demo on Hugging Face Spaces.
