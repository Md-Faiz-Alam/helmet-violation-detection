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

**Main remaining weakness:** false alarms in crowds around bikes (people standing with bicycles, unlabeled pillion riders), and small *With Helmet* heads (val recall 0.69 below 32 px). Pedestrian false alarms mostly disappeared with the new model (66 → 18 on val).

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

## Phase 2: Video system ⭐ in progress

- [x] Video/webcam/RTSP processing with ByteTrack, so each rider is counted once (`scripts/track_video.py`, ~9 FPS on CPU)
- [x] Head-to-vehicle association (`scripts/association.py`, unit-tested) evaluated as a false-alarm filter (`scripts/eval_association.py` → `reports/association.md`): 16 of 19 remaining no-helmet false alarms on val already sit on/next to a bike, so the best filter removes 3 false alarms but costs real violators → **not used as a filter**. Kept for linking a violator to their vehicle (plate OCR)
- [ ] Optional: a temporal version for video (a pedestrian walks beside bikes, a rider moves with one), once there is a clip with pedestrians to measure it
- [x] Violation logic: per-track voting with tuned thresholds (≥5 no-helmet votes and ≥60% of votes), decided when the track ends so ID switches and early frames can't lock in a wrong verdict; border-cut heads don't vote (`scripts/tracking.py`, unit-tested)
- [x] Tested on real footage (Pexels Indian-highway clip, 14 s): fixed duplicate logs (same head boxed as both classes) with class-agnostic NMS before tracking
- [x] **Video evaluation set**: every rider of the clip hand-labelled (`eval/video_gt/`, drafted with `scripts/video_gt_draft.py`, corrected by eye), scored per rider by `scripts/eval_video.py` → `reports/video_eval.md`. Reviewed by the project owner (added 2 missed riders, corrected 1 label): 16 riders, 12 violators. Result (720p): violation precision 5/5, **recall 5/12**; 1080p: 4/4 and 4/12. Of 7 misses: 4 wear headwear (white skullcaps, cloth cap, head cloth) read as a helmet, 2 are far riders (~24-27 px heads), 1 has his head cut by the frame edge
- [ ] **Top priority for the model: non-helmet headwear.** Add labelled images of caps, skullcaps and scarves as "Without Helmet" (decide a policy for turbans, which Indian law exempts), retrain, and re-run `eval_video.py` to show the recall change
- [ ] Add 2-3 more labelled clips (night, rear view, crowded junction, pedestrians) so the video numbers are not from one 14 s clip
- [x] Number-plate reading for violators (`scripts/plate_reader.py` + `scripts/plates.py`, unit-tested): 5 closest frames → rider's vehicle (COCO + association) → RapidOCR (PP-OCRv3, onnxruntime) → Indian-format validation with positional look-alike fixes → vote, reported only if ≥2 frames agree
  - Result on the highway clip: 720p 0 correct / 1 wrong / 4 unread; 1080p 1 correct / 1 wrong / 3 unread. The wrong one (`MP04ZD7116` → `MP02ZD7116`) happens even at 286 px and sharp, so the general OCR model is a limit, not only image quality
  - Split helmet/no-helmet votes (≥40% no-helmet) are now **Uncertain** instead of "With Helmet" (found on the 1080p run: a white-skullcap rider at 17 vs 16 votes)
- [ ] Plate-specific recognizer (e.g. fast-plate-ocr, ONNX): measure on the same saved plate crops before switching. Needs care: it depends on opencv-python-headless, which breaks `--show` when installed next to opencv-python
- [ ] Violation log (SQLite): timestamp, track ID, snapshot, plate text, confidence (today: `tracks.csv` + snapshots + `summary.json`)

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

> Built a helmet-violation detection system (YOLO + ByteTrack + plate OCR). On a held-out image test set I raised no-helmet recall from 74% to 87% while cutting false alarms 3× (model selection across 7 YOLOv8/YOLO11/YOLO26 experiments, class rebalancing, validation-tuned thresholds). Built a per-rider video benchmark with hand-labelled ground truth, which showed 100% violation precision but 42% recall and traced most misses to non-helmet headwear, a training-data gap [then: fixed it to XX%]. Exported to ONNX for X× faster CPU inference; FastAPI, Docker, CI, live demo on Hugging Face Spaces.

_Only claim what is measured: video recall is from one 14 s clip (12 violators), so say so if asked._
    
