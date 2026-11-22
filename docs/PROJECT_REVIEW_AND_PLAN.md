# Helmet Detection: Roadmap

_Last updated: 2026-11-21_

**Goal:** turn this into a portfolio-grade **computer-vision / ML engineering** project for AI/ML roles.
GenAI is covered by separate projects (RAG, VeiSight), so this one focuses on detection, evaluation, video and deployment.

**Name:** *RideSafe AI: Helmet Violation Detection* (repo: `helmet-violation-detection`, renamed on 2026-11-22 after the video system was finished).

---

## Where we are

**Current model:** YOLO26s, trained on the original data (violator images oversampled 2×) plus 2,537 curated "headwear" images from a second dataset; picked on val from 9 experiments (`reports/model_comparison.md`). Without Helmet threshold 0.31 (tuned on val for F2).

| Held-out evaluation | Result | Previous model |
|---|---|---|
| Original test set (63 images), mAP50 / Without Helmet AP50 | **0.937** / 0.889 | 0.917 / 0.881 |
| Headwear test set (635 images), mAP50 / no-helmet recall | **0.810** / 0.763 | 0.320 / 0.285 |
| Video, per rider, 4 test clips (54 riders, 40 violators): violation recall / precision | **18/40 (45%) / 18/24 (75%)** | single clip only: 5/12 at 100% |
| Highway test clip alone: violation recall at 100% precision | **8/12** | 5/12 |

First baseline (YOLOv8s, 30 epochs) for comparison: original test mAP50 0.873, Without Helmet AP50 0.799.

**Main remaining weaknesses:** crowded scenes (Pune junction recall 9/26); pedestrians standing among riders are logged as violators (all 6 false violations on the test clips are non-riders); headscarves, face scarves and woolly caps read as helmets; far-away riders (heads under ~36 px) are not tracked; plate OCR misreads some characters and can take the plate of a closer bike; a rider hidden for a moment can be logged twice.

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

## Phase 2: Video system ✅ done

- [x] Video/webcam/RTSP processing with ByteTrack, so each rider is counted once (`scripts/track_video.py`, ~9 FPS on CPU)
- [x] Head-to-vehicle association (`backend/ridesafe/association.py`, unit-tested) evaluated as a false-alarm filter (`scripts/eval_association.py` → `reports/association.md`): 16 of 19 remaining no-helmet false alarms on val already sit on/next to a bike, so the best filter removes 3 false alarms but costs real violators → **not used as a filter**. Kept for linking a violator to their vehicle (plate OCR)
- [ ] Optional: a temporal version for video (a pedestrian walks beside bikes, a rider moves with one), once there is a clip with pedestrians to measure it
- [x] Violation logic: per-track voting with tuned thresholds (≥5 no-helmet votes and ≥60% of votes), decided when the track ends so ID switches and early frames can't lock in a wrong verdict; border-cut heads don't vote (`backend/ridesafe/tracking.py`, unit-tested)
- [x] Tested on real footage (Pexels Indian-highway clip, 14 s): fixed duplicate logs (same head boxed as both classes) with class-agnostic NMS before tracking
- [x] **Video evaluation set**: every rider of the clip hand-labelled (`eval/video_gt/`, drafted with `scripts/video_gt_draft.py`, corrected by eye), scored per rider by `scripts/eval_video.py` → `reports/video_eval.md`. Reviewed by the project owner (added 2 missed riders, corrected 1 label): 16 riders, 12 violators. Result (720p): violation precision 5/5, **recall 5/12**; 1080p: 4/4 and 4/12. Of 7 misses: 4 wear headwear (white skullcaps, cloth cap, head cloth) read as a helmet, 2 are far riders (~24-27 px heads), 1 has his head cut by the frame edge
- [x] **Top priority for the model: non-helmet headwear**
  - [x] Extra data: "Helmet, Rider, ANPR Detection" v12 (CC BY 4.0, 14.5k images, an aggregate of other datasets). `scripts/mine_headwear.py` keeps one copy per photo, drops junk (product photos, close-ups, riders without head labels) and **231 images that duplicate our own dataset** (by name and perceptual hash), then splits by source family: 2,537 extra-train / 635 held-out headwear test
  - [x] Mining with the current model: the confused no-helmet heads are caps, turbans, hijabs, hoods and **medical face masks**; most "missed" heads are rear views. Baseline on the headwear test: mAP50 **0.32**, no-helmet recall **0.31** (vs 0.92 / 0.87 on our test set)
  - [x] Kaggle: `yolo26s_640_headwear` and `_100ep` (both early-stopped, best epochs 28 / 36). Tied on our val set, the shorter one was **promoted** (threshold re-tuned on val: 0.31). Our test mAP50 0.917 → **0.937**; headwear test 0.32 → **0.81** (no-helmet recall 0.28 → 0.76); video **5/12 → 8/12** violators at 100% precision (720p and 1080p)
  - Trade-offs found: more no-helmet "false alarms" on our test set (9 → 20), mostly unlabelled people in caps/bandanas plus a few pedestrians; far-away riders (27-36 px heads) no longer tracked in the video; R4/R5 logged twice after an occlusion
  - [ ] Next model step: far riders (higher input size or tiled inference; retune `configs/bytetrack.yaml` thresholds for the new model) and a "who is riding" signal for pedestrians
  - Turbans are labelled "without helmet" (visually true); whether a Sikh turban is a violation is a deployment policy, not a model decision
- [x] Video benchmark expanded to 8 Pexels clips (`eval/video_gt/videos.yaml`, `scripts/download_eval_videos.py`): 4 test clips (highway, crowded Pune junction, high-angle junction, night/fog) labelled and reviewed by the project owner (54 riders, 40 violators): violation recall 18/40, precision 18/24, 0 false violations on helmeted riders, 6 on non-riders
- [ ] Label the 4 validation clips (Bangalore junction, two rear-view city roads, railway crossing at dusk) before tuning any video setting
- [ ] "Is riding" signal: all false violations on the test clips are pedestrians or non-riders; measure a temporal head-on-vehicle check on the validation clips
- [x] Number-plate reading for violators (`scripts/plate_reader.py` + `backend/ridesafe/plates.py`, unit-tested): 5 closest frames → rider's vehicle (COCO + association) → RapidOCR (PP-OCRv3, onnxruntime) → Indian-format validation with positional look-alike fixes → vote, reported only if ≥2 frames agree
  - Result on the highway clip: 720p 0 correct / 1 wrong / 4 unread; 1080p 1 correct / 1 wrong / 3 unread. The wrong one (`MP04ZD7116` → `MP02ZD7116`) happens even at 286 px and sharp, so the general OCR model is a limit, not only image quality
  - Split helmet/no-helmet votes (≥40% no-helmet) are now **Uncertain** instead of "With Helmet" (found on the 1080p run: a white-skullcap rider at 17 vs 16 votes)
- [ ] Plate-specific recognizer (e.g. fast-plate-ocr, ONNX): measure on the same saved plate crops before switching. Needs care: it depends on opencv-python-headless, which breaks `--show` when installed next to opencv-python
- [x] Violation log (SQLite, `backend/ridesafe/violation_log.py`): every violation written the moment it is decided (time, track, votes, snapshot, plate filled in by a background thread, review status); WAL mode so an API can read during a stream. Long streams: memory freed per decided rider, tracker scans only open tracks, plate OCR off the main loop
- [ ] Known plate issue: a rider whose bike is hidden behind a closer one can get the closer bike's plate

## Phase 3: Product, engineering & deployment ⭐

**Goal:** a real product, not a tutorial demo: a FastAPI backend and a React web app, deployed on Azure and updated automatically on every push. Streamlit is retired completely.

### Decisions (agreed 2026-11-22)
- **Frontend:** React + TypeScript (Vite), Tailwind CSS, shadcn/ui, lucide icons, Recharts. No Streamlit (`app.py` is removed once the web app works).
- **Look:** dark "traffic control room" theme by default, with a light-mode toggle. Colours carry the same meaning as the video overlays: green = helmet, red = violation, amber = uncertain / needs review (amber is also the brand accent). Fonts: Inter for the UI, JetBrains Mono for data (plates, times, confidences).
- **Backend:** FastAPI with ONNX Runtime (torch-free Docker image), async video jobs, the existing SQLite violation log.
- **Hosting (Azure for Students):** Azure Static Web Apps (frontend, free tier) + Azure Container Apps (API, consumption plan, scale to zero) + GitHub Container Registry for images (free). Azure budget alert set on day one.
- **Deployment:** fully automatic on every push to `main` (details below). No manual deploys.
- **Public demo privacy:** plates partially masked (`MP04 •• 7116`), uploads auto-deleted after 1 hour, full data only when run locally.
- **Reviewing on the live site:** visitors can confirm/reject only their own uploads; the built-in sample run is read-only.
- **AGPL-3.0:** the site footer links to the source code (required for a network service built on Ultralytics).

### Web app pages
1. **Overview:** hero with a looping demo clip, key numbers from the real evaluation (video violation recall/precision, headwear-test mAP50), "how it works" pipeline diagram.
2. **Analyze**
   - Image: drag and drop; boxes drawn in the browser from the API's JSON (hover a box for class and confidence).
   - Video: upload → async job with live progress stages → playback of the original video with boxes drawn on a canvas in sync, plus a clickable timeline of violations. (Drawing in the browser avoids re-encoding video on the server, which browsers could not play anyway in OpenCV's `mp4v` format.)
3. **Review console:** violation cards (snapshot, time, plate, confidence, votes), Confirm / Reject, filters, keyboard shortcuts (J/K next/previous, C confirm, R reject). Uses `review_status` in the SQLite log.
4. **Model card:** the 9-model comparison, headwear-test before/after, video benchmark, known limitations, dataset provenance, privacy and ethics.
5. Link to the API docs (`/docs`, generated by FastAPI).

### Automatic deployment (GitHub Actions)
```
git push to main
  ├─ lint + type check + unit/smoke tests (existing CI): must pass, otherwise nothing deploys
  ├─ backend/ changed  → build Docker image → push to ghcr.io → update the Azure Container Apps revision
  └─ frontend/ changed → build the React app → deploy to Azure Static Web Apps
```
- Only what changed is rebuilt; a docs-only commit deploys nothing.
- GitHub signs in to Azure with OIDC federated credentials (no stored passwords).
- Pull requests get a temporary frontend preview URL (Static Web Apps).
- One-time setup by the project owner: resource group, Container Apps environment, Static Web App, federated credential, GitHub repository variables/secrets, budget alert.

### Steps (in order)
- [x] **ONNX export + benchmark:** PyTorch vs ONNX Runtime on the i7-7700K (latency, CPU FPS), accuracy check that ONNX gives the same metrics; optional INT8. Done 2026-11-22 ([reports/onnx_benchmark.md](../reports/onnx_benchmark.md)): FP32 gives the same boxes as PyTorch and is 1.27x faster (83 vs 106 ms, 12 FPS) with a torch-free detector (`backend/ridesafe/detector.py`); INT8 is 1.9x faster but loses no-helmet recall (0.85 → 0.54), so it is not used
- [x] **Package layout:** `backend/` (FastAPI app + `ridesafe` Python package with the detection, tracking, plates and log code), `frontend/` (React), `scripts/` stays for ML tooling (training, evaluation, mining). Done 2026-11-22: `backend/ridesafe` (detector, tracking, association, plates, violation_log, thresholds) installs in editable mode and imports without PyTorch (its unit tests run in CI without it); `scripts/` imports it. Still on Ultralytics, to be ported in the FastAPI step: the video loop's ByteTrack (`scripts/track_video.py`) and the COCO vehicle detector of the plate reader (`scripts/plate_reader.py`); `frontend/` is created with the React step
- [ ] **FastAPI service:** `POST /detect` (image → boxes JSON), `POST /jobs/video` + `GET /jobs/{id}` (async, progress, per-frame tracks JSON), `GET/PATCH /violations` (review status), `GET /health`, `GET /metrics`; request size limits, upload auto-delete, plate masking in demo mode; API tests
- [ ] **Docker:** multi-stage image (ONNX exported in the build stage, runtime without PyTorch), `docker-compose` for local API + web app; image size before/after recorded
- [ ] **React web app:** the 5 pages above, responsive, dark/light, loading and error states, sample data so the site is never empty
- [x] pytest (unit + `-m smoke` end-to-end) and GitHub Actions CI (ruff, Pyright, unit tests, smoke tests on CPU); AGPL-3.0 LICENSE (required by Ultralytics)
- [ ] **CI/CD to Azure:** deploy workflows as above, OIDC login, path filters, PR previews
- [ ] **Retire Streamlit:** remove `app.py`, `packages.txt` and the Streamlit dependency; update README and smoke tests

## Phase 4: Small GenAI touch (optional)

- [ ] VLM second opinion for low-confidence crops (0.3–0.6): measure the false-alarm reduction and cost per call

## Phase 5: Presentation

- [ ] README hero: demo GIF, live-site link, architecture diagram, results and benchmark tables
- [ ] Limitations & ethics section (privacy of plate data, bias across lighting/regions)
- [ ] Short LinkedIn/Medium write-up: "How I improved recall on the violation class"

---

## Resume line (fill in real numbers as you go)

> Built a helmet-violation detection system (YOLO + ByteTrack + plate OCR). On a held-out image test set I raised no-helmet recall from 74% to 87% while cutting false alarms 3× (model selection across 7 YOLOv8/YOLO11/YOLO26 experiments, class rebalancing, validation-tuned thresholds). Built a per-rider video benchmark with hand-labelled ground truth (54 riders across 4 scenes) and traced the misses to non-helmet headwear, crowds and pedestrians; curated a second public dataset (leak-checked against my test data, hard examples mined with the model), which raised mAP50 on a held-out headwear test set from 0.32 to 0.81; on video the system reaches 75% violation precision with zero false flags on helmeted riders. Shipped it as a product: FastAPI + ONNX Runtime backend (X× faster CPU inference, torch-free Docker image), React review console for human-in-the-loop verification, deployed on Azure with automatic CI/CD on every push.

_Only claim what is measured: the video numbers come from 4 short clips (40 violators); recall ranges from 35% (crowded junction) to 67% (clean highway)._
    
