# 🪖 Helmet Detection with YOLO

Detects riders (motorcycles and bicycles) **with** and **without** helmets in images and videos using a fine-tuned **YOLO26s** model, chosen from 7 trained candidates. It includes evaluation, model comparison, threshold tuning, error analysis and experiment tooling, plus a **Streamlit** web app.

> The class that matters most for road safety is **Without Helmet** (the violation). Missing a violator costs more than a false alarm, so this project optimizes and reports that class separately.

---

## 📊 Results

Model: **YOLO26s** (9.9M params), fine-tuned for 50 epochs (AdamW, lr0=1e-3, imgsz=640, batch 16) on a Tesla T4, with training images that contain violators **repeated 2×** to offset the 2:1 class imbalance.

**Held-out test set** (63 images, 184 boxes, never used for training, tuning or model selection):

| Class | mAP50 | mAP50-95 | vs. first baseline (YOLOv8s) |
|---|---|---|---|
| With Helmet | 0.952 | 0.517 | 0.947 → 0.952 |
| **Without Helmet** | **0.881** | 0.435 | **0.799 → 0.881** |
| **All** | 0.917 | 0.476 | 0.873 → 0.917 |

### Model selection: 7 experiments

All candidates were trained with the same recipe on Kaggle (T4) and compared with [`scripts/benchmark.py`](scripts/benchmark.py). The winner was picked on the **validation** columns only; test is reported once, as the unbiased estimate. CPU latency is batch 1 on an Intel i7-7700K.

| Model | Params (M) | CPU ms/img | val mAP50 | val no-helmet recall | test mAP50 | test no-helmet recall |
|---|---|---|---|---|---|---|
| **YOLO26s + oversampling** ✅ | 9.9 | 108 | **0.925** | **0.843** | 0.917 | 0.869 |
| YOLO26s | 9.9 | 106 | 0.893 | 0.809 | 0.915 | 0.885 |
| YOLO11s | 9.4 | 103 | 0.881 | 0.774 | 0.914 | 0.869 |
| YOLOv8s @ 416 px | 11.1 | 55 | 0.859 | 0.730 | 0.893 | 0.770 |
| YOLOv8n | 3.1 | 49 | 0.854 | 0.774 | 0.871 | 0.803 |
| YOLO26n | 2.5 | 51 | 0.824 | 0.713 | 0.909 | 0.787 |
| YOLOv8s (baseline recipe) | 11.1 | 110 | 0.818 | 0.670 | 0.869 | 0.770 |

No-helmet recall is at confidence 0.25. Full table: [reports/model_comparison.md](reports/model_comparison.md).

- **Architecture mattered most:** YOLO26s beat YOLOv8s by +0.08 val mAP50 at the same speed.
- **Oversampling violators helped further** (+0.03 val mAP50, +0.03 val no-helmet recall), so it was kept.
- Every run's best epoch was 41–48 of 50, so longer training may still help.

### Operating threshold

The violation-class threshold is tuned **on the validation set** to maximize F2 (recall weighted 2×), then checked once on test:

| Without Helmet (test) | Threshold | Precision | Recall | Missed violators | False alarms |
|---|---|---|---|---|---|
| First baseline (YOLOv8s), tuned | 0.06 | 0.634 | 0.852 | 9 of 61 | 30 |
| **Current (YOLO26s + oversampling), tuned** | **0.16** | **0.855** | **0.869** | **8 of 61** | **9** |

The better model catches slightly more violators with about **3× fewer false alarms** to review. On test, the tuned threshold gives the same recall as the default 0.25 (it gained recall on val), which is expected noise on 61 objects. Details: [reports/thresholds.md](reports/thresholds.md).

### What the analysis found

- **The remaining "false alarms" are mostly pedestrians and bystanders without helmets.** The dataset only labels riders, so the model has no notion of who is *riding*. The new model cut no-helmet background false alarms on val from 66 to 18. Next fix: associate heads with a detected bike or motorcycle (video system, below). [Error analysis](reports/error_analysis.md)
- **Small heads improved most:** no-helmet recall on heads under 32 px went from 0.75 to 0.90 (val).
- **Trade-off:** oversampling shifted the model toward the violation class. Missed *With Helmet* heads on val rose from 16 to 22, mostly small ones (recall 0.69 below 32 px).
- **Typical misses:** riders seen from behind in crowded scenes, and face coverings or goggles that look like a helmet.
- **Distribution shift between splits:** train is 79% motorcycle images, but val and test are about 45% bicycle. Recall is similar on both vehicle types (val no-helmet: 0.91 motorcycle, 0.85 bicycle), so the model transfers. [Dataset audit](reports/dataset_audit.md)

**Known limitations**
- The test set is small (63 images), so differences of a few points are within noise.
- The 3,546 training images are Roboflow augmentations (rotation/shear/blur) of a much smaller set of originals, so real-world diversity (night, rain, CCTV angles) is limited.
- Some visible heads in crowded images are unlabeled, which inflates false-positive counts.
- CPU inference takes about 110 ms per image at 640 px (Intel i7-7700K, PyTorch). An ONNX export is planned.

---

## 📂 Project Structure
```
Helmet_Detection/
├── app.py                          # Streamlit web app (upload / webcam / samples)
├── configs/
│   ├── base.yaml                   # shared training recipe
│   ├── experiments/*.yaml          # one file per experiment (YOLOv8 / YOLO11 / YOLO26, imgsz, oversampling)
│   └── thresholds.yaml             # tuned per-class confidence thresholds
├── scripts/
│   ├── train.py                    # config-driven training
│   ├── evaluate.py                 # per-class metrics + confusion matrix / PR curves
│   ├── tune_thresholds.py          # per-class threshold tuning (val) + test report
│   ├── error_analysis.py           # error breakdown, recall by object size, image gallery
│   ├── dataset_audit.py            # split composition (motorcycle/bicycle) + recall per vehicle type
│   ├── benchmark.py                # model comparison: accuracy, size, CPU latency
│   ├── make_oversampled_split.py   # oversample violation images (class imbalance)
│   ├── detect.py                   # inference on an image, folder, or video
│   ├── metrics.py                  # shared prediction-to-ground-truth matching
│   └── utils.py                    # helpers
├── notebook/
│   ├── helmet_detection_training.ipynb   # data exploration, baseline training, evaluation (Kaggle or local)
│   └── experiments_kaggle.ipynb          # Kaggle GPU: train and compare all experiment configs
├── reports/                        # generated analysis reports + error gallery
├── models/best.pt                  # trained weights, YOLO26s (20 MB)
├── data/data.yaml                  # dataset config (images are downloaded separately)
├── samples/                        # demo images for the app
├── docs/PROJECT_REVIEW_AND_PLAN.md # roadmap
├── requirements.txt
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
git clone https://github.com/Md-Faiz-Alam/helmet_detection.git
cd helmet_detection
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

### Inference
```bash
python scripts/detect.py --source data/test/images --n_samples 5   # tuned thresholds from configs/thresholds.yaml
python scripts/detect.py --source samples/image1.jpg --conf 0.4    # one threshold for all classes
python scripts/detect.py --source path/to/video.mp4                # video
```
Annotated outputs go to `results/detect/run_<timestamp>/`, and a zip of them goes to `light_bundles/`.

### Evaluate and analyse
```bash
python scripts/evaluate.py --split test        # mAP, per-class metrics, confusion matrix
python scripts/tune_thresholds.py              # -> configs/thresholds.yaml + reports/thresholds.md
python scripts/error_analysis.py               # -> reports/error_analysis.md + gallery (val split)
python scripts/dataset_audit.py                # -> reports/dataset_audit.md
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
2. **Video system:** rider tracking (ByteTrack), head-to-vehicle association, violation logic, number-plate OCR
3. **Deployment:** FastAPI + Docker, ONNX optimization with benchmarks, CI, live demo

---

## 🙌 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) (YOLOv8 / YOLO11 / YOLO26)
- [Roboflow Universe](https://universe.roboflow.com/), the Bike Helmet Detection dataset by Syed Salman Reza (Public Domain)
