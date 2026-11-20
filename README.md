# 🪖 Helmet Detection with YOLO

Detects riders (motorcycles and bicycles) **with** and **without** helmets in images and videos using a fine-tuned **YOLOv8s** model. It includes evaluation, threshold tuning, error analysis and experiment tooling, plus a **Streamlit** web app.

> The class that matters most for road safety is **Without Helmet** (the violation). Missing a violator costs more than a false alarm, so this project optimizes and reports that class separately.

---

## 📊 Results

Model: YOLOv8s (11.1M params), fine-tuned for 30 epochs (AdamW, lr0=1e-3, imgsz=640, batch 16) on a Tesla T4.

**Held-out test set** (63 images, 184 boxes, never used for training, tuning or model selection):

| Class | mAP50 | mAP50-95 |
|---|---|---|
| With Helmet | 0.947 | 0.520 |
| **Without Helmet** | 0.799 | 0.400 |
| **All** | 0.873 | 0.460 |

### Operating threshold: catching more violators

One global confidence threshold (0.25) misses about 1 in 4 riders without a helmet. The violation-class threshold was re-tuned **on the validation set** to maximize F2 (recall weighted 2×), then checked once on the test set:

| Without Helmet (test) | Threshold | Precision | Recall | Missed violators |
|---|---|---|---|---|
| Default | 0.25 | 0.833 | 0.738 | 16 of 61 |
| **Tuned** | **0.06** | 0.634 | **0.852** | **9 of 61** |

The tuned threshold catches more violators at the cost of more false alarms that a human reviewer would need to dismiss. Details: [reports/thresholds.md](reports/thresholds.md).

### What the analysis found

- **Many "false alarms" are pedestrians and bystanders without helmets.** The dataset only labels riders, so the model has no notion of who is *riding*. Fix: associate heads with a detected bike or motorcycle (Phase 2). [Error analysis](reports/error_analysis.md)
- **Typical misses:** riders seen from behind in crowded scenes, small far-away heads, and goggles or sunglasses on the head mistaken for a helmet.
- **Distribution shift between splits:** train is 79% motorcycle images, but val and test are about 45% bicycle. Recall turned out to be similar on both vehicle types (~0.84 for Without Helmet), so the model transfers. Its weakest images are close-ups with no vehicle visible. [Dataset audit](reports/dataset_audit.md)

**Known limitations**
- The test set is small (63 images), so differences of a few points are within noise.
- The 3,546 training images are Roboflow augmentations (rotation/shear/blur) of a much smaller set of originals, so real-world diversity (night, rain, CCTV angles) is limited.
- Some visible heads in crowded images are unlabeled, which inflates false-positive counts.
- CPU inference takes about 140 ms per image at 640 px (Intel i7-7700K).

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
├── models/best.pt                  # trained weights (22 MB)
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
python scripts/train.py --config configs/experiments/yolo26s_640.yaml --device 0
python scripts/benchmark.py                    # -> reports/model_comparison.md
```
- Runs are saved to `results/train/<experiment>/`. Add `--promote` to copy the best weights to `models/best.pt`, which the app uses.
- No GPU? Run [notebook/experiments_kaggle.ipynb](notebook/experiments_kaggle.ipynb) on Kaggle (free GPU, 30 h/week). It trains all experiments in one background run and saves the results as the notebook's Output.
- Experiment tracking: `pip install mlflow`, then set `MLFLOW_TRACKING_URI=sqlite:///mlflow.db`. Ultralytics logs every run to MLflow automatically; browse them with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

---

## 🗺️ Roadmap

See [docs/PROJECT_REVIEW_AND_PLAN.md](docs/PROJECT_REVIEW_AND_PLAN.md).
1. **ML rigor** (in progress): threshold tuning ✅, error analysis ✅, dataset audit ✅, model comparison (YOLOv8 / YOLO11 / YOLO26) and class rebalancing ⏳
2. **Video system:** rider tracking (ByteTrack), head-to-vehicle association, violation logic, number-plate OCR
3. **Deployment:** FastAPI + Docker, ONNX optimization with benchmarks, CI, live demo

---

## 🙌 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) (YOLOv8 / YOLO11 / YOLO26)
- [Roboflow Universe](https://universe.roboflow.com/), the Bike Helmet Detection dataset by Syed Salman Reza (Public Domain)
