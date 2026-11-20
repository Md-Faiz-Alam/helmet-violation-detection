import os
from collections import Counter

import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO

from scripts.utils import apply_thresholds, load_thresholds

MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")
SAMPLE_DIR = "samples"
VIOLATION_CLASS = "Without Helmet"

st.set_page_config(page_title="Helmet Detection", page_icon="🪖", layout="centered")


@st.cache_resource
def load_model(path):
    """Load the model once per server process instead of on every interaction."""
    return YOLO(path)


def run_detection(model, pil_image, thresholds):
    """Return (annotated RGB image, Counter of detected classes). thresholds: {class name: min confidence}."""
    image = ImageOps.exif_transpose(pil_image).convert("RGB")  # phone photos: apply EXIF rotation
    result = model.predict(image, conf=min(thresholds.values()), verbose=False)[0]
    result = apply_thresholds(result, thresholds)
    counts = Counter(result.names[int(c)] for c in result.boxes.cls)
    annotated = result.plot()[:, :, ::-1]  # BGR -> RGB
    return annotated, counts


def show_result(model, pil_image, thresholds, caption):
    annotated, counts = run_detection(model, pil_image, thresholds)
    st.image(annotated, caption=caption, width="stretch")

    with_helmet = counts.get("With Helmet", 0)
    without_helmet = counts.get(VIOLATION_CLASS, 0)
    col1, col2 = st.columns(2)
    col1.metric("✅ With helmet", with_helmet)
    col2.metric("🚨 Without helmet", without_helmet)

    if not counts:
        st.info("No riders/heads detected. Try a lower threshold or using a clearer image.")
    elif without_helmet:
        st.error(f"Violation: {without_helmet} person(s) detected without a helmet.")
    else:
        st.success("Everyone detected is wearing a helmet.")


st.title("🪖 Helmet Detection")
st.write("Detect riders **with** and **without** helmets using a fine-tuned YOLO model.")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model weights not found at `{MODEL_PATH}`. Train a model or set the MODEL_PATH environment variable.")
    st.stop()

model = load_model(MODEL_PATH)

with st.sidebar:
    st.header("Settings")
    tuned = load_thresholds()
    options = (["Tuned per class (recommended)"] if tuned else []) + ["Single threshold"]
    threshold_mode = st.radio("Confidence thresholds", options,
                              help="Tuned thresholds come from scripts/tune_thresholds.py: the violation class "
                                   "uses a low threshold so fewer riders without helmets are missed.")
    if threshold_mode.startswith("Tuned"):
        thresholds = tuned
        for name, t in thresholds.items():
            st.caption(f"{name}: {t:.2f}")
    else:
        conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05,
                         help="Lower = catch more violations but more false alarms.")
        thresholds = {name: conf for name in model.names.values()}
    st.caption(f"Model: `{MODEL_PATH}`")

mode = st.radio("Input", ["Upload", "Webcam", "Sample"], horizontal=True)

if mode == "Upload":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        show_result(model, Image.open(uploaded_file), thresholds, "Detection result")

elif mode == "Webcam":
    camera_image = st.camera_input("Take a photo")
    if camera_image:
        show_result(model, Image.open(camera_image), thresholds, "Detection result")

elif mode == "Sample":
    sample_images = sorted(f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    choice = st.selectbox("Choose a sample image", sample_images)
    if choice:
        show_result(model, Image.open(os.path.join(SAMPLE_DIR, choice)), thresholds, f"Sample: {choice}")
