# RideSafe AI API image: FastAPI on ONNX Runtime, no PyTorch in the final image.
#
#   docker build -f backend/Dockerfile -t ridesafe-api .      (from the repository root)
#   docker run -p 8000:8000 ridesafe-api                      -> http://localhost:8000/docs
#
# Stage 1 ("export") installs PyTorch + Ultralytics only to convert models/best.pt (and the COCO yolo26s.pt
# used to find a violator's vehicle for plate reading) to ONNX. Stage 2 copies just the .onnx files.

ARG PYTHON=3.13-slim

# ------------------------------------------------------------------ stage 1: export the models to ONNX
FROM python:${PYTHON} AS export
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# CPU-only PyTorch first: a plain `pip install ultralytics` pulls the multi-GB CUDA build
RUN pip install --no-cache-dir torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir ultralytics==8.4.216 onnx==1.23.2 onnxslim==0.1.99
WORKDIR /build
COPY models/best.pt models/best.pt
COPY scripts/export_onnx.py scripts/export_onnx.py
# yolo26s.pt (COCO) is downloaded by Ultralytics during the export
RUN python scripts/export_onnx.py --vehicles \
    && ls -l models

# ------------------------------------------------------------------ stage 2: runtime
FROM python:${PYTHON} AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY backend/requirements-api.txt backend/requirements-api.txt
RUN pip install -r backend/requirements-api.txt \
    && pip install --no-deps rapidocr_onnxruntime==1.2.3

COPY backend/pyproject.toml backend/pyproject.toml
COPY backend/ridesafe backend/ridesafe
RUN pip install --no-deps ./backend && rm -rf backend

COPY configs/thresholds.yaml configs/bytetrack.yaml configs/
COPY --from=export /build/models/best.onnx /build/models/vehicles.onnx models/

# Unprivileged user; uploads, job results and the SQLite log live in /data (mount a volume to keep them)
RUN useradd --create-home --uid 10001 ridesafe && mkdir -p /data && chown ridesafe /data
USER ridesafe
ENV RIDESAFE_DATA_DIR=/data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
# One worker process: video jobs and their state live in this process (the API runs jobs on a thread)
CMD ["uvicorn", "ridesafe.api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
