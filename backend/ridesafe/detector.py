"""Torch-free helmet detector on ONNX Runtime: only numpy, OpenCV and onnxruntime are needed.

    from ridesafe.detector import OnnxDetector
    det = OnnxDetector("models/best.onnx")
    boxes = det(cv2.imread("samples/image1.jpg"))   # (M, 6) [x1, y1, x2, y2, conf, cls] in image pixels

This is the inference path of the API's Docker image (no PyTorch). It reproduces the Ultralytics
pipeline: letterbox (grey 114 padding), RGB 0-1 input, class-aware NMS at IoU 0.5 (Ultralytics' predict default), at most 300 boxes.
A model exported with a dynamic input size is letterboxed like Ultralytics does for PyTorch models: the
long side to `imgsz` and the short side padded only up to a multiple of 32 (a 16:9 frame becomes
640x384 instead of 640x640), which gives the same boxes as PyTorch and is faster on wide frames. Export the model with scripts/export_onnx.py; scripts/benchmark_onnx.py checks
that it gives the same accuracy as PyTorch.
"""

import ast
import os

import cv2
import numpy as np
import onnxruntime as ort


def letterbox(img, size, color=114, stride=None):
    """Resize keeping the aspect ratio, then pad to size x size, or only up to a multiple of `stride` when
    it is given (minimal padding). Returns (padded, ratio, (pad_x, pad_y))."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = size - nw, size - nh
    if stride:
        dw, dh = dw % stride, dh % stride
    dw, dh = dw / 2, dh / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(color, color, color))
    return img, r, (left, top)


def nms(boxes, scores, iou_thr):
    """Greedy NMS on xyxy boxes (N, 4). Returns kept indices, highest score first."""
    order = np.argsort(-scores)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xy1 = np.maximum(boxes[i, :2], boxes[rest, :2])
        xy2 = np.minimum(boxes[i, 2:], boxes[rest, 2:])
        inter = np.clip(xy2 - xy1, 0, None).prod(axis=1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return np.array(keep, dtype=int)


class OnnxDetector:
    """YOLO detector exported by Ultralytics (raw head output, NMS done here)."""

    def __init__(self, path, threads=0, conf=0.25, iou=0.5, max_det=300):
        if not os.path.exists(path):
            raise FileNotFoundError(f"ONNX model not found: {path} (export it with scripts/export_onnx.py)")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads  # 0 = ONNX Runtime default (one thread per physical core)
        self.session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        meta = self.session.get_modelmeta().custom_metadata_map
        self.dynamic = not isinstance(inp.shape[2], int)  # dynamic export: minimal (rectangular) padding
        self.imgsz = max(ast.literal_eval(meta["imgsz"])) if self.dynamic else int(inp.shape[2])
        self.stride = int(meta.get("stride", 32))
        if meta.get("end2end", "False") == "True":
            raise ValueError("End-to-end (NMS-free) exports are not supported; export with end2end disabled")
        self.names = {int(k): v for k, v in ast.literal_eval(meta["names"]).items()}
        self.conf, self.iou, self.max_det = conf, iou, max_det

    def preprocess(self, img):
        """BGR uint8 image -> (1, 3, S, S) float32 RGB tensor in 0-1, plus the letterbox ratio and padding."""
        padded, r, pad = letterbox(img, self.imgsz, stride=self.stride if self.dynamic else None)
        x = padded[:, :, ::-1].transpose(2, 0, 1)  # BGR HWC -> RGB CHW
        x = np.ascontiguousarray(x, dtype=np.float32)[None] / 255.0
        return x, r, pad

    def postprocess(self, out, r, pad, shape, conf, agnostic=False):
        """Raw (1, 4 + nc, N) output -> (M, 6) [x1, y1, x2, y2, conf, cls] in original image pixels."""
        p = out[0].T  # (N, 4 + nc): cx, cy, w, h, class scores
        scores = p[:, 4:]
        cls = scores.argmax(axis=1)
        best = scores[np.arange(len(p)), cls]
        m = best > conf
        p, cls, best = p[m], cls[m], best[m]
        if not len(p):
            return np.zeros((0, 6), dtype=np.float32)
        xy, wh = p[:, :2], p[:, 2:4] / 2
        boxes = np.concatenate([xy - wh, xy + wh], axis=1)
        # class-aware NMS shifts each class's boxes so different classes never overlap; agnostic NMS keeps
        # one box per object whatever its class (used for tracking: one head must not become two riders)
        offset = 0.0 if agnostic else cls[:, None] * 7680.0
        keep = nms(boxes + offset, best, self.iou)[: self.max_det]
        boxes, best, cls = boxes[keep], best[keep], cls[keep]
        boxes -= np.array([pad[0], pad[1], pad[0], pad[1]], dtype=np.float32)
        boxes /= r
        h, w = shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        return np.concatenate([boxes, best[:, None], cls[:, None]], axis=1).astype(np.float32)

    def __call__(self, img, conf=None, agnostic=False):
        """Detect on one BGR image (as read by cv2.imread)."""
        x, r, pad = self.preprocess(img)
        out = self.session.run(None, {self.input_name: x})[0]
        return self.postprocess(out, r, pad, img.shape, self.conf if conf is None else conf, agnostic)
