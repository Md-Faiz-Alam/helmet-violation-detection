"""Export the trained model to ONNX for torch-free CPU inference (and optionally an INT8 copy).

    python scripts/export_onnx.py                       # models/best.pt -> models/best.onnx
    python scripts/export_onnx.py --int8                # also models/best.int8.onnx (static INT8)

The export has a fixed 1x3x640x640 input and the raw detection head output (NMS is done by
scripts/onnx_detector.py). INT8 uses static quantization (QDQ, per-channel weights) calibrated on
training images, so no validation or test image is seen. Compare the files with scripts/benchmark_onnx.py.
"""

import argparse
import os
import random
import shutil
import tempfile
from pathlib import Path

import cv2


def export_fp32(weights, out, imgsz):
    from ultralytics import YOLO

    # Ultralytics writes the export next to the weights; export from a temporary copy so an existing
    # models/best.onnx is never overwritten when --out points elsewhere
    with tempfile.TemporaryDirectory() as tmp:
        src = shutil.copy(weights, os.path.join(tmp, "model.pt"))
        path = YOLO(src).export(format="onnx", imgsz=imgsz, simplify=True, dynamic=False, verbose=False)
        shutil.move(path, out)
    print(f"FP32 model: {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


def calibration_images(data_dir, n, seed=0):
    paths = sorted(Path(data_dir).glob("*.jpg"))
    if not paths:
        raise SystemExit(f"No calibration images in {data_dir}")
    random.Random(seed).shuffle(paths)
    return paths[:n]


def head_decode_nodes(path):
    """Nodes of the detection head that decode boxes and scores (everything after its conv branches).

    They stay in FP32: the final Concat mixes box coordinates (0-640 px) and class scores (0-1), and one
    INT8 scale for both rounds every score to zero.
    """
    import onnx

    graph = onnx.load(path).graph
    out_name = graph.output[0].name
    head = next(nd.name for nd in graph.node if out_name in nd.output).rsplit("/", 1)[0] + "/"  # e.g. /model.23/
    return [nd.name for nd in graph.node
            if nd.name.startswith(head) and "/cv2." not in nd.name and "/cv3." not in nd.name]


def export_int8(fp32, out, calib_dir, n):
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    from onnx_detector import OnnxDetector

    det = OnnxDetector(fp32)

    class Reader(CalibrationDataReader):
        def __init__(self, paths):
            self.it = iter(paths)

        def get_next(self):  # pyright: ignore[reportIncompatibleMethodOverride]  # None ends calibration (ORT API)
            p = next(self.it, None)
            if p is None:
                return None
            img = cv2.imread(str(p))
            return {det.input_name: det.preprocess(img)[0]}

    prep = out + ".prep.onnx"
    quant_pre_process(fp32, prep)
    quantize_static(prep, out, Reader(calibration_images(calib_dir, n)), quant_format=QuantFormat.QDQ,
                    per_channel=True, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax, nodes_to_exclude=head_decode_nodes(prep))
    os.remove(prep)
    print(f"INT8 model: {out} ({os.path.getsize(out) / 1e6:.1f} MB, calibrated on {n} train images)")


def main(args):
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    export_fp32(args.weights, args.out, args.imgsz)
    if args.int8:
        export_int8(args.out, args.out.replace(".onnx", ".int8.onnx"), args.calib_dir, args.calib_images)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export the helmet model to ONNX")
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--out", default="models/best.onnx")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--int8", action="store_true", help="Also write a static INT8 model (<out>.int8.onnx)")
    ap.add_argument("--calib-dir", default="data/train/images", help="Calibration images (training split)")
    ap.add_argument("--calib-images", type=int, default=200)
    main(ap.parse_args())
