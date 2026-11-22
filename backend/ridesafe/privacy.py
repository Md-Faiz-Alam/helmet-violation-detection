"""Blur faces and number plates in a video job, so it can be published as the public sample.

    python -m ridesafe.jobs --make-sample <video> --out samples/demo_job --anonymize

The analysis runs on the original video; only what is published is blurred: the video the web app plays,
the violation snapshots, and the plate data (crops removed, text masked). Everything uses models the project
already ships, so no extra download:

  - faces: the helmet model finds every head; the lower part of each head box (the face) is blurred and the top
    is kept, so a viewer can still see whether there is a helmet, which is what the demo is about. People whose
    head the helmet model does not find (pedestrians) get the head end of their COCO "person" box blurred;
  - plates: the COCO model finds vehicles, and two finders look inside each one: the OCR text detector (no
    reading), and a plate-shape check (a white or yellow rectangle of plate proportions with dense edges, which
    catches dirty two-line plates the text detector misses). Everything they find is blurred (plates, and also
    brand names or a bright patch of clothing: harmless);
  - flicker: a detector that misses a plate in one frame would let it flash readable, so every frame is blurred
    with the regions found in its neighbouring frames too, slightly enlarged to cover the motion in between.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from ridesafe.detector import OnnxDetector

PLATE_VEHICLES = ("bicycle", "motorcycle", "car", "bus", "truck")
CAPTION_PX = 28  # the caption bar ridesafe.video.label_snapshot adds under a snapshot


def face_region(head, top=0.35, side=0.1, below=0.15):
    """The face part of a head box: everything but the top of the head (helmet or hair stays visible)."""
    x1, y1, x2, y2 = head
    w, h = x2 - x1, y2 - y1
    return (x1 - side * w, y1 + top * h, x2 + side * w, y2 + below * h)


def plate_shaped(img, vehicle, max_share=0.04, min_edges=0.08):
    """Boxes inside a vehicle that look like a number plate: a white or yellow, near-rectangular area of plate
    proportions (one-line ~4.5:1 to two-line ~1.3:1) with the dense edges of characters. The characters are
    holes in the bright area; only outer contours are used, so they do not split the plate."""
    H, W = img.shape[:2]
    x1, y1 = max(0, int(vehicle[0])), max(0, int(vehicle[1]))
    x2, y2 = min(W, int(vehicle[2])), min(H, int(vehicle[3]))
    if x2 - x1 < 16 or y2 - y1 < 16:
        return []
    roi = img[y1:y2, x1:x2]
    h, s, v = cv2.split(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))
    white = (v > 140) & (s < 70)
    yellow = (h >= 15) & (h <= 40) & (s > 90) & (v > 120)  # commercial vehicles
    mask = ((white | yellow) * 255).astype(np.uint8)
    edges = cv2.Canny(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 60, 160)
    vehicle_area = (x2 - x1) * (y2 - y1)
    out = []
    for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        bx, by, bw, bh = cv2.boundingRect(c)
        if bh < 8 or bw < 16 or not 1.2 <= bw / bh <= 5.5:
            continue
        if not 0.002 * vehicle_area <= bw * bh <= max_share * vehicle_area:
            continue
        if cv2.contourArea(c) / (bw * bh) < 0.6 or edges[by:by + bh, bx:bx + bw].mean() / 255 < min_edges:
            continue
        out.append((bx + x1, by + y1, bx + bw + x1, by + bh + y1))
    return out


def grow(box, f):
    x1, y1, x2, y2 = box
    dx, dy = f * (x2 - x1), f * (y2 - y1)
    return (x1 - dx, y1 - dy, x2 + dx, y2 + dy)


def pad_text(box, f=0.3, max_px=8):
    """A text box with a margin: f of its height, at most max_px (neighbouring frames cover larger motion)."""
    m = min(f * (box[3] - box[1]), max_px)
    return (box[0] - m, box[1] - m, box[2] + m, box[3] + m)


def blur_boxes(img, boxes):
    """Blur each box in place, irreversibly: shrink to a few pixels, then scale back up."""
    H, W = img.shape[:2]
    for b in boxes:
        x1, y1 = max(0, int(b[0])), max(0, int(b[1]))
        x2, y2 = min(W, int(np.ceil(b[2]))), min(H, int(np.ceil(b[3])))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = img[y1:y2, x1:x2]
        k = max(4.0, min(x2 - x1, y2 - y1) / 5)  # about 5 blocks across the short side: no letter survives
        small = cv2.resize(roi, (max(1, round((x2 - x1) / k)), max(1, round((y2 - y1) / k))),
                           interpolation=cv2.INTER_AREA)
        img[y1:y2, x1:x2] = cv2.GaussianBlur(cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR),
                                             (0, 0), max(1.0, k / 3))
    return img


def neighbours_union(per_frame, window):
    """For each frame, its own regions plus those of the `window` frames before and after it."""
    n = len(per_frame)
    return [[b for j in range(max(0, i - window), min(n, i + window + 1)) for b in per_frame[j]] for i in range(n)]


class Anonymizer:
    def __init__(self, helmet_model="models/best.onnx", coco_model="models/vehicles.onnx", head_conf=0.1,
                 person_conf=0.15, vehicle_conf=0.2, min_vehicle_px=48, text_height=320):
        from rapidocr_onnxruntime import RapidOCR  # optional dependency, installed in the API image

        self.heads = OnnxDetector(helmet_model)
        self.coco = OnnxDetector(coco_model)
        self.text = RapidOCR()
        names = self.coco.names
        self.person_id = next(i for i, n in names.items() if n == "person")
        self.vehicle_ids = {i for i, n in names.items() if n in PLATE_VEHICLES}
        self.head_conf, self.person_conf, self.vehicle_conf = head_conf, person_conf, vehicle_conf
        self.min_vehicle_px, self.text_height = min_vehicle_px, text_height

    def faces(self, img):
        heads = [tuple(p[:4]) for p in self.heads(img, conf=self.head_conf, agnostic=True)]
        # A head cut off by the top of the frame has its face higher in the box than usual: blur all of it
        regions = [face_region(h) if h[1] > 2 else face_region(h, top=0) for h in heads]
        coco = self.coco(img, conf=min(self.person_conf, self.vehicle_conf))
        for p in coco:
            if int(p[5]) != self.person_id or p[4] < self.person_conf:
                continue
            x1, y1, x2, y2 = p[:4]
            # Whole body: the head is the top fifth. Only the upper body visible (a driver behind a windscreen,
            # someone cut by the frame): the head can be anywhere in the top three quarters
            head_part = 0.2 if (y2 - y1) >= 1.5 * (x2 - x1) else 0.75
            top = y1 + max(0.3, head_part) * (y2 - y1)
            if any(x1 <= (h[0] + h[2]) / 2 <= x2 and y1 - 0.1 * (y2 - y1) <= (h[1] + h[3]) / 2 <= top for h in heads):
                continue  # this person's head is already handled (its helmet stays visible)
            regions.append((x1, y1, x2, y1 + head_part * (y2 - y1)))
        vehicles = [tuple(p[:4]) for p in coco if int(p[5]) in self.vehicle_ids and p[4] >= self.vehicle_conf]
        return regions, vehicles

    def text_boxes(self, img, region=None):
        """Text boxes (image coordinates) inside region (default: the whole image), upscaled when small."""
        H, W = img.shape[:2]
        rx1, ry1, rx2, ry2 = region or (0, 0, W, H)
        rx1, ry1, rx2, ry2 = max(0, int(rx1)), max(0, int(ry1)), min(W, int(rx2)), min(H, int(ry2))
        if rx2 - rx1 < 16 or ry2 - ry1 < 16:
            return []
        roi = img[ry1:ry2, rx1:rx2]
        scale = min(3.0, max(1.0, self.text_height / roi.shape[0]))
        if scale > 1:
            roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        quads, _ = self.text.text_detector(roi)  # detection only, no reading
        out = []
        for q in quads if quads is not None else []:
            xs, ys = q[:, 0] / scale + rx1, q[:, 1] / scale + ry1
            out.append((xs.min(), ys.min(), xs.max(), ys.max()))
        return out

    def plates(self, img, vehicles, text=True):
        """Plate regions in the given vehicles; text=False runs only the fast plate-shape check."""
        boxes = []
        for v in vehicles:
            vw, vh = v[2] - v[0], v[3] - v[1]
            if vh < self.min_vehicle_px:
                continue  # too small for a plate to be readable
            boxes += [pad_text(b) for b in plate_shaped(img, v)]
            if text:
                # A plate is a small part of its vehicle (sometimes merged with the headlight above it); the text
                # detector also returns boxes around a whole rider or bike, which would blur away the demo
                boxes += [pad_text(b) for b in self.text_boxes(img, grow(v, 0.05))
                          if b[3] - b[1] <= 0.5 * vh and b[2] - b[0] <= 0.8 * vw]
        return boxes

    def image(self, img, whole_image_text=True):
        """Blurred copy of one image (a snapshot: no neighbouring frames to borrow from)."""
        faces, vehicles = self.faces(img)
        boxes = faces + self.plates(img, vehicles)
        if whole_image_text:  # snapshots are small: also catch a plate on a vehicle the detector missed
            boxes += [pad_text(b) for b in self.text_boxes(img)]
        return blur_boxes(img.copy(), boxes)

    def video(self, src, dst, text_every=3, window=None, progress=True):
        """Write a blurred H.264 copy of src (same size, frame count and frame rate, so tracks stay in sync)."""
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        faces, plates, i = [], [], 0
        while True:  # pass 1: find what to blur in every frame (text every few frames: the slow part)
            ok, frame = cap.read()
            if not ok:
                break
            f, vehicles = self.faces(frame)
            faces.append([grow(b, 0.1) for b in f])
            plates.append(self.plates(frame, vehicles, text=i % text_every == 0))
            i += 1
            if progress and i % 50 == 0:
                print(f"  {i} frames scanned")
        cap.release()
        face_regions = neighbours_union(faces, 2)
        plate_regions = neighbours_union(plates, window or text_every + 1)

        cap = cv2.VideoCapture(str(src))
        write, close = h264_writer(dst, fps, size)
        for k in range(i):  # pass 2: blur and write
            ok, frame = cap.read()
            if not ok:
                break
            write(blur_boxes(frame, face_regions[k] + plate_regions[k]))
        cap.release()
        close()
        return i


def h264_writer(dst, fps, size, crf=26):
    """(write(frame), close()) for an H.264 MP4 that browsers play and that starts before it is fully loaded.

    Uses ffmpeg's x264 encoder from the imageio-ffmpeg package (pip install imageio-ffmpeg) at a normal web
    quality; without it, OpenCV's own H.264 encoder, which works but ignores quality settings (~7x larger files).
    """
    try:
        import imageio_ffmpeg
    except ImportError:
        out = cv2.VideoWriter(str(dst), cv2.VideoWriter.fourcc(*"avc1"), fps, size)
        if not out.isOpened():
            raise RuntimeError("Cannot write H.264 video: pip install imageio-ffmpeg") from None
        return out.write, out.release
    gen = imageio_ffmpeg.write_frames(str(dst), size, fps=fps, codec="libx264", pix_fmt_in="bgr24",
                                      pix_fmt_out="yuv420p", macro_block_size=1,
                                      quality=None,  # pyright: ignore[reportArgumentType]  (None: -crf decides)
                                      output_params=["-crf", str(crf), "-preset", "slow", "-movflags", "+faststart"])
    gen.send(None)
    return gen.send, gen.close


def anonymize_sample(sample_dir, anonymizer=None):
    """Blur a sample job folder in place: video, snapshots; drop plate crops; mask plate text."""
    from ridesafe.api import mask_plate

    d = Path(sample_dir)
    a = anonymizer or Anonymizer()
    job = json.loads((d / "job.json").read_text(encoding="utf-8"))
    src = d / job["video"]
    tmp = d / "blurred.mp4"
    n = a.video(src, tmp)
    src.unlink()
    job["video"] = "input.mp4"
    tmp.replace(d / job["video"])

    violations = json.loads((d / "violations.json").read_text(encoding="utf-8"))
    for v in violations:
        if v.get("snapshot"):
            p = d / v["snapshot"]
            img = cv2.imread(str(p))
            if img is None:
                raise RuntimeError(f"Cannot read snapshot {p}: refusing to publish it unblurred")
            body, bar = img[:-CAPTION_PX], img[-CAPTION_PX:]
            cv2.imwrite(str(p), np.vstack([a.image(body), bar]), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if v.get("plate_image"):
            (d / v["plate_image"]).unlink(missing_ok=True)
            v["plate_image"] = None
        if v.get("plate"):
            v["plate"] = mask_plate(v["plate"])
    (d / "violations.json").write_text(json.dumps(violations, indent=1), encoding="utf-8")
    job["anonymized"] = True
    (d / "job.json").write_text(json.dumps(job, indent=1), encoding="utf-8")
    print(f"Blurred {n} frames and {sum(1 for v in violations if v.get('snapshot'))} snapshots in {d}")
