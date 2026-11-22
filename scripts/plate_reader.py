"""Read the number plate of the vehicle a given rider is on (vehicle detector + OCR).

    reader = PlateReader()
    crop, head = PlateReader.rider_region(frame, head_box)   # cut out the rider and their vehicle
    reading = reader.read(crop, head)                        # {"plate": "MP04ZD7116", "conf": 0.7, ...} or None

Steps for one crop:
  1. a COCO detector finds bicycles/motorcycles in the crop, and backend/ridesafe/association.py picks the
     one this head is on (so a neighbour's plate is not read by mistake);
  2. RapidOCR (PaddleOCR models on onnxruntime, CPU) reads text in that vehicle box, upscaled
     when small; if no vehicle is found, it reads the area below the head instead;
  3. backend/ridesafe/plates.py joins two-line plates, fixes look-alike characters and validates the format.
read_track() does this for a few frames of one rider and votes; a plate is only reported if at
least `min_agree` frames read the same text (one lucky or unlucky frame is not evidence).
"""

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from ridesafe.association import VEHICLE_CLASSES, AssociationRule, associate
from ridesafe.plates import plate_candidates, vote

from metrics import predict_array


# Looser vehicle confidence than the default rule: a partly cropped bike is still the rider's bike
PLATE_RULE = AssociationRule(vehicle_conf=0.15)


class PlateReader:
    def __init__(self, vehicle_weights="yolo26s.pt", fmt="india", device=None, rule=PLATE_RULE, min_region_height=320):
        from rapidocr_onnxruntime import RapidOCR  # imported here so the rest of the project works without it

        self.ocr = RapidOCR()
        # Resolve against the project root, so running from another folder doesn't trigger
        # Ultralytics' automatic download of a second copy
        root_copy = Path(__file__).resolve().parents[1] / vehicle_weights
        self.vehicles = YOLO(str(root_copy) if not Path(vehicle_weights).exists() and root_copy.exists() else vehicle_weights)
        self.fmt, self.device, self.rule = fmt, device, rule
        self.min_region_height = min_region_height

    @staticmethod
    def rider_region(frame, head, pad_x=2.5, pad_up=0.5, pad_down=6.0):
        """Crop the rider and their vehicle around a head box. Returns (crop, head box in crop coordinates)."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = head
        hw, hh = x2 - x1, y2 - y1
        X1, Y1 = max(0, int(x1 - pad_x * hw)), max(0, int(y1 - pad_up * hh))
        X2, Y2 = min(w, int(x2 + pad_x * hw)), min(h, int(y2 + pad_down * hh))
        return frame[Y1:Y2, X1:X2].copy(), (x1 - X1, y1 - Y1, x2 - X1, y2 - Y1)

    def _vehicle_for(self, crop, head):
        pred = predict_array(self.vehicles, crop, conf=self.rule.vehicle_conf, device=self.device)
        vehicles = [(tuple(p[:4]), float(p[4])) for p in pred if self.vehicles.names[int(p[5])] in VEHICLE_CLASSES]
        j = associate([head], vehicles, self.rule)[0]
        return vehicles[j][0] if j >= 0 else None

    def read(self, crop, head):
        """Best plate reading for the rider whose head box (crop coordinates) is given, or None."""
        H, W = crop.shape[:2]
        hx1, hy1, hx2, hy2 = head
        vehicle = self._vehicle_for(crop, head)
        if vehicle is not None:
            vx1, vy1, vx2, vy2 = vehicle
            mx, my = 0.05 * (vx2 - vx1), 0.05 * (vy2 - vy1)
            region = (vx1 - mx, vy1 - my, vx2 + mx, vy2 + my)
        else:  # no vehicle found: read below the head
            hw = hx2 - hx1
            region = (hx1 - 1.5 * hw, hy2, hx2 + 1.5 * hw, H)
        rx1, ry1 = max(0, int(region[0])), max(0, int(region[1]))
        rx2, ry2 = min(W, int(region[2])), min(H, int(region[3]))
        if rx2 - rx1 < 16 or ry2 - ry1 < 16:
            return None
        roi = crop[ry1:ry2, rx1:rx2]
        scale = min(3.0, max(1.0, self.min_region_height / roi.shape[0]))
        if scale > 1:
            roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        result, _ = self.ocr(roi)
        boxes = []
        for quad, text, conf in result or []:
            xs, ys = [p[0] for p in quad], [p[1] for p in quad]
            boxes.append((min(xs) / scale + rx1, min(ys) / scale + ry1, max(xs) / scale + rx1, max(ys) / scale + ry1,
                          text, float(conf)))
        cands = plate_candidates(boxes, self.fmt)
        if not cands:
            return None

        # Several plates in view: prefer the one horizontally closest to this rider's head
        hcx, hw = (hx1 + hx2) / 2, max(hx2 - hx1, 1)
        def score(c):
            bx1, _, bx2, _ = c[2]
            return c[1] - 0.1 * abs((bx1 + bx2) / 2 - hcx) / hw
        plate, conf, pbox = max(cands, key=score)
        return {"plate": plate, "conf": conf, "plate_box": pbox, "vehicle_box": vehicle}

    def read_track(self, crops, min_agree=2):
        """Read several (crop, head) pairs of one rider and vote. Returns a dict, or None if no plate
        text was read in at least `min_agree` frames."""
        readings = []
        for crop, head in crops:
            r = self.read(crop, head)
            if r:
                readings.append((r, crop))
        best = vote([(r["plate"], r["conf"]) for r, _ in readings])
        if best is None:
            return None
        plate, total, n = best
        if n < min_agree:
            return None
        r, crop = max(((r, c) for r, c in readings if r["plate"] == plate), key=lambda rc: rc[0]["conf"])
        x1, y1, x2, y2 = (int(v) for v in r["plate_box"])
        pad = max(4, (y2 - y1) // 2)
        plate_img = crop[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad].copy()
        return {"plate": plate, "conf": round(total / n, 3), "frames_read": n, "frames_tried": len(crops),
                "plate_width_px": x2 - x1,
                "plate_image": plate_img if plate_img.size else np.zeros((1, 1, 3), np.uint8)}
