"""Number-plate text logic (pure Python, no model code): clean up OCR output, validate the
format, pick the best frames to read from, and vote across frames.

Indian registration format: SS DD L{0,3} NNNN, e.g. MP 04 ZD 7116 -> "MP04ZD7116"
    SS state code (2 letters), DD district (1-2 digits), L series (0-3 letters), NNNN number.
    Bharat series: YY BH NNNN L{1,2}, e.g. "22BH1234AB".
OCR often confuses look-alike characters (O/0, I/1, S/5, B/8, Z/2). Because we know which
positions must be letters and which digits, we can fix those confusions safely.
"""

import re
from collections import defaultdict

PLATE_PATTERNS = {
    "india": [re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}$"), re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$")],
    "any": [re.compile(r"^(?=.*\d)(?=.*[A-Z])[A-Z0-9]{4,10}$")],
}

# Registration state / union-territory codes (old and new forms)
INDIA_STATES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL",
    "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UA",
    "UK", "UP", "WB",
}

TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"}
# Series letters never use I or O (they look like 1 and 0), so a 0/O there is almost always a D
TO_SERIES = {**TO_LETTER, "0": "D", "O": "D", "Q": "D"}
TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "U": "0", "I": "1", "L": "1", "J": "1", "Z": "2", "A": "4",
            "S": "5", "G": "6", "T": "7", "B": "8"}


def normalize(text):
    """Upper-case and keep only A-Z / 0-9 ('MP 04-zd 7116' -> 'MP04ZD7116')."""
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def _map(s, table):
    return "".join(table.get(c, c) for c in s)


def is_valid(plate, fmt="india"):
    if fmt != "india":
        return any(p.match(plate) for p in PLATE_PATTERNS[fmt])
    if PLATE_PATTERNS["india"][1].match(plate):  # Bharat series
        return True
    if not PLATE_PATTERNS["india"][0].match(plate) or plate[:2] not in INDIA_STATES:
        return False
    one_digit_district = not plate[3].isdigit()
    return not (one_digit_district and plate[2] == "0")  # a district code is never just "0"


def coerce_india(raw):
    """Read OCR text as an Indian plate, fixing look-alike characters by position, or return None.

    Tries a 2-digit district code first (most common), then 1 digit (e.g. DL 3C ...).
    """
    s = normalize(raw)
    if PLATE_PATTERNS["india"][1].match(s):
        return s
    if not 6 <= len(s) <= 11:
        return None
    state, number, middle = _map(s[:2], TO_LETTER), _map(s[-4:], TO_DIGIT), s[2:-4]
    for k in (2, 1):
        if len(middle) < k or len(middle) - k > 3:
            continue
        cand = state + _map(middle[:k], TO_DIGIT) + _map(middle[k:], TO_SERIES) + number
        if is_valid(cand):
            return cand
    return None


def clean_plate(raw, fmt="india"):
    """Normalized, validated plate text, or None if it doesn't look like a plate."""
    if fmt == "india":
        return coerce_india(raw)
    s = normalize(raw)
    return s if is_valid(s, fmt) else None


def plate_candidates(ocr_boxes, fmt="india"):
    """Turn OCR text boxes into valid plate strings.

    ocr_boxes: list of (x1, y1, x2, y2, text, conf). Plates can be one line ("MP04ZD7116") or two
    stacked lines ("MP04" over "RB0693"), and OCR may split a line into pieces, so we try every
    single box, every line (boxes at the same height, joined left to right) and every two
    consecutive lines. Returns [(plate, conf, (x1, y1, x2, y2))] with conf = mean box confidence.
    """
    boxes = sorted(ocr_boxes, key=lambda b: (b[1] + b[3]) / 2)
    lines = []
    for b in boxes:  # group boxes whose vertical centres are within half a box height
        cy, h = (b[1] + b[3]) / 2, b[3] - b[1]
        for line in lines:
            lcy = sum((x[1] + x[3]) / 2 for x in line) / len(line)
            if abs(cy - lcy) <= 0.5 * h:
                line.append(b)
                break
        else:
            lines.append([b])
    lines = [sorted(line, key=lambda b: b[0]) for line in lines]

    groups = [[b] for b in boxes] + [line for line in lines if len(line) > 1]
    groups += [lines[i] + lines[i + 1] for i in range(len(lines) - 1)]
    out = {}
    for g in groups:
        plate = clean_plate("".join(b[4] for b in g), fmt)
        if plate:
            conf = sum(b[5] for b in g) / len(g)
            box = (min(b[0] for b in g), min(b[1] for b in g), max(b[2] for b in g), max(b[3] for b in g))
            if plate not in out or conf > out[plate][0]:
                out[plate] = (conf, box)
    return [(p, c, b) for p, (c, b) in out.items()]


def vote(readings):
    """Combine plate readings from several frames: [(plate, conf), ...] -> (plate, total_conf, n_frames) or None.

    The plate read in the most frames wins (ties: higher summed confidence).
    """
    score, count = defaultdict(float), defaultdict(int)
    for plate, conf in readings:
        score[plate] += conf
        count[plate] += 1
    if not score:
        return None
    best = max(score, key=lambda p: (count[p], score[p]))
    return best, score[best], count[best]


class BestFrames:
    """Keep the k largest (= closest to the camera) observations of a track, at least `min_gap` frames apart."""

    def __init__(self, k=5, min_gap=3):
        self.k, self.min_gap = k, min_gap
        self.items = []  # (frame_idx, area, payload)

    def offer(self, frame_idx, area, make_payload):
        """make_payload is only called when the frame is kept (cropping is skipped for losers)."""
        near = [it for it in self.items if abs(it[0] - frame_idx) < self.min_gap]
        if near:
            if area <= max(it[1] for it in near):
                return False
            self.items = [it for it in self.items if it not in near]
        elif len(self.items) >= self.k:
            smallest = min(self.items, key=lambda it: it[1])
            if area <= smallest[1]:
                return False
            self.items.remove(smallest)
        self.items.append((frame_idx, area, make_payload()))
        return True

    def payloads(self):
        return [it[2] for it in sorted(self.items, key=lambda it: -it[1])]
