"""Score a track_video.py run against hand-checked video ground truth (eval/video_gt/*.yaml).

    python scripts/track_video.py --source data/videos/indian_highway_riders.mp4 --vid-stride 2
    python scripts/eval_video.py --gt eval/video_gt/indian_highway_riders.yaml --run results/video/<that run>

Several clips: repeat --gt/--run in the same order. Report: reports/video_eval.md.

How tracks are matched to riders: every ground-truth rider has head boxes at keyframes. A track
matches a rider at a keyframe if it has a box within +-`frame_tol` source frames with IoU >= `iou`;
each track is assigned to the rider it matches most often. Boxes are rescaled when the run's
video has another resolution than the ground truth (e.g. the 1080p copy of the same clip).

Bicycle riders (`vehicle: bicycle`) need no helmet: they are never violators, and a violation logged on one
counts as a false log.

Metrics (riders whose head is never >= `min_head` px tall are ignored, as are helmet: 'unclear'
riders for the violation metrics):
  riders        recall = riders with at least one *counted* track; extra tracks = duplicate counts;
                false riders = counted tracks that match no rider
  violations    per rider: TP = a violator with a logged violation, FN = a violator without one,
                FP = a logged violation on a rider with a helmet or on no rider at all;
                duplicates = more than one violation logged for the same rider
  plates        for detected violators whose plate a person can read: correct / wrong / not read
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import yaml


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def assign_tracks(riders, track_boxes, frame_tol=8, min_iou=0.3):
    """riders: {rider id: {frame: box}}; track_boxes: {track id: {frame: box}} -> {track id: rider id}."""
    votes = defaultdict(Counter)
    for rid, keyframes in riders.items():
        for f, gbox in keyframes.items():
            for tid, boxes in track_boxes.items():
                near = [g for g in range(f - frame_tol, f + frame_tol + 1) if g in boxes]
                if near and iou(boxes[min(near, key=lambda g: abs(g - f))], gbox) >= min_iou:
                    votes[tid][rid] += 1
    return {tid: c.most_common(1)[0][0] for tid, c in votes.items()}


def score_clip(gt, track_boxes, tracks, min_head=24, frame_tol=8, min_iou=0.3, scale=1.0):
    """gt: parsed ground-truth YAML; track_boxes: {tid: {frame: box}} in run pixels;
    tracks: {tid: {"counted", "violation", "label", "plate"}} from tracks.csv. Returns a result dict."""
    riders, info, ignored = {}, {}, set()
    for r in gt["riders"]:
        kf = {int(f): [v * scale for v in b] for f, b in r["keyframes"].items()}
        riders[r["id"]] = kf
        info[r["id"]] = r
        if max(b[3] - b[1] for b in kf.values()) < min_head * scale:
            ignored.add(r["id"])
    owner = assign_tracks(riders, track_boxes, frame_tol, min_iou)

    by_rider = defaultdict(list)
    for tid, rid in owner.items():
        by_rider[rid].append(tid)
    counted = [t for t, v in tracks.items() if v["counted"]]
    logged = [t for t, v in tracks.items() if v["violation"]]

    evaluable = [rid for rid in riders if rid not in ignored]
    rows, res = [], Counter()
    for rid in evaluable:
        tids = sorted(by_rider.get(rid, []))
        c_tids = [t for t in tids if tracks.get(t, {}).get("counted")]
        v_tids = [t for t in tids if tracks.get(t, {}).get("violation")]
        helmet = info[rid]["helmet"]
        res["riders"] += 1
        res["riders_found"] += bool(c_tids)
        res["extra_counts"] += max(0, len(c_tids) - 1)
        outcome = ""
        if info[rid].get("vehicle") == "bicycle":  # no helmet rule for bicycles: never a violator
            res["cyclists"] += 1
            if v_tids:
                res["fp"] += len(v_tids)
                res["fp_cyclist"] += len(v_tids)
                outcome = "FP (cyclist: no helmet required)"
            else:
                outcome = "not a violator (cyclist)"
        elif helmet is False:
            res["violators"] += 1
            if v_tids:
                res["tp"] += 1
                res["dup_logs"] += len(v_tids) - 1
                outcome = "TP"
                gt_plate = info[rid].get("plate")
                if gt_plate:
                    res["plates_readable"] += 1
                    read = [tracks[t].get("plate") for t in v_tids if tracks[t].get("plate")]
                    res["plates_correct" if gt_plate in read else "plates_wrong" if read else "plates_unread"] += 1
            else:
                res["fn"] += 1
                labels = sorted({tracks[t]["label"] or "no votes" for t in tids if t in tracks})
                outcome = "FN (" + (", ".join(labels) if tids else "never tracked") + ")"
                res["fn_flagged_uncertain"] += any(tracks[t]["label"] == "Uncertain" for t in tids if t in tracks)
        elif helmet is True:
            res["helmeted"] += 1
            if v_tids:
                res["fp"] += len(v_tids)
                res["fp_helmeted"] += len(v_tids)
                outcome = "FP (helmet worn)"
            else:
                outcome = "TN"
        else:
            outcome = "not scored (helmet unclear)"
        rows.append({"rider": rid, "helmet": helmet, "tracks": tids, "counted_tracks": len(c_tids),
                     "labels": [tracks[t]["label"] or "no votes" for t in tids if t in tracks], "outcome": outcome,
                     "description": info[rid].get("description", "")})

    on_ignored = {t for t in owner if owner[t] in ignored}
    res["false_riders"] = sum(1 for t in counted if t not in owner)
    fp_nobody = [t for t in logged if t not in owner]
    res["fp"] += len(fp_nobody)
    res["fp_nobody"] = len(fp_nobody)
    res["counted_tracks"] = len(counted)
    res["ignored_riders"] = len(ignored)
    res["tracks_on_ignored"] = len(on_ignored)
    return {"totals": res, "rows": rows, "fp_unmatched_tracks": fp_nobody, "ignored": sorted(ignored)}


def load_run(run_dir):
    run_dir = Path(run_dir)
    boxes = defaultdict(dict)
    with open(run_dir / "track_boxes.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            boxes[int(r["track_id"])][int(r["frame"])] = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
    tracks = {}
    with open(run_dir / "tracks.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tracks[int(r["track_id"])] = {"counted": r["counted"] == "True", "violation": r["violation"] == "True",
                                          "label": r["label"], "plate": r.get("plate") or None}
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return boxes, tracks, summary


def video_size(path):
    import cv2  # only needed to read the run's resolution
    cap = cv2.VideoCapture(str(path))
    size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return size


def ratio(num, den):
    return f"{num / den:.0%} ({num}/{den})" if den else "n/a"


def main(args):
    if len(args.gt) != len(args.run):
        raise SystemExit("Give one --run per --gt, in the same order.")
    sections, total = [], Counter()
    for gt_path, run_dir in zip(args.gt, args.run, strict=True):
        gt = yaml.safe_load(Path(gt_path).read_text(encoding="utf-8"))
        boxes, tracks, summary = load_run(run_dir)
        w, _ = video_size(summary["source"]) if Path(summary["source"]).exists() else (gt["resolution"][0], 0)
        scale = w / gt["resolution"][0] if w else 1.0
        out = score_clip(gt, boxes, tracks, args.min_head, args.frame_tol, args.iou, scale)
        t = out["totals"]
        total.update(t)
        rows = ["| Rider | Helmet | Tracks | Counted tracks | System labels | Outcome |", "|---|---|---|---|---|---|"]
        for r in out["rows"]:
            helmet = {True: "yes", False: "**no**"}.get(r["helmet"], str(r["helmet"]))
            rows.append(f"| {r['rider']}: {r['description']} | {helmet} | {', '.join(map(str, r['tracks'])) or '-'} "
                        f"| {r['counted_tracks']} | {', '.join(sorted(set(r['labels']))) or '-'} | {r['outcome']} |")
        s = summary["settings"]
        sections.append(f"""### {gt['video']} → run `{Path(run_dir).name}`

Run: `{Path(summary['source']).name}` at {int(w) if w else '?'} px wide, vid_stride {s['vid_stride']}, min_frames {s['min_frames']}, min_ratio {s['min_ratio']}, {summary['processing_fps']} FPS on CPU. Ignored (head never ≥ {args.min_head} px): {', '.join(out['ignored']) or 'none'}.

{chr(10).join(rows)}

Logged violations matching no rider: {out['fp_unmatched_tracks'] or 'none'}. Counted tracks matching no rider (false riders): {t['false_riders']}.
""")

    t = total
    prec = ratio(t["tp"], t["tp"] + t["fp"])
    summary_rows = [
        "| Metric | Value |", "|---|---|",
        f"| Riders found (≥1 counted track) | {ratio(t['riders_found'], t['riders'])} |",
        f"| Duplicate counts (extra counted tracks for a found rider) | {t['extra_counts']} |",
        f"| False riders (counted tracks on no rider) | {t['false_riders']} |",
        f"| **Violation recall** (violators with a logged violation) | **{ratio(t['tp'], t['violators'])}** |",
        f"| **Violation precision** (logged violations that are real) | **{prec}** |",
        f"| Missed violators shown as Uncertain (flagged for review) | {t['fn_flagged_uncertain']} of {t['fn']} |",
        f"| Duplicate violation logs for one rider | {t['dup_logs']} |",
        f"| False violation logs: on a helmeted rider / on a cyclist / on no rider | "
        f"{t['fp_helmeted']} / {t['fp_cyclist']} / {t['fp_nobody']} |",
        f"| Plates (readable by a person, on detected violators): correct / wrong / not read | "
        f"{t['plates_correct']} / {t['plates_wrong']} / {t['plates_unread']} of {t['plates_readable']} |",
    ]
    report = f"""# Video evaluation

_Generated by `scripts/eval_video.py` on {date.today()}. Ground truth: {', '.join(f'`{p}`' for p in args.gt)}. Matching: IoU ≥ {args.iou} within ±{args.frame_tol} frames._

**Why:** the image test set says how well single photos are classified; this measures what the video system actually outputs per rider (counting, violations, plates), against hand-checked ground truth.

## Summary

{chr(10).join(summary_rows)}

**Caveat:** {t['riders']} riders and {t['violators']} violators in total, so every percentage has a wide margin of error. The ground truth was drafted from tracker boxes and checked by eye (see each YAML's `annotation` block); riders the tracker never saw would be missing from it, which is why the overview frames were also checked.

## Per clip

{chr(10).join(sections)}"""
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved {args.report}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate video tracking/violation output against ground truth")
    ap.add_argument("--gt", nargs="+", required=True, help="Ground-truth YAML file(s)")
    ap.add_argument("--run", nargs="+", required=True, help="track_video.py output folder(s), same order as --gt")
    ap.add_argument("--min-head", type=int, default=24, help="Ignore riders whose head is never this tall (px)")
    ap.add_argument("--frame-tol", type=int, default=8, help="Frames a track box may be away from a keyframe")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--report", default="reports/video_eval.md")
    main(ap.parse_args())
