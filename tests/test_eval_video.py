from eval_video import assign_tracks, iou, score_clip

HEAD = [100, 100, 140, 150]  # 40 x 50 head box


def shifted(box, dx):
    return [box[0] + dx, box[1], box[2] + dx, box[3]]


def gt_with(*riders):
    return {"riders": [{"id": rid, "helmet": helmet, "plate": plate, "keyframes": kf}
                       for rid, helmet, plate, kf in riders]}


def track(counted=True, violation=False, label="With Helmet", plate=None):
    return {"counted": counted, "violation": violation, "label": label, "plate": plate}


def test_iou():
    assert iou(HEAD, HEAD) == 1.0
    assert iou(HEAD, shifted(HEAD, 100)) == 0.0


def test_track_matches_rider_within_frame_tolerance():
    riders = {"R1": {10: HEAD}}
    assert assign_tracks(riders, {7: {14: shifted(HEAD, 3)}}, frame_tol=8) == {7: "R1"}
    assert assign_tracks(riders, {7: {30: HEAD}}, frame_tol=8) == {}          # too far in time
    assert assign_tracks(riders, {7: {10: shifted(HEAD, 60)}}, frame_tol=8) == {}  # different place


def test_violation_tp_fn_fp_and_plates():
    gt = gt_with(("A", False, "MP04ZD7116", {10: HEAD}),
                 ("B", False, None, {10: shifted(HEAD, 300)}),
                 ("C", True, None, {10: shifted(HEAD, 600)}))
    boxes = {1: {10: HEAD}, 2: {10: shifted(HEAD, 300)}, 3: {10: shifted(HEAD, 600)}, 4: {10: shifted(HEAD, 900)}}
    tracks = {1: track(violation=True, label="Without Helmet", plate="MP02ZD7116"),  # TP, wrong plate
              2: track(label="Uncertain"),                                         # FN, flagged
              3: track(violation=True, label="Without Helmet"),                    # FP on a helmeted rider
              4: track(violation=True, label="Without Helmet")}                    # FP on nobody
    t = score_clip(gt, boxes, tracks)["totals"]
    assert (t["tp"], t["fn"], t["fp"], t["fp_helmeted"], t["fp_nobody"]) == (1, 1, 2, 1, 1)
    assert t["fn_flagged_uncertain"] == 1
    assert (t["plates_correct"], t["plates_wrong"], t["plates_unread"]) == (0, 1, 0)
    assert t["false_riders"] == 1  # track 4 is counted but matches no rider


def test_cyclists_need_no_helmet():
    gt = gt_with(("A", False, None, {10: HEAD}), ("B", False, None, {10: shifted(HEAD, 300)}))
    gt["riders"][0]["vehicle"] = "bicycle"  # A rides a bicycle: never a violator
    boxes = {1: {10: HEAD}, 2: {10: shifted(HEAD, 300)}}
    tracks = {1: track(violation=True, label="Without Helmet"), 2: track(violation=True, label="Without Helmet")}
    t = score_clip(gt, boxes, tracks)["totals"]
    assert (t["violators"], t["tp"], t["fn"], t["fp"], t["fp_cyclist"], t["cyclists"]) == (1, 1, 0, 1, 1, 1)
    assert t["riders_found"] == 2  # a cyclist still counts as a rider found


def test_fragmented_rider_counts_extra_and_duplicate_logs():
    gt = gt_with(("A", False, None, {10: HEAD, 100: shifted(HEAD, 50)}))
    boxes = {1: {10: HEAD}, 2: {100: shifted(HEAD, 50)}}
    tracks = {1: track(violation=True, label="Without Helmet"), 2: track(violation=True, label="Without Helmet")}
    t = score_clip(gt, boxes, tracks)["totals"]
    assert (t["riders_found"], t["extra_counts"], t["tp"], t["dup_logs"]) == (1, 1, 1, 1)


def test_small_and_unclear_riders_are_not_scored():
    tiny = [100, 100, 110, 115]  # 15 px tall head
    gt = gt_with(("far", False, None, {10: tiny}), ("blurry", "unclear", None, {10: shifted(HEAD, 300)}))
    boxes = {1: {10: tiny}, 2: {10: shifted(HEAD, 300)}}
    tracks = {1: track(violation=True, label="Without Helmet"), 2: track(violation=True, label="Without Helmet")}
    out = score_clip(gt, boxes, tracks, min_head=24)
    t = out["totals"]
    assert out["ignored"] == ["far"]
    assert (t["riders"], t["violators"], t["tp"], t["fp"]) == (1, 0, 0, 0)


def test_resolution_scaling():
    gt = gt_with(("A", False, None, {10: HEAD}))
    boxes = {1: {10: [v * 1.5 for v in HEAD]}}  # same clip at 1.5x resolution
    t = score_clip(gt, boxes, {1: track(violation=True, label="Without Helmet")}, scale=1.5)["totals"]
    assert t["tp"] == 1
