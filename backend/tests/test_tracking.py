from ridesafe.tracking import CYCLIST, NOT_RIDING, SAFE_CLASS, UNCERTAIN, VIOLATION_CLASS, ViolationTracker

THR = {SAFE_CLASS: 0.25, VIOLATION_CLASS: 0.16}
BOX = (10, 10, 30, 30)


def feed(tracker, frames, start=1, frame_size=None):
    """frames: list of lists of (track_id, class, conf[, box]). Returns violations finalized during the feed."""
    logged = []
    for i, dets in enumerate(frames, start=start):
        logged += tracker.update(i, [(d[0], d[1], d[2], d[3] if len(d) > 3 else BOX) for d in dets], frame_size)
    return logged


def test_violation_needs_min_frames():
    tr = ViolationTracker(THR, min_frames=5)
    feed(tr, [[(1, VIOLATION_CLASS, 0.8)]] * 4)
    assert tr.finish() == []
    tr = ViolationTracker(THR, min_frames=5)
    feed(tr, [[(1, VIOLATION_CLASS, 0.8)]] * 5)
    assert [t.track_id for t in tr.finish()] == [1]


def test_violation_logged_once_when_track_ends():
    tr = ViolationTracker(THR, min_frames=3, finalize_after=5)
    logged = feed(tr, [[(1, VIOLATION_CLASS, 0.9)]] * 10)
    assert logged == []                       # still in view: nothing logged yet
    logged = feed(tr, [[]] * 6, start=11)     # rider leaves; finalized after 5 unseen frames
    assert [t.track_id for t in logged] == [1]
    assert tr.tracks[1].first_violation_frame == 3
    assert tr.finish() == [] and len(tr.events) == 1


def test_early_no_helmet_frames_do_not_lock_in():
    # ID switch case: the first frames say no-helmet, the rest of the track clearly has a helmet
    tr = ViolationTracker(THR, min_frames=5, min_ratio=0.6)
    feed(tr, [[(1, VIOLATION_CLASS, 0.7)]] * 9 + [[(1, SAFE_CLASS, 0.9)]] * 90)
    assert tr.label(tr.tracks[1]) == SAFE_CLASS
    assert tr.finish() == []
    assert tr.summary() == {"riders": 1, "with_helmet": 1, "violations": 0, "uncertain": 0}


def test_flicker_does_not_confirm():
    tr = ViolationTracker(THR, min_frames=3, min_ratio=0.6)
    feed(tr, [[(1, SAFE_CLASS, 0.8)], [(1, SAFE_CLASS, 0.8)], [(1, VIOLATION_CLASS, 0.5)]] * 4)
    assert tr.finish() == []
    assert tr.label(tr.tracks[1]) == SAFE_CLASS


def test_low_confidence_frames_do_not_vote():
    tr = ViolationTracker(THR, min_frames=3)
    feed(tr, [[(1, VIOLATION_CLASS, 0.10)]] * 10)  # below the 0.16 threshold
    t = tr.tracks[1]
    assert t.frames_seen == 10 and t.total_votes == 0
    assert tr.finish() == [] and tr.summary()["riders"] == 0


def test_boxes_at_frame_edge_do_not_vote():
    tr = ViolationTracker(THR, min_frames=3, edge_margin=2)
    cut_off = (0, 0, 20, 20)  # touches the top-left border
    feed(tr, [[(1, VIOLATION_CLASS, 0.9, cut_off)]] * 10, frame_size=(100, 100))
    assert tr.tracks[1].total_votes == 0 and tr.finish() == []


def test_short_tracks_not_counted():
    tr = ViolationTracker(THR, min_frames=5)
    feed(tr, [[(1, SAFE_CLASS, 0.9), (2, SAFE_CLASS, 0.9)]] * 2 + [[(1, SAFE_CLASS, 0.9)]] * 3)
    assert tr.summary() == {"riders": 1, "with_helmet": 1, "violations": 0, "uncertain": 0}


def test_summary_counts_and_uncertain():
    tr = ViolationTracker(THR, min_frames=3, min_ratio=0.6)
    frames = [[(1, SAFE_CLASS, 0.9), (2, VIOLATION_CLASS, 0.7)]] * 4
    # track 3: 3 no-helmet vs 2 helmet votes -> ratio 0.6 -> violation
    # track 4: 3 no-helmet vs 3 helmet votes -> ratio 0.5 -> uncertain (shown orange, not logged)
    frames += [[(3, VIOLATION_CLASS, 0.7), (4, VIOLATION_CLASS, 0.9)]] * 3
    frames += [[(3, SAFE_CLASS, 0.3), (4, SAFE_CLASS, 0.3)]] * 2 + [[(4, SAFE_CLASS, 0.3)]]
    feed(tr, frames)
    assert sorted(t.track_id for t in tr.finish()) == [2, 3]
    assert tr.summary() == {"riders": 4, "with_helmet": 1, "violations": 2, "uncertain": 1}
    assert tr.label(tr.tracks[4]) == UNCERTAIN


def test_split_vote_is_uncertain_not_helmet():
    # Real case (1080p clip, white skullcap): 17 helmet vs 16 no-helmet votes
    tr = ViolationTracker(THR, min_frames=5, min_ratio=0.6, uncertain_ratio=0.4)
    feed(tr, [[(1, SAFE_CLASS, 0.8)]] * 17 + [[(1, VIOLATION_CLASS, 0.8)]] * 16)
    assert tr.finish() == []
    assert tr.label(tr.tracks[1]) == UNCERTAIN
    assert tr.summary() == {"riders": 1, "with_helmet": 0, "violations": 0, "uncertain": 1}


def test_reused_id_keeps_old_track():
    tr = ViolationTracker(THR, min_frames=2, finalize_after=2)
    feed(tr, [[(1, VIOLATION_CLASS, 0.9)]] * 3 + [[]] * 3)   # track 1 finalized as a violation
    feed(tr, [[(1, SAFE_CLASS, 0.9)]] * 3, start=7)          # same ID reused by a new rider
    tr.finish()
    assert [t.violation for t in tr.all_tracks()] == [True, False]


def test_best_violation_frame_kept():
    tr = ViolationTracker(THR, min_frames=2)
    feed(tr, [[(7, VIOLATION_CLASS, 0.4)], [(7, VIOLATION_CLASS, 0.9)], [(7, VIOLATION_CLASS, 0.6)]])
    t = tr.tracks[7]
    assert t.best_violation_conf == 0.9 and t.best_violation_frame == 2


def test_only_open_tracks_are_scanned():
    tr = ViolationTracker(THR, min_frames=2, finalize_after=2)
    feed(tr, [[(1, SAFE_CLASS, 0.9), (2, SAFE_CLASS, 0.9)]] * 3 + [[(2, SAFE_CLASS, 0.9)]] * 4)
    assert set(tr.open) == {2}          # track 1 left and was decided
    assert set(tr.tracks) == {1, 2}     # still kept for the final report
    tr.finish()
    assert tr.open == {}


def riding_track(vehicles, ratio=0.5):
    """One clear no-helmet track whose riding checks found these vehicles ('motorcycle', 'bicycle', None)."""
    tr = ViolationTracker(THR, min_frames=5, min_ride_ratio=ratio, min_ride_checks=2)
    feed(tr, [[(1, VIOLATION_CLASS, 0.9)]] * 10)
    for v in vehicles:
        tr.observe_riding(1, v)
    return tr


def test_riding_check_off_by_default():
    tr = ViolationTracker(THR, min_frames=5)
    feed(tr, [[(1, VIOLATION_CLASS, 0.9)]] * 10)
    tr.observe_riding(1, None)  # recorded, but without min_ride_ratio it decides nothing
    assert [t.track_id for t in tr.finish()] == [1]


def test_motorcyclist_without_helmet_is_a_violation():
    tr = riding_track(["motorcycle", "motorcycle", None, "motorcycle"])
    assert tr.label(tr.tracks[1]) == VIOLATION_CLASS
    assert [t.track_id for t in tr.finish()] == [1]


def test_pedestrian_is_not_riding():
    tr = riding_track([None, None, "motorcycle", None])  # next to a bike once, but not on one
    assert tr.label(tr.tracks[1]) == NOT_RIDING
    assert tr.finish() == []


def test_cyclist_needs_no_helmet():
    tr = riding_track(["bicycle", "bicycle", None])
    assert tr.label(tr.tracks[1]) == CYCLIST
    assert tr.finish() == []


def test_too_few_riding_checks_is_not_a_violation():
    tr = riding_track(["motorcycle"])  # 1 check, 2 needed
    assert tr.label(tr.tracks[1]) == NOT_RIDING and tr.finish() == []


def test_helmeted_pedestrian_keeps_its_helmet_label():
    tr = ViolationTracker(THR, min_frames=5, min_ride_ratio=0.5)
    feed(tr, [[(1, SAFE_CLASS, 0.9)]] * 10)
    tr.observe_riding(1, None)
    tr.observe_riding(1, None)
    assert tr.label(tr.tracks[1]) == SAFE_CLASS  # only no-helmet tracks are relabelled


def moving(tid, start_x, frames, step=4, y=100, h=40, cls=VIOLATION_CLASS):
    """A head moving right `step` px per frame: list of per-frame detections for one track."""
    return [[(tid, cls, 0.9, (start_x + i * step, y, start_x + i * step + h, y + h))] for i in range(frames)]


def test_a_rider_lost_for_a_moment_is_logged_once():
    # Track 1 is lost behind a bus for 10 frames; track 2 picks up where its movement was heading
    tr = ViolationTracker(THR, min_frames=3, finalize_after=5, dedupe_gap=60)
    feed(tr, moving(1, 0, 20))
    feed(tr, [[]] * 10, start=21)
    feed(tr, moving(2, 120, 20), start=31)   # 20 frames x 4 px + 10 hidden frames x 4 px = x 120
    feed(tr, [[]] * 6, start=51)
    assert [t.track_id for t in tr.events] == [1]
    t2 = tr.tracks[2]
    assert t2.duplicate_of == 1 and not t2.violation and tr.label(t2) == VIOLATION_CLASS
    assert tr.summary()["riders"] == 1 and tr.summary()["violations"] == 1


def test_duplicates_chain_back_to_the_first_log():
    tr = ViolationTracker(THR, min_frames=3, finalize_after=5, dedupe_gap=60)
    feed(tr, moving(1, 0, 10) + [[]] * 3 + moving(2, 52, 10) + [[]] * 3 + moving(3, 104, 10) + [[]] * 6)
    assert [t.track_id for t in tr.events] == [1]
    assert tr.tracks[2].duplicate_of == 1 and tr.tracks[3].duplicate_of == 1


def test_a_different_rider_nearby_is_logged_separately():
    # A pillion right beside the driver (1.5 head heights), a rider elsewhere, a much smaller (farther) head,
    # and a rider at the same spot but long after: all logged on their own
    for second in ([(2, VIOLATION_CLASS, 0.9, (60, 100, 100, 140))],              # 1.5 heads to the right
                   [(2, VIOLATION_CLASS, 0.9, (400, 300, 440, 340))],            # elsewhere
                   [(2, VIOLATION_CLASS, 0.9, (0, 100, 15, 115))]):              # 0.375x the head size
        tr = ViolationTracker(THR, min_frames=3, finalize_after=5, dedupe_gap=60)
        feed(tr, [[(1, VIOLATION_CLASS, 0.9, (0, 100, 40, 140))]] * 10 + [[]] * 2 + [second] * 10 + [[]] * 6)
        assert [t.track_id for t in tr.events] == [1, 2], second
    tr = ViolationTracker(THR, min_frames=3, finalize_after=5, dedupe_gap=20)
    feed(tr, [[(1, VIOLATION_CLASS, 0.9, (0, 100, 40, 140))]] * 10 + [[]] * 30
         + [[(2, VIOLATION_CLASS, 0.9, (0, 100, 40, 140))]] * 10 + [[]] * 6)
    assert [t.track_id for t in tr.events] == [1, 2]


def test_duplicate_check_is_off_by_default():
    tr = ViolationTracker(THR, min_frames=3, finalize_after=5)
    feed(tr, moving(1, 0, 10) + [[]] * 3 + moving(2, 52, 10) + [[]] * 6)
    assert [t.track_id for t in tr.events] == [1, 2]
