from ridesafe.plates import BestFrames, clean_plate, coerce_india, is_valid, normalize, plate_candidates, vote


def test_normalize():
    assert normalize("mp 04-zd 7116") == "MP04ZD7116"


def test_valid_formats():
    assert is_valid("MP04ZD7116")
    assert is_valid("DL3C1234")       # 1-digit district, 1-letter series
    assert is_valid("KA011234")       # no series letters
    assert is_valid("22BH1234AB")     # Bharat series
    assert not is_valid("HPOZYC")
    assert not is_valid("9190322")
    assert not is_valid("XX04ZD7116")     # not a state code
    assert not is_valid("MP0ZZD7116")     # 1-digit district can't be 0


def test_coerce_fixes_lookalikes_by_position():
    assert coerce_india("MPO4ZD7116") == "MP04ZD7116"   # O in the district -> 0
    assert coerce_india("MP04ZD711G") == "MP04ZD7116"   # G in the number -> 6
    assert coerce_india("MP04Z07116") == "MP04ZD7116"   # series never uses O, so 0 there is a D (real OCR output)
    assert coerce_india("MP0420711B") == "MP04ZD7118"   # series 2/0 -> Z/D, B in the number -> 8
    assert coerce_india("DL3CAB1234") == "DL3CAB1234"   # 1-digit district stays
    assert coerce_india("HP-OZYC") is None


def test_clean_plate_any_format():
    assert clean_plate("ab-123", fmt="any") == "AB123"
    assert clean_plate("ABCDEF", fmt="any") is None    # needs a digit


def box(x1, y1, x2, y2, text, conf=0.8):
    return (x1, y1, x2, y2, text, conf)


def test_two_line_plate_is_joined():
    cands = plate_candidates([box(1032, 382, 1140, 418, "MP04"), box(1005, 418, 1180, 459, "RB0693")])
    assert [c[0] for c in cands] == ["MP04RB0693"]


def test_split_single_line_is_joined_left_to_right():
    cands = plate_candidates([box(700, 540, 770, 571, "7116"), box(614, 540, 695, 571, "MP04ZD")])
    assert [c[0] for c in cands] == ["MP04ZD7116"]


def test_noise_is_ignored():
    assert plate_candidates([box(0, 0, 50, 20, "9190322"), box(0, 40, 50, 60, "RE")]) == []


def test_vote_prefers_most_frames():
    result = vote([("MP04ZD7116", 0.7), ("MP04ZD7118", 0.95), ("MP04ZD7116", 0.6)])
    assert result is not None
    plate, score, n = result
    assert (plate, round(score, 6), n) == ("MP04ZD7116", 1.3, 2)
    assert vote([]) is None


def test_best_frames_keeps_largest_and_spaced():
    bf = BestFrames(k=2, min_gap=3)
    for idx, area in [(1, 10), (2, 50), (3, 20), (10, 30), (20, 5), (21, 40)]:
        bf.offer(idx, area, lambda idx=idx: idx)
    assert bf.payloads() == [2, 21]   # frame 2 (50) beats its neighbours 1 and 3; 21 (40) beats 10 (30)


def test_best_frames_skips_payload_for_losers():
    bf = BestFrames(k=1)
    bf.offer(1, 100, lambda: "kept")
    calls = []
    bf.offer(10, 5, lambda: calls.append(1))
    assert calls == [] and bf.payloads() == ["kept"]
