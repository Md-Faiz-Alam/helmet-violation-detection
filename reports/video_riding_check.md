# Riding check: only riders on a motorcycle can be violators

The helmet model sees heads, not riders, so pedestrians without a helmet get logged as violators. The riding check
([`tracking.py`](../backend/ridesafe/tracking.py), [`video.py`](../backend/ridesafe/video.py)) uses the COCO model
that plate reading already ships: every 2nd analysed frame, while a track has a no-helmet vote, it finds the
motorcycles and bicycles and asks whether each head sits where a rider's head would be
([`association.py`](../backend/ridesafe/association.py)). A track can only be a violation if it was on a motorcycle in
at least half of at least 2 checks; mostly on a bicycle is a **Cyclist** (no helmet rule), too little evidence of
riding is **Not riding**. `scripts/track_video.py --riding-check`.

Chosen on the **validation** clips, then checked once on the **test** clips. All runs: every 2nd frame, no plate
reading, the production tracker and violation rule, scored per rider by `scripts/eval_video.py`; speed on an Intel
i7-7700K CPU.

## Validation (5 clips: 59 riders, 32 violators)

A fifth validation clip was added for this: a crowded market street in Lucknow (Pexels, Hindustani Lens) with
pedestrians walking among riders, people at stalls and cycle-rickshaws. The first 4 validation clips have almost no
pedestrians, which is why they could not warn about the larger input size ([video_input_size.md](video_input_size.md)).

| Setting | Riders found | Violation recall | Violation precision | Duplicate logs | False logs (helmet / cyclist / non-rider) | FPS |
|---|---|---|---|---|---|---|
| 640 (production) | 34/59 | 14/32 (44%) | 14/24 (58%) | 8 | 0 / 1 / 9 | 16.1 |
| 640 + riding check | 34/59 | 14/32 (44%) | 14/21 (67%) | 8 | 0 / 1 / 6 | 11.2 |
| 960 | 42/59 | 18/32 (56%) | 18/27 (67%) | 10 | 0 / 1 / 8 | 8.6 |
| 960 + riding check | 42/59 | 18/32 (56%) | 18/22 (82%) | 10 | 0 / 1 / 3 | 6.6 |

The check never removed a real violator. It did not fix the one cyclist log: the COCO model mostly calls that bicycle a
motorcycle (where it boxes one vehicle as both, the more confident box is now kept).

## Test (4 clips: 54 riders, 40 violators), checked once

| Setting | Riders found | Violation recall | Violation precision | Duplicate logs | False logs (helmet / cyclist / non-rider) | FPS |
|---|---|---|---|---|---|---|
| 640 (production) | 28/54 | 18/40 (45%) | 18/23 (78%) | 6 | 0 / 0 / 5 | 16.9 |
| **640 + riding check** | 28/54 | **18/40 (45%)** | **18/21 (86%)** | 5 | 0 / 0 / 3 | 11.0 |
| 960 + riding check | 32/54 | 17/40 (42%) | 17/35 (49%) | 8 | 1 / 0 / 17 | 6.3 |

## Findings

- **At 640 the riding check helps on both sets, at no cost in recall:** precision 78% → 86% on test, 2 of 5
  pedestrian logs removed, at about two thirds of the speed (it only runs while someone has a no-helmet vote).
- **It is not enough to make 960 pay off.** At the crowded Pune junction pedestrians stand in the traffic, among the
  bikes, so their heads often do sit over a motorcycle. A head-over-vehicle test cannot tell them apart; whether the
  person moves *with* the vehicle could. The Lucknow clip has walking pedestrians but not that case, so validation
  still overrated 960.
- The remaining false logs and the cyclist log share the same cause: the vehicle check is geometric and per frame.

Note on method: the test clips have now been used twice in this line of work, once to reject 960 alone and once for
this check. Further choices about the riding check should be made on validation only.

_Runs: `scripts/track_video.py --vid-stride 2 --no-plates [--imgsz N] [--riding-check]`, scored with
`scripts/eval_video.py` on 2026-11-22, against labels reviewed by the author._
