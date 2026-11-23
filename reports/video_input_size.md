# Video input size: 640 vs 960 vs 1280

Can a larger detector input catch the far-away riders the video system misses? Chosen on the **validation** clips,
then checked once on the **test** clips. All runs: every 2nd frame, no plate reading, the production tracker and
violation rule (`configs/bytetrack.yaml`, at least 5 no-helmet votes and 60% of a rider's votes), scored per rider by
`scripts/eval_video.py`; speed on an Intel i7-7700K CPU.

## Validation (4 clips: 43 riders, 23 violators)

| Input size | Riders found | Violation recall | Violation precision | Duplicate logs | False logs (helmet / cyclist / non-rider) | FPS |
|---|---|---|---|---|---|---|
| 640 (production) | 25/43 | 12/23 (52%) | 12/16 (75%) | 8 | 0 / 1 / 3 | 14.6 |
| **960** | 31/43 | **16/23 (70%)** | **16/18 (89%)** | 10 | 0 / 1 / 1 | 7.5 |
| 1280 | 32/43 | 16/23 (70%) | 16/19 (84%) | 12 | 1 / 1 / 1 | 4.2 |

At 960, three tracker variants (lower thresholds, a longer memory for lost tracks, both) changed nothing: same recall
and precision, 10-11 duplicate logs. The 7 violators still missed at 960 are 6 riders the model never tracks (2 far
away, 4 bareheaded riders seen from behind at dusk, whose heads score 0.01-0.05) and 1 child pillion whose track breaks
into 13 pieces. The duplicate logs are riders split into several tracks by occlusion or ID swaps between neighbours.

## Test (4 clips: 54 riders, 40 violators), checked once

| Input size | Riders found | Violation recall | Violation precision | Duplicate logs | False logs (helmet / cyclist / non-rider) | FPS |
|---|---|---|---|---|---|---|
| **640 (production)** | 28/54 | **18/40 (45%)** | **18/23 (78%)** | 6 | 0 / 0 / 5 | 16.9 |
| 960 | 32/54 | 17/40 (42%) | 17/42 (40%) | 8 | 1 / 0 / 24 | 8.7 |

Per clip at 960: the highway clip improves (8 → 9 of 12 violators), the crowded Pune junction gets worse (9 → 7 of 26),
and tracks counted on people who are not riding rise from 8 to 36.

## Decision

**Production stays at 640.** A larger input finds more small heads, riders and pedestrians alike, and the system has
no signal for who is riding, so on crowded junctions the extra pedestrian heads become false violation logs. The
validation clips have few pedestrians, so they could not show this; the held-out test clips did.

Next: a "who is riding" check (a person on a two-wheeler, as used to propose missed riders while labelling the
validation clips), then 960 again on top of it. The validation set also needs a crowded, pedestrian-heavy clip of its
own, so the next choice does not depend on the test clips.

Caveat: part of the validation labels' boxes came from 1280 px runs, which slightly favours larger inputs in the
matching (IoU ≥ 0.3); it does not change the decision.

_Runs: `scripts/track_video.py --no-plates --no-riding-check --dedupe-gap 0 [--imgsz N]` (every 2nd frame; the riding
and duplicate checks did not exist yet), scored with `scripts/eval_video.py` on 2026-11-22._
