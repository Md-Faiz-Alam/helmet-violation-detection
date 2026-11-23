# Duplicate logs: one violator, one log

When a rider is hidden for a moment (behind a bus or their pillion, or blurred), ByteTrack loses them and gives
them a new ID when they reappear. Both tracks are judged on their own votes, so the same violator was logged twice
or three times: 10 extra logs on the 5 validation clips, more than half the number of violators caught.

The duplicate check ([`tracking.py`](../backend/ridesafe/tracking.py), `RIDESAFE_DEDUPE_GAP`) runs when a track is
about to be logged as a violation. It links the track to an earlier violator instead of logging it again when all
of these hold:

- it starts at most `dedupe_gap` analysed frames after that violator's track ended (60 = 4 s at 30 FPS, every 2nd frame);
- its first head box is within 1 head height of where the violator was last seen, **or** of where the violator's
  last movement was heading (riders keep moving while hidden);
- its head is between 1/1.4 and 1.4 times the violator's last head height.

The linked track keeps its red "no helmet" label in the web app and records `duplicate_of`; it is not logged, and
it is not counted as another rider. Chains link back to the first log.

## Choosing the limits (validation only)

For every logged violation on the 5 validation clips (production 640 px, riding check on), every track that started
within 60 analysed frames after it ended was measured against the labels:

- **Same rider:** 0.27-1.43 head heights from the last position or the predicted one (all but one within 0.93),
  head size 0.87-1.33 times, gaps of 1-58 frames.
- **Different riders:** the closest pair is a motorcycle driver and his pillion, 1.37 head heights apart at the same
  size. Other nearby different riders had a much smaller head (0.26-0.45 times: someone farther away).

So 1 head height and 1.4 times the size separate them with a margin. The predicted position matters: one rider
reappeared 2 head heights from where he vanished but 0.69 from where he was heading.

## Result (validation: 5 clips, 61 scored riders, 34 violators)

| Setting | Violation recall | Violation precision | Duplicate logs | False logs (helmet / cyclist / non-rider) |
|---|---|---|---|---|
| production before (640 + riding check) | 16/34 (47%) | 16/19 (84%) | 10 | 0 / 1 / 2 |
| + duplicate check, 30 frames | 16/34 (47%) | 16/19 (84%) | 5 | 0 / 1 / 2 |
| **+ duplicate check, 60 frames** | **16/34 (47%)** | **16/18 (89%)** | **3** | 0 / 1 / 1 |
| + duplicate check, 90 frames | 16/34 (47%) | 16/18 (89%) | 2 | 0 / 1 / 1 |

All 8 links at 60 frames were checked against the labels: 7 join two tracks of the same rider. The 8th joins a
pedestrian, logged as a violator, to a rider without a helmet who had just passed the same spot. That link is wrong,
though it happens to remove a false log, so the precision gain from 84% to 89% is partly luck; the duplicate
reduction (10 → 3) is the real effect. No violator was lost.

60 frames is the production setting: the longest same-rider gap seen was 58, and 90 frames removes only one more
duplicate while doubling the window for linking two people who pass the same spot. The 3 duplicates left:
a pillion who reappeared 1.43 head heights from her predicted position, a rider hidden for 125 analysed frames
(about 8 s), and a rider tracked twice *at the same time* for a few seconds (two overlapping IDs on one head), which
a rule about tracks that end and restart cannot see.

The test clips were not used for this choice.

## Test (4 clips: 54 riders, 40 violators), checked once after the choice

| Setting | Violation recall | Violation precision | Duplicate logs | False logs (helmet / cyclist / non-rider) |
|---|---|---|---|---|
| production before (640 + riding check) | 18/40 (45%) | 18/21 (86%) | 5 | 0 / 0 / 3 |
| + duplicate check, 60 frames | 17/40 (42%) | 17/18 (94%) | 3 | 0 / 0 / 1 |

The scored recall drops by one, so every link was checked by eye. One link on the night clip joins two tracks of
the same rider. The other links are at the Pune junction, and the tracks they join are all on **one man standing on
the pavement**, who stays put for at least 6 s while traffic passes in front of him. Motorcycles keep passing under his
head, so the riding check takes him for a rider, and he was logged three times. The duplicate check now logs him once.

Those three logs had been scored as three caught violators (P18, P5, P8): riders who passed directly in front of him
at the moment of one of their keyframes, so the box on his head matched theirs (P18 and P8 have a single keyframe
each, at exactly that moment; P5 at frame 329). So the recall drop is credit that the earlier pipeline got from a
pedestrian's track, not a violator who was caught before and missed now. What the result does show is the remaining
Pune problem: a person standing in the traffic still passes the riding check.

Checking this also found 2 wrong keyframes for P5 (frames 325 and 331 were on her driver's head); they were
redrawn on hers (noted in the clip's `annotation.corrections`). Re-scoring both runs on the corrected labels gives the
same numbers as the table above.

The duplicate check stays on in production. Note on method: the test clips have now been used four times in this
line of work.

_Runs: `scripts/track_video.py --no-plates [--dedupe-gap N]` (every 2nd frame, riding check on), scored with
`scripts/eval_video.py` on 2026-11-23 against the corrected validation labels._
