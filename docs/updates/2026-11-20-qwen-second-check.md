# Update 2026-11-20: event fix + Qwen3-VL second check

Yesterday's results are unchanged in this repo. This page adds what changed today and compares it with yesterday.

## What changed
1. Event-logic fix: clips with only one scoring window (shorter than about 4 s) could never alert. Fixed in `vdet/stream.py` (patch: `2026-11-20-event-fix.patch`).
2. Second check: Qwen3-VL-8B (4-bit, runs locally on a MacBook Air) re-checks every VideoMAE alert. It can only confirm or dismiss an alert, never raise one alone.
3. New script `scripts/check_clip.py`: runs both stages on one video and writes an annotated video, a timeline, and a scene description (school-safety prompt from Pankaj).
4. Tested pose features (YOLO pose + tracking) and fusion with VideoMAE: not worth it for now (see below).

## Comparison: yesterday vs today

SURV (real CCTV, 150 fights / 150 non-fights, about 2 s clips)

| Setup | Fights caught | False alarms | Precision |
|---|---|---|---|
| Yesterday: VideoMAE alert as shipped | 2/150 (1.3%) | 0/150 | 100% |
| Today: + event fix | 138/150 (92%) | 24/150 (16%) | 85.2% |
| Today: + Qwen "simple" prompt (default) | 135/150 (90%) | 13/150 (8.7%) | 91.2% |
| Today: + Qwen "cues" prompt | 129/150 (86%) | 8/150 (5.3%) | 94.2% |

AIRTLab (acted, 230 fights / 120 non-fights)

| Setup | Fights caught | False alarms | Precision |
|---|---|---|---|
| Yesterday: VideoMAE alert as shipped | 151/230 (65.7%) | 41/120 (34.2%) | 78.6% |
| Today: + event fix | 166/230 (72.2%) | 46/120 (38.3%) | 78.3% |
| Today: + Qwen "simple" | 69/230 (30.0%) | 0/120 | 100% |
| Today: + Qwen "cues" | 37/230 (16.1%) | 0/120 | 100% |

## Reading the numbers
- Real CCTV (SURV): the biggest gain is the event fix (2 to 138 fights caught). Qwen then cuts false alarms from 24 to 13 while losing only 3 fights.
- AIRTLab: Qwen removed every false alarm (hugs, greetings, jumps, high-fives) but also dismissed most acted fights as playful or choreographed. AIRTLab fights are staged, so this does not mean Qwen fails on real fights.
- Pose features and fusion with VideoMAE learned the dataset's staging and did not transfer across datasets (false alarms up to 75% when trained on one dataset and tested on the other). Not adopted.

## Single-clip checks (children, YouTube Shorts, run locally)
- Classroom fight clip: VideoMAE 99%, Qwen confirmed fight, alert raised.
- Slap-boxing clip (children, restroom): VideoMAE 99%, Qwen called it play, alert dismissed. The separate scene description called it an altercation. Plan: merge into one call returning fight / rough play / normal.

## Limits
- No dataset here has both real fights and real hugs/play, and none has children. That is the test that decides whether this works in schools.
- Results are from a laptop run; no child footage is stored in this repo.

## Next
- Decide with the team whether rough play should alert in schools.
- One merged Qwen prompt (description + fight / rough play / normal), re-tested on SURV and AIRTLab.
- Realistic test set with real fights and real hugs/play.
