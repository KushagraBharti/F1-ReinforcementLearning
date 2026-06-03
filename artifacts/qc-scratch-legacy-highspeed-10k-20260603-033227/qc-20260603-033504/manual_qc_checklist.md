# Manual QC Checklist

Run these checks before starting the next long RL goal.

## Manual Driving
- `uv run f1-manual`
- Confirm the car points forward at spawn.
- Confirm full throttle visually matches the HUD speed.
- Confirm braking, steering, and coast feel plausible.
- Confirm rays render from the car nose and rotate with the car.
- Confirm the HUD shows speed, progress, checkpoint/lap state, reward, and reason fields clearly enough for debugging.

## Ghost Overlay
- `uv run f1-manual --ghost-reference`
- Confirm the manual timer advances in real time.
- Confirm the reference ghost timer advances in the same time base as replay.
- Confirm the ghost is visibly faster for legitimate speed/line reasons, not renderer timing drift.

## Replay
- `uv run f1-replay artifacts\reference-ghost-20260602-093941\steps.jsonl`
- Confirm replay timing matches real seconds unless `--speed` or `--no-timing` is used.
- Confirm rays, car orientation, and track scale match manual mode.

## Lap Validity
- Drive off track intentionally and confirm immediate termination.
- Drive a clean partial lap and confirm checkpoint count increases in order.
- Confirm benchmark/QC JSON reports missed checkpoints and invalid laps when they occur.

## Physics Feel
- Compare manual top speed, braking distance, and corner speeds against the Fast-F1 reference ghost qualitatively.
- Do not expect exact F1 lap time from manual control; use the reference ghost as calibration and the scripted driver as a physically controlled sanity baseline.
