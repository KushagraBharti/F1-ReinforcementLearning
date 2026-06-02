# Plan

## Milestone 1: Archive and Reset
- Archive legacy runtime, tests, docs, dependency manifests, and old artifacts into `archive/legacy-20260424/`.
- Keep `AGENTS.md`, `.git`, `.gitignore`, and source images in `imgs/`.
- Create a clean `src/f1rl/`, `tests/`, `assets/tracks/monza/`, and `artifacts/` structure.

Validation:
- `git status --short --branch`

## Milestone 2: Track Build
- Build `TrackSpec` from Monza contour/background/source images.
- Persist `assets/tracks/monza/track_spec.npz` and `track_manifest.json`.
- Include centerline, cumulative arc length, boundaries, drivable mask, checkpoints, start pose, finish line, and Monza scale.

Validation:
- `uv run python -m f1rl.track_build`
- `uv run pytest -q tests/test_track_build.py`

## Milestone 3: Physics and Simulator
- Implement fixed-step bicycle-style physics.
- Implement collision/off-track termination, ray sensors, centerline projection progress, checkpoint indexing, and stable reward accounting.

Validation:
- `uv run pytest -q tests/test_physics.py tests/test_sim.py`

## Milestone 4: Manual, Scripted, Telemetry, Replay
- Implement Pygame full-track manual mode with ray overlay and HUD.
- Implement deterministic scripted controller.
- Persist per-step JSONL and episode summary in all modes.
- Implement telemetry replay.
- Add reference ghost overlay and flying-start comparison in manual mode.
- Keep manual/replay timing aligned to real wall-clock time.

Validation:
- `uv run python -m f1rl.manual --headless --max-steps 20`
- `uv run python -m f1rl.scripted --steps 20`
- `uv run python -m f1rl.scripted --steps 18000 --no-telemetry`
- `uv run pytest -q tests/test_telemetry.py tests/test_scripted_replay.py`

## Milestone 5: Gymnasium and PPO
- Implement Gymnasium env using the shared simulator.
- Implement Stable-Baselines3 PPO training and checkpointing.
- Implement checkpoint evaluation with telemetry.

Validation:
- `uv run pytest -q tests/test_env.py`
- `uv run python -m f1rl.train --timesteps 128 --n-envs 1 --device auto`
- `uv run python -m f1rl.eval --checkpoint latest --steps 40`

## Milestone 6: Final Quality
- Rewrite README, runbook, and status docs.
- Regenerate `uv.lock`.
- Run lint, type checks, tests, hardware check, and smoke commands.
- Current status: active rebuild is implemented; latest audit validation found `uv run pytest -q` passing with `19 passed`.
- Current caveat: PPO smoke training works, but no trained RL policy completes clean laps yet.

Validation:
- `uv sync --active --all-extras --all-packages`
- `uv run ruff check .`
- `uv run pyright src/f1rl`
- `uv run pytest -q`
- `uv run python -m f1rl.hardware --json`

## Milestone 7: Calibration and Analysis Additions
- Status: implemented extensions. These are intentional improvements over the original minimal rebuild plan, not regressions or scope drift.
- Integrate Fast-F1 Monza reference telemetry as a ghost/reference baseline.
- Add calibration report for speed, braking, and cornering estimates.
- Expand telemetry with g-forces, curvature, control deltas, racing-line deviation, braking zones, sector speeds, and corner summaries.
- Keep richer episode summaries for sector timing, braking zones, corner entry/apex/exit speeds, smoothness metrics, g-force aggregates, and ghost-gap aggregates.
- Keep manual-mode reference ghost overlay, flying-start comparison, and replay timestamp interpolation/speed controls.
- Make CPU/GPU placement explicit: simulator/rendering/telemetry on CPU, PyTorch policy training/inference on CUDA when available.
- Keep future evolutionary search as an architecture boundary only; no evolutionary runner is active yet.

Validation:
- `uv run f1-calibration`
- `uv run f1-reference-agent --mode ghost`
- `uv run f1-hardware-check --json --require-gpu`

## Current Remaining Work
- Train PPO for serious timesteps and prove whether it can complete clean laps.
- Benchmark PPO against random, scripted, and Fast-F1 ghost-reference baselines.
- Add learning-curve plots, crash-rate/completion-rate summaries, and replay videos.
- Add a final README results table once real training/evaluation metrics exist.
- Add stronger tests around reward edge cases, collision progression, checkpoint/lap validity, reset determinism, and benchmark reporting.
