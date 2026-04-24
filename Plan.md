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
- Implement collision/off-track termination, ray sensors, centerline projection progress, checkpoint guardrails, and stable reward accounting.

Validation:
- `uv run pytest -q tests/test_physics.py tests/test_sim.py`

## Milestone 4: Manual, Scripted, Telemetry, Replay
- Implement Pygame full-track manual mode with ray overlay and HUD.
- Implement deterministic scripted controller.
- Persist per-step JSONL and episode summary in all modes.
- Implement telemetry replay.

Validation:
- `uv run python -m f1rl.manual --headless --max-steps 20`
- `uv run python -m f1rl.scripted --steps 20`
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

Validation:
- `uv sync --active --all-extras --all-packages`
- `uv run ruff check .`
- `uv run pyright src/f1rl`
- `uv run pytest -q`
- `uv run python -m f1rl.hardware --json`
