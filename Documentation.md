# Documentation

## Session
- Date: 2026-04-24
- Objective: simplify and rebuild the project around explicit Monza geometry, shared car physics, manual mode, telemetry, Gymnasium, and PPO.

## Decisions
- Archived the previous complex implementation into `archive/legacy-20260424/`.
- Kept source assets in `imgs/`.
- Removed active Ray/RLlib, imitation, campaign, swarm, image observation, and custom torch-native PPO paths.
- Chose Stable-Baselines3 PPO as the first trainer.
- Chose discrete actions for v1.
- Kept future evolutionary search support as an architecture boundary, not an active implementation.
- CPU is used for simulation/rendering/telemetry/vector env workers.
- GPU is used for PyTorch policy training/inference when available.

## Current Active Stack
- Python `>=3.11,<3.13`
- `uv`
- NumPy
- OpenCV for one-time track preprocessing
- Pygame for manual/replay rendering
- Gymnasium for env API
- PyTorch + Stable-Baselines3 for PPO

## Commands to Run
1. `uv sync --active --all-extras --all-packages`
2. `uv run python -m f1rl.track_build`
3. `uv run pytest -q`
4. `uv run ruff check .`
5. `uv run pyright src/f1rl`
6. `uv run python -m f1rl.hardware --json`
7. `uv run python -m f1rl.manual --headless --max-steps 60`
8. `uv run python -m f1rl.train --timesteps 128 --n-envs 1 --device auto`
9. `uv run python -m f1rl.eval --checkpoint latest --steps 40 --device auto`

## Validation Log
- `uv sync --active --all-extras --all-packages` -> pass after removing stale legacy package metadata inside `.venv`; SB3, TensorBoard, and updated dev tooling installed.
- `uv run python -m f1rl.track_build` -> pass; wrote `assets/tracks/monza/track_spec.npz`.
- `.venv\Scripts\python.exe -m pytest -q tests/test_track_build.py tests/test_physics.py tests/test_sim.py` -> pass (`6 passed`).
- `.venv\Scripts\python.exe -m pytest -q tests/test_env.py tests/test_telemetry.py tests/test_scripted_replay.py` -> pass (`4 passed`).
- `.venv\Scripts\python.exe -m f1rl.manual --headless --max-steps 20` -> pass; wrote manual-headless telemetry.
- `uv run ruff check .` -> pass.
- `uv run pyright src/f1rl` -> pass (`0 errors`).
- `uv run pytest -q` -> pass (`11 passed`).
- `uv run python -m f1rl.hardware --json --require-gpu` -> pass; detected `NVIDIA GeForce RTX 4060 Laptop GPU`, Torch `2.10.0+cu128`, CUDA `12.8`.
- `uv run python -m f1rl.train --timesteps 128 --n-envs 1 --max-steps 80 --device auto` -> pass; wrote `artifacts/train-20260424-015021/checkpoints/final_model.zip` on CUDA.
- `uv run python -m f1rl.eval --checkpoint latest --steps 40 --device auto` -> pass; wrote `artifacts/eval-20260424-015046/steps.jsonl`.
- `uv run python -m f1rl.replay artifacts/eval-20260424-015046/steps.jsonl --headless` -> pass; loaded 40 steps.
- `uv run f1-reference-agent --mode ghost` -> pass; wrote a perfect Fast-F1 ghost lap at `79.662s`, `259.9 kph` average, `348.0 kph` max.
- `uv run f1-replay artifacts/reference-ghost-20260424-025933/steps.jsonl --headless` -> pass; loaded 610 reference ghost steps.
- `uv run f1-reference-agent --mode control --steps 7200` -> diagnostic run; controller left track early, so this is not yet a tuned autonomous driver.
- Fixed replay timing so interactive playback respects telemetry `sim_time_s` instead of advancing one row per rendered frame.

## Issues
- During exact `uv sync`, the old environment had several stale dist-info folders from archived packages with missing `RECORD` files. They were removed only after verifying their paths were inside this repo's `.venv`.
- Stable-Baselines3 warns that MLP PPO on GPU may have poor utilization. The project still defaults policy training/inference to CUDA when available, matching the rebuild requirement; env stepping remains CPU-bound.

## Fast-F1 Monza Calibration - 2026-04-24
### Source
- Inspected `C:\Users\kushagra\OneDrive\Documents\CS Projects\Fast-F1`.
- Ran `examples\telemetry\plot_monza_car_timings.py` with the local Fast-F1 package through `uv run`.
- Exported source telemetry from Fast-F1:
  - `C:\Users\kushagra\OneDrive\Documents\CS Projects\Fast-F1\exports\monza_2024_Q_VER_telemetry.csv`
- Copied calibration input into this repo:
  - `assets/reference/monza_2024_Q_VER_telemetry.csv`
  - `assets/reference/monza_2024_Q_VER_summary.json`

### Real Targets
- Event: 2024 Italian Grand Prix qualifying
- Driver: VER
- Lap time: `79.662s`
- Telemetry distance: `5745.669m`
- Minimum speed: `75.0 kph`
- Maximum speed: `348.0 kph`
- Mean speed: `259.9 kph`
- P90 speed: `336.2 kph`
- Derived turning targets from `X`, `Y`, and `Distance`:
  - curvature p90: `0.0128 rad/m`
  - curvature p95: `0.0204 rad/m`
  - radius p05: `49.1m`
  - radius p10: `77.9m`
  - lateral-g p90: `3.28g`
  - lateral-g p95: `4.09g`

### Simulator Tuning
- Added `f1rl.calibration` / `f1-calibration`.
- Tuned default `CarParams` toward the real Monza telemetry:
  - terminal speed estimate: `354.6 kph`
  - 5s full-throttle estimate: `298.1 kph`
  - 8s full-throttle estimate: `339.8 kph`
  - braking estimate `330 -> 150 kph`: `66.7m`
  - braking estimate `330 -> 100 kph`: `78.4m`
- Tuned turning model:
  - grip cap: `4.1g`, matching Fast-F1 p95 derived lateral load
  - max steering: `18 deg`
  - steering response: `6 rad/s`
  - simulator cornering limits:
    - `100 kph`: radius `19.2m`
    - `150 kph`: radius `43.2m`
    - `200 kph`: radius `76.7m`
    - `250 kph`: radius `119.9m`
    - `300 kph`: radius `172.7m`
- The raw maximum lateral-g derived from telemetry is noisy, so tuning uses p90/p95 percentiles rather than raw spikes.

### Validation
- `uv run python -m f1rl.calibration` -> pass.
- `uv run ruff check src tests` -> pass.
- `uv run pyright src/f1rl` -> pass.
- `uv run pytest -q` -> pass (`13 passed`).

## Fast-F1 Reference Agent - 2026-04-24
### Implementation
- Added `f1rl.reference_agent` / `f1-reference-agent`.
- `--mode ghost` maps Fast-F1 telemetry distance onto the simulator Monza centerline and copies real speed, throttle, brake, lap time, and derived steering/yaw-rate proxies into the standard telemetry schema.
- `--mode control` uses the same reference profile as a target for a pure-pursuit and speed-chasing controller through the real simulator physics.
- Ghost mode is the performance oracle/reference lap. Control mode is intentionally separate because it reveals physics/controller mismatch.

### Validation
- `uv run f1-reference-agent --mode ghost` -> pass.
  - target lap: `79.662s`
  - simulated ghost lap: `79.662s`
  - target/sim average speed: `259.9 kph`
  - target/sim max speed: `348.0 kph`
  - artifact: `artifacts/reference-ghost-20260424-025933/`
- `uv run f1-reference-agent --mode control --steps 7200` -> generated telemetry but did not complete a lap; current result ended `off_track`.
- `uv run f1-replay artifacts/reference-ghost-20260424-025933/steps.jsonl --headless` -> pass; loaded 610 steps.
- `uv run ruff check src tests` -> pass.
- `uv run pyright src/f1rl` -> pass.
- `uv run pytest -q` -> pass (`17 passed`).

## Replay Timing Fix - 2026-04-24
- Root cause: `f1-replay` rendered one telemetry row per frame. The Fast-F1 ghost lap has 610 telemetry samples, so at 120 FPS it appeared to finish in about 5 seconds even though the file spans `79.662s`.
- Fix: replay now uses each row's `sim_time_s` timestamp by default.
- Added `--speed` for intentional playback scaling and `--no-timing` for old fast-scrub behavior.
- `uv run f1-replay artifacts/reference-ghost-20260424-025933/steps.jsonl --headless` -> pass; reports `duration=79.662s`.
