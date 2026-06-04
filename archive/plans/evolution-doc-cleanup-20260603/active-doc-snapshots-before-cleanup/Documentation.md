# Documentation

## Current Status - 2026-06-02

The simplified Monza simulator and RL proof of concept is implemented and verified end to end.

Active architecture:

```text
track geometry -> car physics -> shared simulator -> manual/scripted -> telemetry -> Gymnasium -> PPO -> eval/replay
```

Verified components:

- Monza track preprocessing and persisted `TrackSpec`
- 2D physics with dynamic/aero grip, traction/braking limits, drag, rolling resistance, and speed-sensitive steering
- checkpoint/lap validity fields and QC coverage
- collision/off-track/no-progress/lap/max-step termination
- ray-cast sensors
- manual driving
- reference ghost overlay and flying-start comparison
- scripted pure-pursuit baseline
- Fast-F1 reference ghost replay
- telemetry/replay system
- Gymnasium env
- PPO train/eval/checkpoint path
- curriculum/segment spawning
- benchmark harness
- QC report/dashboard generation
- CUDA-required training smoke
- TensorBoard scalar verification

Latest validation:

- `uv run --no-sync ruff check .` -> pass
- `uv run --no-sync pyright src/f1rl` -> pass, `0 errors`
- `uv run --no-sync pytest -q` -> pass, `29 passed`

Current proof-of-concept results:

- Observation/action dimensions: `18` / `9`
- Track: `120` checkpoints, `1800` boundary segments
- Fast-F1 reference ghost: `79.662s`, `259.9 kph` average, `348.0 kph` max
- Scripted baseline: valid lap, `214.47s`
- PPO scratch initial: `0.0m`, no-progress
- PPO final checkpoint: `431.4m`, off-track
- PPO best checkpoint: `797.6m`, collision
- CUDA smoke: `device=cuda`, `vec_env=subproc`, `2048` timesteps, `51.15` training FPS

Current gap:

- PPO has a measurable learning signal but has not completed a clean or valid full lap.

Current active plan:

- `Plan.md` now describes the next serious PPO training goal.
- `LearningPlan.md` defines the strict goal-mode learning loop: TensorBoard-first, scratch PPO baseline, full-lap PPO, curriculum PPO, telemetry diagnosis, focused experiments, and a single success criterion of a valid normal-start Monza lap in `<=80.0s`.
- The completed proof-of-concept goal plan is archived at `archive/plans/Plan-rl-poc-goal-completed-20260602.md`.

## Learning Goal Run - 2026-06-02

Strict success criterion:

- PPO completes a valid normal-start Monza lap near the Fast-F1 ghost target with lap time `<=80.0s`.

Runtime expectation:

- Run iteratively for hours and hours; `8+` hours is a persistence reference, not a success criterion.
- Partial progress is not completion.

Setup and validation:

- Started TensorBoard:
  - `uv run --no-sync tensorboard --logdir artifacts --host 127.0.0.1 --port 6006`
  - Browser verified at `http://127.0.0.1:6006/?darkMode=true#timeseries`
- CUDA hardware check:
  - `uv run --no-sync python -m f1rl.hardware --json`
  - Result: CUDA available, `NVIDIA GeForce RTX 4060 Laptop GPU`, Torch `2.10.0+cu128`, CUDA `12.8`
- Validation:
  - `uv run --no-sync ruff check .` -> pass
  - `uv run --no-sync pyright src/f1rl` -> pass, `0 errors`
  - `uv run --no-sync pytest -q` -> pass, `29 passed`

Scratch control run:

- Command:
  - `uv run --no-sync python -m f1rl.train --timesteps 2048 --seed 0 --n-envs 2 --max-steps 600 --device auto --require-gpu --vec-env subproc --curriculum none --checkpoint-every 1024 --eval-every 1024 --eval-episodes 1 --telemetry selected --run-name scratch-control-smoke-goal`
- Artifact:
  - `artifacts/scratch-control-smoke-goal-20260602-165322`
- Result:
  - `device=cuda`
  - `vec_env=subproc`
  - `2048` timesteps
  - `78.6` training FPS
  - scratch initialization confirmed in `run_metadata.json`

Scratch benchmark results:

- Initial checkpoint benchmark:
  - `artifacts/benchmark-20260602-165413`
  - `20/20` no-progress terminations
  - average progress `1.71m`
  - best progress `1.71m`
  - `0` checkpoints
  - valid lap rate `0.0`
- Tiny final checkpoint benchmark:
  - `artifacts/benchmark-20260602-165806`
  - `20/20` no-progress terminations
  - average progress `0.0m`
  - best progress `0.0m`
  - `0` checkpoints
  - valid lap rate `0.0`

Interpretation:

- The scratch baseline has no usable driving behavior, which is the intended zero-knowledge control.
- Any later progress beyond no-progress / zero checkpoint behavior is attributable to training rather than ghost/scripted/imitation initialization.

Active full-lap PPO run:

- Command:
  - `uv run --no-sync python -m f1rl.train --timesteps 1000000 --seed 10 --n-envs 8 --max-steps 3600 --device auto --require-gpu --vec-env subproc --curriculum none --checkpoint-every 50000 --eval-every 50000 --eval-episodes 5 --telemetry selected --telemetry-every 10 --run-name ppo-full-scratch-goal-1m`
- Artifact:
  - `artifacts/ppo-full-scratch-goal-1m-20260602-165924`
- Early evidence:
  - run started successfully
  - SB3 confirms PPO model is using CUDA, with expected low-utilization MLP PPO warning
  - CPU vector workers are active
  - GPU memory is allocated

## Documentation Cleanup - 2026-06-02

- Rewrote `README.md` around the verified current implementation, commands, results, and next goal.
- Rewrote `Prompt.md` so future agents start from the current implemented state instead of the original rebuild request.
- Rewrote `Implement.md` as the current runbook for setup, validation, manual/replay, benchmarks, PPO training, QC, and TensorBoard.
- Replaced active `Plan.md` with the forward training plan focused on serious PPO, clean/valid laps, benchmark matrices, performance polish, and recruiting outputs.
- Archived stale completed plan:
  - `archive/plans/Plan-rl-poc-goal-completed-20260602.md`
- Preserved generated artifacts in Git because artifacts are intentionally part of the repository evidence trail for this project.

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
- Treat Fast-F1 calibration/reference tooling, richer physics, richer telemetry, ghost overlays, replay interpolation, and explicit hardware policy reporting as intentional implemented extensions. These improve calibration, debugging, and resume-quality evidence without reintroducing the old orchestration complexity.

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

## RL Completion Plan - 2026-06-02
- Archived the previous rebuild-focused `Plan.md` snapshot to `archive/plans/Plan-pre-rl-training-20260602.md`.
- Replaced `Plan.md` with the forward RL completion plan used for the proof-of-concept goal.
- The phases in that plan are now implemented and verified pragmatically: minimal strict lap validity, minimal Monza collision robustness, lightweight debug HUD, benchmark harness, real PPO training infrastructure, and curriculum/segment spawning.
- The active `Plan.md` was later replaced again with the next serious PPO training plan, and the completed proof-of-concept plan was archived at `archive/plans/Plan-rl-poc-goal-completed-20260602.md`.
- Overnight `1M+` / `3M+` serious training, plots, videos, imitation learning, continuous actions, and evolutionary search remain later phases.
- Training telemetry policy: full per-step telemetry for eval/replay/benchmark/selected training episodes; lightweight aggregate logging by default during PPO training so throughput is not crushed by JSONL writes.
- Expanded `Plan.md` into the Goal contract: completion criteria, GPU/CPU compute rules, command QC, benchmark/training artifact requirements, measurable-learning proof-of-concept bar, learning-failure triage, and blocked-report requirements now live in the plan so the Codex `/goal` prompt can stay concise.
- Added compute optimization detail to `Plan.md`: SB3 MLP PPO may not saturate GPU, so short CPU/CUDA and dummy/subproc vector-env throughput experiments are required before long runs. GPU-required commands still must abort on CPU fallback and run on CUDA when available.

## Issues
- During exact `uv sync`, the old environment had several stale dist-info folders from archived packages with missing `RECORD` files. They were removed only after verifying their paths were inside this repo's `.venv`.
- Stable-Baselines3 warns that MLP PPO on GPU may have poor utilization. The project still defaults policy training/inference to CUDA when available, matching the rebuild requirement; env stepping remains CPU-bound.

## Implemented Extensions vs Original Minimal Plan
- The active package intentionally includes `calibration` and `reference_agent` in addition to the original minimal modules.
- The CLI intentionally includes `f1-calibration` and `f1-reference-agent`.
- Physics is intentionally richer than the first plan: dynamic grip, aero grip, traction-circle style acceleration/braking/cornering limits, and speed-sensitive steering.
- `CarParams`, `StepTelemetry`, and `EpisodeSummary` intentionally contain more fields than the original interface sketch so runs can be analyzed without changing schemas later.
- The scripted baseline intentionally uses continuous controls through `step_controls`; it can still map to discrete actions when needed, but the clean-lap diagnostic benefits from finer control.
- Manual mode intentionally includes reference ghost overlay and flying-start comparison.
- Replay intentionally includes timestamp interpolation and playback speed controls.
- Fast-F1 calibration/reference ghost integration is now part of the active project because it gives a real-world Monza baseline.
- Python `>=3.11,<3.13` remains the project target; local Python 3.12 runtime/cache artifacts are acceptable.
- PPO uses CUDA when requested/available, while simulator stepping, rendering, geometry, and telemetry stay on CPU. Low MLP PPO GPU utilization is acceptable under this compute policy.
- Future evolutionary search support remains architectural only through the shared simulator/policy/telemetry/replay boundaries.

## Fast-F1 Monza Calibration - 2026-04-24
Note: the later `Physics, Telemetry, and Compute Policy - 2026-04-24` section supersedes the older simulator tuning numbers in this section. The current terminal speed estimate is `351.1 kph`.

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
- Follow-up fix: timed replay now interpolates between telemetry rows at render time. The Fast-F1 samples are about `7.7 Hz`, so direct row rendering looked jumpy and visually faster than manual mode even at correct total duration.
- Added `--speed` for intentional playback scaling and `--no-timing` for old fast-scrub behavior.
- `uv run f1-replay artifacts/reference-ghost-20260424-025933/steps.jsonl --headless` -> pass; reports `duration=79.662s`.

## Unified Rendering Contract - 2026-04-24
- Manual mode and replay now share one default renderer surface.
- Render FPS is `60`, matching the simulator physics timestep of `1/60s`.
- Renderer now uses `SimConfig.car_image` instead of randomly choosing a car sprite, so manual and replay use the same Ferrari sprite by default.
- Added a regression test to keep render FPS aligned with physics Hz.
- Added `f1-manual --ghost-reference` to overlay the Fast-F1 reference lap directly inside manual mode.
- Added `f1-manual --ghost-reference --flying-start` for a fair comparison against the reference lap, which starts around `322 kph` because it is a flying qualifying lap.
- Fixed manual timing so interactive mode advances physics from wall-clock time with fixed `1/60s` catch-up steps instead of advancing exactly one physics step per rendered frame.
- Cached sensor ray distances per simulator state to avoid repeated raycasts during observation, telemetry, and rendering.
- Changed telemetry writer to keep the JSONL file open for the run instead of opening and closing it every step.
- User-measured issue before fix: `11-12s` real time produced only about `3s` of manual-mode simulation time.

## Physics, Telemetry, and Compute Policy - 2026-04-24
### Physics
- Kept the model as a simple top-down bicycle model.
- Added dynamic grip:
  - lower mechanical grip at low speed
  - aero-grip growth with speed
  - capped max grip
- Added a simple traction-circle style limit so acceleration/braking and cornering compete for available grip.
- Added speed-sensitive steering effectiveness.
- Kept Monza top-speed calibration near the Fast-F1 target:
  - target max: `348.0 kph`
  - simulator terminal estimate: `351.1 kph`

### Compute Policy
- Added explicit CPU/GPU policy reporting to `f1-hardware-check --json`.
- CPU-owned work:
  - env stepping
  - physics
  - geometry
  - rendering
  - keyboard input
  - telemetry
  - track preprocessing
  - vector env workers
- GPU-owned work:
  - PyTorch neural policy training
  - PyTorch neural inference
- Added `f1-train --require-gpu` so training can fail fast if CUDA is not available.

### Telemetry Expansion
- Per-step telemetry now includes:
  - acceleration
  - longitudinal g
  - lateral g
  - curvature
  - throttle/brake/steering deltas
  - racing-line deviation
  - optional reference progress/speed/gap fields
- Episode summaries now include:
  - sector times
  - sector speeds
  - braking zone count/details
  - racing-line deviation aggregates
  - lateral/longitudinal g aggregates
  - steering/throttle/brake smoothness
  - ghost gap aggregates when available
  - corner entry/apex/exit speed summaries

### Scripted Baseline
- Replaced binary scripted steering with a conservative continuous pure-pursuit controller.
- Fixed a start-line centerline projection continuity issue that could snap progress to the wrong closed-loop segment.
- Validation:
  - `uv run f1-scripted --steps 18000 --no-telemetry` -> pass; `lap_complete`, `5800.8m`, `222.3s`.
  - `uv run f1-scripted --steps 3600` -> pass; telemetry run reached max steps without off-track.

### Validation
- `uv run ruff check src tests` -> pass.
- `uv run pyright src/f1rl` -> pass.
- `uv run pytest -q` -> pass (`19 passed`).
- `uv run f1-calibration` -> pass; terminal estimate `351.1 kph`.
- `uv run f1-hardware-check --json --require-gpu` -> pass; CUDA visible and compute policy reported.
- `uv run f1-reference-agent --mode ghost` -> pass; reference ghost still matches `79.662s`.
- `uv run f1-manual --headless --ghost-reference --flying-start --max-steps 60` -> pass.
- `uv run f1-train --timesteps 64 --n-envs 1 --max-steps 80 --device auto --require-gpu` -> pass; final checkpoint written on CUDA.
- `uv run f1-eval --checkpoint latest --steps 80 --device auto` -> pass.
- `uv run f1-replay artifacts/reference-ghost-20260424-064405/steps.jsonl --headless` -> pass; duration `79.662s`.

## RL Goal Contract Update - 2026-06-02
- Expanded `Plan.md` so the next Goal-mode run has a stricter implementation and experiment contract.
- Added explicit experiment policy for:
  - full-lap PPO
  - curriculum/segment PPO
  - branch comparison
  - best/failure replay extraction
  - deferred evolutionary/elite-selection training
- Clarified that vectorized PPO is the current practical version of "many agents trying things," while literal "keep the best agents" is deferred as evolutionary search.
- Added GPU/CPU optimization caveats:
  - CUDA must be used when `--require-gpu` is passed.
  - low GPU utilization can still be correct for small MLP PPO because rollout collection is CPU-bound.
  - fastest wall-clock configuration must be measured across CPU/CUDA and dummy/subprocess vector envs.
- Added detailed PPO training metadata, selected telemetry, best-model ranking, curriculum reset/reward policy, stage metrics, branch decision rules, optional overnight scale run rules, and recruiter-facing results artifact guidance.
- No code validation was run for this documentation-only update.

## Phase 1-6 Implementation Progress - 2026-06-02
### Code Changes
- Added `src/f1rl/curriculum.py` for segment curriculum stage definitions, reset sampling, and stage metadata.
- Added `src/f1rl/benchmark.py` as the headless benchmark harness for `random`, `scripted`, `ppo`, and `reference_ghost` policies.
- Added `f1-benchmark` CLI entry in `pyproject.toml`.
- Extended `MonzaSim.reset(seed, options)` with curriculum/segment spawn options:
  - `start_progress_m`
  - `start_checkpoint`
  - `start_speed_kph`
  - `segment_length_m`
  - `position_noise_m`
  - `heading_noise_deg`
  - `speed_noise_kph`
  - `curriculum_stage`
- Added lap-validity state to simulator telemetry/info:
  - `next_checkpoint_index`
  - `checkpoints_passed`
  - `missed_checkpoint_count`
  - `valid_lap`
  - `finish_crossed`
  - `segment_complete`
  - `curriculum_stage`
  - `segment_target_progress_m`
- Added ordered-checkpoint validity checks and finish-crossing requirement for strict-enough lap completion.
- Added segment completion termination for curriculum episodes while preserving normal-start full-lap eval.
- Expanded the manual HUD with compact reward/progress/ray/lap-validity/ghost/curriculum debug fields.
- Updated `MonzaEnv` to support curriculum reset sampling through the Gymnasium `reset(..., options=...)` path.
- Upgraded PPO training artifacts:
  - run metadata
  - final/checkpoint models
  - best model
  - TensorBoard logs
  - eval metrics JSONL
  - selected telemetry
  - device/vector-env throughput metadata
- Added dual eval for curriculum training:
  - `ppo_full_lap` strict normal-start eval rows
  - `ppo_curriculum_segment` segment-reset eval rows
- Added relative curriculum metrics:
  - `segment_progress_delta_m`
  - `segment_target_progress_m`
  - `segment_remaining_m`
- Updated checkpoint discovery so `--checkpoint latest` works for named training runs, not only `train-*` run IDs.
- Fixed SB3 action conversion so scalar/array model predictions are handled safely.

### Tests Added Or Expanded
- Expanded `tests/test_sim.py` for lap-validity telemetry fields and checkpoint-skip invalidation.
- Added `tests/test_curriculum.py` for curriculum sampler promotion, env reset metadata, and segment completion termination.
- Added `tests/test_benchmark.py` for benchmark JSON/CSV/JSONL and selected telemetry artifact output.
- Expanded `tests/test_policy_train_smoke.py` to verify metadata, eval metrics, best/final models, selected telemetry, and named-run latest checkpoint discovery.

### API References Rechecked
- Gymnasium Env API: `reset(seed, options)` returns `(obs, info)` and `step(action)` returns `(obs, reward, terminated, truncated, info)`.
- Stable-Baselines3 callbacks: `CheckpointCallback`, `CallbackList`, and callback `num_timesteps` behavior match the training callback implementation.
- Stable-Baselines3 vector envs: `DummyVecEnv` and `SubprocVecEnv` are the right comparison points for the throughput matrix; subprocess envs should not exceed logical CPU cores.
- uv CLI docs: `uv run --no-sync` is appropriate for repeated validation once the environment is already synced.
- References:
  - https://gymnasium.farama.org/api/env/
  - https://stable-baselines3.readthedocs.io/en/v2.8.0/guide/callbacks.html
  - https://stable-baselines3.readthedocs.io/en/v2.8.0/guide/vec_envs.html
  - https://docs.astral.sh/uv/reference/cli/#uv-run

### Validation Commands
- `uv run --no-sync ruff check .` -> pass.
- `uv run --no-sync pyright src/f1rl` -> pass.
- `uv run --no-sync pytest -q` -> pass, `25 passed`.
- `uv run --no-sync f1-hardware-check --json --require-gpu` -> pass:
  - GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
  - Torch: `2.10.0+cu128`
  - CUDA: `12.8`
  - policy/model device: `cuda`
  - simulator/physics/rendering/telemetry/vector workers: `cpu`
- `uv run --no-sync f1-scripted --steps 18000 --no-telemetry` -> pass:
  - `reason=lap_complete`
  - `completed=True`
  - `progress=5800.3m`
  - `time=214.5s`
- `uv run --no-sync f1-reference-agent --mode ghost --no-telemetry` -> pass.

### Benchmark Artifacts
- Baseline benchmark:
  - command: `uv run --no-sync f1-benchmark --policies random scripted reference_ghost --episodes 3 --max-steps 600 --telemetry selected`
  - artifact: `artifacts/benchmark-20260602-071146`
  - required files present: `config.json`, `summary.json`, `summary.csv`, `per_episode.jsonl`, selected telemetry JSONL files.
  - random:
    - completion rate: `0.0`
    - avg progress: `35.76m`
    - best progress: `52.33m`
    - crash/off-track rate: `0.333`
  - scripted under short `600`-step cap:
    - completion rate: `0.0`
    - avg progress: `252.09m`
    - termination: `max_steps`
  - reference ghost:
    - completion rate: `1.0`
    - valid lap rate: `1.0`
    - best lap: `79.662s`
    - avg/best progress: `5793.0m`
- PPO benchmark:
  - command: `uv run --no-sync f1-benchmark --policies ppo --checkpoint latest --episodes 2 --max-steps 300 --device auto --telemetry selected`
  - artifact: `artifacts/benchmark-20260602-072147`
  - purpose: verified latest named-run checkpoint loading and selected PPO benchmark telemetry.

### PPO And Curriculum Smoke Artifacts
- CUDA full-lap PPO smoke:
  - command: `uv run --no-sync f1-train --timesteps 1024 --n-envs 1 --max-steps 120 --device auto --require-gpu --checkpoint-every 512 --eval-every 512 --eval-episodes 1 --telemetry selected --run-name smoke-gpu`
  - artifact: `artifacts/smoke-gpu-20260602-071339`
  - resolved device: `cuda`
  - vector env: `dummy`
  - training FPS: `37.5`
  - eval result: tiny smoke policy went `off_track`; no learning claim.
- Checkpoint eval:
  - command: `uv run --no-sync f1-eval --checkpoint latest --steps 120 --device auto`
  - artifact: `artifacts/eval-20260602-071441`
  - result: `off_track`; checkpoint load/eval path works.
- Curriculum dual-eval smoke:
  - command: `uv run --no-sync f1-train --timesteps 2048 --n-envs 2 --max-steps 600 --device auto --require-gpu --curriculum segments --checkpoint-every 1024 --eval-every 1024 --eval-episodes 1 --telemetry selected --run-name curriculum-dual-eval-smoke`
  - artifact: `artifacts/curriculum-dual-eval-smoke-20260602-071809`
  - resolved device: `cuda`
  - vector env: `dummy`
  - training FPS: `41.0`
  - verified eval artifacts include both `ppo_full_lap` and `ppo_curriculum_segment` selected telemetry.
- Curriculum metric smoke after adding relative segment metrics:
  - command: `uv run --no-sync f1-train --timesteps 1024 --n-envs 2 --max-steps 300 --device auto --require-gpu --curriculum segments --checkpoint-every 512 --eval-every 512 --eval-episodes 1 --telemetry selected --run-name curriculum-metric-smoke`
  - artifact: `artifacts/curriculum-metric-smoke-20260602-072104`
  - resolved device: `cuda`
  - vector env: `dummy`
  - training FPS: `38.8`
  - strict full-lap eval: `no_progress`, `0.0m` best progress; no learning claim.
  - curriculum segment eval examples:
    - `curriculum_stage=A-short-low-speed`
    - `segment_progress_delta_m=16.94m`, then `24.36m`
    - `segment_remaining_m=282.72m`, then `275.27m`
    - termination reasons: `off_track`, then `no_progress`
  - interpretation: curriculum reset/eval telemetry works; 1,024 timesteps is still just a smoke run and does not prove useful learning.

### Throughput Smoke Matrix
Short `512`-timestep throughput sanity checks were run before the full long-run matrix:
- `uv run --no-sync f1-train --timesteps 512 --n-envs 2 --max-steps 120 --device cpu --vec-env dummy --telemetry none --run-name throughput-smoke-cpu-dummy`
  - artifact: `artifacts/throughput-smoke-cpu-dummy-20260602-072212`
  - result: `44.8 fps`
- `uv run --no-sync f1-train --timesteps 512 --n-envs 2 --max-steps 120 --device cuda --require-gpu --vec-env dummy --telemetry none --run-name throughput-smoke-cuda-dummy`
  - artifact: `artifacts/throughput-smoke-cuda-dummy-20260602-072236`
  - result: `56.3 fps`
- `uv run --no-sync f1-train --timesteps 512 --n-envs 2 --max-steps 120 --device cpu --vec-env subproc --telemetry none --run-name throughput-smoke-cpu-subproc`
  - artifact: `artifacts/throughput-smoke-cpu-subproc-20260602-072300`
  - result: `71.9 fps`
- `uv run --no-sync f1-train --timesteps 512 --n-envs 2 --max-steps 120 --device cuda --require-gpu --vec-env subproc --telemetry none --run-name throughput-smoke-cuda-subproc`
  - artifact: `artifacts/throughput-smoke-cuda-subproc-20260602-072327`
  - result: `80.7 fps`
- Interpretation:
  - Subprocess vector env works under Windows spawn for this repo.
  - Short smoke matrix suggests `cuda + subproc` is fastest so far.
  - This is not yet the full `4096`-timestep throughput matrix required before serious long training.
  - SB3 emits the expected warning that MLP PPO may not use GPU efficiently; CUDA-required runs still correctly resolve to CUDA.

### Current Status
- Phases 1-6 infrastructure is materially implemented and smoke-validated.
- The project now has stronger tests and command-level artifacts for lap validity, curriculum reset/segment eval, benchmark output, GPU-required PPO, checkpoint load/eval, and throughput switching.
- This is not yet Goal-complete:
  - full `4096`-timestep throughput matrix is still pending
  - short full-lap PPO baseline at serious timesteps is still pending
  - mini/serious curriculum PPO baseline at serious timesteps is still pending
  - final random/scripted/PPO/reference branch comparison is still pending
  - measurable learning evidence is not yet established
  - README first results section is still pending

## Full PPO/Curriculum Baseline Pass - 2026-06-02
### User Constraint
- For actual learning baselines, use a fresh scratch PPO initialization.
- Throughput-run checkpoints are timing artifacts only and must not be treated as learning baselines.
- PPO benchmarks should use explicit checkpoint paths from the intended scratch full-lap or scratch curriculum run, not a vague `latest` checkpoint after throughput runs.
- Updated trainer contract: every PPO training run now saves `initial_model.zip` before learning starts and, when eval is enabled, writes a timestep-0 `initial_scratch` eval row. This lets us benchmark the "knows nothing" current PPO policy directly against the trained-from-scratch `final_model.zip` and `best_model.zip`.
- Updated `Plan.md` and `README.md` so learning-result benchmarks use explicit scratch checkpoint paths. `--checkpoint latest` remains acceptable only for smoke/infrastructure validation, not for PPO learning claims.

### Full Throughput Matrix
Commands:
- `uv run --no-sync f1-train --timesteps 4096 --n-envs 8 --max-steps 600 --device cpu --vec-env dummy --telemetry none --run-name throughput-cpu-dummy`
- `uv run --no-sync f1-train --timesteps 4096 --n-envs 8 --max-steps 600 --device cuda --require-gpu --vec-env dummy --telemetry none --run-name throughput-cuda-dummy`
- `uv run --no-sync f1-train --timesteps 4096 --n-envs 8 --max-steps 600 --device cpu --vec-env subproc --telemetry none --run-name throughput-cpu-subproc`
- `uv run --no-sync f1-train --timesteps 4096 --n-envs 8 --max-steps 600 --device cuda --require-gpu --vec-env subproc --telemetry none --run-name throughput-cuda-subproc`

Results:
- `artifacts/throughput-cpu-dummy-20260602-072659`
  - device: CPU
  - vec env: dummy
  - training FPS: `43.6`
  - env steps/sec: `349.1`
- `artifacts/throughput-cuda-dummy-20260602-072847`
  - device: CUDA
  - vec env: dummy
  - training FPS: `41.1`
  - env steps/sec: `328.9`
- `artifacts/throughput-cpu-subproc-20260602-073045`
  - device: CPU
  - vec env: subproc
  - training FPS: `218.5`
  - env steps/sec: `1747.7`
- `artifacts/throughput-cuda-subproc-20260602-073124`
  - device: CUDA
  - vec env: subproc
  - training FPS: `224.5`
  - env steps/sec: `1796.4`
- Interpretation:
  - `subproc` is materially faster than dummy vector env on this Windows machine.
  - CUDA + subproc is the fastest required-GPU option measured so far.
  - SB3 still emits the expected MLP PPO warning; CUDA-required mode nevertheless resolves to CUDA correctly.

### Scratch Full-Lap PPO Baseline
Command:
- `uv run --no-sync f1-train --timesteps 250000 --n-envs 8 --max-steps 3600 --device auto --require-gpu --vec-env subproc --checkpoint-every 25000 --eval-every 25000 --eval-episodes 5 --telemetry selected --run-name ppo-full-lap-baseline-scratch`

Artifact:
- `artifacts/ppo-full-lap-baseline-scratch-20260602-073219`

Result:
- fresh PPO initialization
- device: CUDA
- vec env: subproc
- timesteps: `250000`
- training FPS: `272.4`
- final strict eval:
  - completion rate: `0.0`
  - avg/best progress: `0.0m`
  - termination: `no_progress`
- Interpretation:
  - full-lap-from-scratch PPO collapsed to a no-progress local optimum.
  - This is the expected failure evidence justifying curriculum.

Benchmark:
- command: `uv run --no-sync f1-benchmark --policies ppo --checkpoint "artifacts\ppo-full-lap-baseline-scratch-20260602-073219\final_model.zip" --episodes 20 --max-steps 3600 --device auto --telemetry selected`
- artifact: `artifacts/benchmark-20260602-074813`
- summary:
  - episodes: `20`
  - completion rate: `0.0`
  - valid lap rate: `0.0`
  - crash/off-track rate: `0.0`
  - avg progress: `0.0m`
  - best progress: `0.0m`
  - avg reward: `-10.0`
  - termination reasons: `no_progress: 20`

### Scratch Curriculum PPO Baseline
Command:
- `uv run --no-sync f1-train --timesteps 250000 --n-envs 8 --max-steps 1200 --device auto --require-gpu --vec-env subproc --curriculum segments --checkpoint-every 25000 --eval-every 25000 --eval-episodes 5 --telemetry selected --run-name ppo-curriculum-baseline-scratch`

Artifact:
- `artifacts/ppo-curriculum-baseline-scratch-20260602-074919`

Result:
- fresh PPO initialization
- device: CUDA
- vec env: subproc
- timesteps: `250000`
- training FPS: `181.7`
- selected eval checkpoints:
  - `25000`: full-lap progress `0.0m`; segment progress delta `153.1m`
  - `150000`: full-lap progress `370.9m`; segment completion rate `0.4`; segment progress delta `229.5m`
  - `225000`: full-lap progress `431.4m`; segment completion rate `0.2`; segment progress delta `180.5m`
  - `250000`: full-lap progress `431.0m`; segment completion rate `0.0`; segment progress delta `169.7m`
- Interpretation:
  - curriculum produced measurable learning evidence.
  - the policy can move from normal start into the first sector, but it is still unstable and does not complete a lap.
  - later checkpoints partly regress on segment completion, so best-model tracking matters.

Final checkpoint benchmark:
- command: `uv run --no-sync f1-benchmark --policies ppo --checkpoint "artifacts\ppo-curriculum-baseline-scratch-20260602-074919\final_model.zip" --episodes 20 --max-steps 3600 --device auto --telemetry selected`
- artifact: `artifacts/benchmark-20260602-081303`
- summary:
  - episodes: `20`
  - completion rate: `0.0`
  - valid lap rate: `0.0`
  - crash/off-track rate: `1.0`
  - avg progress: `431.0m`
  - best progress: `431.0m`
  - avg reward: `9.5`
  - termination reasons: `off_track: 20`

Best-model benchmark:
- command: `uv run --no-sync f1-benchmark --policies ppo --checkpoint "artifacts\ppo-curriculum-baseline-scratch-20260602-074919\best_model.zip" --episodes 20 --max-steps 3600 --device auto --telemetry selected`
- artifact: `artifacts/benchmark-20260602-081641`
- summary:
  - episodes: `20`
  - completion rate: `0.0`
  - valid lap rate: `0.0`
  - crash/off-track rate: `0.0`
  - avg progress: `370.9m`
  - best progress: `370.9m`
  - avg reward: `19.7`
  - termination reasons: `no_progress: 20`

### Random/Scripted/Reference Anchors
Command:
- `uv run --no-sync f1-benchmark --policies random scripted reference_ghost --episodes 3 --max-steps 18000 --telemetry selected --telemetry-every 3`

Artifact:
- `artifacts/benchmark-20260602-082103`

Results:
- random:
  - episodes: `3`
  - completion rate: `0.0`
  - crash/off-track rate: `1.0`
  - avg progress: `100.6m`
  - best progress: `167.7m`
  - avg reward: `-17.0`
- scripted:
  - episodes: `3`
  - completion rate: `1.0`
  - valid lap rate: `1.0`
  - avg/best progress: `5800.3m`
  - best lap: `214.47s`
  - avg reward: `564.0`
- reference ghost:
  - episodes: `3`
  - completion rate: `1.0`
  - valid lap rate: `1.0`
  - progress: `5793.0m`
  - lap: `79.662s`

### First Results Summary
- Full-lap scratch PPO did not learn: `0.0m` avg/best progress over 20 strict eval episodes.
- Curriculum scratch PPO did learn something measurable:
  - normal-start strict eval improved to `431.0m` avg/best progress over 20 episodes
  - segment eval reached `40%` completion at the best eval checkpoint
  - segment progress delta reached `229.5m` at the best eval checkpoint
- This satisfies the proof-of-concept learning-signal requirement, but not the clean-lap requirement.
- Next best engineering step is not blind overnight scaling. The policy is overdriving/off-tracking early, so the next run should tune curriculum/reward/action behavior before a longer overnight run:
  - reduce early curriculum spawn speed or noise
  - increase stage A stability before promotion
  - consider no-progress penalty timing/magnitude
  - consider a mild off-track/collision shaping adjustment
  - inspect selected telemetry around the `431m` failure point

### README Update
- Added benchmark command examples.
- Added first PPO results table with throughput, baselines, PPO full-lap scratch, and PPO curriculum scratch.
- Clearly states that the RL agent has not completed a full lap yet.

### Final Validation For This Pass
- `uv run --no-sync ruff check .` -> pass.
- `uv run --no-sync pyright src/f1rl` -> pass.
- `uv run --no-sync pytest -q` -> pass, `25 passed`.

### Remaining Goal Work
- The proof-of-concept learning signal exists, so the project is no longer blocked at "does PPO learn anything?"
- The agent is still not good enough for a completed goal:
  - no PPO clean lap
  - no PPO valid lap
  - final curriculum checkpoint goes off-track around `431m`
  - best curriculum checkpoint avoids off-track but times out around `371m`
- Next implementation/training pass should tune the curriculum/reward before overnight scaling:
  - reduce stage A speed/noise or add an easier A0 stage
  - keep stage A longer before progression
  - add stage-level metric aggregation to benchmark output if useful
  - inspect selected telemetry around `370-431m`
  - rerun a shorter tuned curriculum slice before any `1M+` overnight run

## Scratch PPO Baseline Contract And Balanced Curriculum Pass - 2026-06-02

### User Constraint
- Baselines must use the current PPO agent from scratch, with no warm start, no reference ghost pretraining, and no inherited checkpoint.
- The untrained PPO policy must be visible as its own baseline so progress is measured against an agent that has no driving knowledge.

### Implementation Updates
- `src/f1rl/train.py` now saves:
  - `initial_model.zip` at the run root
  - `checkpoints/initial_model.zip`
  - `final_model.zip`
  - `checkpoints/final_model.zip`
- When eval is enabled, training now writes a timestep-0 eval row:
  - `phase="initial_scratch"`
  - `scratch_initial_policy=true`
- `run_metadata.json` now records:
  - `scratch_initialization=true`
  - `initial_checkpoint`
  - PPO hyperparameters
- PPO now uses a small entropy coefficient:
  - `ent_coef=0.02`
  - reason: keep the discrete policy exploring instead of collapsing immediately to no-progress.
- Reward rebalance:
  - collision penalty: `-60.0`
  - off-track penalty: `-60.0`
  - no-progress penalty: `-90.0`
  - reason: sitting still should be worse than a short failed exploratory attempt, but crashing should still not be profitable.
- Early curriculum stages were softened:
  - `A0-short-control`: `120m`, `10-30 kph`
  - `A1-short-low-speed`: `240m`, `20-55 kph`
  - `B-medium-control`: `500m`, `35-85 kph`

### Validation
- `uv run --no-sync ruff check .` -> pass.
- `uv run --no-sync pyright src/f1rl` -> pass.
- `uv run --no-sync pytest -q` -> pass, `25 passed`.
- The smoke test now verifies `initial_model.zip` and the timestep-0 `initial_scratch` eval row.

### Negative Tuning Result
- Run: `artifacts/ppo-curriculum-tuned-scratch-20260602-084051`
- Command:
  - `uv run --no-sync f1-train --timesteps 150000 --n-envs 8 --max-steps 1200 --device auto --require-gpu --vec-env subproc --curriculum segments --checkpoint-every 25000 --eval-every 25000 --eval-episodes 5 --telemetry selected --run-name ppo-curriculum-tuned-scratch`
- Result:
  - CUDA/subproc training completed at `206.5` FPS.
  - Stronger `-75` collision/off-track penalties with only `-40` no-progress penalty caused the policy to prefer no-progress after 50k.
  - Final 150k eval: full-lap progress `0.0m`; segment progress delta `78.9m`; segment completion `0.0`.
- Decision:
  - Do not scale this configuration overnight.
  - Use the balanced reward where no-progress is worse than a short failed attempt.

### Balanced Scratch Curriculum Run
- Run: `artifacts/ppo-curriculum-balanced-scratch-20260602-085522`
- Command:
  - `uv run --no-sync f1-train --timesteps 150000 --n-envs 8 --max-steps 1200 --device auto --require-gpu --vec-env subproc --curriculum segments --checkpoint-every 25000 --eval-every 25000 --eval-episodes 5 --telemetry selected --run-name ppo-curriculum-balanced-scratch`
- Result:
  - CUDA/subproc training completed at `139.8` FPS.
  - Initial scratch PPO eval was written before training.
  - Eval curve:
    - `0`: full-lap `0.0m`, reward `-90.0`, segment delta `4.8m`, segment completion `0.0`
    - `25k`: full-lap `647.5m`, reward `-8.2`, segment delta `81.6m`, segment completion `0.0`
    - `50k`: full-lap `431.4m`, reward `-25.5`, segment delta `110.6m`, segment completion `0.8`
    - `75k`: full-lap `431.4m`, reward `-25.5`, segment delta `99.8m`, segment completion `0.4`
    - `100k`: full-lap `431.4m`, reward `-25.5`, segment delta `115.7m`, segment completion `0.8`
    - `125k`: full-lap `431.4m`, reward `-25.5`, segment delta `114.3m`, segment completion `0.8`
    - `150k`: full-lap `797.6m`, reward `3.8`, segment delta `120.6m`, segment completion `1.0`
- Interpretation:
  - This is the strongest PPO proof-of-concept so far.
  - Scratch PPO improved from `0.0m` to `797.6m` strict normal-start progress.
  - Curriculum segment eval reached `100%` completion on the current easiest stage.
  - PPO still does not complete a clean or valid full lap.

### Explicit Checkpoint Benchmarks
- Initial scratch PPO:
  - command: `uv run --no-sync f1-benchmark --policies ppo --checkpoint artifacts\ppo-curriculum-balanced-scratch-20260602-085522\initial_model.zip --episodes 20 --max-steps 3600 --device auto --telemetry selected`
  - artifact: `artifacts/benchmark-20260602-091409`
  - result: completion `0.0`, valid `0.0`, crash/off-track `0.0`, avg/best progress `0.0m`, avg reward `-90.0`, termination `no_progress:20`
- Final trained PPO:
  - command: `uv run --no-sync f1-benchmark --policies ppo --checkpoint artifacts\ppo-curriculum-balanced-scratch-20260602-085522\final_model.zip --episodes 20 --max-steps 3600 --device auto --telemetry selected`
  - artifact: `artifacts/benchmark-20260602-091449`
  - result: completion `0.0`, valid `0.0`, crash/off-track `1.0`, avg/best progress `431.4m`, avg reward `-25.5`, termination `off_track:20`
- Best trained PPO:
  - command: `uv run --no-sync f1-benchmark --policies ppo --checkpoint artifacts\ppo-curriculum-balanced-scratch-20260602-085522\best_model.zip --episodes 20 --max-steps 3600 --device auto --telemetry selected`
  - artifact: `artifacts/benchmark-20260602-091800`
  - result: completion `0.0`, valid `0.0`, crash/off-track `1.0`, avg/best progress `797.6m`, avg reward `3.8`, termination `collision:20`

### Refreshed Non-PPO Baselines
- Random:
  - command: `uv run --no-sync f1-benchmark --policies random --episodes 3 --max-steps 600 --telemetry selected --telemetry-every 3`
  - artifact: `artifacts/benchmark-20260602-093506`
  - result: completion `0.0`, valid `0.0`, crash/off-track `0.33`, avg progress `35.8m`, best progress `52.3m`, avg reward `-17.1`
- Scripted:
  - command: `uv run --no-sync f1-benchmark --policies scripted --episodes 1 --max-steps 18000 --telemetry selected --telemetry-every 1`
  - artifact: `artifacts/benchmark-20260602-093545`
  - result: completion `1.0`, valid `1.0`, avg/best progress `5800.3m`, avg reward `564.0`, best lap `214.47s`
- Fast-F1 reference ghost:
  - command: `uv run --no-sync f1-benchmark --policies reference_ghost --episodes 1 --max-steps 18000 --telemetry selected --telemetry-every 1`
  - artifact: `artifacts/benchmark-20260602-093941`
  - result: completion `1.0`, valid `1.0`, progress `5793.0m`, avg reward `563.4`, lap `79.662s`
- Note:
  - A combined 3-episode `random scripted reference_ghost` benchmark was stopped because it was taking too long for the incremental value.
  - Smaller targeted baseline benchmarks completed and are the current refreshed baseline artifacts.

### Current Status
- The scratch PPO baseline protocol is implemented.
- The project now has a direct untrained-vs-trained PPO comparison from the same current codepath.
- The proof-of-concept RL learning signal is solid:
  - initial PPO: `0.0m`
  - best trained PPO: `797.6m`
  - random: `35.8m` average over the refreshed small baseline
  - segment completion reached `100%` in training eval
- The PPO policy still fails by collision/off-track and has not completed a valid lap.
- Next best step is full-lap fine-tuning from the best curriculum checkpoint or a longer curriculum run with stricter transfer-to-normal-start evaluation.

## QC And Manual-Review Hardening Pass - 2026-06-02

### Implementation Updates
- Added `checkpoint_lateral_limit_m=24.0` to `SimConfig`.
- Checkpoint progression now invalidates the lap if a checkpoint threshold is crossed while the car is too far from the centerline corridor.
- Expanded benchmark aggregate summaries with:
  - `finish_crossed_rate`
  - `invalid_lap_rate`
  - `avg_checkpoints_passed`
  - `avg_missed_checkpoint_count`
  - `max_missed_checkpoint_count`
  - `avg_steps`
  - `max_steps_observed`
  - `avg_elapsed_time_s`
- Added `f1rl.qc` / `f1-qc` quality-control report generator.
- Added `tests/test_qc.py`.

### QC Command
- Command:
  - `uv run --no-sync python -m f1rl.qc --telemetry artifacts\benchmark-20260602-091800\selected_telemetry\ppo-episode-000-steps.jsonl --run-scripted --scripted-steps 18000`
- Artifact:
  - `artifacts/qc-20260602-130020`
- Files:
  - `qc_report.json`
  - `qc_report.md`
  - `telemetry_dashboard.html`
  - `manual_qc_checklist.md`
- Entry point note:
  - `f1-qc` is declared in `pyproject.toml`.
  - `uv run --no-sync python -m f1rl.qc --help` works.
  - Historical issue: `uv run f1-qc --help` previously attempted to rebuild/reinstall the package and failed on Windows with stale `.dist-info` access denied. The environment was later repaired; module commands remain preferred for repeatable local runs.

### QC Results
- Track:
  - checkpoints: `120`
  - boundary segments: `1800`
  - observation/action dimensions: `18` / `9`
  - sensor rays: `7`
  - checkpoint lateral validity limit: `24.0m`
- Lap validity:
  - huge progress jump invalidates lap: `true`
  - wide checkpoint crossing invalidates lap: `true`
  - forced full-lap skip produced missed checkpoint count: `119`
  - forced wide checkpoint crossing produced missed checkpoint count: `1`
- Physics calibration:
  - Fast-F1 reference lap: `79.662s`
  - Fast-F1 reference max speed: `348.0kph`
  - simulator estimated terminal speed: `351.1kph`
  - simulator 8s full-throttle speed: `336.1kph`
  - simulator braking `330->150kph`: `66.7m`
- Scripted sanity:
  - termination: `lap_complete`
  - completed lap: `true`
  - valid lap: `true`
  - finish crossed: `true`
  - elapsed time: `214.47s`
  - progress: `5800.3m`
  - missed checkpoints: `0`
- PPO telemetry dashboard source:
  - `artifacts\benchmark-20260602-091800\selected_telemetry\ppo-episode-000-steps.jsonl`
  - steps: `678`
  - termination: `collision`
  - best progress: `797.6m`
  - checkpoints passed: `16`
  - missed checkpoints: `0`
  - max speed: `346.1kph`
  - max lateral g: `3.06`
  - min ray distance: `0.07m`

### Status After This Pass
- Lap/checkpoint validity is stronger and has explicit QC coverage.
- Benchmark artifacts now include enough lap-validity fields to debug invalid laps without video.
- Physics has a repeatable calibration report against Fast-F1 reference targets.
- Telemetry has a lightweight local HTML dashboard for immediate inspection.
- Manual testing remains intentionally human-in-the-loop using `manual_qc_checklist.md`.
- PPO still has not completed a clean or valid lap; that belongs to the next training goal.

## Reference Ghost Benchmark Checkpoint Fix - 2026-06-02

### Issue
- Fresh reference ghost benchmark summaries reported `avg_checkpoints_passed: 0.0` even though the reference ghost completed a valid lap and the standalone reference episode summary reported `119` checkpoints passed.
- Root cause: reference ghost telemetry used modulo checkpoint index for `checkpoints_passed`; at the finish line the current checkpoint index wraps back to `0`.
- The benchmark episode reducer also trusted only the final telemetry row for checkpoint count, which made it vulnerable to any finish-line wrap.

### Fix
- Updated reference ghost telemetry so:
  - `checkpoints_passed` records the clamped passed-checkpoint count.
  - final reference ghost telemetry reports `checkpoints_passed=119`.
  - final `next_checkpoint_index` reports `120` instead of wrapping to `1`.
- Updated benchmark episode metrics to use the maximum `checkpoints_passed` and `missed_checkpoint_count` seen across the full episode.
- Added focused tests for:
  - benchmark aggregation when the final row wraps checkpoint index to zero
  - reference ghost telemetry preserving finish checkpoint count

### Validation
- `uv run --no-sync pytest tests\test_benchmark.py tests\test_reference_agent.py -q`
  - passed: `8`
- `uv run --no-sync ruff check src\f1rl\benchmark.py src\f1rl\reference_agent.py tests\test_benchmark.py tests\test_reference_agent.py`
  - passed
- `uv run --no-sync pyright src/f1rl`
  - passed: `0 errors`
- `uv run --no-sync pytest -q`
  - passed: `29`
- Fresh benchmark command:
  - `uv run --no-sync python -m f1rl.benchmark --policies reference_ghost --episodes 1 --max-steps 18000 --telemetry selected --telemetry-every 1`
- Fresh benchmark artifact:
  - `artifacts\benchmark-20260602-153332`
- Confirmed summary fields:
  - `avg_checkpoints_passed: 119.0`
  - `avg_missed_checkpoint_count: 0.0`
  - `completion_rate: 1.0`
  - `valid_lap_rate: 1.0`
  - `finish_crossed_rate: 1.0`
  - `best_lap_time_s: 79.662`
- Confirmed final reference telemetry fields:
  - `checkpoint_index: 0`
  - `next_checkpoint_index: 120`
  - `checkpoints_passed: 119`
  - `missed_checkpoint_count: 0`
  - `valid_lap: true`
  - `finish_crossed: true`
  - `termination_reason: lap_complete`

## PPO Benchmark, GPU Smoke, And TensorBoard Verification - 2026-06-02

### PPO Checkpoint Benchmarks
- Initial scratch checkpoint command:
  - `uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\ppo-curriculum-balanced-scratch-20260602-085522\initial_model.zip" --episodes 20 --max-steps 3600 --device auto --telemetry selected`
- Artifact:
  - `artifacts\benchmark-20260602-154222`
- Result:
  - episodes: `20`
  - completion rate: `0.0`
  - valid lap rate: `0.0`
  - average/best progress: `0.0m` / `0.0m`
  - checkpoints passed: `0.0`
  - termination reasons: `no_progress: 20`
  - average elapsed time: `3.0s`
  - average reward: `-90.0`

- Final checkpoint command:
  - `uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\ppo-curriculum-balanced-scratch-20260602-085522\final_model.zip" --episodes 20 --max-steps 3600 --device auto --telemetry selected`
- Artifact:
  - `artifacts\benchmark-20260602-154253`
- Result:
  - episodes: `20`
  - completion rate: `0.0`
  - valid lap rate: `0.0`
  - average/best progress: `431.4m` / `431.4m`
  - checkpoints passed: `8.0`
  - missed checkpoints: `0.0`
  - termination reasons: `off_track: 20`
  - average elapsed time: `7.18s`
  - average reward: `-25.49`
  - max speed in episode 0: `331.8kph`

- Best checkpoint command:
  - `uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\ppo-curriculum-balanced-scratch-20260602-085522\best_model.zip" --episodes 20 --max-steps 3600 --device auto --telemetry selected`
- Artifact:
  - `artifacts\benchmark-20260602-154555`
- Result:
  - episodes: `20`
  - completion rate: `0.0`
  - valid lap rate: `0.0`
  - average/best progress: `797.6m` / `797.6m`
  - checkpoints passed: `16.0`
  - missed checkpoints: `0.0`
  - termination reasons: `collision: 20`
  - average elapsed time: `11.3s`
  - average reward: `3.81`
  - max speed in episode 0: `346.1kph`

### Interpretation
- The PPO checkpoint ordering is coherent:
  - scratch checkpoint has no movement and terminates by no-progress
  - final checkpoint learns forward progress but leaves the track at `431.4m`
  - best checkpoint reaches `797.6m`, passes `16` checkpoints, and crashes
- This remains proof of a learning signal, not a solved lap-driving policy.
- Full-lap PPO still needs longer training and/or curriculum-to-full-lap transfer work.

### GPU Training Smoke
- Command:
  - `uv run --no-sync python -m f1rl.train --timesteps 2048 --n-envs 2 --max-steps 600 --device auto --require-gpu --vec-env subproc --curriculum segments --checkpoint-every 1024 --eval-every 1024 --eval-episodes 1 --telemetry selected --run-name manual-qc-smoke`
- Artifact:
  - `artifacts\manual-qc-smoke-20260602-155050`
- Terminal result:
  - `device=cuda`
  - `vec_env=subproc`
  - `fps=51.2`
- Metadata result:
  - timesteps: `2048`
  - envs: `2`
  - training fps: `51.15`
  - env steps/sec: `102.31`
  - compute policy recorded CUDA for neural training/inference and CPU for env/physics/rendering/telemetry
- Required files confirmed:
  - `initial_model.zip`
  - `final_model.zip`
  - `checkpoints\final_model.zip`
  - `eval\eval_metrics.jsonl`
  - `eval\best_eval_summary.json`
  - TensorBoard event file under `tensorboard\ppo_1`
- Smoke eval result at `2048` timesteps:
  - full-lap mean reward: `-25.49`
  - full-lap mean best progress: `431.4m`
  - segment completion rate: `1.0`
  - segment progress delta: `120.7m`

### TensorBoard
- Initial TensorBoard attempt failed because the venv's `setuptools` install was missing `pkg_resources`.
- Environment repair:
  - stopped stale `uv.exe` metadata checks
  - removed stale `setuptools` package/metadata inside this repo's `.venv`
  - installed `setuptools==80.9.0`
  - verified `pkg_resources` imports
- TensorBoard command:
  - `uv run --no-sync tensorboard --logdir "C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts" --host 127.0.0.1 --port 6006`
- TensorBoard status:
  - started at `http://127.0.0.1:6006/`
  - ran with reduced feature set because TensorFlow is not installed, which is acceptable for scalar inspection
- Browser verification:
  - Time Series tab loaded
  - run list included `manual-qc-smoke-20260602-155050\tensorboard\ppo_1`
  - scalar cards rendered, including `rollout/ep_len_mean` and `rollout/ep_rew_mean`
  - scalar endpoint exposed tags:
    - `time/fps`
    - `rollout/ep_len_mean`
    - `rollout/ep_rew_mean`
    - `train/approx_kl`
    - `train/clip_fraction`
    - `train/clip_range`
    - `train/entropy_loss`
    - `train/explained_variance`
    - `train/learning_rate`
    - `train/loss`
    - `train/policy_gradient_loss`
    - `train/value_loss`
- Confirmed scalar datapoints for `manual-qc-smoke-20260602-155050\tensorboard\ppo_1`:
  - `time/fps`: steps `256` through `2048`, final value `51.0`
  - `rollout/ep_len_mean`: steps `1024` through `2048`, final value `536.0`
  - `rollout/ep_rew_mean`: improved from `-55.77` to `-15.17`
  - `train/loss` and `train/value_loss` had non-empty datapoints
- Screenshot saved:
  - `artifacts\tensorboard-qc-20260602-155050.png`
- TensorBoard process was stopped after verification.

## 2026-06-02 Long PPO Goal Run

### Strict Completion Criterion
- The only success criterion is: PPO completes a valid normal-start Monza lap near the Fast-F1 ghost target, with lap time `<=80.0s`.
- The `8+` hour runtime target is a persistence expectation, not a success criterion.
- Partial progress, higher reward, longer episodes, checkpoints passed, slow valid laps, or elapsed training time do not count as completion.

### TensorBoard
- Command:
  - `uv run --no-sync tensorboard --logdir artifacts --host 127.0.0.1 --port 6006`
- Status:
  - running at `http://127.0.0.1:6006/`
  - in-app browser verified the active run appears in the TensorBoard run list
  - active run: `ppo-full-scratch-goal-1m-20260602-165924\tensorboard\ppo_1`

### Pre-Run Validation
- Hardware check confirmed CUDA:
  - device: `NVIDIA GeForce RTX 4060 Laptop GPU`
  - torch: `2.10.0+cu128`
  - CUDA: `12.8`
- Validation commands passed:
  - `uv run --no-sync ruff check .`
  - `uv run --no-sync pyright src/f1rl`
  - `uv run --no-sync pytest -q`

### Scratch Control Baseline
- Command:
  - `uv run --no-sync python -m f1rl.train --timesteps 2048 --seed 0 --n-envs 2 --max-steps 600 --device auto --require-gpu --vec-env subproc --curriculum none --checkpoint-every 1024 --eval-every 1024 --eval-episodes 1 --telemetry selected --run-name scratch-control-smoke-goal`
- Artifact:
  - `artifacts\scratch-control-smoke-goal-20260602-165322`
- Result:
  - device: `cuda`
  - vec env: `subproc`
  - training fps: `78.6`
  - initial checkpoint benchmark: no useful movement, `0` checkpoints, no-progress termination
  - final checkpoint benchmark: no useful movement, `0` checkpoints, no-progress termination
- Interpretation:
  - confirms the serious baseline starts from scratch/random PPO behavior, not ghost/scripted/imitation knowledge.

### Serious Full-Lap Scratch PPO Run
- Command:
  - `uv run --no-sync python -m f1rl.train --timesteps 1000000 --seed 10 --n-envs 8 --max-steps 3600 --device auto --require-gpu --vec-env subproc --curriculum none --checkpoint-every 50000 --eval-every 50000 --eval-episodes 5 --telemetry selected --telemetry-every 10 --run-name ppo-full-scratch-goal-1m`
- Artifact:
  - `artifacts\ppo-full-scratch-goal-1m-20260602-165924`
- Metadata:
  - device: `cuda`
  - `--require-gpu`: enabled
  - vec env: `subproc`
  - env workers: `8`
  - curriculum: `none`
  - initial scratch policy: `true`

### Full-Lap PPO Early Results
- Initial scratch eval at timestep `0`:
  - mean reward: `-89.71`
  - mean best progress: `3.65m`
  - checkpoints passed: `0`
  - termination: no-progress
- Training scalar at timestep `49,152`:
  - rollout reward mean: `-51.02`
  - rollout episode length mean: `492.15`
  - fps: `208`
- Deterministic eval at timestep `50,000`:
  - mean reward: `-25.49`
  - mean best progress: `431.38m`
  - checkpoints passed: `8`
  - average speed: `216.1kph`
  - max speed: `331.8kph`
  - termination: off-track
  - interpretation: policy learned high-throttle forward motion, but not enough braking/turning for the first major section.
- Training scalar at timestep `100,352`:
  - rollout reward mean: `-46.07`
  - rollout episode length mean: `789.21`
  - fps: `198`
- Deterministic eval at timestep `100,000`:
  - mean reward: `-90.0`
  - mean best progress: `0.0m`
  - checkpoints passed: `0`
  - termination: no-progress
  - interpretation: full-lap scratch PPO is showing stochastic rollout improvement but unstable deterministic eval behavior. Continue to the next eval boundary before deciding whether to cut over to curriculum.
- Training scalar at timestep `155,648`:
  - rollout reward mean: `-37.98`
  - rollout episode length mean: `1298.06`
  - fps: `210`
- Deterministic eval at timestep `150,000`:
  - mean reward: `-90.0`
  - mean best progress: `0.0m`
  - checkpoints passed: `0`
  - termination: no-progress
  - interpretation: deterministic eval failed for two consecutive scheduled evals after the `50,000` throttle-only improvement. The run was stopped early and the next experiment moved to segment curriculum from scratch.

### Segment Curriculum Scratch PPO Run
- Command:
  - `uv run --no-sync python -m f1rl.train --timesteps 1000000 --seed 20 --n-envs 8 --max-steps 3600 --device auto --require-gpu --vec-env subproc --curriculum segments --checkpoint-every 50000 --eval-every 50000 --eval-episodes 5 --telemetry selected --telemetry-every 10 --run-name ppo-segments-scratch-goal-1m`
- Artifact:
  - `artifacts\ppo-segments-scratch-goal-1m-20260602-171306`
- Status:
  - running
  - device guarded with `--require-gpu`
  - CUDA memory allocated on the RTX 4060 Laptop GPU
  - TensorBoard remains running on `http://127.0.0.1:6006/`

### Segment Curriculum Early Results
- Initial eval at timestep `0`:
  - full-lap mean reward: `-90.0`
  - full-lap mean best progress: `0.0m`
  - full-lap checkpoints passed: `0`
  - full-lap termination: no-progress
  - segment completion rate: `0.0`
  - mean segment progress delta: `55.92m`
  - interpretation: scratch full-lap behavior still has no movement; segment eval spawns into local curriculum stages and must not be counted as normal-start lap progress.
- Training scalar at timestep `50,176`:
  - rollout reward mean: `-34.62`
  - rollout episode length mean: `538.68`
  - fps: `179`
- Deterministic eval at timestep `50,000`:
  - full-lap mean reward: `-56.08`
  - full-lap mean best progress: `49.01m`
  - full-lap checkpoints passed: `1`
  - full-lap termination: off-track
  - segment completion rate: `0.0`
  - mean segment progress delta: `49.89m`
  - segment stage: `A0-short-control`
  - segment failure mode: off-track after partial local progress
  - interpretation: curriculum has not solved the easiest segment yet; current gap is local steering/control stability.
- Training scalar near timestep `100,000`:
  - rollout reward mean: `-19.29`
  - rollout episode length mean: `956.93`
  - fps: `198`
- Deterministic eval at timestep `100,000`:
  - full-lap mean reward: `-90.0`
  - full-lap mean best progress: `0.0m`
  - full-lap checkpoints passed: `0`
  - full-lap termination: no-progress
  - segment completion rate: `0.4`
  - mean segment progress delta: `70.20m`
  - sample completed segment:
    - stage: `A0-short-control`
    - termination: segment_complete
    - segment progress delta: `120.88m`
    - average/max speed: `139.8kph` / `237.4kph`
  - interpretation: curriculum is now learning local segment control, but the learned behavior has not transferred to deterministic normal-start full-lap driving.
- Deterministic eval at timestep `150,000`:
  - full-lap mean reward: `-90.0`
  - full-lap mean best progress: `0.0m`
  - full-lap checkpoints passed: `0`
  - full-lap termination: no-progress
  - segment completion rate: `0.0`
  - mean segment progress delta: `16.98m`
  - interpretation: segment performance regressed after the `100,000` checkpoint. The likely issue is that the default curriculum promotes stages by reset count instead of measured stage mastery, so the run was stopped and replaced with a focused A0-only experiment.

### Trainer Controls Added
- Added CLI controls for focused experiments:
  - `--curriculum-stage-count`
  - `--curriculum-promotion-resets`
  - `--n-steps`
  - `--batch-size`
  - `--n-epochs`
  - `--learning-rate`
  - `--gamma`
  - `--ent-coef`
- Validation after code change:
  - `uv run --no-sync ruff check .` passed
  - `uv run --no-sync pyright src/f1rl` passed
  - `uv run --no-sync pytest -q` passed, `30` tests

### Focused A0-Only Scratch PPO Run
- Command:
  - `uv run --no-sync python -m f1rl.train --timesteps 500000 --seed 30 --n-envs 8 --max-steps 3600 --device auto --require-gpu --vec-env subproc --curriculum segments --curriculum-stage-count 1 --curriculum-promotion-resets 1000000 --n-steps 256 --batch-size 256 --n-epochs 6 --learning-rate 0.0003 --gamma 0.995 --ent-coef 0.005 --checkpoint-every 25000 --eval-every 25000 --eval-episodes 5 --telemetry selected --telemetry-every 10 --run-name ppo-a0-focus-scratch-goal-500k`
- Artifact:
  - `artifacts\ppo-a0-focus-scratch-goal-500k-20260602-173212`
- Metadata:
  - device: `cuda`
  - `--require-gpu`: enabled
  - curriculum stage count: `1`
  - promotion resets: `1000000`
  - only active curriculum stage: `A0-short-control`
  - PPO hyperparameters:
    - `n_steps`: `256`
    - `batch_size`: `256`
    - `n_epochs`: `6`
    - `learning_rate`: `0.0003`
    - `gamma`: `0.995`
    - `ent_coef`: `0.005`
- Initial eval at timestep `0`:
  - full-lap mean reward: `-6.87`
  - full-lap mean best progress: `664.08m`
  - full-lap checkpoints passed: `13`
  - full-lap termination: off-track
  - segment completion rate: `1.0`
  - mean segment progress delta: `120.29m`
  - interpretation: this seed produced an unusually active random initial deterministic policy; it is a baseline, not learned behavior.
- Deterministic eval at timestep `25,000`:
  - full-lap mean reward: `-90.0`
  - full-lap mean best progress: `0.0m`
  - segment completion rate: `0.0`
  - mean segment progress delta: `11.32m`
  - TensorBoard rollout reward mean near `25,000`: `14.60`
  - interpretation: stochastic rollout behavior improved, but deterministic eval collapsed to no-progress. Entropy remained near maximum, so this run was stopped in favor of a lower-entropy focused experiment.

### Focused A0 Low-Entropy Scratch PPO Plan
- Reason:
  - sampled PPO rollouts are improving, but deterministic policy evaluation remains weak because action entropy stays high and argmax often collapses to no useful control.
- Next experiment:
  - keep A0-only curriculum
  - keep CUDA required
  - set `ent_coef=0.0`
  - increase update pressure with larger rollout batches and more epochs

### Focused A0 Low-Entropy Scratch PPO Run
- Command:
  - `uv run --no-sync python -m f1rl.train --timesteps 200000 --seed 40 --n-envs 8 --max-steps 3600 --device auto --require-gpu --vec-env subproc --curriculum segments --curriculum-stage-count 1 --curriculum-promotion-resets 1000000 --n-steps 512 --batch-size 256 --n-epochs 10 --learning-rate 0.001 --gamma 0.995 --ent-coef 0.0 --checkpoint-every 10000 --eval-every 10000 --eval-episodes 5 --telemetry selected --telemetry-every 10 --run-name ppo-a0-lowentropy-scratch-goal-200k`
- Artifact:
  - `artifacts\ppo-a0-lowentropy-scratch-goal-200k-20260602-173740`
- Metadata:
  - device: `cuda`
  - `--require-gpu`: enabled
  - curriculum stage count: `1`
  - promotion resets: `1000000`
  - only active curriculum stage: `A0-short-control`
  - PPO hyperparameters:
    - `n_steps`: `512`
    - `batch_size`: `256`
    - `n_epochs`: `10`
    - `learning_rate`: `0.001`
    - `gamma`: `0.995`
    - `ent_coef`: `0.0`
- Deterministic eval at timestep `10,000`:
  - full-lap mean reward: `-90.0`
  - full-lap mean best progress: `0.006m`
  - full-lap termination: no-progress
  - segment completion rate: `0.2`
  - mean segment progress delta: `76.53m`
- Deterministic eval at timestep `20,000`:
  - full-lap mean reward: `-25.49`
  - full-lap mean best progress: `431.38m`
  - full-lap checkpoints passed: `8`
  - full-lap termination: off-track
  - segment completion rate: `0.6`
  - mean segment progress delta: `106.54m`
  - interpretation: lower-entropy A0 training recovered deterministic full-lap forward behavior and improved A0 segment completion, but still fails by leaving the track around the first major section.
- Deterministic eval at timestep `30,000`:
  - full-lap mean reward: `-25.49`
  - full-lap mean best progress: `431.38m`
  - full-lap checkpoints passed: `8`
  - full-lap termination: off-track
  - segment completion rate: `1.0`
  - mean segment progress delta: `120.38m`
  - interpretation: A0 short-control is now solved in deterministic eval, but normal-start full-lap transfer is still stuck at the `431m` off-track failure. Next useful step is broader-stage curriculum or full-lap fine-tuning from this checkpoint.
- Deterministic eval at timestep `40,000`:
  - full-lap mean reward: `-88.92`
  - full-lap mean best progress: `13.54m`
  - segment completion rate: `0.0`
  - mean segment progress delta: `14.44m`
  - interpretation: the run regressed after the `30,000` checkpoint. The run was stopped and the `30,000` checkpoint is treated as the measured best checkpoint for transfer.

### Resume / Fine-Tune Support Added
- Added `--resume-checkpoint` to `f1rl.train`.
- Metadata records:
  - `scratch_initialization: false`
  - `resume_checkpoint: <path>`
- Validation after code change:
  - `uv run --no-sync ruff check .` passed
  - `uv run --no-sync pyright src/f1rl` passed
  - `uv run --no-sync pytest -q` passed, `31` tests

### Shaped A0/A1 Scratch PPO Run
- Command:
  - `uv run --no-sync python -m f1rl.train --timesteps 300000 --seed 60 --n-envs 8 --max-steps 3600 --device auto --require-gpu --vec-env subproc --curriculum segments --curriculum-stage-count 2 --curriculum-promotion-resets 1000000 --n-steps 512 --batch-size 256 --n-epochs 8 --learning-rate 0.0005 --gamma 0.995 --ent-coef 0.001 --checkpoint-every 25000 --eval-every 25000 --eval-episodes 5 --telemetry selected --telemetry-every 10 --run-name ppo-shaped-a0a1-scratch-goal-300k`
- Artifact:
  - `artifacts\ppo-shaped-a0a1-scratch-goal-300k-20260602-180421`
- Metadata:
  - device: `cuda`
  - `--require-gpu`: enabled
  - `scratch_initialization`: `true`
  - curriculum stage count: `2`
  - active curriculum stages:
    - `A0-short-control`
    - `A1-short-low-speed`
  - reward shaping active:
    - `lateral_penalty_scale`: `0.025`
    - `track_limit_penalty_scale`: `0.03`
- Deterministic eval at timestep `25,000`:
  - full-lap mean reward: `-90.0`
  - full-lap mean best progress: `0.0m`
  - full-lap termination: no-progress
  - checkpoint benchmark of `ppo_monza_25000_steps.zip` showed action `8` (`brake_right`) on every step
  - segment completion rate: `0.0`
  - mean segment progress delta: `21.90m`
- Deterministic eval at timestep `50,000`:
  - full-lap mean reward: `-90.0`
  - full-lap mean best progress: `0.0m`
  - full-lap termination: no-progress
  - segment completion rate: `0.0`
  - mean segment progress delta: `8.02m`
- Interpretation:
  - dense track-discipline reward overcorrected the full-throttle exploit and produced a stationary deterministic normal-start policy.
  - the run was stopped after `50,000` timesteps rather than burning the full budget.
  - best PPO progress remains the earlier `797.6m` / `16` checkpoint benchmark; this shaped run is not promoted.

### Experiment Controls Added
- Added curriculum mixed-start control:
  - `--curriculum-normal-start-probability`
  - when enabled with segment curriculum, resets can include true normal starts while still sampling segment starts.
  - intended use: train local control on segments without losing exposure to the real normal-start distribution.
- Added trainer reward override controls:
  - `--reward-progress-scale`
  - `--reward-finish-bonus`
  - `--reward-collision-penalty`
  - `--reward-off-track-penalty`
  - `--reward-no-progress-penalty`
  - `--reward-lateral-deadzone-m`
  - `--reward-lateral-penalty-scale`
  - `--reward-track-limit-safe-ray-m`
  - `--reward-track-limit-penalty-scale`
  - `--reward-smoothness-penalty`
- Validation after code change:
  - `uv run --no-sync ruff check .` passed
  - `uv run --no-sync pyright src/f1rl` passed
  - `uv run --no-sync pytest -q` passed, `32` tests

### A0 -> A1 Transfer PPO Run
- Command:
  - `uv run --no-sync python -m f1rl.train --timesteps 300000 --seed 50 --n-envs 8 --max-steps 3600 --device auto --require-gpu --vec-env subproc --curriculum segments --curriculum-stage-count 2 --curriculum-promotion-resets 1000000 --resume-checkpoint "artifacts\ppo-a0-lowentropy-scratch-goal-200k-20260602-173740\checkpoints\ppo_monza_30000_steps.zip" --n-steps 512 --batch-size 256 --n-epochs 8 --learning-rate 0.0003 --gamma 0.995 --ent-coef 0.002 --checkpoint-every 25000 --eval-every 25000 --eval-episodes 5 --telemetry selected --telemetry-every 10 --run-name ppo-a0a1-transfer-goal-300k`
- Artifact:
  - `artifacts\ppo-a0a1-transfer-goal-300k-20260602-174803`
- Metadata:
  - device: `cuda`
  - `--require-gpu`: enabled
  - `scratch_initialization`: `false`
  - resume checkpoint: `artifacts\ppo-a0-lowentropy-scratch-goal-200k-20260602-173740\checkpoints\ppo_monza_30000_steps.zip`
  - curriculum stage count: `2`
  - active curriculum stages:
    - `A0-short-control`
    - `A1-short-low-speed`
  - PPO hyperparameters:
    - `n_steps`: `512`
    - `batch_size`: `256`
    - `n_epochs`: `8`
    - `learning_rate`: `0.0003`
    - `gamma`: `0.995`
    - `ent_coef`: `0.002`
- Initial loaded-checkpoint eval at timestep `0`:
  - full-lap mean reward: `-25.49`
  - full-lap mean best progress: `431.38m`
  - full-lap checkpoints passed: `8`
  - full-lap termination: off-track
  - segment completion rate: `0.4`
  - note: this row was produced before the eval phase label fix, so it is incorrectly labeled `initial_scratch`; metadata correctly records `scratch_initialization: false`.
- Deterministic eval at timestep `25,000`:
  - full-lap mean reward: `-25.49`
  - full-lap mean best progress: `431.38m`
  - full-lap checkpoints passed: `8`
  - full-lap termination: off-track
  - segment completion rate: `0.2`
  - mean segment progress delta: `100.18m`
  - interpretation: A0+A1 transfer preserved the normal-start `431m` behavior but did not improve past it.
- Deterministic eval at timestep `50,000`:
  - full-lap mean reward: `-25.49`
  - full-lap mean best progress: `431.38m`
  - full-lap checkpoints passed: `8`
  - full-lap termination: off-track
  - segment completion rate: `0.6`
  - mean segment progress delta: `93.35m`
  - interpretation: still stuck at the same full-throttle off-track failure, so the run was stopped for direct telemetry diagnosis.

### 431m Failure Diagnosis
- Selected telemetry analyzed:
  - `artifacts\ppo-a0a1-transfer-goal-300k-20260602-174803\eval\selected_telemetry\ppo_full_lap-episode-000-steps.jsonl`
- Finding:
  - action `1` (`throttle`) was used on all `431` steps
  - throttle: `1.0`
  - brake: `0.0`
  - steering: `0.0`
  - final speed: `331.8kph`
  - final progress: `431.38m`
  - final lateral error: `15.51m`
  - terminal reason: off-track
- Interpretation:
  - the policy is exploiting progress reward by driving straight at full throttle until the terminal off-track penalty.
  - the old reward only penalized the terminal off-track event; it did not give enough dense feedback while the car drifted toward the boundary.

### Dense Track-Discipline Reward Added
- Added reward components:
  - `lateral`: penalizes lateral error beyond a `4.0m` deadzone
  - `track_limit`: penalizes low minimum ray distance below `10.0m`, scaled by speed
- Updated reward schema is still emitted in every mode through `REWARD_COMPONENT_KEYS`.
- Added resume eval label fix:
  - resumed runs now use `phase: initial_resume` instead of incorrectly labeling loaded checkpoints as `initial_scratch`.
- Validation after code change:
  - `uv run --no-sync ruff check .` passed
  - `uv run --no-sync pyright src/f1rl` passed
  - `uv run --no-sync pytest -q` passed, `31` tests

### Expanded-Action Scratch PPO Diagnosis
- Added an expanded action table for explicit soft throttle/steer/brake combinations while preserving legacy action IDs `0..8`.
- Validation after action expansion:
  - `uv run --no-sync python -m f1rl.hardware --json` passed; CUDA available on `NVIDIA GeForce RTX 4060 Laptop GPU`
  - `uv run --no-sync ruff check .` passed
  - `uv run --no-sync pyright src/f1rl` passed
  - `uv run --no-sync pytest -q` passed, `33` tests
- A0 expanded-actions mixed-start run:
  - command: `uv run --no-sync python -m f1rl.train --timesteps 120000 --seed 80 --n-envs 8 --max-steps 3600 --device auto --require-gpu --vec-env subproc --curriculum segments --curriculum-stage-count 1 --curriculum-promotion-resets 1000000 --curriculum-normal-start-probability 0.15 --reward-lateral-penalty-scale 0.0 --reward-track-limit-penalty-scale 0.0 --n-steps 512 --batch-size 256 --n-epochs 10 --learning-rate 0.001 --gamma 0.995 --ent-coef 0.02 --checkpoint-every 10000 --eval-every 10000 --eval-episodes 5 --telemetry selected --telemetry-every 10 --run-name ppo-a0-expanded-actions-scratch-goal-120k`
  - artifact: `artifacts\ppo-a0-expanded-actions-scratch-goal-120k-20260602-183508`
  - 10k deterministic full-lap eval: `0.0m`, `0/120` checkpoints, no-progress, mean reward `-90.0`
  - segment eval at 10k: completion rate `0.2`, mean segment delta `73.98m`
  - selected telemetry showed normal-start deterministic action collapsed to `coast`; segment sample used repeated `half_throttle`
  - result: not promoted
- Full-lap expanded-actions scratch run:
  - command: `uv run --no-sync python -m f1rl.train --timesteps 200000 --seed 90 --n-envs 8 --max-steps 3600 --device auto --require-gpu --vec-env subproc --curriculum none --reward-lateral-penalty-scale 0.0 --reward-track-limit-penalty-scale 0.0 --n-steps 512 --batch-size 256 --n-epochs 10 --learning-rate 0.0007 --gamma 0.995 --ent-coef 0.02 --checkpoint-every 10000 --eval-every 10000 --eval-episodes 5 --telemetry selected --telemetry-every 10 --run-name ppo-full-expanded-actions-scratch-goal-200k`
  - artifact: `artifacts\ppo-full-expanded-actions-scratch-goal-200k-20260602-185814`
  - initial deterministic eval: `431.38m`, `8/120` checkpoints, off-track, inherited random full-throttle style behavior
  - 10k deterministic eval: `12.87m`, `0/120` checkpoints, off-track
  - 20k deterministic eval: `13.84m`, `0/120` checkpoints, collision
  - selected telemetry at 20k was dominated by hard right-launch actions, especially `throttle_right`
  - result: stopped at 20k and not promoted
- A0 low-entropy expanded-actions scratch run:
  - command: `uv run --no-sync python -m f1rl.train --timesteps 120000 --seed 110 --n-envs 8 --max-steps 3600 --device auto --require-gpu --vec-env subproc --curriculum segments --curriculum-stage-count 1 --curriculum-promotion-resets 1000000 --reward-lateral-penalty-scale 0.0 --reward-track-limit-penalty-scale 0.0 --n-steps 512 --batch-size 256 --n-epochs 10 --learning-rate 0.001 --gamma 0.995 --ent-coef 0.0 --checkpoint-every 10000 --eval-every 10000 --eval-episodes 5 --telemetry selected --telemetry-every 10 --run-name ppo-a0-lowentropy-expanded-actions-scratch-goal-120k`
  - artifact: `artifacts\ppo-a0-lowentropy-expanded-actions-scratch-goal-120k-20260602-190404`
  - eval sequence:
    - 0: `12.51m`, `0/120`, collision; segment completion `0.2`
    - 10k: `161.09m`, `3/120`, off-track; segment completion `0.2`
    - 20k: `37.54m`, `0/120`, off-track; segment completion `0.8`
    - 30k: `0.0m`, `0/120`, no-progress; segment completion `0.2`
    - 40k: `0.0m`, `0/120`, no-progress; segment completion `0.0`
  - selected telemetry at 40k showed the full-lap sample dominated by `half_throttle_soft_left` and braking/turning actions, while the segment sample collapsed to repeated `brake_right`
  - result: stopped at 40k and not promoted
- Interpretation:
  - the expanded 21-action space mechanically works, but deterministic PPO repeatedly collapses to idle, braking, or hard-steering attractors.
  - the best measured normal-start PPO progress remains the earlier `797.6m` / `16` checkpoint benchmark; none of the expanded-action experiments improved it.

### Action-Set Selector Added
- Added `SimConfig.action_set` with selectable action spaces:
  - `legacy`: original 9 discrete actions; now the default
  - `expanded`: 21 actions including soft throttle/steer/brake controls
- Added trainer CLI option:
  - `--action-set legacy|expanded`
- Updated `MonzaEnv.action_space`, `MonzaSim.action_dim`, rollout evals, and training metadata to use the selected action set.
- Rationale:
  - legacy 9-action PPO produced the strongest measured scratch progress so far and is compatible with older/better checkpoints.
  - expanded actions remain available for explicit experiments without changing the default PPO model shape.
- Validation after selector change:
  - `uv run --no-sync ruff check .` passed
  - `uv run --no-sync pyright src/f1rl` passed
  - `uv run --no-sync pytest -q` passed, `35` tests
  - `uv run --no-sync python -m f1rl.hardware --json` passed; CUDA available on `NVIDIA GeForce RTX 4060 Laptop GPU`

### Best-Checkpoint Legacy Continuation Rejected
- Best measured normal-start PPO checkpoint before this loop:
  - run: `artifacts\ppo-curriculum-balanced-scratch-20260602-085522`
  - checkpoint: `checkpoints\ppo_monza_150000_steps.zip`
  - normal-start eval: `797.58m`, `16/120` checkpoints, collision at `11.30s`, no valid lap
  - selected telemetry action mix: mostly `throttle`, then late `brake_right`
- Resumed continuation command:
  - `uv run --no-sync python -m f1rl.train --timesteps 500000 --seed 121 --n-envs 8 --max-steps 5000 --device auto --require-gpu --vec-env subproc --action-set legacy --curriculum segments --curriculum-promotion-resets 300 --curriculum-normal-start-probability 0.05 --resume-checkpoint "artifacts\ppo-curriculum-balanced-scratch-20260602-085522\checkpoints\ppo_monza_150000_steps.zip" --reward-lateral-penalty-scale 0.0 --reward-track-limit-penalty-scale 0.0 --n-steps 256 --batch-size 128 --n-epochs 4 --learning-rate 0.0001 --gamma 0.995 --ent-coef 0.01 --checkpoint-every 25000 --eval-every 25000 --eval-episodes 1 --telemetry selected --telemetry-every 1 --run-name ppo-best797-legacy-normalmix-goal-500k`
- Artifact:
  - `artifacts\ppo-best797-legacy-normalmix-goal-500k-20260602-192141`
- Metadata verified:
  - device: `cuda`
  - `--require-gpu`: enabled
  - `scratch_initialization`: `false`
  - `action_set`: `legacy`
  - reward dense penalties disabled:
    - `lateral_penalty_scale`: `0.0`
    - `track_limit_penalty_scale`: `0.0`
- Deterministic normal-start eval sequence:
  - `0`: `797.58m`, `16/120`, collision, no valid lap
  - `25k`: `431.38m`, `8/120`, off-track, no valid lap
  - `50k`: `431.38m`, `8/120`, off-track, no valid lap
  - `75k`: `431.38m`, `8/120`, off-track, no valid lap
  - `100k`: `431.38m`, `8/120`, off-track, no valid lap
  - `125k`: `431.38m`, `8/120`, off-track, no valid lap
  - `150k`: `431.38m`, `8/120`, off-track, no valid lap
- Telemetry diagnosis:
  - the regressed policy used `throttle` on all `431` full-lap steps.
  - final speed: `331.8kph`
  - final lateral error: `15.51m`
  - terminal reason: off-track
- Interpretation:
  - this continuation was negative for normal-start performance and was stopped at `150k`.
  - segment evals stayed positive for several checkpoints, but that did not transfer to normal-start driving.
  - best measured PPO progress remains `797.58m`; no PPO checkpoint has completed a valid normal-start lap.

### Focus-Window Curriculum Added
- Added focused segment reset controls:
  - `--curriculum-focus-start-progress-m`
  - `--curriculum-focus-window-m`
  - `--curriculum-focus-segment-length-m`
  - `--curriculum-focus-min-speed-kph`
  - `--curriculum-focus-max-speed-kph`
  - `--curriculum-focus-position-noise-m`
  - `--curriculum-focus-heading-noise-deg`
  - `--curriculum-focus-speed-noise-kph`
- Behavior:
  - when focus start progress is set, segment curriculum samples `start_progress_m` inside the requested progress window instead of random checkpoint starts.
  - normal-start mix remains available through `--curriculum-normal-start-probability`.
  - run metadata records `focus_start_progress_m` and `focus_window_m`.
- Rationale:
  - the current best `797.58m` failure is localized around progress `730-800m`, where the policy brakes/steers too late at roughly `280-330kph` and collides.
  - focused resets should create more gradient signal around that failure section than random checkpoint sampling.
- Validation after focus-window change:
  - `uv run --no-sync ruff check .` passed
  - `uv run --no-sync pyright src/f1rl` passed
  - `uv run --no-sync pytest -q` passed, `37` tests

### Strategy Reset: Full-Lap-First PPO Experiments
- User direction:
  - stop optimizing tiny `~10m` local gains.
  - zoom out, run targeted experiments, use online research, and change strategy toward materially better learning.
- Online sources checked on 2026-06-02:
  - Stable-Baselines3 PPO docs: `https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html`
    - PPO supports both `Discrete` and `Box` action spaces.
    - PPO supports multiprocessing and gSDE options.
    - SB3 warns that MLP PPO is primarily CPU-oriented, but this project still requires CUDA-backed PyTorch policy training for goal runs.
  - Stable-Baselines3 RL tips: `https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html`
    - evaluate with a separate test environment and deterministic `predict`.
    - PPO/A2C are reasonable for multiprocessed continuous-control experiments.
    - normalization is critical for continuous-action algorithms.
    - continuous action spaces should be normalized and symmetric, typically rescaled to `[-1, 1]` inside the environment.
    - shaped rewards and simplified problems are recommended for custom environments.
  - Stable-Baselines3 custom env docs: `https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html`
    - SB3-compatible discrete spaces should start at `0`.
  - Gymnasium Env docs: `https://gymnasium.farama.org/api/env/`
    - active `step`/`reset` contract remains `(obs, reward, terminated, truncated, info)`.
  - Gymnasium spaces docs: `https://gymnasium.farama.org/api/spaces/fundamental/`
    - active `Box` and `Discrete` spaces match Gymnasium semantics.
  - uv docs: `https://docs.astral.sh/uv/`
    - `uv` remains the project/package manager.
  - Pygame docs: `https://www.pygame.org/docs/`
    - current manual/render loop expectations remain compatible with Pygame 2.x.
- Local package versions checked:
  - `uv 0.9.26`
  - `gymnasium 1.2.3`
  - `pygame 2.6.1`
  - `stable_baselines3 2.8.0`
  - `torch 2.10.0+cu128`
  - CUDA device: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Focus-window results before reset:
  - `ppo-focus-650-legacy-resume-goal-200k-20260602-194242`
    - best deterministic normal-start eval: `959.13m`, `19/120`, off-track/collision around `13.1s`.
  - `ppo-focus-850-legacy-resume-goal-120k-20260602-195514`
    - best benchmarked checkpoint: `966.32m`, `20/120`, off-track at `13.07s`, repeated identically across `5` deterministic episodes.
  - `ppo-focus-900-heading-legacy-resume-goal-80k-20260602-200700`
    - single eval high: `970.10m`, `20/120`, collision at `13.03s`.
    - stopped because the gain was tiny and did not change the learning regime.
- Global discrete speed-target resume rejected:
  - artifact: `artifacts\ppo-discrete-speedtarget-best966-resume-goal-150k-20260602-202301`
  - command used legacy discrete actions, speed-target/heading penalties, normal-start mix, and resumed the `966.32m` checkpoint.
  - eval sequence:
    - `0`: `966.32m`, `20/120`, off-track
    - `10k`: `966.95m`, `20/120`, off-track
    - `20k`: `967.13m`, `20/120`, off-track; segment eval completed a short A0 segment
    - `30k`: `959.13m`, `19/120`, collision
  - result: stopped at `30k`; no promotion.
  - important diagnosis: the old `best_model` score let easy segment completion dominate normal-start full-lap progress, so best-checkpoint selection was misaligned with the real goal.
- Continuous drive/brake scratch experiment rejected:
  - artifact: `artifacts\ppo-continuous-speedtarget-scratch-goal-250k-20260602-201820`
  - technically valid `Box([-1, 1], shape=(2,))` continuous path with gSDE.
  - deterministic full-lap eval stayed at `0.0m`/no-progress through `20k`.
  - likely cause: policy mean `0` mapped to coast in the drive/brake scheme, creating a no-movement attractor.
- Strategic code changes made:
  - added `SimConfig.continuous_action_scheme`.
  - preserved `drive_brake`: action `0` maps to coast; negative drive maps to brake.
  - added `throttle_bias`: action mean `0` maps to `0.5` throttle and `0` brake while remaining a normalized scratch PPO policy.
  - added optional trainer reward normalization through SB3 `VecNormalize` with observation normalization disabled.
  - saved `vecnormalize.pkl` and `best_vecnormalize.pkl` when reward normalization is active.
  - changed best-checkpoint scoring to prioritize valid full-lap completion and normal-start best progress; segment completion is now only a tiny diagnostic tie-breaker.
  - changed segment evaluation to use the segment curriculum with `normal_start_probability=0.0`, so normal-start mix can no longer leak into segment diagnostics.
- Validation after strategy-reset code:
  - `uv run --no-sync ruff check .` passed.
  - `uv run --no-sync pyright src/f1rl` passed.
  - `uv run --no-sync pytest -q` passed, `39` tests.
- New targeted experiment launched:
  - artifact: `artifacts\ppo-continuous-throttlebias-norm-scratch-goal-120k-20260602-203340`
  - command:
    ```powershell
    uv run --no-sync python -m f1rl.train --timesteps 120000 --seed 310 --n-envs 8 --max-steps 5000 --device auto --require-gpu --vec-env subproc --action-mode continuous --continuous-action-scheme throttle_bias --normalize-reward --curriculum segments --curriculum-promotion-resets 500 --curriculum-normal-start-probability 0.20 --reward-lateral-penalty-scale 0.003 --reward-track-limit-penalty-scale 0.003 --reward-heading-deadzone-deg 8 --reward-heading-penalty-scale 0.001 --reward-speed-target-min-kph 85 --reward-speed-target-max-kph 320 --reward-speed-target-heading-scale 3.0 --reward-speed-target-deadzone-kph 20 --reward-speed-target-penalty-scale 0.0015 --n-steps 1024 --batch-size 512 --n-epochs 5 --learning-rate 0.0001 --gamma 0.997 --ent-coef 0.004 --use-sde --sde-sample-freq 16 --checkpoint-every 10000 --eval-every 10000 --eval-episodes 1 --telemetry selected --telemetry-every 1 --run-name ppo-continuous-throttlebias-norm-scratch-goal-120k
    ```
  - status: running.
  - promotion rule: do not promote unless normal-start progress materially improves or a valid full lap is completed.
- Continuous run crash and fix:
  - initial deterministic eval before training:
    - normal-start progress: `958.92m`
    - checkpoints: `19/120`
    - termination: collision at `18.12s`
    - average speed: `191.3kph`
    - max speed: `247.6kph`
    - A0 segment eval completed `120.02m`
  - interpretation:
    - the `throttle_bias` mapping fixed the no-progress continuous-policy attractor, but initial scratch behavior still collides before the second chicane.
  - crash:
    - at the first training eval, selected telemetry tried to rewrite the same `ppo_curriculum_segment-episode-000-steps.jsonl` path used by the initial eval.
    - on the OneDrive-backed workspace this failed with `OSError: [Errno 22] Invalid argument`.
  - fix:
    - selected telemetry filenames now include the eval phase and timestep, e.g. `ppo_full_lap_train_00010000-episode-000-steps.jsonl`.
    - this also makes per-eval telemetry auditable instead of overwriting previous evidence.
  - validation after fix:
    - `uv run --no-sync ruff check .` passed.
    - `uv run --no-sync pyright src/f1rl` passed.
    - `uv run --no-sync pytest tests/test_policy_train_smoke.py tests/test_sim.py -q` passed, `14` tests.

### Continuous Throttle-Bias Reward-Normalized PPO Rejected
- Artifact:
  - `artifacts\ppo-continuous-throttlebias-norm-scratch-goal-120k-r2-20260602-203742`
- Command:
  - same strategy as the first continuous throttle-bias run, relaunched with seed `311` and run name `ppo-continuous-throttlebias-norm-scratch-goal-120k-r2`.
- Eval sequence:
  - `0`: `284.74m`, `5/120`, off-track at `7.90s`; A0 segment completed `120.08m`.
  - `10k`: `88.46m`, `1/120`, collision at `6.12s`; segment failed at `88.61m`.
  - `20k`: `0.00m`, `0/120`, no-progress; segment completed slowly at `30.7kph` average.
  - `30k`: `0.00m`, `0/120`, no-progress; segment failed at `3.15m`.
  - `40k`: `14.10m`, `0/120`, no-progress; segment failed at `38.96m`.
- Result:
  - stopped and rejected.
  - no checkpoint improved the robust `966.32m` discrete PPO benchmark.
- Diagnosis:
  - the continuous action path is mechanically valid, and `throttle_bias` fixed the initial no-progress attractor.
  - training still converged toward stopping/crawling because the default segment curriculum starts at very slow A0 behavior for too long.
  - the current default curriculum is misaligned with the `<=80s` target; for a racing policy, the next serious run should skip directly to longer/faster stages or use a custom fast curriculum.

### Fast-Stage Curriculum Offset Added
- Added trainer CLI option:
  - `--curriculum-start-stage-index`
- Behavior:
  - segment curriculum now slices `DEFAULT_SEGMENT_STAGES` from the requested start index before applying `--curriculum-stage-count`.
  - example: `--curriculum-start-stage-index 3 --curriculum-stage-count 4` selects `C-long`, `D-random-checkpoint`, `E-flying-lap`, and `F-normal-lap`.
- Rationale:
  - A0/A1 stages are useful smoke curricula, but they are teaching low-speed/no-progress behavior during serious racing runs.
  - a fast-stage run should produce gradients around long, high-speed survival and normal-start transfer rather than repeated 120m crawling.

### Fast-Stage Discrete Resume Rejected
- Artifact:
  - `artifacts\ppo-faststage-discrete-norm-best966-resume-goal-120k-20260602-204447`
- Command:
  - used legacy discrete actions, reward normalization, `--curriculum-start-stage-index 3`, `--curriculum-stage-count 4`, normal-start mix `0.35`, and resumed from the benchmarked `966.32m` checkpoint.
- Eval sequence:
  - `0`: `966.32m`, `20/120`, off-track at `13.07s`.
  - `10k`: `966.99m`, `20/120`, off-track at `13.05s`.
  - `20k`: `968.51m`, `20/120`, collision at `13.03s`.
  - `30k`: `959.13m`, `19/120`, off-track at `13.10s`.
  - `40k`: `959.13m`, `19/120`, off-track at `13.10s`.
- Result:
  - stopped after two consecutive evals below the `966.32m` starting benchmark.
  - best observed checkpoint was `20k`, but this is still a tiny local improvement and no lap was completed.
- Diagnosis:
  - skipping A0/A1/B avoided the continuous run's no-progress curriculum collapse.
  - the policy still drives into the same second-chicane failure envelope at roughly `270kph` average and `350kph` max.
  - next strategy should make speed discipline/braking materially stronger, not keep nudging the local trajectory by meters.

### Strong Speed-Discipline Resume Rejected
- Artifact:
  - `artifacts\ppo-speeddiscipline-discrete-best966-resume-goal-100k-20260602-205132`
- Command:
  - resumed from the same `966.32m` checkpoint.
  - used fast-stage curriculum, no reward normalization, and much stronger speed/heading shaping:
    - `reward_speed_target_penalty_scale=0.01`
    - `reward_speed_target_max_kph=300`
    - `reward_speed_target_heading_scale=3.5`
    - `reward_speed_target_deadzone_kph=5`
    - `reward_heading_penalty_scale=0.003`
- Eval sequence:
  - `0`: `966.32m`, `20/120`, off-track; speed-target reward total `-409.77`.
  - `10k`: `969.19m`, `20/120`, collision; speed-target reward total `-412.33`.
  - `20k`: `966.31m`, `20/120`, off-track; speed-target reward total `-410.07`.
  - `30k`: `959.40m`, `19/120`, collision; speed-target reward total `-408.38`.
- Result:
  - stopped at `30k`.
  - best row was `10k`, but it remained a tiny local improvement with the same failure mode.
- Telemetry action diagnosis:
  - fast-stage `20k`: actions were `throttle` `768/782` steps, `brake_right` `11/782`, `brake_left` `3/782`.
  - speed-discipline `10k`: actions were `throttle` `767/781`, `brake_right` `11/781`, `brake_left` `3/781`.
  - speed-discipline `30k`: actions were `throttle` `757/785`, `brake_right` `20/785`, `brake_left` `8/785`.
  - final failure remained around lateral error `19-21m` and heading error about `-52deg`.
- Interpretation:
  - stronger speed penalties changed reward accounting but did not change deterministic control.
  - the discrete PPO policy is stuck in a strong full-throttle attractor; late brake/steer actions are not enough to alter the trajectory before the second chicane.
  - next strategic change should alter the learning problem more directly, for example by adding braking-relevant observation features or using a staged normal-start/failure-zone curriculum with explicit earlier braking signal.

### Overspeed Action Credit Added
- Added reward component:
  - `overspeed_action`
- New reward fields:
  - `overspeed_throttle_penalty_scale`
  - `overspeed_brake_reward_scale`
- Behavior:
  - uses the same speed-target calculation as `speed_target`.
  - when current speed is above target plus deadzone, throttle gets an immediate penalty.
  - braking while overspeeding gets immediate positive credit.
- Rationale:
  - prior strong speed-target penalties changed reward totals but did not change deterministic action selection.
  - telemetry showed almost all steps were still `throttle`, with only a few late brake/steer actions near impact.
  - this directly attacks credit assignment for braking before the chicane.
- Validation:
  - `uv run --no-sync ruff check .` passed.
  - `uv run --no-sync pyright src/f1rl` passed.
  - `uv run --no-sync pytest -q` passed, `40` tests.

### Brake-Credit Resume Rejected
- Artifact:
  - `artifacts\ppo-brakecredit-discrete-best966-resume-goal-80k-20260602-210050`
- Command:
  - resumed from the `966.32m` checkpoint.
  - used fast-stage curriculum and new overspeed action shaping:
    - `reward_speed_target_penalty_scale=0.006`
    - `reward_overspeed_throttle_penalty_scale=0.8`
    - `reward_overspeed_brake_reward_scale=0.35`
- Eval sequence:
  - `0`: `966.32m`, `20/120`, off-track; `overspeed_action=-311.08`.
  - `10k`: `966.28m`, `20/120`, off-track; `overspeed_action=-313.69`.
  - `20k`: `960.82m`, `19/120`, off-track; `overspeed_action=-282.79`.
- Action diagnosis:
  - `10k`: `throttle` `767/784`, `brake_right` `14/784`, `brake_left` `3/784`.
  - `20k`: `throttle` `757/787`, `brake_right` `23/787`, `brake_left` `7/787`.
- Result:
  - stopped and rejected.
  - action-credit shaping increased late brake actions slightly, but did not introduce early braking or reduce the failure speed envelope enough.
- Current diagnosis:
  - reward-only changes are not breaking the entrenched discrete policy.
  - the next meaningful strategy likely needs a different policy/data regime: reset exactly before the braking zone with a segment target beyond the chicane, reduce initial speed to force learnable braking/turning sequence, then test transfer back to normal start; or add explicit brake-zone/target-speed observation features and train a new policy shape from scratch.

### Brake-Zone Focus Curriculum Rejected
- Artifact:
  - `artifacts\ppo-brakezone-focus600-discrete-best966-resume-goal-80k-20260602-210614`
- Command:
  - resumed from the `966.32m` checkpoint.
  - focus window centered at `600m`, width `120m`, segment length `900m`, start speed `120-220kph`, normal-start mix `0.25`.
  - used overspeed action credit.
- Eval sequence:
  - `0`: full lap `966.32m`; focus segment delta `325.44m`, collision.
  - `10k`: full lap `959.13m`; focus segment delta `391.42m`, collision.
  - `20k`: full lap `959.13m`; focus segment delta `380.21m`, collision.
- Result:
  - stopped and rejected.
  - focused reset distribution slightly improved local segment progress but did not complete the target segment and regressed normal-start performance.
- Diagnosis:
  - reward/focus changes are still asking the existing policy to infer braking from indirect signals.
  - next change: add explicit target-speed/brake-demand observation features and train a new policy shape, rather than trying to preserve compatibility with the entrenched 966m discrete checkpoint.

### Brake Observation Profile Added
- Added `SimConfig.observation_profile`:
  - `base`: existing observation shape; default; compatible with old checkpoints.
  - `brake`: appends three normalized features:
    - target speed from upcoming heading change.
    - current speed minus target speed.
    - brake demand above target plus deadzone.
- Added trainer CLI:
  - `--observation-profile base|brake`
- Rationale:
  - previous runs showed reward-only braking signals did not change the entrenched full-throttle policy.
  - the new profile makes braking demand observable directly, at the cost of requiring a new policy shape trained from scratch.
- Validation:
  - `uv run --no-sync ruff check .` passed.
  - `uv run --no-sync pyright src/f1rl` passed.
  - `uv run --no-sync pytest -q` passed, `41` tests.

### Brake-Observation Discrete Scratch Rejected
- Artifact:
  - `artifacts\ppo-brakeobs-discrete-scratch-goal-120k-20260602-211429`
- Command:
  - trained from scratch with `--observation-profile brake`, legacy discrete actions, stages `B` through `F`, normal-start mix `0.30`, speed-target and overspeed-action shaping.
- Eval sequence:
  - `0`: full lap `0.0m`, no-progress; segment delta `19.68m`, off-track.
  - `10k`: full lap `0.0m`, no-progress; segment delta `11.89m`, collision.
- Result:
  - stopped and rejected.
  - new observation features are valid, but scratch discrete PPO with legacy actions still collapsed into no-progress.
- Diagnosis:
  - the observation change alone is not enough if the action distribution can settle on non-driving actions.
  - next test combines brake observations with the continuous `throttle_bias` action scheme so the deterministic policy starts with forward drive.

### Step-Back Protocol And Milestone 0 Baseline Audit
- User direction:
  - pause the active PPO loop.
  - keep the original `LearningPlan.md` success criterion active.
  - treat `fine-tuned learning plan.md` as an inserted course correction, not a reset.
  - do not launch another serious PPO run until eval truth, observability, curriculum, and section training are improved.
- Docs reread:
  - `AGENTS.md`
  - `Prompt.md`
  - `Plan.md`
  - `Implement.md`
  - `LearningPlan.md`
  - `Documentation.md`
  - `fine-tuned learning plan.md`
  - `README.md`
- Process audit:
  - no active `f1rl.train` process remained after stopping the previous launch-guard/guidance run.
  - the only process-list match was the process-list command itself.
- Active code path remapped:
  - track geometry: `src/f1rl/track_build.py`, `track_model.py`, `geometry.py`, `config.py`
  - physics: `src/f1rl/physics.py`
  - shared simulator: `src/f1rl/sim.py`
  - manual/scripted/reference: `manual.py`, `scripted.py`, `reference_agent.py`
  - telemetry/QC: `telemetry.py`, `qc.py`
  - Gymnasium wrapper: `env.py`
  - PPO training: `train.py`
  - eval/benchmark/replay: `eval.py`, `benchmark.py`, `replay.py`, `policy_io.py`
- Current best robust PPO checkpoint:
  - `artifacts\ppo-focus-850-legacy-resume-goal-120k-20260602-195514\checkpoints\ppo_monza_40000_steps.zip`
  - benchmark artifact: `artifacts\ppo-focus-850-legacy-resume-goal-120k-20260602-195514\benchmark_40000`
  - result over `5` deterministic normal-start episodes:
    - valid lap rate: `0.0`
    - completion rate: `0.0`
    - best/mean progress: `966.317m`
    - checkpoints passed: `20/120`
    - termination: `off_track`
    - elapsed time: `13.067s`
    - average speed: `269.8kph`
    - max speed: `347.2kph`
  - selected telemetry:
    - `artifacts\ppo-focus-850-legacy-resume-goal-120k-20260602-195514\benchmark_40000\selected_telemetry\ppo_focus_40000_full_lap-episode-000-steps.jsonl`
- Single higher but unpromoted eval:
  - `artifacts\ppo-focus-900-heading-legacy-resume-goal-80k-20260602-200700`
  - `10k` eval reached `970.104m`, `20/120`, collision.
  - this is not the robust best because it is a single eval row and did not materially change the failure mode.
- Most recent stopped scout:
  - `artifacts\ppo-guidance-launchguard-p1-multidiscrete-scratch-goal-100k-20260602-220326`
  - initial scratch eval: `31.455m`, `0/120`, collision.
  - `10k` eval: `12.013m`, `0/120`, collision.
  - result is negative and unpromoted.
- Current failure statement:
  - no PPO checkpoint has completed a valid normal-start lap.
  - the robust best reaches roughly the second-chicane region and exits the track at very high speed.
  - QC reports `966.317m`, `20/120`, no missed checkpoints, `off_track`, max speed `347.2kph`, minimum ray distance `0.337m`.
  - broader training history shows a full-throttle/late-braking attractor; reward-only penalties and small focus-window nudges did not break it.
- Validation commands:
  - `uv run --no-sync ruff check .` -> passed.
  - `uv run --no-sync pyright src/f1rl` -> `0` errors.
  - `uv run --no-sync pytest -q` -> passed; only the known SB3 `VecMonitor` warning appeared.
  - `uv run --no-sync python -m f1rl.hardware --json` -> CUDA available on `NVIDIA GeForce RTX 4060 Laptop GPU`, Torch `2.10.0+cu128`, CUDA `12.8`.
  - `uv run --no-sync python -m f1rl.scripted --steps 18000 --no-telemetry` -> `lap_complete`, `5800.3m`, `214.5s`.
  - `uv run --no-sync python -m f1rl.qc --telemetry "artifacts\ppo-focus-850-legacy-resume-goal-120k-20260602-195514\benchmark_40000\selected_telemetry\ppo_focus_40000_full_lap-episode-000-steps.jsonl" --run-scripted --scripted-steps 18000` -> passed.
- QC artifact:
  - `artifacts\qc-20260602-221250\qc_report.json`
  - `artifacts\qc-20260602-221250\qc_report.md`
  - `artifacts\qc-20260602-221250\telemetry_dashboard.html`
- Live API check:
  - official Stable-Baselines3 PPO docs still show PPO support for `Discrete`, `Box`, and `MultiDiscrete` action spaces, with a note that MLP PPO is often CPU-oriented even though this project intentionally requires CUDA for goal training.
  - official SB3 custom env docs still require `Discrete` and `MultiDiscrete` starts compatible with SB3 and recommend `check_env`.
  - official SB3 tips still recommend separate deterministic evaluation, shaped rewards for custom problems, and normalized symmetric continuous action spaces.
  - Gymnasium docs still use the `step -> (obs, reward, terminated, truncated, info)` and `reset -> (obs, info)` API.
- Decision:
  - Milestone 0 is complete.
  - Do not start another PPO training run yet.
  - Next implementation milestone is `fine-tuned learning plan.md` Milestone 1: make `eval.py` and `benchmark.py` load trained-run metadata, action mode/action set, observation profile, reward overrides, and `VecNormalize` stats, and fail loudly on model/environment shape mismatches.

### Milestone 1: Metadata-Faithful Eval And Benchmark
- Purpose:
  - prevent standalone eval/benchmark from silently scoring a PPO checkpoint with the wrong action mode, action set, observation profile, reward settings, or normalization stats.
- Implementation:
  - expanded `src\f1rl\policy_io.py` with a shared `PpoEvalConfig` resolver.
  - checkpoint inputs now support:
    - `latest`
    - exact `.zip` model paths
    - artifact directories such as `artifacts\ppo-focus-850-legacy-resume-goal-120k-20260602-195514`
  - artifact directories resolve in priority order:
    - `best_model.zip`
    - `final_model.zip`
    - `checkpoints\best_model.zip`
    - `checkpoints\final_model.zip`
    - newest `checkpoints\*.zip`
    - newest root `*.zip`
  - `run_metadata.json` is loaded by default when found.
  - metadata restores:
    - `action_mode`
    - `action_set`
    - `continuous_action_scheme`
    - `observation_profile`
    - launch guard fields
    - reward settings
    - car/sensor/lookahead settings
  - `max_steps` remains an eval/benchmark override so short smoke checks can run without changing trained-run metadata.
  - `best_vecnormalize.pkl` or `vecnormalize.pkl` is loaded when present.
  - observations are normalized before `model.predict()` when VecNormalize stats are loaded.
  - model observation/action spaces are validated before rollout; mismatches raise a clear `ValueError`.
  - `src\f1rl\eval.py` now reports `config_source` and VecNormalize path.
  - `src\f1rl\benchmark.py` now writes `ppo_eval_config` into `config.json` and `summary.json`, and labels PPO episode rows with action mode/action set/observation profile/config source.
  - added `--metadata-mode auto|require|ignore` to eval and benchmark.
    - `auto`: default; load metadata when available.
    - `require`: fail if metadata is missing.
    - `ignore`: use explicit CLI flags; intended only for compatibility tests.
- Tests:
  - added `tests\test_policy_io.py`.
  - coverage includes:
    - metadata resolution from checkpoint path.
    - artifact-directory checkpoint selection.
    - metadata ignore mode.
    - observation shape mismatch failure.
    - action-space mismatch failure.
    - matching `MultiDiscrete` validation.
- Validation commands:
  - `uv run --no-sync ruff check src/f1rl/policy_io.py src/f1rl/eval.py src/f1rl/benchmark.py tests/test_policy_io.py tests/test_benchmark.py` -> passed.
  - `uv run --no-sync pytest tests/test_policy_io.py tests/test_benchmark.py -q` -> passed.
  - `uv run --no-sync pyright src/f1rl` -> `0` errors.
  - `uv run --no-sync python -m f1rl.eval --checkpoint "artifacts\ppo-focus-850-legacy-resume-goal-120k-20260602-195514\checkpoints\ppo_monza_40000_steps.zip" --steps 50 --seed 17 --device auto --metadata-mode require` -> passed, wrote `artifacts\eval-20260602-222415`, reported `config_source=run_metadata`.
  - `uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "artifacts\ppo-focus-850-legacy-resume-goal-120k-20260602-195514" --episodes 1 --max-steps 80 --seed 23 --device auto --telemetry none --metadata-mode require` -> passed, wrote `artifacts\benchmark-20260602-222405`, reported legacy discrete/base profile from metadata.
  - `uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "artifacts\ppo-brakeobs-continuous-throttlebias-norm-scratch-goal-80k-20260602-211740" --episodes 1 --max-steps 20 --seed 31 --device auto --telemetry none --metadata-mode require` -> passed, wrote `artifacts\benchmark-20260602-222501`, loaded continuous `throttle_bias`, `brake` observation profile, and `best_vecnormalize.pkl`.
  - intentional mismatch check:
    - `uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "artifacts\ppo-brakeobs-continuous-throttlebias-norm-scratch-goal-80k-20260602-211740\best_model.zip" --episodes 1 --max-steps 5 --seed 31 --device auto --telemetry none --metadata-mode ignore`
    - expected failure occurred: `ValueError: Model observation shape does not match eval environment: model=(21,) env=(18,)`.
  - full validation after the milestone:
    - `uv run --no-sync ruff check .` -> passed.
    - `uv run --no-sync pyright src/f1rl` -> `0` errors.
    - `uv run --no-sync pytest -q` -> passed; known SB3 `VecMonitor` warning only.
- Decision:
  - Milestone 1 is complete.
  - Next milestone is failure-first observability: section definitions, per-section summaries, first-bad-event detection, and QC failure tables.

### Milestone 2: Failure-First Observability
- Purpose:
  - make telemetry answer why PPO failed, not only how far it went.
  - support the next section curriculum by identifying the failed section, first bad event, action distribution before failure, and section speed/brake behavior.
- Implementation:
  - added `src\f1rl\section_analysis.py`.
  - defined Monza section boundaries by progress distance:
    - `start_finish_straight`: `0-450m`
    - `rettifilo_chicane`: `450-1150m`
    - `curva_grande_roggia_run`: `1150-1850m`
    - `roggia_chicane`: `1850-2500m`
    - `lesmo_1`: `2500-3150m`
    - `lesmo_2_serraglio`: `3150-3850m`
    - `ascari_approach`: `3850-4450m`
    - `ascari_chicane`: `4450-5150m`
    - `parabolica_finish`: `5150-5793m`
  - per-section summaries now include:
    - entry/exit/min/max/average speed,
    - brake start progress,
    - throttle reapplication progress,
    - average throttle and brake,
    - action histogram,
    - control histogram,
    - max speed surplus vs section target,
    - minimum ray distance,
    - lateral/heading error aggregates,
    - reward totals,
    - section termination reason.
  - first-bad-event detection now flags:
    - `throttle_during_brake_demand`,
    - `no_brake_before_turn_in`,
    - `overspeed_at_braking_zone`,
    - `boundary_contact_risk`,
    - `excessive_lateral_error`,
    - `wrong_heading`,
    - terminal `collision`, `off_track`, or `no_progress`.
  - QC now accepts either a telemetry file or a telemetry folder.
  - QC writes:
    - `telemetry_reports`,
    - `failure_table`,
    - section summaries inside the primary `telemetry` report,
    - scripted section comparison when `--run-scripted` is enabled.
  - QC Markdown and HTML dashboard now include a compact failure table and section summary table.
- Tests:
  - added `tests\test_section_analysis.py`.
  - updated `tests\test_qc.py`.
  - coverage includes:
    - section lookup for the `966m` failure region,
    - detection of throttle during brake demand,
    - action histogram before failure,
    - section speed/brake summaries,
    - QC JSON failure-table output.
- Validation commands:
  - `uv run --no-sync ruff check src/f1rl/section_analysis.py src/f1rl/qc.py tests/test_section_analysis.py tests/test_qc.py` -> passed after import-order fix.
  - `uv run --no-sync pytest tests/test_section_analysis.py tests/test_qc.py -q` -> passed.
  - `uv run --no-sync pyright src/f1rl` -> `0` errors.
  - full validation after the milestone:
    - `uv run --no-sync ruff check .` -> passed.
    - `uv run --no-sync pyright src/f1rl` -> `0` errors.
    - `uv run --no-sync pytest -q` -> passed; known SB3 `VecMonitor` warning only.
- Real artifact validation:
  - command:
    ```powershell
    uv run --no-sync python -m f1rl.qc --telemetry "artifacts\ppo-focus-850-legacy-resume-goal-120k-20260602-195514\benchmark_40000\selected_telemetry" --max-telemetry-files 2 --run-scripted --scripted-steps 18000
    ```
  - artifact:
    - `artifacts\qc-20260602-223327`
  - result:
    - telemetry files analyzed: `2`
    - robust-best PPO failed in `rettifilo_chicane`.
    - first bad event: `throttle_during_brake_demand`.
    - first bad event location/speed: `521.4m`, `333.2kph`.
    - reason: car is overspeed in a braking zone while still applying throttle.
    - actions before failure: `{'brake_right': 3, 'throttle': 177}`.
    - terminal event: `off_track` at `966.3m`, `325.0kph`.
    - section summary:
      - `start_finish_straight`: entry `1.3kph`, exit `328.7kph`, avg brake `0.01`.
      - `rettifilo_chicane`: entry `328.8kph`, min `325.0kph`, avg brake `0.04`, max speed surplus `232.2kph`, min ray `0.337m`, terminal `off_track`.
    - scripted comparison: `lap_complete`, valid lap, `5800.3m`.
- Decision:
  - Milestone 2 is complete.
  - The next section curriculum should target the first heavy braking/chicane failure window beginning around `520m`, not the old vague `~966m` terminal point.
  - Next milestone is the racing observation profile: signed lateral error, previous throttle/brake, target-speed/lookahead features, and braking-gate distance.

### Milestone 3: Racing Observation Profile
- Purpose:
  - give PPO direct state needed for high-speed braking and car placement while preserving existing checkpoint compatibility.
- Implementation:
  - added `src\f1rl\track_sections.py` as shared Monza section/braking-gate metadata for simulator observations and QC analysis.
  - kept existing observation profiles stable:
    - `base`: `18`
    - `brake`: `21`
    - `guidance`: `23`
  - added `observation_profile="racing"` with dimension `31`.
  - racing profile includes the existing base signals plus:
    - brake profile features: target speed, speed surplus, brake demand,
    - guidance features: target steer and steer error,
    - signed lateral error,
    - previous throttle,
    - previous brake,
    - per-lookahead target-speed features,
    - normalized distance to the next Monza braking gate.
  - legacy base lateral observation now explicitly uses the absolute lateral magnitude, preserving old behavior even though signed lateral is computed for racing.
- Tests:
  - added Gymnasium env-check coverage for `observation_profile="racing"`.
  - verified old profile dimensions remain unchanged and `racing` is `base + 13`.
  - verified racing observations stay bounded in `[-1, 1]`.
  - verified signed lateral observation flips sign on opposite sides of the centerline.
  - added braking-gate distance wraparound coverage.
- Validation commands:
  - `uv run --no-sync ruff check src/f1rl/sim.py src/f1rl/scripted.py src/f1rl/section_analysis.py src/f1rl/track_sections.py tests/test_env.py tests/test_sim.py tests/test_section_analysis.py` -> passed.
  - `uv run --no-sync pyright src/f1rl` -> `0` errors.
  - `uv run --no-sync pytest tests/test_env.py tests/test_sim.py tests/test_section_analysis.py tests/test_qc.py -q` -> passed.
  - `uv run --no-sync python -c "from f1rl.config import SimConfig, OBSERVATION_PROFILES; from f1rl.sim import MonzaSim; print(sorted(OBSERVATION_PROFILES)); print({p: MonzaSim(SimConfig(observation_profile=p)).observation_dim for p in sorted(OBSERVATION_PROFILES)})"` -> `{'base': 18, 'brake': 21, 'guidance': 23, 'racing': 31}`.
  - full validation after the milestone:
    - `uv run --no-sync ruff check .` -> passed.
    - `uv run --no-sync pyright src/f1rl` -> `0` errors.
    - `uv run --no-sync pytest -q` -> passed; known SB3 `VecMonitor` warning only.
- Decision:
  - Milestone 3 is complete.
  - The next milestone is successful-state snapshots and state-library curriculum, so section starts can come from physically coherent lap states instead of arbitrary progress-only resets.

### Milestone 4: Successful-State Snapshots And State Libraries
- Purpose:
  - let curricula start from physically coherent successful states instead of only approximate checkpoint/progress resets.
  - support future chicane and elite-search curricula with exact simulator states from scripted, manual, reference, PPO, or search telemetry.
- Implementation:
  - added `src\f1rl\state_snapshot.py`.
  - `StateSnapshot` stores:
    - `x`, `y`, `heading_rad`, `speed_mps`, `yaw_rate_rps`, `steering_rad`,
    - raw and monotonic progress,
    - checkpoint/lap validity bookkeeping,
    - previous throttle/brake/steer/action,
    - source metadata.
  - added `MonzaSim.reset(..., options={"state_snapshot": ...})`.
    - restores physical state and checkpoint/lap fields,
    - keeps elapsed steps fresh for the new curriculum episode,
    - supports optional position/heading/speed noise,
    - still supports `segment_length_m` targets after restore.
  - added `src\f1rl\state_library.py` / `f1-state-library`.
    - `--source scripted` runs the scripted controller and samples snapshots.
    - `--source telemetry` samples snapshots from telemetry JSONL files or folders, covering manual/reference/PPO artifacts.
  - added curriculum integration:
    - `CurriculumConfig.start_mode="state_library"`.
    - `--curriculum-state-library PATH`.
    - `--curriculum-state-library-segment-length-m`.
    - optional state-library position/heading/speed noise flags.
    - run metadata records `curriculum_state_library`, `start_mode`, and `state_library_count`.
- Tests:
  - added `tests\test_state_library.py`.
  - updated `tests\test_curriculum.py`.
  - coverage includes:
    - simulator reset from a snapshot and continuing,
    - library generation from telemetry JSONL,
    - scripted library write/load,
    - curriculum sampler state-library options,
    - `MonzaEnv` reset from a state library.
- Validation commands:
  - `uv run --no-sync ruff check src/f1rl/state_snapshot.py src/f1rl/state_library.py src/f1rl/sim.py src/f1rl/curriculum.py src/f1rl/train.py tests/test_state_library.py tests/test_curriculum.py` -> passed after formatter fixes.
  - `uv run --no-sync pytest tests/test_state_library.py tests/test_curriculum.py -q` -> passed.
  - `uv run --no-sync pyright src/f1rl` -> `0` errors.
  - state-library CLI smoke:
    - `uv run --no-sync python -m f1rl.state_library --source scripted --steps 120 --sample-every-steps 30 --sample-every-m 0 --output artifacts\state-library-smoke-20260602-m4\state_library.json`
    - artifact: `artifacts\state-library-smoke-20260602-m4\state_library.json`
    - snapshots: `5`.
  - training integration smoke:
    - `uv run --no-sync python -m f1rl.train --timesteps 64 --seed 42 --n-envs 1 --max-steps 60 --device cpu --checkpoint-every 64 --eval-every 32 --eval-episodes 1 --telemetry none --curriculum segments --curriculum-state-library "artifacts\state-library-smoke-20260602-m4\state_library.json" --curriculum-state-library-segment-length-m 30 --run-name state-library-curriculum-smoke`
    - artifact: `artifacts\state-library-curriculum-smoke-20260602-225555`
    - metadata check: `start_mode=state_library`, `state_library_count=5`.
  - full scripted state library:
    - `uv run --no-sync python -m f1rl.state_library --source scripted --steps 18000 --sample-every-m 100 --sample-every-steps 0 --output artifacts\state-library-scripted-full-m4-20260602\state_library.json`
    - artifact: `artifacts\state-library-scripted-full-m4-20260602\state_library.json`
    - snapshots: `59`
    - final snapshot: `5800.35m`, `checkpoint_index=0`, `next_checkpoint_index=120`, `valid_lap=True`, `completed_lap=True`, source step `12868`.
  - full validation after the milestone:
    - `uv run --no-sync ruff check .` -> passed.
    - `uv run --no-sync pyright src/f1rl` -> `0` errors.
    - `uv run --no-sync pytest -q` -> passed; known SB3 `VecMonitor` warning only.
- Decision:
  - Milestone 4 is complete.
  - Next milestone is the chicane-specific skill curriculum. It should use the new `racing` observation profile and state-library starts, with honest normal-start eval kept separate from section diagnostics.

### Milestone 5: Chicane-Specific Skill Curriculum
- Purpose:
  - replace a vague focus window with staged chicane skill tasks: approach/brake, turn-in, apex, exit, post-exit, full chicane, and mixed normal-start/chicane training.
  - support both the current Rettifilo first-bad-event window and the later Roggia/second-chicane section that will become relevant after the first chicane is solved.
- Implementation:
  - extended `CurriculumStage` with optional:
    - `start_min_progress_m`,
    - `start_max_progress_m`,
    - `target_progress_m`.
  - added `CHICANE_SKILL_STAGES`:
    - `rettifilo`: approach/brake, turn-in, apex, exit, post-exit, full-chicane stages.
    - `roggia`: approach/brake, turn-in, apex, exit, post-exit, full-chicane stages.
  - added `--curriculum-preset chicane-skill`.
  - added `--curriculum-chicane rettifilo|roggia`.
  - chicane stages can sample:
    - progress ranges when no state library is provided,
    - filtered successful-state snapshots when `--curriculum-state-library` is provided.
  - state-library chicane sampling respects `--curriculum-start-stage-index` and `--curriculum-stage-count`.
  - added `curriculum_stage_metrics` to eval summaries:
    - episodes per stage,
    - segment completion rate,
    - mean segment progress delta,
    - mean best progress,
    - termination reason counts.
- Tests:
  - updated `tests\test_curriculum.py`.
  - coverage includes:
    - progress-range sampling for Rettifilo,
    - state-library filtering by stage progress window,
    - dynamic segment target length from snapshot to target gate,
    - actual per-stage success metric aggregation.
- Validation commands:
  - `uv run --no-sync ruff check src/f1rl/curriculum.py src/f1rl/train.py tests/test_curriculum.py` -> passed after formatter fix.
  - `uv run --no-sync pytest tests/test_curriculum.py -q` -> passed.
  - `uv run --no-sync pyright src/f1rl` -> `0` errors.
  - chicane-skill integration smoke:
    - `uv run --no-sync python -m f1rl.train --timesteps 64 --seed 43 --n-envs 1 --max-steps 120 --device cpu --checkpoint-every 64 --eval-every 32 --eval-episodes 2 --telemetry none --curriculum segments --curriculum-preset chicane-skill --curriculum-chicane rettifilo --curriculum-stage-count 2 --curriculum-state-library "artifacts\state-library-scripted-full-m4-20260602\state_library.json" --run-name chicane-skill-curriculum-smoke`
    - artifact: `artifacts\chicane-skill-curriculum-smoke-20260602-230601`
    - metadata check: `start_mode=state_library`, `state_library_count=59`, stages `rettifilo-approach-brake`, `rettifilo-turn-in`.
    - eval summary includes `curriculum_stage_metrics`.
  - full validation after the milestone:
    - `uv run --no-sync ruff check .` -> passed.
    - `uv run --no-sync pyright src/f1rl` -> `0` errors.
    - `uv run --no-sync pytest -q` -> passed; known SB3 `VecMonitor` warning only.
- Decision:
  - Milestone 5 is complete as infrastructure.
  - It does not prove the PPO agent can complete the chicane yet; it makes the next scaffolded section-training experiment measurable and reproducible.
  - Next milestone is training-only brake/exit scaffold rewards, with honest unassisted eval remaining separate.

### Milestone 6: Temporary Brake/Exit Scaffold Rewards
- Purpose:
  - add explicit training-only reward scaffolds for braking, turn-in speed, clean apex passage, and exit quality.
  - keep them zero by default and make scaffold-free eval explicit.
- Implementation:
  - added zero-default reward config fields:
    - `scaffold_brake_reward_scale`,
    - `scaffold_no_throttle_penalty_scale`,
    - `scaffold_turn_in_speed_penalty_scale`,
    - `scaffold_apex_clean_reward_scale`,
    - `scaffold_exit_alignment_reward_scale`,
    - `scaffold_exit_speed_reward_scale`,
    - `scaffold_scale`.
  - added reward components:
    - `scaffold_brake`,
    - `scaffold_no_throttle`,
    - `scaffold_turn_in_speed`,
    - `scaffold_apex_clean`,
    - `scaffold_exit_alignment`,
    - `scaffold_exit_speed`.
  - scaffold rewards are active only in sections with braking/turn-in metadata and only when explicit scales are nonzero.
  - added `MonzaEnv.set_reward_scaffold_scale()` for schedule callbacks.
  - added optional linear scaffold schedule:
    - `--reward-scaffold-final-scale`,
    - `--reward-scaffold-schedule-timesteps`.
  - training metadata records:
    - `scaffold_rewards_enabled`,
    - `scaffold_reward_schedule`.
  - added `--disable-scaffold-rewards` to `f1rl.eval` and `f1rl.benchmark`.
    - this loads checkpoint metadata normally but zeros training-only scaffold fields before rollout/reward accounting.
- Tests:
  - updated `tests\test_sim.py`:
    - scaffold components are zero by default,
    - braking is credited in the chicane brake zone when enabled,
    - throttle is penalized in the chicane brake zone when enabled.
  - updated `tests\test_env.py`:
    - env method clamps and applies scaffold scale.
- Validation commands:
  - `uv run --no-sync ruff check src/f1rl/config.py src/f1rl/telemetry.py src/f1rl/sim.py src/f1rl/env.py src/f1rl/train.py src/f1rl/eval.py src/f1rl/benchmark.py tests/test_sim.py tests/test_env.py` -> passed after formatter fixes.
  - `uv run --no-sync pytest tests/test_sim.py tests/test_env.py tests/test_policy_train_smoke.py -q` -> passed; known SB3 warning only.
  - `uv run --no-sync pyright src/f1rl` -> `0` errors.
  - scaffold training smoke:
    - `uv run --no-sync python -m f1rl.train --timesteps 64 --seed 44 --n-envs 1 --max-steps 120 --device cpu --checkpoint-every 64 --eval-every 32 --eval-episodes 2 --telemetry none --curriculum segments --curriculum-preset chicane-skill --curriculum-chicane rettifilo --curriculum-stage-count 2 --curriculum-state-library "artifacts\state-library-scripted-full-m4-20260602\state_library.json" --reward-scaffold-brake-reward-scale 0.4 --reward-scaffold-no-throttle-penalty-scale 0.6 --reward-scaffold-turn-in-speed-penalty-scale 0.2 --reward-scaffold-apex-clean-reward-scale 0.05 --reward-scaffold-exit-alignment-reward-scale 0.05 --reward-scaffold-exit-speed-reward-scale 0.05 --reward-scaffold-final-scale 0.0 --reward-scaffold-schedule-timesteps 64 --run-name scaffold-reward-smoke`
    - artifact: `artifacts\scaffold-reward-smoke-20260602-231335`
    - metadata check: `scaffold_rewards_enabled=True`, schedule `1.0 -> 0.0` over `64` timesteps.
  - scaffold-free benchmark smoke:
    - `uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "artifacts\scaffold-reward-smoke-20260602-231335" --episodes 1 --max-steps 5 --seed 55 --device cpu --telemetry none --metadata-mode require --disable-scaffold-rewards`
    - artifact: `artifacts\benchmark-20260602-231438`
    - config and summary record `disable_scaffold_rewards=True`.
  - full validation after the milestone:
    - `uv run --no-sync ruff check .` -> passed.
    - `uv run --no-sync pyright src/f1rl` -> `0` errors.
    - `uv run --no-sync pytest -q` -> passed; known SB3 `VecMonitor` warning only.
- Decision:
  - Milestone 6 is complete as infrastructure.
  - Scaffolded training must still be reported as assisted training.
  - Promotion requires normal-start benchmark/eval with scaffold rewards disabled.

### Milestone 7: Training-Only Forced Exploration
- Purpose:
  - make lazy full-throttle chicane behavior fail or become costly during assisted section training.
  - keep those gates explicitly disabled for honest full-lap eval.
- Implementation:
  - added `AssistConfig` to `SimConfig`.
  - added zero-default assist telemetry/reward components:
    - `assist_overspeed_gate`,
    - `assist_throttle_brake_demand`,
    - `assist_no_brake_gate`,
    - `assist_virtual_corridor`.
  - added simulator assist logic:
    - overspeed near turn-in can terminate with `assist_overspeed_gate`,
    - throttle during brake demand can be penalized,
    - no-brake behavior before turn-in can be penalized,
    - an optional virtual corridor can penalize or terminate lateral excursions.
  - added training CLI flags:
    - `--assist-enabled`,
    - `--assist-overspeed-turn-in-terminate`,
    - `--assist-overspeed-turn-in-margin-kph`,
    - `--assist-overspeed-turn-in-penalty`,
    - `--assist-throttle-brake-demand-penalty-scale`,
    - `--assist-no-brake-penalty`,
    - `--assist-no-brake-min-brake`,
    - `--assist-virtual-corridor-m`,
    - `--assist-virtual-corridor-penalty`,
    - `--assist-virtual-corridor-terminate`.
  - training metadata records:
    - `training_assists_enabled`,
    - `assist_config`.
  - added `--disable-training-assists` to `f1rl.eval` and `f1rl.benchmark`.
  - `policy_io` now restores `AssistConfig` from metadata.
- Tests:
  - updated `tests\test_sim.py`:
    - assist components are zero by default,
    - throttle/no-brake penalties fire in the chicane brake zone,
    - overspeed turn-in gate can terminate assisted episodes.
  - updated `tests\test_policy_io.py`:
    - assist config survives metadata resolution.
- Validation commands:
  - `uv run --no-sync ruff check src/f1rl/config.py src/f1rl/telemetry.py src/f1rl/sim.py src/f1rl/train.py src/f1rl/eval.py src/f1rl/benchmark.py src/f1rl/policy_io.py tests/test_sim.py tests/test_policy_io.py` -> passed after formatter fix.
  - `uv run --no-sync pytest tests/test_sim.py tests/test_policy_io.py tests/test_env.py -q` -> passed.
  - `uv run --no-sync pyright src/f1rl` -> `0` errors.
  - assisted training smoke:
    - `uv run --no-sync python -m f1rl.train --timesteps 64 --seed 45 --n-envs 1 --max-steps 120 --device cpu --checkpoint-every 64 --eval-every 32 --eval-episodes 2 --telemetry none --curriculum segments --curriculum-preset chicane-skill --curriculum-chicane rettifilo --curriculum-stage-count 2 --curriculum-state-library "artifacts\state-library-scripted-full-m4-20260602\state_library.json" --assist-enabled --assist-overspeed-turn-in-terminate --assist-overspeed-turn-in-margin-kph 20 --assist-throttle-brake-demand-penalty-scale 0.8 --assist-no-brake-penalty -8 --run-name forced-exploration-smoke`
    - artifact: `artifacts\forced-exploration-smoke-20260602-232142`
    - metadata check: `training_assists_enabled=True`, `enabled=True`, `overspeed_turn_in_terminate=True`, `throttle_brake_demand_penalty_scale=0.8`, `no_brake_penalty=-8.0`.
  - assist-free benchmark smoke:
    - `uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "artifacts\forced-exploration-smoke-20260602-232142" --episodes 1 --max-steps 5 --seed 56 --device cpu --telemetry none --metadata-mode require --disable-training-assists`
    - artifact: `artifacts\benchmark-20260602-232229`
    - config and summary record `disable_training_assists=True`.
  - full validation after the milestone:
    - `uv run --no-sync ruff check .` -> passed.
    - `uv run --no-sync pyright src/f1rl` -> `0` errors.
    - `uv run --no-sync pytest -q` -> passed; known SB3 `VecMonitor` warning only.
- Decision:
  - Milestone 7 is complete as infrastructure.
  - Assisted training artifacts must not be presented as unassisted performance.
  - Promotion remains based on normal-start benchmark/eval with assists disabled.

### Milestone 8: Segment Elite Search
- Purpose:
  - discover high-quality section exits through many short attempts and save the best resulting states for future curricula.
  - provide an artifact path from section attempts to new state-library starts.
- Implementation:
  - added `src\f1rl\elite_search.py` / `f1-elite-search`.
  - inputs:
    - source state library,
    - progress filter,
    - segment length,
    - attempt count,
    - top-K elite count,
    - scripted/random/optional PPO policy mode,
    - optional action/start perturbations.
  - outputs:
    - `attempts.jsonl`,
    - `selected_telemetry\elite-rank-...jsonl`,
    - `elite_state_library.json`,
    - `elite_search_summary.json`.
  - scoring favors:
    - segment completion,
    - valid/no-collision/no-offtrack behavior,
    - progress delta,
    - exit speed,
    - heading alignment,
    - low lateral error,
    - no missed checkpoints.
  - PPO modes are available through:
    - `--policy ppo-deterministic`,
    - `--policy ppo-stochastic`,
    - `--checkpoint`,
    - metadata-aware policy loading.
- Tests:
  - added `tests\test_elite_search.py`.
  - coverage verifies:
    - runner writes an elite library,
    - selected telemetry is produced for top attempts,
    - summary and attempts artifacts are written.
- Validation commands:
  - `uv run --no-sync ruff check src/f1rl/elite_search.py tests/test_elite_search.py` -> passed.
  - `uv run --no-sync pytest tests/test_elite_search.py -q` -> passed.
  - `uv run --no-sync pyright src/f1rl` -> `0` errors.
  - Roggia elite-search artifact:
    - `uv run --no-sync python -m f1rl.elite_search --state-library "artifacts\state-library-scripted-full-m4-20260602\state_library.json" --output-dir artifacts\elite-search-roggia-m8-20260602 --attempts 6 --max-steps 240 --top-k 3 --segment-length-m 800 --start-min-progress-m 1850 --start-max-progress-m 2020 --policy scripted --action-noise 0.05 --seed 80`
    - artifact: `artifacts\elite-search-roggia-m8-20260602`
    - result: `6` attempts, `3` elite states.
    - top attempt final progress range: about `2028.4m` to `2126.9m`; no full segment completion claimed.
  - QC display validation:
    - `uv run --no-sync python -m f1rl.qc --telemetry "artifacts\elite-search-roggia-m8-20260602\selected_telemetry" --max-telemetry-files 3`
    - artifact: `artifacts\qc-20260602-232815`
  - full validation after the milestone:
    - `uv run --no-sync ruff check .` -> passed.
    - `uv run --no-sync pyright src/f1rl` -> `0` errors.
    - `uv run --no-sync pytest -q` -> passed; known SB3 `VecMonitor` warning only.
- Decision:
  - Milestone 8 is complete as infrastructure.
  - The elite search produced seed states, not proof of chicane mastery.
  - Next milestone is a properly shaped continuous-control retry using racing observations, state-library starts, scaffold/assist controls, VecNormalize, and gSDE.

### Milestone 9: Continuous-Control Retry
- Purpose:
  - retry continuous control only after adding racing observations, metadata-faithful eval, state-library curriculum, section stages, scaffold rewards, and assist gates.
- Experiment 1: hard-assisted continuous retry:
  - command:
    ```powershell
    uv run --no-sync python -m f1rl.train --timesteps 20000 --seed 90 --n-envs 8 --max-steps 900 --device auto --require-gpu --vec-env subproc --action-mode continuous --continuous-action-scheme throttle_bias --observation-profile racing --curriculum segments --curriculum-preset chicane-skill --curriculum-chicane rettifilo --curriculum-stage-count 2 --curriculum-state-library "artifacts\state-library-scripted-full-m4-20260602\state_library.json" --reward-scaffold-brake-reward-scale 0.4 --reward-scaffold-no-throttle-penalty-scale 0.6 --reward-scaffold-turn-in-speed-penalty-scale 0.2 --reward-scaffold-apex-clean-reward-scale 0.05 --reward-scaffold-exit-alignment-reward-scale 0.05 --reward-scaffold-exit-speed-reward-scale 0.05 --reward-scaffold-final-scale 0.1 --reward-scaffold-schedule-timesteps 20000 --assist-enabled --assist-overspeed-turn-in-terminate --assist-overspeed-turn-in-margin-kph 25 --assist-throttle-brake-demand-penalty-scale 0.8 --assist-no-brake-penalty -8 --normalize-reward --use-sde --sde-sample-freq 16 --n-steps 256 --batch-size 256 --n-epochs 3 --learning-rate 0.0001 --gamma 0.995 --ent-coef 0.004 --checkpoint-every 5000 --eval-every 5000 --eval-episodes 2 --telemetry selected --telemetry-every 1 --run-name ppo-continuous-racing-rettifilo-m9-20k
    ```
  - artifact: `artifacts\ppo-continuous-racing-rettifilo-m9-20k-20260602-233058`
  - result:
    - initial section delta `183.5m`, no completion, assisted overspeed-gate terminations.
    - final section delta `59.9m`.
    - final normal-start eval `43.6m`.
  - decision: rejected; hard assist termination collapsed continuous behavior.
- Experiment 2: discrete-expanded comparison:
  - artifact: `artifacts\ppo-discrete-expanded-racing-rettifilo-m9-10k-20260602-233450`
  - result:
    - initial section delta `30.7m`.
    - `5k/10k` section delta `183.2m`, still no section completion.
    - honest scaffold/assist-disabled benchmark: `artifacts\benchmark-20260602-233826`, `431.60m`, `8` checkpoints, off-track.
  - decision: better than collapsed hard-assisted continuous, but still far below robust `966.32m`.
- Experiment 3: soft continuous retry:
  - command:
    ```powershell
    uv run --no-sync python -m f1rl.train --timesteps 10000 --seed 94 --n-envs 8 --max-steps 900 --device auto --require-gpu --vec-env subproc --action-mode continuous --continuous-action-scheme throttle_bias --observation-profile racing --curriculum segments --curriculum-preset chicane-skill --curriculum-chicane rettifilo --curriculum-stage-count 1 --curriculum-state-library "artifacts\state-library-scripted-full-m4-20260602\state_library.json" --reward-scaffold-brake-reward-scale 0.4 --reward-scaffold-no-throttle-penalty-scale 0.6 --reward-scaffold-turn-in-speed-penalty-scale 0.2 --reward-scaffold-apex-clean-reward-scale 0.05 --reward-scaffold-exit-alignment-reward-scale 0.05 --reward-scaffold-exit-speed-reward-scale 0.05 --reward-scaffold-final-scale 0.2 --reward-scaffold-schedule-timesteps 10000 --normalize-reward --use-sde --sde-sample-freq 16 --n-steps 256 --batch-size 256 --n-epochs 3 --learning-rate 0.00005 --gamma 0.995 --ent-coef 0.008 --checkpoint-every 5000 --eval-every 5000 --eval-episodes 2 --telemetry selected --telemetry-every 1 --run-name ppo-continuous-racing-rettifilo-soft-m9-10k
    ```
  - artifact: `artifacts\ppo-continuous-racing-rettifilo-soft-m9-10k-20260602-234003`
  - result:
    - section eval completed the approach/brake stage at `0`, `5k`, and `10k`.
    - normal-start eval reached `770.2m` at `10k`.
  - honest benchmark:
    - command:
      ```powershell
      uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "artifacts\ppo-continuous-racing-rettifilo-soft-m9-10k-20260602-234003" --episodes 3 --max-steps 1200 --seed 95 --device auto --telemetry selected --telemetry-every 1 --metadata-mode require --disable-scaffold-rewards --disable-training-assists
      ```
    - artifact: `artifacts\benchmark-20260602-234321`
    - result: best/avg `951.76m`, `19/120`, collision, no valid lap.
  - QC:
    - artifact: `artifacts\qc-20260602-234452`
    - first bad event: `throttle_during_brake_demand` at `521.1m`, `244.0kph`.
    - terminal: collision at `951.76m`, `252.9kph`.
- Decision:
  - Milestone 9 is complete as a controlled retry and comparison.
  - The soft continuous setup is the best M9 candidate, but it is not promoted over the robust `966.32m` baseline.
  - Continuous control is not rejected globally; hard assist termination is rejected for now.
  - Next step is Milestone 10: use the soft continuous setup for full-lap transfer/scaffold reduction, but benchmark promotion only with scaffold/assist disabled.

### Milestone 10: Full-Lap Transfer And Scaffold Removal
- Purpose:
  - transfer the best M9 section behavior back toward normal-start full-lap performance while decaying scaffolds and keeping forced assists disabled.
- Transfer run:
  - command:
    ```powershell
    uv run --no-sync python -m f1rl.train --timesteps 20000 --seed 96 --n-envs 8 --max-steps 1500 --device auto --require-gpu --vec-env subproc --resume-checkpoint "artifacts\ppo-continuous-racing-rettifilo-soft-m9-10k-20260602-234003\best_model.zip" --vec-normalize-path "artifacts\ppo-continuous-racing-rettifilo-soft-m9-10k-20260602-234003\best_vecnormalize.pkl" --action-mode continuous --continuous-action-scheme throttle_bias --observation-profile racing --curriculum segments --curriculum-preset chicane-skill --curriculum-chicane rettifilo --curriculum-stage-count 2 --curriculum-normal-start-probability 0.5 --curriculum-state-library "artifacts\state-library-scripted-full-m4-20260602\state_library.json" --reward-scaffold-brake-reward-scale 0.25 --reward-scaffold-no-throttle-penalty-scale 0.35 --reward-scaffold-turn-in-speed-penalty-scale 0.1 --reward-scaffold-apex-clean-reward-scale 0.03 --reward-scaffold-exit-alignment-reward-scale 0.03 --reward-scaffold-exit-speed-reward-scale 0.03 --reward-scaffold-final-scale 0.0 --reward-scaffold-schedule-timesteps 20000 --normalize-reward --use-sde --sde-sample-freq 16 --n-steps 256 --batch-size 256 --n-epochs 3 --learning-rate 0.00003 --gamma 0.997 --ent-coef 0.006 --checkpoint-every 5000 --eval-every 5000 --eval-episodes 3 --telemetry selected --telemetry-every 1 --run-name ppo-continuous-racing-full-transfer-m10-20k
    ```
  - artifact: `artifacts\ppo-continuous-racing-full-transfer-m10-20k-20260602-234628`
  - result:
    - initial resume: `951.8m`, segment completion rate `1.0`.
    - `5k`: `204.4m`.
    - `10k`: `99.1m`.
    - `15k`: `73.6m`.
    - `20k`: `60.3m`.
  - decision: continuation settings were destructive; final checkpoint rejected.
- Preserved-best honest benchmark:
  - command:
    ```powershell
    uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "artifacts\ppo-continuous-racing-full-transfer-m10-20k-20260602-234628" --episodes 3 --max-steps 1500 --seed 97 --device auto --telemetry selected --telemetry-every 1 --metadata-mode require --disable-scaffold-rewards --disable-training-assists
    ```
  - artifact: `artifacts\benchmark-20260602-235140`
  - result: `951.76m`, `19/120`, collision, no valid lap.
- Decision:
  - Milestone 10 is complete as a transfer attempt.
  - No checkpoint from M9/M10 is promoted above the robust `966.32m` baseline.
  - Fine-tuned course-correction milestones are now complete as infrastructure plus controlled experiments.
  - Return to the original `LearningPlan.md` goal loop. The strict target remains unchanged: valid normal-start PPO lap in `<=80.0s`.

### Original Learning Loop Resumed: Turn-In Continuation Rejected
- Purpose:
  - after completing the fine-tuned milestones, test a narrower continuation that teaches only the Rettifilo turn-in stage from the soft continuous M9 checkpoint.
  - avoid the destructive M10 normal-start mixed transfer settings.
- Command:
  ```powershell
  uv run --no-sync python -m f1rl.train --timesteps 10000 --seed 98 --n-envs 8 --max-steps 900 --device auto --require-gpu --vec-env subproc --resume-checkpoint "artifacts\ppo-continuous-racing-rettifilo-soft-m9-10k-20260602-234003\best_model.zip" --vec-normalize-path "artifacts\ppo-continuous-racing-rettifilo-soft-m9-10k-20260602-234003\best_vecnormalize.pkl" --action-mode continuous --continuous-action-scheme throttle_bias --observation-profile racing --curriculum segments --curriculum-preset chicane-skill --curriculum-chicane rettifilo --curriculum-start-stage-index 1 --curriculum-stage-count 1 --curriculum-state-library "artifacts\state-library-scripted-full-m4-20260602\state_library.json" --reward-scaffold-brake-reward-scale 0.15 --reward-scaffold-no-throttle-penalty-scale 0.25 --reward-scaffold-turn-in-speed-penalty-scale 0.15 --reward-scaffold-apex-clean-reward-scale 0.05 --reward-scaffold-exit-alignment-reward-scale 0.05 --reward-scaffold-exit-speed-reward-scale 0.04 --reward-scaffold-final-scale 0.1 --reward-scaffold-schedule-timesteps 10000 --normalize-reward --use-sde --sde-sample-freq 16 --n-steps 256 --batch-size 256 --n-epochs 3 --learning-rate 0.00002 --gamma 0.997 --ent-coef 0.006 --checkpoint-every 5000 --eval-every 5000 --eval-episodes 2 --telemetry selected --telemetry-every 1 --run-name ppo-continuous-racing-rettifilo-turnin-resume-goal-10k
  ```
- Artifact:
  - `artifacts\ppo-continuous-racing-rettifilo-turnin-resume-goal-10k-20260602-235421`
- Result:
  - initial resume: normal-start eval `770.2m`; turn-in segment completion `1.0`.
  - `5k`: normal-start eval `447.4m`; turn-in segment completion `1.0`, but terminations were collisions.
  - `10k`: normal-start eval `285.4m`; turn-in segment completion `0.0`.
- Decision:
  - rejected.
  - continuing PPO from the soft continuous candidate remains unstable even with a narrow turn-in stage and lower learning rate.
  - robust best remains `966.32m`, `20/120`, off-track from `artifacts\ppo-focus-850-legacy-resume-goal-120k-20260602-195514`.

### Original Learning Loop Continued: Strategy Reset Experiments - 2026-06-03
- Validation gate before new work:
  - `uv run --no-sync python -m f1rl.hardware --json` -> CUDA available on `NVIDIA GeForce RTX 4060 Laptop GPU`.
  - `uv run --no-sync ruff check .` -> passed.
  - `uv run --no-sync pyright src/f1rl` -> `0` errors.
  - `uv run --no-sync pytest -q` -> passed; known SB3 `VecMonitor` warning only.
  - `uv run --no-sync python -m f1rl.scripted --steps 18000 --no-telemetry` -> `lap_complete`, valid slow lap `214.5s`.
  - TensorBoard was still running at `127.0.0.1:6006`.
- New implementation:
  - added continuous action scheme `exclusive_throttle_bias`.
    - positive drive maps to throttle only,
    - negative drive maps to brake only,
    - zero drive maps to a small launch throttle,
    - this avoids the old `throttle_bias` behavior where braking actions still carried throttle unless drive saturated to `-1`.
  - added PPO transfer initialization through `f1rl.train --initialize-from-checkpoint`.
    - creates a fresh PPO model for the requested environment.
    - copies compatible SB3 MLP policy tensors from a source checkpoint.
    - expands first-layer observation-input tensors when the target observation space is larger.
    - zero-initializes new observation columns so transferred behavior initially matches the source policy while new features remain trainable.
    - metadata records `transfer_initialization`, `initialize_from_checkpoint`, and `transfer_weight_report`.
  - tests:
    - `tests\test_sim.py` covers `exclusive_throttle_bias`.
    - `tests\test_policy_train_smoke.py` covers transfer initialization from `base` observations into `racing` observations.
- Focused validation for new code:
  - `uv run --no-sync ruff check src/f1rl/config.py src/f1rl/sim.py tests/test_sim.py` -> passed.
  - `uv run --no-sync pytest tests/test_sim.py -q` -> passed.
  - `uv run --no-sync ruff check src/f1rl/train.py tests/test_policy_train_smoke.py` -> passed.
  - `uv run --no-sync pytest tests/test_policy_train_smoke.py -q` -> passed.
  - `uv run --no-sync pytest tests/test_policy_train_smoke.py::test_ppo_transfer_initialization_can_expand_observation_inputs -q` -> passed after zero-initializing new observation columns.
- Experiment: exclusive continuous control smoke:
  - artifact: `artifacts\exclusive-throttlebias-smoke-20260603-000620`.
  - result: smoke train completed on CPU; metadata confirms `continuous_action_scheme=exclusive_throttle_bias`, `observation_profile=racing`, state-library curriculum count `59`.
- Experiment: exclusive continuous Rettifilo state-library run:
  - artifact: `artifacts\ppo-continuous-exclusive-racing-rettifilo-soft-goal-40k-20260603-000722`.
  - stopped after `10k`.
  - result: normal-start eval reached only `168.1m`, off-track on the start-finish straight.
  - diagnosis: the new action scheme worked, but the policy learned tiny steering and drifted off before testing Rettifilo.
  - decision: rejected.
- Experiment: exclusive continuous prefix-start steering run:
  - artifact: `artifacts\ppo-continuous-exclusive-racing-prefix-steering-goal-60k-20260603-001222`.
  - stopped after `10k`.
  - result: selected 10k telemetry regressed to `72.6m`, collision, very low throttle, small brake, crawl/placement behavior.
  - decision: rejected; steering/placement penalties overpowered forward progress.
- Experiment: racing-discrete prefix-start runs:
  - artifact without launch guard: `artifacts\ppo-racingdiscrete-racingobs-prefix-goal-80k-20260603-001928`.
    - initial deterministic eval no-progressed; stopped.
  - artifact with launch guard: `artifacts\ppo-racingdiscrete-racingobs-prefix-launchguard-goal-60k-20260603-002116`.
    - `10k`: `12.4m`, off-track.
  - decision: both rejected; scratch racing-discrete is not competitive with the robust legacy checkpoint.
- Experiment: legacy robust checkpoint exit-transfer focus:
  - artifact: `artifacts\ppo-legacy-best966-rettifilo-exit-transfer-goal-100k-20260603-002435`.
  - stopped after `10k`.
  - result: `970.3m`, `20/120`, collision.
  - telemetry: same first bad event, `throttle_during_brake_demand` at `520.2m`, `335.5kph`; brake still starts around `960.8m`.
  - decision: rejected as another small local movement without braking improvement.
- Experiment: legacy robust checkpoint soft assist brake-gate:
  - artifact: `artifacts\ppo-legacy-best966-softassist-brakegate-goal-60k-20260603-002832`.
  - stopped after `20k`.
  - result: still `966.3m`, off-track.
  - telemetry at `10k`: still throttle through the brake zone; Rettifilo reward totals included large training-only penalties:
    - `assist_throttle_brake_demand=-343.7`,
    - `assist_no_brake_gate=-254.0`,
    - `assist_overspeed_gate=-1100.0`,
    - but deterministic policy still selected throttle.
  - decision: soft reward pressure alone did not move deterministic action selection.
- Experiment: legacy robust checkpoint hard assist brake-gate:
  - artifact: `artifacts\ppo-legacy-best966-hardassist-brakegate-goal-40k-20260603-003346`.
  - stopped after `10k`.
  - result: assisted eval still terminated at `686.5m` with `assist_overspeed_gate`.
  - decision: rejected; low-LR hard gate did not shift deterministic braking.
- Experiment: legacy robust checkpoint aggressive hard assist brake-gate:
  - artifact: `artifacts\ppo-legacy-best966-hardassist-brakegate-aggressive-goal-30k-20260603-003724`.
  - stopped after `10k`.
  - result: still terminated at the assisted overspeed gate around `686m`.
  - direct policy-probability inspection at a `520m`, `333kph` brake-zone state:
    - robust checkpoint deterministic action: `throttle`.
    - action probabilities were close: `throttle=0.1317`, `brake_left=0.1176`, `brake_right=0.1165`, but training barely moved them by `10k`.
  - decision: the old base-observation policy is stuck in a brittle throttle argmax; reward-only continuation is weak.
- Experiment: transfer-initialized racing-observation soft assist:
  - artifact: `artifacts\ppo-transfer-racingobs-best966-softassist-goal-60k-20260603-005219`.
  - stopped after `20k`.
  - initial transfer reproduced the robust baseline: `966.3m`, `20/120`, off-track, `transfer_initialization=True`, `observation_profile=racing`.
  - `10k`: still `966.3m`; segment completion improved to `1.0`.
  - `20k`: regressed to `960.7m`, collision.
  - decision: promising infrastructure, but this soft-assist run did not improve normal-start behavior.
- Experiment: transfer-initialized racing-observation aggressive hard assist:
  - artifact: `artifacts\ppo-transfer-racingobs-best966-hardassist-aggressive-goal-20k-20260603-005723`.
  - stopped after `10k`.
  - initial transfer with hard assist terminated at `686.5m`, as expected.
  - `5k`: still `assist_overspeed_gate`.
  - `10k`: collapsed to `3.6m`, max-steps.
  - decision: rejected; hard assist was destructive even with racing observations.
- Current best remains unchanged:
  - robust best checkpoint: `artifacts\ppo-focus-850-legacy-resume-goal-120k-20260602-195514\checkpoints\ppo_monza_40000_steps.zip`.
  - robust best benchmark: `966.32m`, `20/120`, off-track, no valid PPO lap.
- Current diagnosis:
  - the repeated first bad event remains `throttle_during_brake_demand` in `rettifilo_chicane` around `520-521m`.
  - robust policy action probabilities at the brake-zone state are close, but deterministic argmax still picks `throttle`.
  - reward/assist continuation has not reliably moved deterministic action preference.
  - transfer initialization is now available and preserves old competence when expanding from `base` to `racing` observations, but the attempted soft/hard transfer schedules were not sufficient.
  - strict goal remains unmet: no PPO valid normal-start lap, and no lap at `<=80.0s`.

### Fine-Tuned Plan Closure Audit And Stochastic Check - 2026-06-03
- Plan file audit:
  - Found `LearningPlan.md` and `fine-tuned learning plan.md`.
  - No separate `finetune learning plan.md` file exists at the repo root or in the active tracked file list; the user wording is treated as referring to the original learning plan plus the fine-tuned course-correction plan.
- Runtime state:
  - TensorBoard remains live on `127.0.0.1:6006`.
  - Active Python processes are TensorBoard wrappers only; no PPO training process is currently running.
- Fine-tuned milestone status:
  - Milestone 0 baseline audit: complete. Robust best is `artifacts\ppo-focus-850-legacy-resume-goal-120k-20260602-195514\checkpoints\ppo_monza_40000_steps.zip`, `966.32m`, `20/120`, off-track, no valid lap.
  - Milestone 1 metadata-faithful eval/benchmark: complete. PPO eval/benchmark resolve `run_metadata.json`, trained action/observation config, and VecNormalize stats when present, and fail on space mismatches.
  - Milestone 2 failure-first observability: complete. QC/section analysis identify the current first bad event as `throttle_during_brake_demand` in `rettifilo_chicane` near `520m`.
  - Milestone 3 racing observation profile: complete. `racing` observations preserve `base`/`brake` behavior and add signed lateral, prior control, target-speed, brake-demand, and braking-gate features.
  - Milestone 4 successful-state curriculum: complete. `StateSnapshot`, telemetry/scripted state libraries, and state-library curriculum starts exist and are tested.
  - Milestone 5 chicane-specific skill curriculum: complete as infrastructure. Both `rettifilo` and `roggia` staged curricula exist; current measured PPO failure is Rettifilo, not Roggia.
  - Milestone 6 temporary scaffold rewards: complete. Brake, no-throttle, turn-in, apex, exit alignment, and exit speed scaffolds are zero by default, schedulable, logged, and disableable for honest eval.
  - Milestone 7 forced exploration gates: complete. Training-only assists are explicit, logged, and disableable for honest eval.
  - Milestone 8 segment elite search: complete. `f1rl.elite_search` writes attempts, selected telemetry, summaries, and elite state libraries.
  - Milestone 9 continuous-control retry: complete as controlled experiments. `exclusive_throttle_bias` was added, but current continuous retries are rejected because they underperform the robust `966.32m` discrete checkpoint.
  - Milestone 10 full-lap transfer/scaffold removal: complete as controlled experiments, not as goal completion. Transfer-initialized racing-observation attempts preserved the baseline initially but did not improve honest normal-start progress; no valid PPO lap exists.
- Stochastic PPO eval/benchmark implementation:
  - Added `--ppo-deterministic` / `--ppo-stochastic` flags to `f1rl.eval` and `f1rl.benchmark`.
  - Benchmark/eval config and per-episode PPO rows now record `ppo_deterministic`.
  - Focused validation:
    - `uv run --no-sync ruff check src/f1rl/benchmark.py src/f1rl/eval.py tests/test_benchmark.py` -> passed.
    - `uv run --no-sync pytest tests/test_benchmark.py -q` -> passed.
- Stochastic benchmark evidence:
  - Interrupted exploratory artifact: `artifacts\benchmark-20260603-010726`.
    - It produced partial selected telemetry but no complete summary.
    - Partial behavior was poor: mostly max-step crawling under about `200m` or early failures.
  - Complete artifact: `artifacts\benchmark-20260603-011203`.
    - Command shape: benchmark robust best checkpoint with `--metadata-mode require --ppo-stochastic --episodes 4 --max-steps 1000 --telemetry selected --telemetry-every 1`.
    - Result: `0/4` valid laps, `0/4` finish crossings, `0.0` completion rate.
    - Average progress: `73.74m`.
    - Best progress: `129.31m`.
    - Terminations: `4/4` `max_steps`.
    - Recorded config confirms `ppo_deterministic=false`, `ppo_action_set=legacy`, `ppo_observation_profile=base`, and `config_source=run_metadata`.
  - Decision: stochastic inference is not a hidden solution for the current robust checkpoint. It is worse than deterministic normal-start evaluation and should not be promoted.
- Online research notes used for the next strategy:
  - Gymnasium official docs still define the `reset()`/`step()` API and reinforce keeping env checker coverage for custom environments: https://gymnasium.farama.org/api/env/
  - Stable-Baselines3 PPO docs confirm the current PPO constructor surface, including `policy_kwargs`, gSDE, `device`, and load/eval behavior: https://stable-baselines3.readthedocs.io/en/v2.5.0/modules/ppo.html
  - Stable-Baselines3 VecNormalize docs confirm normalization stats are separate from model files and must be loaded with the eval vector env, with training disabled for evaluation: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
  - Stable-Baselines3 custom policy docs support `policy_kwargs`/custom MLP structure as the low-complexity way to change policy capacity without reviving archive-era systems: https://stable-baselines3.readthedocs.io/en/v2.3.0/guide/custom_policy.html
  - Reverse Curriculum Generation supports the state-library/restart direction already implemented here: https://arxiv.org/abs/1707.05300
  - Potential-based reward-shaping work reinforces why scaffold rewards must remain training-only and why honest unassisted evaluation is the promotion gate: https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf
- Strategic decision after the closure audit:
  - The fine-tuned infrastructure milestones are complete enough to return to the original `LearningPlan.md` loop.
  - The next serious run should not be another small `10m` focus-window tweak.
  - Next candidate strategy should combine:
    - metadata-faithful transfer from the robust `966.32m` checkpoint,
    - `racing` observations,
    - normal-start pressure from the beginning,
    - Rettifilo state-library/chicane starts as a minority curriculum source,
    - no hard assist termination,
    - mild scaffold schedule only if it does not damage normal-start eval,
    - and promotion only if deterministic honest normal-start progress clears the robust `966.32m` checkpoint by a meaningful margin or produces a valid lap.
- Validation gate after the audit/docs update:
  - `uv run --no-sync ruff check .` -> passed.
  - `uv run --no-sync pyright src/f1rl` -> `0` errors, `0` warnings.
  - `uv run --no-sync pytest -q` -> passed; known SB3 `VecMonitor` warning only.
  - `uv run --no-sync python -m f1rl.hardware --json` -> CUDA available on `NVIDIA GeForce RTX 4060 Laptop GPU`; policy training/inference on CUDA and simulator work on CPU.
  - `uv run --no-sync python -m f1rl.scripted --steps 18000 --no-telemetry` -> `lap_complete`, `completed=True`, progress `5800.3m`, time `214.5s`.
- Fresh deterministic robust-checkpoint benchmark and QC anchor:
  - Benchmark command:
    ```powershell
    uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "artifacts\ppo-focus-850-legacy-resume-goal-120k-20260602-195514\checkpoints\ppo_monza_40000_steps.zip" --episodes 1 --max-steps 1500 --seed 9400 --device auto --telemetry selected --telemetry-every 1 --metadata-mode require --ppo-deterministic
    ```
  - Benchmark artifact: `artifacts\benchmark-20260603-012314`.
  - Result: `966.317m`, `20/120`, `off_track`, no finish crossing, no valid lap.
  - Per-episode config: `ppo_deterministic=true`, `ppo_config_source=run_metadata`, `ppo_action_set=legacy`, `ppo_observation_profile=base`.
  - QC command:
    ```powershell
    uv run --no-sync python -m f1rl.qc --telemetry "artifacts\benchmark-20260603-012314\selected_telemetry" --max-telemetry-files 1
    ```
  - QC artifact: `artifacts\qc-20260603-012346`.
  - QC failure table:
    - failed section: `rettifilo_chicane`.
    - first bad event: `throttle_during_brake_demand`.
    - first bad progress: `521.414m`.
    - first bad speed: `333.213kph`.
    - actions before failure: `177` `throttle`, `3` `brake_right`.
    - terminal event: `off_track` at `966.317m`, `324.984kph`.
  - Rettifilo section summary:
    - entry speed `328.85kph`.
    - min speed `324.98kph`.
    - max speed `347.18kph`.
    - average throttle `0.964`.
    - average brake `0.036`.
    - termination `off_track`.
  - Decision:
    - This is the current anchor failure artifact for the resumed original learning loop.
    - The policy does not have a subtle line-choice issue at Rettifilo; it has a high-speed braking/action-selection issue.

### Original Learning Loop Resumed: Action-Head Transfer - 2026-06-03
- Reason for code change:
  - Previous transfer initialization could preserve a robust policy when only the observation space expanded (`base` -> `racing`).
  - It could not preserve behavior when changing discrete action spaces (`legacy` -> `racing` or `expanded`) because SB3 `action_net` tensors changed from `9` logits to a larger logit count and were left random.
  - This made richer-action experiments behave like scratch policies even when `--initialize-from-checkpoint` was used.
- Implementation:
  - `src\f1rl\train.py` now expands discrete PPO action heads during transfer initialization.
  - Exact target actions copy the matching source action row, e.g. `throttle`, `brake_left`, `brake_right`.
  - New target actions copy the nearest source action row by throttle/brake/steer distance and receive an initial bias penalty so they do not steal deterministic behavior before training.
  - Transfer reports now include `expanded_discrete_action_head` rows in `run_metadata.json`.
- Tests:
  - Added `tests\test_policy_train_smoke.py::test_transfer_initialization_can_expand_discrete_action_head`.
  - Validated that exact rows copy and approximate rows get the bias-penalty mode.
- Focused validation:
  - `uv run --no-sync ruff check src/f1rl/train.py tests/test_policy_train_smoke.py` -> passed.
  - `uv run --no-sync pyright src/f1rl` -> `0` errors, `0` warnings.
  - `uv run --no-sync pytest tests/test_policy_train_smoke.py::test_transfer_initialization_can_expand_discrete_action_head -q` -> passed.
  - `uv run --no-sync pytest tests/test_policy_train_smoke.py::test_ppo_transfer_initialization_can_expand_observation_inputs -q` -> passed; known SB3 `VecMonitor` warning only.
- Full validation after code change:
  - `uv run --no-sync ruff check .` -> passed.
  - `uv run --no-sync pyright src/f1rl` -> `0` errors, `0` warnings.
  - `uv run --no-sync pytest -q` -> passed; known SB3 `VecMonitor` warning only.
- Transfer probe:
  - Command:
    ```powershell
    uv run --no-sync python -m f1rl.train --timesteps 64 --seed 950 --n-envs 2 --max-steps 1500 --device auto --require-gpu --vec-env subproc --initialize-from-checkpoint "artifacts\ppo-focus-850-legacy-resume-goal-120k-20260602-195514\checkpoints\ppo_monza_40000_steps.zip" --action-mode discrete --action-set racing --observation-profile racing --curriculum none --checkpoint-every 64 --eval-every 64 --eval-episodes 1 --telemetry selected --telemetry-every 1 --run-name ppo-transfer-racing-actionhead-probe-goal-64
    ```
  - Artifact: `artifacts\ppo-transfer-racing-actionhead-probe-goal-64-20260603-012854`.
  - Metadata confirms:
    - `transfer_initialization=true`.
    - `observation_profile=racing`.
    - `action_set=racing`.
    - transfer report includes both `expanded_input` and `expanded_discrete_action_head`.
  - Initial-transfer eval:
    - `966.107m`, `20/120`, `collision`, no valid lap.
  - Decision:
    - The new transfer path preserves the robust baseline while exposing the richer `racing` action set.
    - This is a materially better starting point than previous scratch `racing` action-set experiments.
    - Next experiment should train from this transfer path with strong normal-start pressure and a minority of Rettifilo state-library starts, without hard assist termination.

### Racing Action-Head Transfer Run And QC Truth Fix - 2026-06-03
- Serious transfer run:
  - Command:
    ```powershell
    uv run --no-sync python -m f1rl.train --timesteps 60000 --seed 960 --n-envs 8 --max-steps 5000 --device auto --require-gpu --vec-env subproc --initialize-from-checkpoint "artifacts\ppo-focus-850-legacy-resume-goal-120k-20260602-195514\checkpoints\ppo_monza_40000_steps.zip" --action-mode discrete --action-set racing --observation-profile racing --curriculum segments --curriculum-preset chicane-skill --curriculum-chicane rettifilo --curriculum-stage-count 6 --curriculum-state-library "artifacts\state-library-scripted-full-m4-20260602\state_library.json" --curriculum-promotion-resets 120 --curriculum-normal-start-probability 0.65 --reward-lateral-penalty-scale 0.004 --reward-track-limit-penalty-scale 0.004 --reward-heading-deadzone-deg 8 --reward-heading-penalty-scale 0.001 --reward-speed-target-min-kph 80 --reward-speed-target-max-kph 340 --reward-speed-target-heading-scale 2.5 --reward-speed-target-deadzone-kph 18 --reward-speed-target-penalty-scale 0.001 --reward-overspeed-throttle-penalty-scale 0.003 --reward-overspeed-brake-reward-scale 0.001 --reward-scaffold-brake-reward-scale 0.10 --reward-scaffold-no-throttle-penalty-scale 0.15 --reward-scaffold-turn-in-speed-penalty-scale 0.05 --reward-scaffold-apex-clean-reward-scale 0.02 --reward-scaffold-exit-alignment-reward-scale 0.02 --reward-scaffold-exit-speed-reward-scale 0.02 --reward-scaffold-final-scale 0.0 --reward-scaffold-schedule-timesteps 60000 --normalize-reward --n-steps 512 --batch-size 256 --n-epochs 4 --learning-rate 0.00005 --gamma 0.997 --ent-coef 0.004 --checkpoint-every 10000 --eval-every 10000 --eval-episodes 2 --telemetry selected --telemetry-every 1 --run-name ppo-transfer-racing-actionhead-rettifilo-mix-goal-60k
    ```
  - Artifact: `artifacts\ppo-transfer-racing-actionhead-rettifilo-mix-goal-60k-20260603-013341`.
  - Runtime: CUDA, subproc vector env, `60000` timesteps, approximately `133` training fps.
  - Eval progression:
    - initial transfer: `966.107m`, `20/120`, `collision`, no valid lap.
    - `10000`: `968.478m`, `20/120`, `collision`.
    - `20000`: `970.775m`, `20/120`, `collision`, best checkpoint.
    - `30000`: regressed to `283.987m`, `5/120`, `collision`.
    - `40000`: `283.496m`, `5/120`, `collision`.
    - `50000`: `320.510m`, `6/120`, `off_track`.
    - `60000`: `326.170m`, `6/120`, `off_track`.
  - Decision:
    - Do not promote this run as goal progress. The best checkpoint improved the deterministic distance by only about `4.5m` over the robust anchor and still fails the same Rettifilo braking behavior.
    - The late collapse after `20000` confirms that the current scaffold/curriculum mix can damage normal-start behavior if left running.
    - Use it as evidence for a strategy change: preserve transfer initialization, reduce destructive scaffold pressure, and optimize for earlier braking rather than marginal extra meters.
- Honest benchmark of the best model:
  - Command:
    ```powershell
    uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "artifacts\ppo-transfer-racing-actionhead-rettifilo-mix-goal-60k-20260603-013341" --episodes 5 --max-steps 5000 --seed 9700 --device auto --telemetry selected --telemetry-every 1 --metadata-mode require --disable-scaffold-rewards --disable-training-assists --ppo-deterministic
    ```
  - Artifact: `artifacts\benchmark-20260603-014242`.
  - Result: `0/5` valid laps, `0/5` finish crossings, `5/5` collisions, average/best progress `970.775m`, `20/120`.
- Observability bug found:
  - The simulator persisted only `action_id` in telemetry.
  - `f1rl.qc`/section analysis fell back to legacy action labels, which misnamed `racing` action-set IDs in reports.
  - This did not affect controls or benchmark metrics, but it made human failure diagnosis misleading.
- Implementation fix:
  - `src\f1rl\telemetry.py` `StepTelemetry` now records `action_name`.
  - `src\f1rl\sim.py` emits action names from the configured action set for discrete policies, describes multidiscrete actions, and labels scripted/reference actions explicitly.
  - `src\f1rl\reference_agent.py` writes `action_name="reference_ghost"`.
  - `src\f1rl\section_analysis.py` prefers telemetry `action_name` and falls back to legacy labels only for old telemetry files.
  - Tests cover racing discrete telemetry labels, multidiscrete labels, and QC failure reports preferring telemetry labels.
- Validation after the action-name fix:
  - `uv run --no-sync ruff check src/f1rl/telemetry.py src/f1rl/sim.py src/f1rl/reference_agent.py src/f1rl/section_analysis.py tests/test_sim.py tests/test_section_analysis.py` -> passed.
  - `uv run --no-sync pyright src/f1rl` -> `0` errors, `0` warnings.
  - `uv run --no-sync pytest tests/test_sim.py tests/test_section_analysis.py tests/test_reference_agent.py -q` -> passed.
  - Full validation:
    - `uv run --no-sync ruff check .` -> passed.
    - `uv run --no-sync pyright src/f1rl` -> `0` errors, `0` warnings.
    - `uv run --no-sync pytest -q` -> passed; known SB3 `VecMonitor` warning only.
- Fresh post-fix benchmark/QC:
  - Benchmark command:
    ```powershell
    uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "artifacts\ppo-transfer-racing-actionhead-rettifilo-mix-goal-60k-20260603-013341" --episodes 1 --max-steps 1500 --seed 9800 --device auto --telemetry selected --telemetry-every 1 --metadata-mode require --disable-scaffold-rewards --disable-training-assists --ppo-deterministic
    ```
  - Benchmark artifact: `artifacts\benchmark-20260603-015016`.
  - QC command:
    ```powershell
    uv run --no-sync python -m f1rl.qc --telemetry "artifacts\benchmark-20260603-015016\selected_telemetry" --max-telemetry-files 1
    ```
  - QC artifact: `artifacts\qc-20260603-015045`.
  - Confirmed QC labels are now correct for the `racing` action set:
    - first bad event: `throttle_during_brake_demand`.
    - first bad progress: `520.806m`.
    - first bad speed: `334.094kph`.
    - actions before failure: `177` `throttle`, `3` `brake_right`.
    - terminal event: `collision` at `970.775m`, `273.825kph`.
    - terminal action: `brake_left`.
  - Interpretation:
    - The transferred racing-action policy learned some late braking compared with the robust legacy checkpoint, but it still delays braking until too late and collides in Rettifilo.
    - Next training must target brake timing and retention, not merely more normal-start distance.

### Retention Runs Rejected And Curriculum Truth Bug Fixed - 2026-06-03
- User progress concern:
  - The best honest PPO distance was still about `970m` after several hours.
  - That concern was correct: raw normal-start driving progress was effectively flat.
  - The useful progress in this loop was diagnostic and infrastructural, not a new lap-distance breakthrough.
- Failed retention experiments:
  - `artifacts\ppo-racing-best970-brakedemand-retention-goal-20k-20260603-015740`
    - stopped because the assist overspeed gate inherited a destructive default penalty and produced extremely large negative rewards.
    - not used as learning evidence.
  - `artifacts\ppo-racing-best970-brakedemand-retention-v2-goal-20k-20260603-020001`
    - started from the `970.775m` best racing-action checkpoint and matching `best_vecnormalize.pkl`.
    - initial eval preserved `970.775m`.
    - `5000` eval regressed to `967.8m`.
    - `10000` eval was `968.7m`.
    - `15000` eval collapsed to about `209.5m`; run was stopped.
    - honest benchmark/QC of the `10000` checkpoint remained the same root failure: first bad event around `520.76m` at about `334.18kph`, with actions before failure dominated by throttle.
  - Decision:
    - Neither retention run is promoted.
    - More training against the same signal is not expected to break the plateau.
- `racing_v2` observation profile:
  - Added `observation_profile="racing_v2"` while preserving old `base` and `racing` profile compatibility.
  - `racing_v2` adds explicit section-skill features:
    - named-section target speed normalized;
    - speed surplus relative to the named-section target;
    - in-brake-zone flag;
    - brake-zone phase.
  - Rationale:
    - The older `racing` profile exposed distance to the next braking gate, but after crossing the gate that feature jumped to the next gate.
    - The policy therefore had weak explicit evidence that it was inside the Rettifilo brake zone while QC was already demanding braking.
  - Transfer probe:
    - `artifacts\ppo-racingv2-transfer-probe-goal-64-20260603-020826`.
    - Initial transfer preserved `970.773m`, proving the observation expansion did not destroy the current behavior.
  - Failed `racing_v2` training run:
    - `artifacts\ppo-racingv2-brakedemand-transfer-goal-10k-20260603-021215`.
    - Initial eval: `970.775m`, `20/120`, collision.
    - `2504` timesteps: unchanged.
    - `5008` timesteps: collapsed to about `213.22m`.
    - `7512` timesteps: still collapsed; run was killed.
  - Decision:
    - `racing_v2` is retained as a useful observation upgrade, but observation features alone did not fix the learning loop.
- Root-cause diagnosis:
  - The chicane-skill segment curriculum counted completion using progress only.
  - Example: `rettifilo-approach-brake` could start around `450-560m`, target `720m`, and count as `segment_complete` even if the policy arrived at `720m` at roughly `330kph`.
  - That made segment completion metrics look successful while teaching the exact behavior that later fails the real normal-start lap.
  - This explains why repeated runs could preserve or return to the `966-971m` range without learning the missing braking behavior.
- Implementation fix:
  - `CurriculumStage` now supports `target_max_speed_kph`.
  - Chicane-skill stages for Rettifilo and Roggia now have target max speeds at approach, turn-in, apex, exit, post-exit, and full-chicane targets.
  - `CurriculumSampler.sample_options` emits `segment_target_max_speed_kph` when a stage has a speed gate.
  - `MonzaSim` stores the segment speed gate from reset options, includes it in `info`, and marks `segment_complete` only when both progress and speed requirements are met.
  - State restoration clears stale segment speed gates.
  - State-library chicane curriculum now preserves the target max speed in train-time stage construction.
- Validation:
  - `uv run --no-sync ruff check src/f1rl/curriculum.py src/f1rl/sim.py src/f1rl/train.py tests/test_curriculum.py` -> passed.
  - `uv run --no-sync pyright src/f1rl` -> `0` errors, `0` warnings.
  - `uv run --no-sync pytest tests/test_curriculum.py -q` -> passed.
  - `uv run --no-sync pytest -q` -> passed; known SB3 `VecMonitor` warnings only.
- Current status after this fix:
  - Strict goal remains unmet: no valid normal-start PPO lap and no `<=80.0s` lap.
  - Best honest candidate remains `970.775m`, `20/120`, collision.
  - The next PPO run must use the speed-gated chicane curriculum so segment success means braking skill, not merely reaching the target distance.

### Racing-v2 VecNormalize Compatibility And Speed-Gated Probe - 2026-06-03
- Probe failure found:
  - Command attempted to initialize `observation_profile=racing_v2` from the best `racing` checkpoint while loading `best_vecnormalize.pkl`.
  - SB3 rejected the old VecNormalize object because the saved observation shape was `(31,)` and `racing_v2` uses `(35,)`.
  - This was an infrastructure issue, not a PPO behavior result.
- Implementation fix:
  - Added `_load_vecnormalize_with_reward_fallback` in `src\f1rl\train.py`.
  - Exact VecNormalize loads still use SB3's standard path.
  - If the old VecNormalize file has `norm_obs=False` and only the observation shape changed, training now creates a fresh wrapper for the new env and carries over reward running statistics only.
  - If the old file used observation normalization, the fallback does not apply because old observation statistics would be invalid for the new shape.
  - Run metadata now records `vec_normalize_load_mode`, e.g. `exact`, `fresh`, or `reward_stats_only_observation_shape_changed:(31,)->(35,)`.
- Test added:
  - `tests\test_policy_train_smoke.py::test_vecnormalize_reward_fallback_allows_observation_expansion`.
- Validation:
  - `uv run --no-sync ruff check src/f1rl/train.py tests/test_policy_train_smoke.py` -> passed.
  - `uv run --no-sync pytest tests/test_policy_train_smoke.py::test_vecnormalize_reward_fallback_allows_observation_expansion -q` -> passed.
  - `uv run --no-sync pyright src/f1rl` -> `0` errors, `0` warnings.
  - `uv run --no-sync pytest -q` -> passed; known SB3 `VecMonitor` warnings only.
- End-to-end speed-gated probe:
  - Command:
    ```powershell
    uv run --no-sync python -m f1rl.train --timesteps 64 --seed 989 --n-envs 2 --max-steps 1500 --device auto --require-gpu --vec-env subproc --initialize-from-checkpoint "artifacts\ppo-transfer-racing-actionhead-rettifilo-mix-goal-60k-20260603-013341\best_model.zip" --vec-normalize-path "artifacts\ppo-transfer-racing-actionhead-rettifilo-mix-goal-60k-20260603-013341\best_vecnormalize.pkl" --action-mode discrete --action-set racing --observation-profile racing_v2 --curriculum segments --curriculum-preset chicane-skill --curriculum-chicane rettifilo --curriculum-stage-count 1 --curriculum-state-library "artifacts\state-library-scripted-full-m4-20260602\state_library.json" --curriculum-promotion-resets 1000000 --curriculum-normal-start-probability 0.0 --reward-lateral-penalty-scale 0.004 --reward-track-limit-penalty-scale 0.004 --reward-heading-deadzone-deg 8 --reward-heading-penalty-scale 0.001 --reward-speed-target-min-kph 80 --reward-speed-target-max-kph 340 --reward-speed-target-heading-scale 2.5 --reward-speed-target-deadzone-kph 18 --reward-speed-target-penalty-scale 0.001 --reward-overspeed-throttle-penalty-scale 0.003 --reward-overspeed-brake-reward-scale 0.001 --normalize-reward --n-steps 64 --batch-size 64 --n-epochs 1 --learning-rate 0.00001 --gamma 0.997 --ent-coef 0.002 --checkpoint-every 64 --eval-every 64 --eval-episodes 2 --telemetry selected --telemetry-every 1 --run-name ppo-racingv2-speedgated-curriculum-probe-goal-64
    ```
  - Artifact: `artifacts\ppo-racingv2-speedgated-curriculum-probe-goal-64-20260603-023012`.
  - Metadata:
    - `observation_profile=racing_v2`.
    - `action_set=racing`.
    - `vec_normalize_load_mode=reward_stats_only_observation_shape_changed:(31,)->(35,)`.
    - `transfer_initialization=true`.
  - Eval rows:
    - `0`: normal-start `970.775m`, `0` completion rate, segment completion rate `0.0`.
    - `64`: normal-start `970.775m`, `0` completion rate, segment completion rate `0.0`.
    - `128`: normal-start `970.775m`, `0` completion rate, segment completion rate `0.0`.
  - Stage metrics:
    - `rettifilo-approach-brake`: `2` episodes, segment completion rate `0.0`, mean best progress `968.006m`, termination `off_track`.
  - Interpretation:
    - The old policy still reaches the old near-970m crash region.
    - The corrected speed-gated segment eval no longer counts this as a Rettifilo skill success.
    - This confirms the next run will train against a more truthful curriculum target.

### Transcript-Driven Aggressive Mode And Brake-Zone Assist Levers - 2026-06-03
- User course correction:
  - The active target remains unchanged: a valid normal-start PPO Monza lap near the Fast-F1 reference target.
  - The current loop was too conservative and was stopped.
  - The next milestone is not another small distance gain; it is changing the first bad event so the car brakes before Rettifilo turn-in and exits the section alive.
- Stopped run:
  - `artifacts\ppo-racingv2-speedgated-rettifilo-goal-30k-20260603-023655`.
  - It was interrupted during initial evaluation after the user instructed a pause and transcript reread.
  - It is not used as learning evidence.
- Required files reread:
  - `AGENTS.md`
  - `Prompt.md`
  - `Plan.md`
  - `Implement.md`
  - `LearningPlan.md`
  - `fine-tuned learning plan.md`
  - `transcripts\README.md`
  - `transcripts\01-yosh-trackmania-2023.txt`
  - `transcripts\02-yosh-noseboost.txt`
  - `transcripts\03-yosh-a01.txt`
  - `transcripts\04-yosh-a06.txt`
  - `transcripts\05-f1rl-methods-summary.txt`
- Transcript-derived operating rules now active:
  - Treat the `970m` plateau as a failed training objective, not a runtime shortage.
  - Make the old easy behavior fail quickly during training.
  - Run short aggressive experiments with explicit rejection conditions.
  - Keep section changes only if QC/telemetry show a materially different first bad event.
  - Preserve elite states only when braking, entry, apex, exit, validity, speed, heading, and lateral criteria are actually met.
  - Promote only metadata-faithful normal-start PPO evaluations with scaffold rewards and training assists disabled.
- Implementation:
  - Added training-only `AssistConfig.throttle_brake_demand_terminate`.
  - Added `AssistConfig.throttle_brake_demand_min_throttle`.
  - Added `AssistConfig.brake_zone_progress_multiplier`.
  - Added reward component `assist_brake_zone_progress_suppression`.
  - `MonzaSim` can now:
    - terminate throttle-through-brake-demand immediately during assisted section training;
    - cancel or scale progress reward while the car is overspeed in an active brake zone.
  - Trainer CLI flags added:
    - `--assist-throttle-brake-demand-terminate`
    - `--assist-throttle-brake-demand-min-throttle`
    - `--assist-brake-zone-progress-multiplier`
- Rationale:
  - The current failure is full throttle at about `520.8m`, `334kph`, where the section target is about `115kph`.
  - Existing progress reward still made "go far fast and crash" locally attractive.
  - The new levers directly remove that incentive during training-only Rettifilo experiments.
- Validation:
  - `uv run --no-sync ruff check src/f1rl/config.py src/f1rl/sim.py src/f1rl/train.py src/f1rl/telemetry.py tests/test_sim.py` -> passed.
  - `uv run --no-sync pytest tests/test_sim.py::test_assist_can_terminate_throttle_in_brake_demand tests/test_sim.py::test_assist_can_suppress_brake_zone_progress_reward tests/test_sim.py::test_assist_components_are_zero_by_default -q` -> passed.
  - `uv run --no-sync pyright src/f1rl` -> `0` errors, `0` warnings.
  - `uv run --no-sync pytest -q` -> passed; known SB3 `VecMonitor` warning only.
- Next mini-experiment:
  - A short hard Rettifilo section run from the current best racing-action model.
  - Use `racing_v2`, racing discrete actions, speed-gated Rettifilo curriculum, progress suppression in brake demand, immediate throttle-through-brake-demand termination, hard no-brake penalty, hard overspeed turn-in termination, and high overspeed-action penalties.
  - Rejection condition: if the first trained eval still reports `throttle_during_brake_demand` around `520m`, reject quickly.

### Aggressive Rettifilo Brake-Gate Mini-Experiment 1 Rejected - 2026-06-03
- Hypothesis:
  - If the old full-throttle behavior is terminal during assisted section training, PPO should discover a braking alternative.
- Command:
  ```powershell
  uv run --no-sync python -m f1rl.train --timesteps 5000 --seed 1001 --n-envs 8 --max-steps 1500 --device auto --require-gpu --vec-env subproc --initialize-from-checkpoint "artifacts\ppo-transfer-racing-actionhead-rettifilo-mix-goal-60k-20260603-013341\best_model.zip" --vec-normalize-path "artifacts\ppo-transfer-racing-actionhead-rettifilo-mix-goal-60k-20260603-013341\best_vecnormalize.pkl" --action-mode discrete --action-set racing --observation-profile racing_v2 --curriculum segments --curriculum-preset chicane-skill --curriculum-chicane rettifilo --curriculum-stage-count 2 --curriculum-state-library "artifacts\state-library-scripted-full-m4-20260602\state_library.json" --curriculum-promotion-resets 20 --curriculum-normal-start-probability 0.0 --assist-enabled --assist-throttle-brake-demand-terminate --assist-throttle-brake-demand-min-throttle 0.6 --assist-throttle-brake-demand-penalty-scale 20.0 --assist-brake-zone-progress-multiplier 0.0 --assist-no-brake-penalty -20.0 --assist-no-brake-min-brake 0.10 --assist-overspeed-turn-in-terminate --assist-overspeed-turn-in-margin-kph 20.0 --assist-overspeed-turn-in-penalty -120.0 --reward-progress-scale 0.02 --reward-collision-penalty -200.0 --reward-off-track-penalty -200.0 --reward-lateral-penalty-scale 0.006 --reward-track-limit-penalty-scale 0.006 --reward-heading-deadzone-deg 8 --reward-heading-penalty-scale 0.002 --reward-speed-target-min-kph 70 --reward-speed-target-max-kph 340 --reward-speed-target-heading-scale 2.5 --reward-speed-target-deadzone-kph 8 --reward-speed-target-penalty-scale 0.008 --reward-overspeed-throttle-penalty-scale 0.05 --reward-overspeed-brake-reward-scale 0.01 --normalize-reward --n-steps 256 --batch-size 256 --n-epochs 2 --learning-rate 0.00003 --gamma 0.995 --ent-coef 0.006 --checkpoint-every 2500 --eval-every 2500 --eval-episodes 2 --telemetry selected --telemetry-every 1 --run-name ppo-aggressive-rettifilo-brakegate-goal-5k
  ```
- Artifact:
  - `artifacts\ppo-aggressive-rettifilo-brakegate-goal-5k-20260603-024536`.
- Eval progression:
  - `0`: assisted normal-start best progress `520.81m`; segment completion `0.0`.
  - `2504`: assisted normal-start best progress `520.91m`; segment completion `0.0`.
  - `5008`: assisted normal-start best progress `520.93m`; segment completion `0.0`.
- QC:
  - Command:
    ```powershell
    uv run --no-sync python -m f1rl.qc --telemetry "artifacts\ppo-aggressive-rettifilo-brakegate-goal-5k-20260603-024536\eval\selected_telemetry\ppo_full_lap_train_00005008-episode-000-steps.jsonl" --max-telemetry-files 1
    ```
  - Artifact: `artifacts\qc-20260603-024747`.
  - First bad event: still `throttle_during_brake_demand`.
  - First bad progress: `520.925m`.
  - First bad speed: `333.831kph`.
  - Terminal: `assist_throttle_brake_demand`.
  - Actions before failure: `176` `throttle`, `4` `brake_right`.
- Decision:
  - Rejected by the explicit rejection rule.
  - The assist successfully made the old behavior fail at the brake gate, but PPO did not discover the alternative.
- Important diagnosis:
  - This run used `artifacts\state-library-scripted-full-m4-20260602\state_library.json`.
  - That library has only one Rettifilo approach snapshot in the `450-560m` range: about `501.4m` at `106.8kph`.
  - It therefore did not train the actual high-speed failure condition of `~334kph` at the brake gate.
- Implementation fix for the next experiment:
  - Added focus-target curriculum fields:
    - `--curriculum-focus-target-progress-m`
    - `--curriculum-focus-target-max-speed-kph`
  - Focus windows can now sample high-speed starts near `520m` and require reaching a fixed target such as `720m` under a speed gate such as `190kph`.
- Validation after focus-target change:
  - `uv run --no-sync ruff check src/f1rl/curriculum.py src/f1rl/train.py tests/test_curriculum.py` -> passed.
  - `uv run --no-sync pytest tests/test_curriculum.py::test_training_curriculum_focus_stage_can_target_progress_and_speed_gate -q` -> passed.
  - `uv run --no-sync pyright src/f1rl` -> `0` errors, `0` warnings.
  - `uv run --no-sync pytest -q` -> passed; known SB3 `VecMonitor` warning only.
- Next hypothesis:
  - Training must repeatedly start near the real failure state: `480-560m`, `310-340kph`, target `720m`, target max speed `190kph`.
  - If the policy still chooses throttle at the brake gate after this high-speed focus run, reject again and switch to policy/action-logit intervention or elite action search.

### Active Goal Prompt Reread And Scratch-Curriculum Reset - 2026-06-03
- Active prompt:
  - `goal.md` is now the active replacement goal prompt.
  - The final success criterion is unchanged and strict: scratch/random SB3 PPO on CUDA completes a valid normal-start Monza lap near `<=80.0s`.
  - Scratch means the PPO weights are random, not initialized from ghost, scripted, imitation, or existing trained PPO checkpoints.
  - Curriculum is required and is not a loophole: isolate the current blocker, train the missing section skill, prove it with telemetry/QC/replay, then transfer to linked/full-lap training.
  - Segment-only, assisted, scaffolded, partial-distance, or transferred-policy artifacts do not count as final success.
- Required files reread before additional training:
  - `goal.md`
  - `AGENTS.md`
  - `Prompt.md`
  - `Plan.md`
  - `Implement.md`
  - `LearningPlan.md`
  - `fine-tuned learning plan.md`
  - `Documentation.md` latest audit sections
  - `transcripts\README.md`
  - `transcripts\01-yosh-trackmania-2023.txt`
  - `transcripts\02-yosh-noseboost.txt`
  - `transcripts\03-yosh-a01.txt`
  - `transcripts\04-yosh-a06.txt`
  - `transcripts\05-f1rl-methods-summary.txt`
- Practical transcript interpretation for the next loop:
  - Do not keep replaying the first kilometer when the missing behavior is a section skill.
  - Make the bad local behavior impossible or expensive in focused training.
  - Use starts near the actual failure state, not only convenient low-speed scripted snapshots.
  - Reward and gate braking/entry/exit quality directly during section training.
  - Remove training assists and scaffold rewards before any promotion claim.
  - Preserve successful states only when speed, braking, heading, lateral error, validity, and no-collision criteria are met.
- Online research checked before the next run:
  - Stable-Baselines3 RL tips: custom environments generally need bounded/normalized observations, separate evaluation environments, wrapper-aware eval, multiple runs/seeds, and reward engineering iterations. Source: https://stable-baselines3.readthedocs.io/en/v2.0.0/guide/rl_tips.html
  - Gymnasium custom environment guidance: action and observation spaces must define the contract, and reset options are the right hook for controlled start distributions. Source: https://gymnasium.farama.org/v1.0.0/introduction/create_custom_env/
  - PPO paper: PPO alternates environment sampling with multiple minibatch epochs on a clipped/surrogate policy objective, which supports short iterative experiments but does not fix a bad reward/start distribution by itself. Source: https://arxiv.org/abs/1707.06347
  - Automatic Curriculum Learning survey: curricula shape agent experience by adapting tasks to capability to improve sample efficiency, exploration, generalization, and sparse-reward learning. Source: https://arxiv.org/abs/2003.04664
  - Self-Paced Deep RL: progressively adapting task distributions toward the target task improves learning speed/stability in curriculum settings. Source: https://arxiv.org/abs/2004.11812
  - Reverse Curriculum Generation: start-state distributions should focus on intermediate-difficulty states and expand from mastered/near-goal states; this supports using Rettifilo section starts and elite states rather than blind full-lap PPO. Source: https://bair.berkeley.edu/blog/2017/12/20/reverse-curriculum/
- Local validation before new training:
  - `uv run --no-sync python -m f1rl.hardware --json` -> CUDA available, `NVIDIA GeForce RTX 4060 Laptop GPU`, Torch `2.10.0+cu128`, CUDA `12.8`.
  - Local versions: Stable-Baselines3 `2.8.0`, Gymnasium `1.2.3`, Pygame `2.6.1`.
  - `uv run --no-sync ruff check .` -> passed.
  - `uv run --no-sync pyright src/f1rl` -> `0` errors, `0` warnings.
  - `uv run --no-sync pytest -q` -> passed; known SB3 `VecMonitor` warning only.

### Aggressive Rettifilo High-Speed Focus Diagnostic Rejected - 2026-06-03
- Status:
  - Diagnostic only, not a promotion candidate, because it initialized from an existing trained PPO checkpoint.
- Hypothesis:
  - Starting directly near the real high-speed failure state (`480-560m`, `310-340kph`) would expose the missing braking skill better than low-speed scripted-library starts.
- Artifact:
  - `artifacts\ppo-aggressive-rettifilo-highspeed-focus-goal-6k-20260603-025224`
- Result:
  - Initial assisted full-lap eval: `520.806m`, termination `assist_throttle_brake_demand`.
  - Segment evals: `0.0` segment completion throughout.
  - Final normal-start eval collapsed to about `152.116m`, `3/120`, `off_track`.
  - Best final segment evidence reached only about `523.467m` from the focus start and terminated by `assist_throttle_brake_demand`.
- QC:
  - Artifact: `artifacts\qc-aggressive-highspeed-focus-segment-20260603-025650\qc-20260603-025618`.
  - First bad event changed from pure `throttle_during_brake_demand` to `overspeed_at_braking_zone` while braking at `520.584m`, `314.938kph`.
  - Terminal event remained `assist_throttle_brake_demand` at `523.467m`, `311.829kph`.
  - Actions before failure: `24` throttle, `2` brake_left, `1` brake_right.
- Decision:
  - Rejected as a learning run because there was no segment completion and the full-lap behavior collapsed.
  - Kept as diagnostic evidence: high-speed starts did expose a slightly different behavior, but the policy still failed to retain braking and returned to throttle almost immediately.

### High-Speed Rettifilo Scripted State Library Generated - 2026-06-03
- Purpose:
  - Create physically coherent high-speed Rettifilo states for curriculum/backchaining because the old scripted full-lap library was too slow in the approach range.
- Artifact:
  - `artifacts\highspeed-rettifilo-scripted-library-20260603-030010`
  - State library: `artifacts\highspeed-rettifilo-scripted-library-20260603-030010\state_library.json`
- Contents:
  - `91` snapshots from scripted high-speed Rettifilo attempts.
  - Approach snapshots include speeds up to about `330kph`.
  - Selected rollouts from starts at `480m`, `500m`, `520m`, `540m`, and `560m` could reach the Rettifilo exit/post-exit target under speed gates.
- Guardrail:
  - This state library is a curriculum/debugging resource only.
  - It is not an imitation initialization and does not count toward final success.
  - Final promotion still requires scratch PPO weights and honest normal-start eval with assists/scaffolds disabled.

### Aggressive Rettifilo Backchain Library Diagnostic Rejected - 2026-06-03
- Status:
  - Diagnostic only, not a promotion candidate, because it resumed from a previous trained PPO checkpoint.
- Hypothesis:
  - Backchaining from the high-speed Rettifilo state library would reduce the approach speed enough for PPO to discover sustained braking.
- Artifact:
  - `artifacts\ppo-aggressive-rettifilo-backchain-library-goal-8k-20260603-030432`
- Result:
  - Initial normal-start eval was effectively dead: about `3.595m`, `max_steps`.
  - Final normal-start eval was still collapsed: about `214.165m`, `4/120`, `off_track`.
  - Segment completion remained `0.0`.
  - Final segment evidence reached only `523.775m` and terminated by `assist_throttle_brake_demand`.
- QC:
  - Artifact: `artifacts\qc-aggressive-backchain-segment-20260603-031000\qc-20260603-030948`.
  - First bad event: `overspeed_at_braking_zone` at `521.570m`, `240.316kph`, action `brake_left`.
  - Terminal event: `assist_throttle_brake_demand` at `523.775m`, `238.545kph`, action `throttle`.
  - Actions before failure: only `brake_left: 1`; the very next relevant action returned to throttle.
- Decision:
  - Rejected.
  - Useful signal: the state-library/backchain setup lowered speed from `~315kph` to `~240kph`, but the policy still did not learn sustained braking or a valid section exit.

### Next Scratch Curriculum Experiment - 2026-06-03
- Active hypothesis:
  - The transferred policy is too biased toward the old throttle solution. A scratch PPO policy trained only on the real high-speed Rettifilo brake gate should learn the local "do not throttle, brake and slow down" skill more cleanly.
- Experiment shape:
  - Random PPO weights.
  - `observation_profile=racing_v2`.
  - `action_set=racing`.
  - High-speed focus starts around `520m`, `310-340kph`.
  - Target `720m` with target max speed `190kph`.
  - Strong training-only assist termination for throttle-through-brake-demand.
  - Zero progress reward in active brake demand.
  - High entropy to force exploration.
  - Short eval interval and immediate QC inspection.
- Rejection condition:
  - Reject quickly if segment eval still ends at `~520-524m` with `assist_throttle_brake_demand` and action histograms dominated by throttle.
- Keep condition:
  - Keep and scale only if telemetry shows sustained braking, lower speed at `520-720m`, later first-bad-event, or nonzero speed-gated section completion.

### Scratch Rettifilo High-Speed Brake-Gate Experiment 1 Rejected, 4k Branch Preserved - 2026-06-03
- Hypothesis:
  - The transferred policy was too biased toward the old throttle solution; a random PPO policy trained directly on high-speed Rettifilo starts would discover braking more cleanly.
- Command:
  ```powershell
  uv run --no-sync python -m f1rl.train --timesteps 12000 --seed 1101 --n-envs 8 --max-steps 900 --device auto --require-gpu --vec-env subproc --action-mode discrete --action-set racing --observation-profile racing_v2 --curriculum segments --curriculum-focus-start-progress-m 520 --curriculum-focus-window-m 80 --curriculum-focus-target-progress-m 720 --curriculum-focus-target-max-speed-kph 190 --curriculum-focus-min-speed-kph 310 --curriculum-focus-max-speed-kph 340 --curriculum-focus-position-noise-m 0.3 --curriculum-focus-heading-noise-deg 1.0 --curriculum-focus-speed-noise-kph 2.0 --curriculum-promotion-resets 1000000 --curriculum-normal-start-probability 0.0 --assist-enabled --assist-throttle-brake-demand-terminate --assist-throttle-brake-demand-min-throttle 0.25 --assist-throttle-brake-demand-penalty-scale 40.0 --assist-brake-zone-progress-multiplier 0.0 --assist-no-brake-penalty -40.0 --assist-no-brake-min-brake 0.10 --assist-overspeed-turn-in-terminate --assist-overspeed-turn-in-margin-kph 75.0 --assist-overspeed-turn-in-penalty -160.0 --reward-progress-scale 0.01 --reward-collision-penalty -220.0 --reward-off-track-penalty -220.0 --reward-lateral-penalty-scale 0.006 --reward-track-limit-penalty-scale 0.006 --reward-heading-deadzone-deg 8 --reward-heading-penalty-scale 0.002 --reward-speed-target-min-kph 70 --reward-speed-target-max-kph 340 --reward-speed-target-heading-scale 2.5 --reward-speed-target-deadzone-kph 8 --reward-speed-target-penalty-scale 0.006 --reward-overspeed-throttle-penalty-scale 0.08 --reward-overspeed-brake-reward-scale 0.03 --reward-scaffold-brake-reward-scale 3.0 --reward-scaffold-no-throttle-penalty-scale 4.0 --reward-scaffold-turn-in-speed-penalty-scale 2.0 --reward-scaffold-apex-clean-reward-scale 0.05 --reward-scaffold-exit-alignment-reward-scale 0.05 --reward-scaffold-exit-speed-reward-scale 0.05 --normalize-reward --n-steps 256 --batch-size 256 --n-epochs 4 --learning-rate 0.0001 --gamma 0.995 --ent-coef 0.04 --checkpoint-every 2000 --eval-every 2000 --eval-episodes 4 --telemetry selected --telemetry-every 1 --run-name ppo-scratch-rettifilo-highspeed-brakegate-goal-12k
  ```
- Artifact:
  - `artifacts\ppo-scratch-rettifilo-highspeed-brakegate-goal-12k-20260603-032057`
- Validation/hardware:
  - `device=cuda`, `vec_env=subproc`, training FPS `68.2`, env steps/sec `545.7`.
  - Initial policy was scratch/random: `scratch_initialization=true`, `transfer_initialization=false`, `resume_checkpoint=null`.
- Eval summary:
  - `0`: full-lap `0.0m`; segment completion `0.0`; mean segment best `597.61m`; segment terminations `1` assist throttle, `3` collisions.
  - `2000`: full-lap `0.0m`; segment completion `0.0`; mean segment best `586.09m`; terminations `1` assist throttle, `3` collisions.
  - `4000`: full-lap `0.0m`; segment completion `0.0`; mean segment best `615.71m`; mean segment delta `110.90m`; terminations `4` collisions.
  - `6000`: full-lap `0.0m`; segment completion `0.0`; mean segment best `554.96m`; terminations `4` assist throttle.
  - `8000`: full-lap `27.85m`; segment completion `0.0`; mean segment best `543.75m`; terminations `4` assist throttle.
  - `10000`: full-lap `24.99m`; segment completion `0.0`; mean segment best `536.06m`; terminations `4` assist throttle.
  - `12000`: full-lap `12.58m`; segment completion `0.0`; mean segment best `530.43m`; terminations `4` assist throttle.
- QC at the useful `4000` checkpoint:
  - Telemetry: `artifacts\ppo-scratch-rettifilo-highspeed-brakegate-goal-12k-20260603-032057\eval\selected_telemetry\ppo_curriculum_segment_train_00004000-episode-000-steps.jsonl`.
  - QC artifact: `artifacts\qc-scratch-highspeed-brakegate-4k-20260603-032057\qc-20260603-032448`.
  - First bad event: `overspeed_at_braking_zone` at `526.648m`, `313.340kph`.
  - Terminal: collision at `600.601m`, `258.736kph`.
  - Average throttle/brake in Rettifilo: `0.0` / `0.35`.
  - Action histogram: `soft_brake_soft_right: 59`.
  - Interpretation: the run discovered braking and stopped throttle, but collapsed to a single soft-brake/right-steer action that slowed too little and crashed before turn-in.
- QC at final `12000` checkpoint:
  - QC artifact: `artifacts\qc-scratch-highspeed-brakegate-12k-20260603-032057\qc-20260603-032448`.
  - First bad event: `throttle_during_brake_demand` at `520.072m`, `314.843kph`.
  - Terminal: `assist_throttle_brake_demand` at `520.072m`, `314.843kph`.
  - Average throttle/brake in Rettifilo: `0.5` / `0.0`.
  - Actions before failure: `half_throttle_right: 18`.
  - Interpretation: final policy regressed to the old root failure, now through half-throttle.
- Decision:
  - Final `12k` checkpoint rejected.
  - Experiment not promoted: no segment completion, no valid lap, no normal-start progress.
  - Preserve the `4000` scratch-derived checkpoint as a branch point because it materially changed behavior from throttle to braking and revealed the next blocker: full-brake intensity plus line/turn-in control.
- Next branch hypothesis:
  - Continue from `artifacts\ppo-scratch-rettifilo-highspeed-brakegate-goal-12k-20260603-032057\checkpoints\ppo_monza_4000_steps.zip`, not from the regressed final checkpoint.
  - Increase pressure for full braking instead of soft braking.
  - Add stronger steering/line/corridor pressure so the policy cannot solve the gate by repeating `soft_brake_soft_right`.
  - Keep target `720m <=190kph`; reject if it returns to throttle or a single wrong steering action.

### Scratch-Derived 4k Full-Brake/Line Branch Rejected - 2026-06-03
- Hypothesis:
  - The useful `4000` checkpoint from the scratch run could be refined by penalizing soft braking, adding corridor termination, and adding steering-target pressure.
- Command:
  ```powershell
  uv run --no-sync python -m f1rl.train --timesteps 8000 --seed 1102 --n-envs 8 --max-steps 900 --device auto --require-gpu --vec-env subproc --resume-checkpoint "artifacts\ppo-scratch-rettifilo-highspeed-brakegate-goal-12k-20260603-032057\checkpoints\ppo_monza_4000_steps.zip" --vec-normalize-path "artifacts\ppo-scratch-rettifilo-highspeed-brakegate-goal-12k-20260603-032057\vecnormalize.pkl" --action-mode discrete --action-set racing --observation-profile racing_v2 --curriculum segments --curriculum-focus-start-progress-m 520 --curriculum-focus-window-m 80 --curriculum-focus-target-progress-m 720 --curriculum-focus-target-max-speed-kph 190 --curriculum-focus-min-speed-kph 310 --curriculum-focus-max-speed-kph 340 --curriculum-focus-position-noise-m 0.3 --curriculum-focus-heading-noise-deg 1.0 --curriculum-focus-speed-noise-kph 2.0 --curriculum-promotion-resets 1000000 --curriculum-normal-start-probability 0.0 --assist-enabled --assist-throttle-brake-demand-terminate --assist-throttle-brake-demand-min-throttle 0.25 --assist-throttle-brake-demand-penalty-scale 40.0 --assist-brake-zone-progress-multiplier 0.0 --assist-no-brake-penalty -120.0 --assist-no-brake-min-brake 0.90 --assist-overspeed-turn-in-terminate --assist-overspeed-turn-in-margin-kph 75.0 --assist-overspeed-turn-in-penalty -160.0 --assist-virtual-corridor-m 9.0 --assist-virtual-corridor-penalty -80.0 --assist-virtual-corridor-terminate --reward-progress-scale 0.005 --reward-collision-penalty -240.0 --reward-off-track-penalty -240.0 --reward-lateral-penalty-scale 0.02 --reward-track-limit-penalty-scale 0.02 --reward-heading-deadzone-deg 6 --reward-heading-penalty-scale 0.006 --reward-speed-target-min-kph 70 --reward-speed-target-max-kph 340 --reward-speed-target-heading-scale 2.5 --reward-speed-target-deadzone-kph 4 --reward-speed-target-penalty-scale 0.01 --reward-overspeed-throttle-penalty-scale 0.08 --reward-overspeed-brake-reward-scale 0.08 --reward-steering-target-deadzone 0.08 --reward-steering-target-penalty-scale 0.15 --reward-scaffold-brake-reward-scale 5.0 --reward-scaffold-no-throttle-penalty-scale 5.0 --reward-scaffold-turn-in-speed-penalty-scale 3.0 --reward-scaffold-apex-clean-reward-scale 0.05 --reward-scaffold-exit-alignment-reward-scale 0.08 --reward-scaffold-exit-speed-reward-scale 0.05 --normalize-reward --n-steps 256 --batch-size 256 --n-epochs 3 --learning-rate 0.00005 --gamma 0.995 --ent-coef 0.02 --checkpoint-every 2000 --eval-every 2000 --eval-episodes 4 --telemetry selected --telemetry-every 1 --run-name ppo-scratch-rettifilo-4k-fullbrake-line-goal-8k
  ```
- Artifact:
  - `artifacts\ppo-scratch-rettifilo-4k-fullbrake-line-goal-8k-20260603-032828`
- Eval summary:
  - Initial resume: full-lap `0.0m`; segment completion `0.0`; mean segment best `594.63m`; terminations `4` virtual corridor.
  - `2000`: full-lap `0.0m`; segment completion `0.0`; mean segment best `595.49m`; terminations `4` virtual corridor.
  - `4000`: full-lap `0.0m`; segment completion `0.0`; mean segment best `577.74m`; terminations `4` virtual corridor.
  - `6000`: full-lap `0.0m`; segment completion `0.0`; mean segment best `535.66m`; terminations `4` assist throttle.
  - `8000`: full-lap `0.0m`; segment completion `0.0`; mean segment best `536.10m`; terminations `4` assist throttle.
- QC at `2000`:
  - Artifact: `artifacts\qc-scratch-4k-fullbrake-line-2k-20260603-032828\qc-20260603-033046`.
  - First bad event: `overspeed_at_braking_zone` at `539.500m`, `321.985kph`.
  - Terminal: `assist_virtual_corridor` at `591.507m`, `281.921kph`.
  - Action histogram: `soft_brake_soft_right: 39`.
  - Max lateral error: `9.030m`; max heading error: `17.383deg`.
- QC at `8000`:
  - Artifact: `artifacts\qc-scratch-4k-fullbrake-line-8k-20260603-032828\qc-20260603-033046`.
  - First bad event: `throttle_during_brake_demand` at `543.302m`, `315.608kph`.
  - Terminal: `assist_throttle_brake_demand`.
  - Action histogram: `half_throttle_left: 1`.
- Decision:
  - Rejected.
  - The `racing` action set still encourages soft/half action local optima under this setup.
  - Stronger no-brake penalties and corridor termination did not produce full braking or a speed-gated Rettifilo approach success.
- Next axis:
  - Run a fresh scratch high-speed focus with `action_set=legacy` to remove half-throttle and soft-brake actions from the local search.
  - Keep the same honest guardrail: section training can be assisted/scaffolded, but no final promotion can use assists/scaffolds or transferred/scripted initialization.

### Scratch Legacy-Action High-Speed Rettifilo Experiment Rejected - 2026-06-03
- Hypothesis:
  - Removing half-throttle and soft-brake actions would stop the local optima seen with the richer `racing` action set and force PPO to choose full braking in the high-speed Rettifilo brake gate.
- Artifact:
  - `artifacts\ppo-scratch-legacy-rettifilo-highspeed-brakegate-goal-10k-20260603-033227`
- Result:
  - Scratch/random initial policy: `scratch_initialization=true`, `transfer_initialization=false`, CUDA training.
  - `0`: segment completion `0.0`, mean segment best `538.66m`, mean segment delta `15.89m`.
  - `2000`: segment completion `0.0`, mean segment best `554.49m`, mean segment delta `46.90m`.
  - `4000`: segment completion `0.0`, mean segment best `554.12m`, mean segment delta `0.0m`.
  - `6000`: segment completion `0.0`, mean segment best `535.62m`, mean segment delta `0.76m`.
  - `8000`: segment completion `0.0`, mean segment best `523.02m`, mean segment delta `14.44m`.
  - `10000`: segment completion `0.0`, mean segment best `520.46m`, mean segment delta `19.72m`.
- QC at useful `2000` interval:
  - Artifact: `artifacts\qc-scratch-legacy-highspeed-2k-20260603-033227\qc-20260603-033504`.
  - First bad event: `overspeed_at_braking_zone` at `521.134m`, `284.003kph`.
  - Terminal: `assist_virtual_corridor` at `547.648m`, `264.484kph`.
  - Action histogram: `brake_right: 27`, `right: 22`, `throttle_left: 1`.
  - Average throttle/brake: `0.02` / `0.54`.
  - Max lateral error: `12.195m`; max heading error: `22.338deg`.
- QC at final `10000` interval:
  - Artifact: `artifacts\qc-scratch-legacy-highspeed-10k-20260603-033227\qc-20260603-033504`.
  - First bad event: `throttle_during_brake_demand` at `520.398m`, `302.102kph`.
  - Terminal: `assist_throttle_brake_demand`.
  - Action histogram: `throttle_left: 11`.
- Decision:
  - Rejected.
  - The legacy action space did force full-brake actions early, but the policy coupled braking with steering (`brake_right`) and exited the corridor well before the `720m` turn-in target.
  - Later training regressed back to throttle.
- Probe:
  - A one-off simulator probe showed `_target_steer()` is near straight through the brake phase:
    - `520m`: `0.015`
    - `540m`: `0.015`
    - `560m`: `0.016`
    - `600m`: `-0.004`
    - `650m`: `0.019`
  - Therefore the next experiment can use a much stronger existing `reward_steering_target_penalty_scale` instead of adding a new code path yet.
- Next axis:
  - Decompose Rettifilo further into a pure braking micro-stage.
  - Start around `500-540m`, `310-340kph`.
  - Target `600m` with max speed around `260kph`.
  - Use legacy actions, full-brake pressure, and strong straight-steering penalty.
  - Only after this micro-stage succeeds should turn-in/exit be trained.

### Scratch Legacy Pure-Brake Micro-Stage Rejected - 2026-06-03
- Hypothesis:
  - A smaller Rettifilo micro-stage (`~520m` start, target `600m <=260kph`) with legacy actions, zero progress reward, full-brake pressure, a narrow corridor, and strong target-steer penalty would teach the missing straight-line braking skill before turn-in.
- Artifact:
  - `artifacts\ppo-scratch-legacy-rettifilo-brake-straight-micro-goal-6k-20260603-033723`
- Result:
  - Scratch/random initial policy: `scratch_initialization=true`, `transfer_initialization=false`, CUDA training.
  - Eval table:
    - `0`: segment completion `0.0`, mean segment best `575.66m`, terminations `6` virtual-corridor.
    - `1000`: segment completion `0.0`, mean segment best `573.84m`, terminations `6` virtual-corridor.
    - `2000`: segment completion `0.0`, mean segment best `529.31m`, terminations `6` throttle-brake-demand.
    - `3000`: segment completion `0.0`, mean segment best `529.28m`, terminations `6` throttle-brake-demand.
    - `4000`: segment completion `0.0`, mean segment best `526.46m`, terminations `6` throttle-brake-demand.
    - `5000`: segment completion `0.0`, mean segment best `526.57m`, terminations `6` throttle-brake-demand.
    - `6000`: segment completion `0.0`, mean segment best `522.07m`, terminations `6` throttle-brake-demand.
  - Full-lap diagnostic at final was only `239.04m`, `4/120`, `max_steps`; no valid lap and no finish.
- QC:
  - Initial QC artifact: `artifacts\qc-scratch-legacy-brake-straight-micro-initial-20260603-033723\qc-20260603-034344`.
    - First bad event: `overspeed_at_braking_zone` at `526.979m`, `315.914kph`.
    - Terminal: `assist_virtual_corridor` at `580.577m`, `274.956kph`.
    - Actions: `left: 41`, average throttle/brake `0.0/0.0`.
  - `1000` QC artifact: `artifacts\qc-scratch-legacy-brake-straight-micro-1k-20260603-033723\qc-20260603-034344`.
    - First bad event: `overspeed_at_braking_zone` at `521.034m`, `309.547kph`.
    - Terminal: `assist_virtual_corridor` at `564.798m`, `276.273kph`.
    - Actions: `left: 39`, average throttle/brake `0.0/0.0`.
  - Final QC artifact: `artifacts\qc-scratch-legacy-brake-straight-micro-6k-20260603-033723\qc-20260603-034344`.
    - First bad event: `throttle_during_brake_demand` at `521.384m`, `318.552kph`.
    - Terminal: `assist_throttle_brake_demand` at the same point.
    - Actions: `throttle: 14`, average throttle/brake `1.0/0.0`.
- Diagnosis:
  - The run exposed a reward-design bug in the focused assisted setup.
  - Immediate throttle termination produced a short episode around `-217` reward, while wrong-but-longer coasting/steering attempts accumulated thousands of per-step no-brake and steering penalties before corridor termination.
  - PPO learned the short-episode escape hatch instead of braking.
  - The global speed-target reward did not penalize the straight brake-zone overspeed because the lookahead-derived target remained high on the straight approach; the active section pressure came mostly from action-dependent assist/scaffold terms.
- Decision:
  - Reject the final checkpoint and do not preserve this branch.
  - Next experiment should keep scratch/random weights but change the local objective so `throttle` in brake demand is catastrophically worse than attempting the segment, reduce or remove the per-step no-brake trap that makes long exploration look worse than quick death, and make full straight braking strongly positive.

### Scratch Legacy Anti-Escape Brake Micro-Stage Succeeded - 2026-06-03
- Hypothesis:
  - The prior micro-stage failed because quick throttle termination was less negative than longer exploration. Make throttle during brake demand catastrophically worse, reduce the per-step no-brake trap, keep full-brake reward high, and train the same `520m -> 600m <=220kph` gate from random PPO weights.
- Command:
  ```powershell
  uv run --no-sync python -m f1rl.train --timesteps 8000 --seed 1131 --n-envs 8 --max-steps 180 --device auto --require-gpu --vec-env subproc --action-mode discrete --action-set legacy --observation-profile racing_v2 --curriculum segments --curriculum-focus-start-progress-m 520 --curriculum-focus-window-m 20 --curriculum-focus-target-progress-m 600 --curriculum-focus-target-max-speed-kph 220 --curriculum-focus-min-speed-kph 315 --curriculum-focus-max-speed-kph 335 --curriculum-focus-position-noise-m 0.2 --curriculum-focus-heading-noise-deg 0.5 --curriculum-focus-speed-noise-kph 1.0 --curriculum-promotion-resets 1000000 --curriculum-normal-start-probability 0.0 --assist-enabled --assist-throttle-brake-demand-terminate --assist-throttle-brake-demand-min-throttle 0.25 --assist-throttle-brake-demand-penalty-scale 2000.0 --assist-brake-zone-progress-multiplier 0.0 --assist-no-brake-penalty -8.0 --assist-no-brake-min-brake 0.90 --assist-virtual-corridor-m 12.0 --assist-virtual-corridor-penalty -120.0 --assist-virtual-corridor-terminate --reward-progress-scale 0.0 --reward-collision-penalty -500.0 --reward-off-track-penalty -500.0 --reward-lateral-penalty-scale 0.01 --reward-track-limit-penalty-scale 0.01 --reward-heading-deadzone-deg 5 --reward-heading-penalty-scale 0.004 --reward-speed-target-min-kph 70 --reward-speed-target-max-kph 340 --reward-speed-target-heading-scale 2.5 --reward-speed-target-deadzone-kph 4 --reward-speed-target-penalty-scale 0.0 --reward-overspeed-throttle-penalty-scale 0.0 --reward-overspeed-brake-reward-scale 0.0 --reward-steering-target-deadzone 0.04 --reward-steering-target-penalty-scale 0.3 --reward-scaffold-brake-reward-scale 20.0 --reward-scaffold-no-throttle-penalty-scale 20.0 --reward-scaffold-turn-in-speed-penalty-scale 0.0 --normalize-reward --n-steps 128 --batch-size 128 --n-epochs 4 --learning-rate 0.0001 --gamma 0.99 --ent-coef 0.08 --checkpoint-every 1000 --eval-every 1000 --eval-episodes 8 --telemetry selected --telemetry-every 1 --run-name ppo-scratch-legacy-rettifilo-brake-anti-escape-micro-goal-8k
  ```
- Artifact:
  - `artifacts\ppo-scratch-legacy-rettifilo-brake-anti-escape-micro-goal-8k-20260603-034805`
- Result:
  - Scratch/random initialization confirmed: no resume, no transfer, CUDA training.
  - Eval progression:
    - `0`: segment completion `0.0`, mean segment best `584.61m`, terminations `8` throttle-brake-demand.
    - `1000`: segment completion `0.0`, mean segment best `586.57m`, terminations `8` throttle-brake-demand.
    - `2000`: segment completion `0.0`, mean segment best `597.28m`, terminations `8` virtual-corridor.
    - `3000-7000`: segment completion `0.0`, virtual-corridor failures.
    - `8000`: segment completion `1.0`, mean segment best `600.41m`, terminations `8` segment-complete.
  - Full-lap metric remained `0.0m` throughout and is irrelevant for this micro-stage.
- QC:
  - Useful early diagnostic: `artifacts\qc-scratch-legacy-brake-anti-escape-micro-2k-20260603-034805\qc-20260603-035143`.
    - Full braking appeared, but the policy coupled it with `brake_left/brake_right` and exited the corridor at `~596m`, `~266.5kph`.
  - Success artifact: `artifacts\qc-scratch-legacy-brake-anti-escape-micro-8k-20260603-034805\qc-20260603-040142`.
    - Terminal: `segment_complete` at `600.321m`, `159.150kph`.
    - Actions: `brake: 67`.
    - Average throttle/brake: `0.0` / `1.0`.
    - Max lateral error: `0.167m`.
    - Max heading error: `0.685deg`.
- Decision:
  - Preserve `artifacts\ppo-scratch-legacy-rettifilo-brake-anti-escape-micro-goal-8k-20260603-034805\checkpoints\ppo_monza_8000_steps.zip` as the first scratch-derived Rettifilo straight-braking micro-skill checkpoint.
  - This is not promotion toward final success by itself because it is assisted/scaffolded section training only.
  - Next stage: continue from this scratch-derived checkpoint into a longer braking/turn-in gate, e.g. `520m -> 720m <=190kph`, with scaffold/assist still enabled and immediate QC.

### Brake-Straight Action-Set Probe Rejected - 2026-06-03
- Purpose:
  - Test whether removing combined brake/steer actions would make the straight-brake micro-skill easier.
- Implementation:
  - Added `BRAKE_STRAIGHT_DISCRETE_ACTIONS` and CLI action set `brake_straight`.
  - Actions: coast, throttle, brake, left, right.
  - Focused validation passed:
    - `uv run --no-sync ruff check src\f1rl\config.py tests\test_sim.py` -> passed.
    - `uv run --no-sync pytest tests\test_sim.py -q` -> passed.
    - `uv run --no-sync python -m f1rl.train --help` -> `--action-set {brake_straight,expanded,legacy,racing}`.
- Artifact:
  - `artifacts\ppo-scratch-brakestreet-rettifilo-straight-brake-micro-goal-4k-20260603-035659`
- Result:
  - Stopped after the `2000` eval because `segment_completion_rate` remained `0.0` and mean segment best regressed from `593.50m` to `587.99m`.
  - QC at `1000`: `artifacts\qc-scratch-brakestreet-straight-brake-micro-1k-20260603-035659\qc-20260603-035849`.
    - Actions: `right: 49`, `left: 1`.
    - Average throttle/brake: `0.0` / `0.0`.
    - Terminal: `assist_virtual_corridor` at `588.018m`, `276.973kph`.
- Decision:
  - Reject as a training branch because it did not discover braking quickly.
  - Keep the action set available as an experimental tool, but the active branch is the successful legacy anti-escape `8000` checkpoint.

### Release-Gated Brake-Release Backchain and Rettifilo Corridor Audit - 2026-06-03
- Context:
  - The old honest normal-start PPO ceiling remains no valid lap, with best trusted normal-start progress around `970.775m` and collision near Rettifilo.
  - The active fine-tuned plan required rejecting fake segment success and proving linked brake-release transfer before moving to steering/full Rettifilo.
- Implementation changes:
  - Added strict segment release gating:
    - Reset options: `segment_require_release`, `segment_release_max_brake`, `segment_release_max_throttle`, `segment_release_min_speed_kph`, `segment_release_max_speed_kph`.
    - New terminal reason: `segment_release_gate_failed`.
    - Curriculum/CLI flags: `--curriculum-segment-require-release`, `--curriculum-segment-release-max-brake`.
  - Added strict speed/release support fields to `info`: `segment_require_release`, `segment_release_observed`.
  - Added `brake_release` action set: `brake`, `trail_brake`, `coast`.
  - Added `release_brake` action set: `coast`, `trail_brake`, `brake`.
  - Added dense scaffold `scaffold_brake_curve` to penalize deviation from a section-relative braking speed curve.
  - Added `racing_release` observation profile features for release threshold/surplus/deficit/in-band state.
  - Added `--initialize-source-action-set` so action-head expansion can map source rows by action semantics when transferring from a smaller action set.
  - Added state-library progress/speed filters used for release/backchain libraries.
- Validation:
  - `uv run --no-sync ruff check src\f1rl\config.py src\f1rl\telemetry.py src\f1rl\sim.py src\f1rl\curriculum.py src\f1rl\train.py tests\test_sim.py tests\test_curriculum.py` passed during development.
  - `uv run --no-sync pytest tests\test_sim.py tests\test_curriculum.py -q` passed (`48` focused tests).
  - `uv run --no-sync ruff check src\f1rl\train.py` passed after adding `--initialize-source-action-set`.
- Key rejected experiments:
  - `artifacts\ppo-link520-crispgate-1024-20260603-060019`: `0/8`, all `assist_overbrake_gate` through `1024`; rejected.
  - `artifacts\ppo-link520-brakerelease-scratch-1536-20260603-061056`: `0/8`, all `assist_overbrake_gate` through `512`; rejected early.
  - `artifacts\ppo-link520-brakecurve-term-1024-20260603-061547`: changed no-brake to overbrake but no completion; rejected at `768`.
  - `artifacts\ppo-link520-brakecurve-softover-1024-20260603-061914`: got `8/8` strict speed-gated completions at `384`, but QC showed fake release: `trail_brake:107`, no coast; rejected as promotion but kept as diagnostic.
  - `artifacts\ppo-link520-releasegate-transfer-1024-20260603-062810`: release gate correctly demoted trail-only behavior; `0/8` through `512`, all `segment_release_gate_failed`.
  - `artifacts\ppo-rel570-releasegate-transfer-1024-20260603-063155`: release-gated 570->650 with `brake_release` stayed `0/12`, gate failed; logit inspection showed deterministic action bias away from `coast`.
- Accepted release/backchain milestones:
  - `artifacts\ppo-rel570-releasebrake-gated-scratch-1024-20260603-063822`:
    - Initial random `release_brake` policy completed `12/12` on release-gated 570->650.
    - QC: `artifacts\qc-ppo-rel570-releasebrake-initial-20260603-063822\qc-20260603-063944`.
    - Selected replay: `segment_complete` at `650.522m`, `157.005kph`; actions `brake:10`, `trail_brake:1`, `coast:76`.
  - `artifacts\ppo-rel540-highspeed-softnobrake-1024-20260603-065156`:
    - High-speed 540->571 backchain initially failed speed gate, then solved at `896/1024` with `13/13` completions.
    - QC: `artifacts\qc-ppo-rel540-highspeed-softnobrake-1024-20260603-065156\qc-20260603-070026`.
    - Selected replay: `segment_complete` at `650.694m`, `157.422kph`; actions `brake:37`, `coast:77`.
  - `artifacts\ppo-link520-releasebrake-softnobrake-transfer-1024-20260603-070121`:
    - Strict linked 520->650 release-gated retest completed `8/8` at initial transfer.
    - QC: `artifacts\qc-ppo-link520-releasebrake-softnobrake-initial-20260603-070121\qc-20260603-070238`.
    - Selected replay: `segment_complete` at `650.784m`, `154.489kph`; actions `brake:50`, `coast:86`.
    - This is the first accepted linked brake-release transfer milestone. It is scaffolded/assisted section training, not final success.
- Steering / Rettifilo extension:
  - `artifacts\ppo-turnin760-expanded-transfer-1024-20260603-070512`:
    - Expanded action-set transfer to 520->760 completed `8/8` at initial transfer.
    - QC: `artifacts\qc-ppo-turnin760-expanded-initial-20260603-070512\qc-20260603-070714`.
    - Selected replay used no steering (`brake:43`, `coast:237`, avg abs steering `0.0`) and reached `760.199m` at `129.373kph`; accepted only as early turn-in progress, not steering proof.
  - `artifacts\ppo-rettifilo930-expanded-transfer-1024-20260603-070818`:
    - 520->930 completed `8/8` at initial transfer.
    - QC: `artifacts\qc-ppo-rettifilo930-expanded-initial-20260603-070818\qc-20260603-071158`.
    - Still no steering (`brake:45`, `coast:652`, avg abs steering `0.0`) and very slow terminal speed `75.325kph`; not full Rettifilo behavior.
  - `artifacts\ppo-rettifilo1220-expanded-transfer-1024-20260603-071315`:
    - Full Rettifilo-exit probe failed at initial/128 around `961-966m` with `assist_virtual_corridor`.
    - Training collapsed after `384` to `assist_throttle_brake_demand` around `523-525m`; reject continuation.
    - QC: `artifacts\qc-ppo-rettifilo1220-expanded-initial-20260603-071315\qc-20260603-071850`.
    - Selected replay: `assist_virtual_corridor` at `964.272m`, max lateral error `14.156m`, actions `brake:40`, `coast:808`, avg abs steering `0.0`.
  - `artifacts\ppo-corridor900-1220-expanded-1024-20260603-072009`:
    - 900->1220 corridor rung from the exact 1220 failure replay stayed at `assist_virtual_corridor` through `512`; stopped and rejected.
- Current diagnosis:
  - PPO has made real curriculum progress: it can now complete a strict, release-gated linked 520->650 brake-release segment with brake then coast.
  - It has not completed a normal-start full lap.
  - It has not completed full Rettifilo exit. The current blocker has moved from "does not brake before Rettifilo" to "does not steer/throttle out of Rettifilo; drifts out of virtual corridor around `964m`."
- Next action:
  - Do not continue mild PPO on the same 900->1220 setup.
  - Inspect action logits/actions around `900-965m` and run explicit steering/action-perturbation or elite search to find clean corridor exits.
  - Use that elite corridor state/action evidence to train a steering/throttle rung, then retest 520->1220 and only then return to honest normal-start eval.

### Line-Gated Turn-In / Exit Audit - 2026-06-03
- Context:
  - The accepted linked brake-release milestone remains `artifacts\ppo-link520-releasebrake-softnobrake-transfer-1024-20260603-070121`.
  - No honest normal-start full lap has completed; the trusted honest normal-start best remains around `970.775m`, ending in Rettifilo collision.
  - The immediate goal is not more 520->650 release polishing; it is linked transfer into full Rettifilo behavior.
- Implementation changes:
  - Fixed curriculum checkpoint selection:
    - Full-lap ranking still prioritizes valid full-lap completion.
    - Curriculum runs now select best checkpoints by segment completion, segment progress delta, then segment best progress.
    - This prevents section experiments from saving later collapsed policies over better section policies.
  - Added strict segment target gates:
    - `segment_target_min_speed_kph`
    - `segment_target_max_speed_kph`
    - `segment_target_max_lateral_error_m`
    - `segment_target_max_heading_error_deg`
    - New terminal reasons include `segment_min_speed_gate_failed`, `segment_lateral_gate_failed`, and `segment_heading_gate_failed`.
  - Added dense curriculum scaffold `scaffold_segment_speed` to penalize being outside the target speed band near a configured segment target.
  - Added transfer CLI `--initialize-new-action-bias-penalty` because new action rows were too strongly suppressed by the default `-4.0` bias when expanding from `release_brake`.
  - Added section-only action sets:
    - `turnin_power`: maintenance/half-throttle/soft-brake with soft steering, no pure coast, no full brake.
    - `turnin_micro`: same drive levels with `0.15` steering for smaller line corrections.
  - Added `src\f1rl\action_search.py` / `f1-action-search` for fixed-schedule probes and artifact capture.
- Validation:
  - `uv run --no-sync ruff check src\f1rl\config.py src\f1rl\telemetry.py src\f1rl\sim.py src\f1rl\curriculum.py src\f1rl\train.py src\f1rl\action_search.py tests\test_sim.py tests\test_curriculum.py tests\test_policy_train_smoke.py tests\test_action_search.py` -> passed.
  - `uv run --no-sync pytest tests\test_sim.py tests\test_curriculum.py tests\test_policy_train_smoke.py::test_curriculum_checkpoint_selection_prioritizes_segment_transfer tests\test_policy_train_smoke.py::test_transfer_initialization_can_expand_discrete_action_head tests\test_action_search.py -q` -> passed (`56` focused tests).
- Rejected probes and experiments:
  - Fixed-schedule 900->1220 action probes:
    - `artifacts\action-search-corridor900single-1220-cap8-20260603-miniprobe`
    - `artifacts\action-search-corridor900single-1220-cap20-20260603-poweredsteer`
    - `artifacts\action-search-preexit650-1220-cap24-20260603-linkedrung`
    - `artifacts\action-search-preexit650-1220-delayedturn-20260603-rung`
    - `artifacts\action-search-preexit650-1220-delay120left20-20260603-rung`
    - `artifacts\action-search-preexit650-1220-delay180left20-20260603-rung`
    - `artifacts\action-search-preexit650-1220-delay180left-turn12-20260603-rung`
    - `artifacts\action-search-preexit650-1220-delay180right20-20260603-rung`
    - Best variants changed behavior but still failed before full Rettifilo exit, usually corridor/collision around `850-983m`; rejected as non-transfer.
  - `artifacts\ppo-preexit650-corridorheading-hardgate-512-20260603-rung-20260603-083927`:
    - Corrected selector saved the `256` checkpoint, but selected replay failed at `855.585m` with `assist_virtual_corridor`.
    - First bad event: `wrong_heading` at `854.004m`; actions mostly hard right throttle/brake.
    - Rejected.
  - `artifacts\ppo-turnin650-930-linegated-256-20260603-rung-20260603-085445`:
    - New line/speed gates exposed full-brake collapse.
    - Selected replay: `brake:285`, stopped at `710.442m`, `no_progress`.
    - Rejected.
  - `artifacts\ppo-turnin650-930-releaseinit-overbraketerm-256-20260603-rung-20260603-085810`:
    - Release initialization and terminal overbrake removed brake camping but collapsed to coast.
    - Initial transfer reached the target but failed `segment_min_speed_gate_failed`; trained replay terminated early on overspeed assist.
    - Rejected.
  - `artifacts\ppo-turnin650-930-segspeed-nooverterm-256-20260603-rung-20260603-090322`:
    - Dense segment-speed penalty added.
    - Selected replay still pure `coast`, failed `segment_min_speed_gate_failed` at `930.338m`, `75.398kph`, heading `-10.265deg`.
    - Rejected as no action change.
  - `artifacts\ppo-turnin650-930-newactionbias05-256-20260603-rung-20260603-090813`:
    - Softer new-action transfer bias `-0.5`.
    - Selected replay still pure `coast`, timed out at `911.264m`, `79.636kph`.
    - Rejected.
  - `artifacts\ppo-turnin650-930-nobrakegate-256-20260603-rung-20260603-091818`:
    - Continued from `turnin_power`; no-brake gate made some episodes fail earlier.
    - Best selected behavior stayed straight `maintenance`, reached `930.616m` at `149.989kph`, failed only `segment_heading_gate_failed` at about `-10.122deg`.
    - Rejected as a steering skill, but kept as useful speed-carry evidence.
- Useful retained artifacts:
  - `artifacts\state-library-rettifilo-turnin-650_760-from930-20260603-linegate\state_library.json`: 5 early post-release snapshots.
  - `artifacts\ppo-turnin650-930-turninpower-256-20260603-rung-20260603-091309`:
    - `turnin_power` changed behavior from coast to straight maintenance throttle.
    - Selected replay reached `930.236m` at `165.270kph`, lateral `0.772m`, heading `-10.048deg`.
    - Failed strict speed/heading gates (`segment_speed_gate_failed`) but demonstrates a speed-carry rung; not accepted as steering proof.
  - `artifacts\state-library-rettifilo-exit-900_935-from-turninpower-20260603\state_library.json`: 3 high-speed exit-entry snapshots from the speed-carry rung.
- Deterministic control probes:
  - Constant `0.45` steering from 650/675 is too strong; left/right leave track around `694-720m`.
  - Constant `0.15` micro steering from 650/675 is still too strong if held continuously; left/right leave track around `717-761m`.
  - From 900/910/921, constant straight maintenance crashes around `968.8m`; micro-left can push farther (`~979-987m`) but still fails; soft-brake micro-left can reach about `1010.9m` but stops.
  - Large brute-force phase schedule search was stopped because it was too slow for the experiment loop; use smaller targeted probes or PPO instead.
- Current diagnosis:
  - Positive learning/progress: the current section stack can brake/release 520->650 and carry speed to 930m without overbraking.
  - Negative/blocked behavior: PPO has not learned closed-loop steering for Rettifilo exit. It still cannot complete 900/930->1220, and no full normal-start lap is complete.
  - First bad event has moved from old `throttle_during_brake_demand` to later turn-in/exit issues depending on rung:
    - `segment_min_speed_gate_failed` for pure coast.
    - `no_brake_before_turn_in` / `segment_speed_gate_failed` for straight maintenance.
    - `segment_heading_gate_failed` when speed is acceptable but steering is absent.
- Next action:
  - Treat 650->930 speed-carry as a rung with relaxed heading (`<=12deg`) and speed band (`105-170kph`), not as full Rettifilo success.
  - Train 900/930->1220 directly from `state-library-rettifilo-exit-900_935-from-turninpower-20260603`, using `turnin_micro` or a smaller delayed-steering curriculum.
  - Add or use a cheaper targeted phase-search path only if PPO cannot discover delayed micro-left / brake-left behavior.
  - Promote only after strict linked 520->1220 improves over the old `~970m` normal-start crash, then retest honest normal start.

### Honest Normal-Start Diagnostic Probe - 2026-06-03
- User-directed course correction:
  - Before committing more time to micro-rungs, run a short honest normal-start full-lap probe only as a diagnostic.
  - Use metadata-faithful eval/benchmark, no scaffold rewards, no training assists, no ghost/scripted/imitation initialization, and inspect telemetry/QC/replay/first-bad-event.
  - If the probe reproduces the Rettifilo `~970m` failure, immediately return to the linked Rettifilo transfer ladder.
- Probe target:
  - Checkpoint: `artifacts\ppo-transfer-racing-actionhead-rettifilo-mix-goal-60k-20260603-013341\best_model.zip`.
  - This remains the current honest normal-start PPO candidate; later `delayed_turn`/section-only checkpoints are not normal-start promotable without transfer because their action sets are section-specific.
- Commands:
  - `uv run --no-sync python -m f1rl.benchmark --policies ppo --episodes 3 --max-steps 7000 --seed 8603 --checkpoint artifacts\ppo-transfer-racing-actionhead-rettifilo-mix-goal-60k-20260603-013341\best_model.zip --device auto --telemetry all --telemetry-every 1 --metadata-mode require --disable-scaffold-rewards --disable-training-assists --ppo-deterministic`
  - `uv run --no-sync python -m f1rl.eval --checkpoint artifacts\ppo-transfer-racing-actionhead-rettifilo-mix-goal-60k-20260603-013341\best_model.zip --steps 7000 --seed 9603 --device auto --metadata-mode require --disable-scaffold-rewards --disable-training-assists --ppo-deterministic`
  - `uv run --no-sync python -m f1rl.qc --telemetry artifacts\benchmark-20260603-113711\selected_telemetry --max-telemetry-files 3 --output-dir artifacts\qc-honest-normalstart-probe-20260603-113711`
  - `uv run --no-sync python -m f1rl.replay artifacts\benchmark-20260603-113711\selected_telemetry\ppo-episode-000-steps.jsonl --headless --no-timing`
- Artifacts:
  - Benchmark: `artifacts\benchmark-20260603-113711`.
  - Eval: `artifacts\eval-20260603-113846`.
  - QC: `artifacts\qc-honest-normalstart-probe-20260603-113711\qc-20260603-113926`.
  - Replay source: `artifacts\benchmark-20260603-113711\selected_telemetry\ppo-episode-000-steps.jsonl`.
- Result:
  - Benchmark: `0/3` completed laps, `0/3` valid laps, `3/3` collisions.
  - Best and average progress: `970.7748596765059m`.
  - Checkpoints: `20/120`.
  - Elapsed time before terminal event: `13.566666666666666s`.
  - Metadata was loaded from `run_metadata.json`; effective eval config used `action_set=racing`, `observation_profile=racing`, `max_steps=7000`.
  - `disable_scaffold_rewards=true` and `disable_training_assists=true`.
- First-bad-event/QC:
  - First bad event remains `throttle_during_brake_demand`.
  - Location: `rettifilo_chicane`, `520.8062228014228m`.
  - Speed: `334.0937462671194kph` against `115kph` target.
  - Action/control: `throttle` / `throttle_straight`.
  - Terminal event: `collision` at `970.7748596765059m`, `273.8250186268078kph`, `brake_left`, lateral error `19.251011748766587m`, heading error `-46.941228724576874deg`.
  - Rettifilo section summary: entry `325.77099057844515kph`, max `338.71177209006987kph`, min `273.8250186268078kph`, average brake `0.12465373961218837`, average throttle `0.8753462603878116`.
  - Action histogram in Rettifilo: `throttle:316`, `brake_left:23`, `brake_right:22`.
- Decision:
  - Reject the probe as unchanged honest behavior. It does not improve beyond the old `970.775m` failure and does not complete a lap.
  - Do not launch long full-lap PPO.
  - Return immediately to curriculum and finish the linked Rettifilo transfer ladder:
    - stable yaw/steering handoff,
    - `848/900->1080`,
    - `650->1220`,
    - `520->1220`,
    - then honest normal-start eval again.

### Rejected 945m Handoff Attempts - 2026-06-03
- Purpose:
  - After the honest normal-start probe reproduced the old `~970.775m` collision, isolate the next linked-transfer blocker after clean release around `900->945m`.
  - Target behavior: from clean `944-946m` states, stabilize yaw/heading and survive toward `1080m` without fake segment success.
- State libraries:
  - `artifacts\state-library-rettifilo-clean935_946-from-releaseprobe-20260603\state_library.json`.
    - Built from `artifacts\targeted-probe-release900-945-yawgate-turninpower-20260603-rung\selected_telemetry`.
    - 18 snapshots between `935m` and `946m`, speed `140-170kph`.
  - `artifacts\state-library-rettifilo-clean944_946-rank000-20260603\state_library.json`.
    - Built from `rank-000-maint.jsonl`.
    - 2 clean snapshots around `944-946m`, speed about `162kph`, zero yaw rate, near-straight controls.
- Rejected broad action searches:
  - `artifacts\action-search-clean945-1080-turninpower-cap190-20260603-rung`.
  - `artifacts\action-search-clean943_946-1080-turninpower-cap190-20260603-rung`.
  - `artifacts\action-search-clean944_946-1080-turninpower-cap190-small-20260603-rung`.
  - These sweeps were stopped because they were too slow for the required aggressive mini-experiment loop. Keep the lesson, not the runtime: broad phase schedule search is not the right tool for this handoff unless the schedule space is drastically reduced.
- Rejected PPO handoff run:
  - Artifact: `artifacts\ppo-clean945-1080-delayedturn-tight-512-20260603-rung-20260603-114845`.
  - Start state: clean `946.305m`, `162.081kph`.
  - Action set: `delayed_turn`.
  - Target: `1080m`, max speed `190kph`, strict virtual corridor/yaw/heading gates.
  - Early eval result: `0/8` completions; all selected episodes terminated on `assist_virtual_corridor`.
  - Representative selected telemetry:
    - Terminal progress: `969.218m`.
    - Terminal speed: `162.550kph`.
    - Lateral error: `12.136m`.
    - Heading error: `-45.453deg`.
    - Action histogram: `maintenance_tiny_left: 39`; no braking, no stabilizing correction.
    - First bad event: `no_brake_before_turn_in` on the first step because the policy remained overspeed and chose left-biased maintenance.
- Decision:
  - Reject the run. It changed the action distribution, but in the wrong direction: it learned to hold tiny-left and drift out, not to stabilize the exit.
  - Next experiment should be narrower and earlier: prove `945->1000` stabilization before stretching to `1080`.
  - Forbid or remove early left bias, prefer straight/right/soft-brake stabilization, and only reintroduce left/turn-in after the car remains alive through the handoff.

### Evolutionary Search Pivot - 2026-06-03
- Decision:
  - Stop the active PPO/curriculum micro-rung loop before it spends more time on one-off action sets, gates, and schedule presets.
  - The honest normal-start scoreboard is still unchanged at about `970.775m`, no valid lap, no lap time.
  - The previous loop produced useful local Rettifilo facts, but not linked transfer or full-lap improvement.
- Action taken:
  - Sent the active goal thread a stop/pause prompt.
  - Stopped the active `f1rl.action_search` process tree.
  - Verified no remaining `f1rl.train` or `f1rl.action_search` processes were running.
- New implementation:
  - Added `src/f1rl/evolution_search.py`.
  - Added CLI entrypoint `f1-evolution-search`.
  - Added tests in `tests/test_evolution_search.py`.
- Capability:
  - Phase-based driving genomes.
  - Random population initialization.
  - Elite retention.
  - Mutation and crossover.
  - Random immigrants.
  - Optional multiprocessing workers.
  - Strict segment target gates.
  - Generation summaries.
  - `attempts.jsonl`.
  - `evolution_summary.json`.
  - Selected telemetry for top candidates.
  - `elite_state_library.json` for curriculum starts.
- Validation:
  - `uv run --no-sync ruff check .` passed.
  - `uv run --no-sync pytest -q` passed.
  - `uv run --no-sync pyright src/f1rl` passed.
  - CLI smoke:
    - Command used `python -m f1rl.evolution_search` with a tiny `500m->510m` straight segment.
    - Artifact: `artifacts\evolution-smoke-20260603`.
    - Smoke wrote `evolution_summary.json`, `attempts.jsonl`, selected telemetry, and `elite_state_library.json`.
    - Smoke best attempt completed the target segment.
- Docs:
  - Archived the previous active docs under `archive\plans\evolution-pivot-20260603`.
  - Added `EvolutionSearchPlan.md`.
  - Added `EvolutionGoal.md`.
- Next operating rule:
  - Do not resume the old micro-PPO loop directly.
  - Run population-based evolutionary search first, save elite states/telemetry, then train PPO from search-discovered curriculum starts.
  - Evolution/search can discover behavior and seed curriculum, but final success still requires honest normal-start PPO completion near `<=80.0s` with assists/scaffolds disabled.
