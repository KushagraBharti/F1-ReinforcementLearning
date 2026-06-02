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
- The completed proof-of-concept goal plan is archived at `archive/plans/Plan-rl-poc-goal-completed-20260602.md`.

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
