# F1 Reinforcement Learning Project Audit

Date: 2026-06-04.

This audit treats the repository as if read from zero prior knowledge. Claims below are verified from the current working tree, source code, configuration, generated assets, tests, validation commands, and artifact summaries. The strongest conclusion is positive: this is a substantial custom RL/simulation project with a real Gymnasium environment, calibrated top-down vehicle physics, Monza track geometry, Fast-F1 reference telemetry, SB3 PPO infrastructure, an evolution-first trajectory search engine, telemetry/replay tooling, and meaningful generated results. The project is already portfolio-grade as a simulation, ML systems, and search/optimization project. The remaining unsolved pieces are best framed as clear next milestones, not as a reason to downplay the engineering already built.

## Executive Verdict

You built a custom top-down Formula 1 Monza simulation and learning stack: image-derived track geometry, deterministic bicycle-model car physics, ray-cast observations, Gymnasium/SB3 PPO training, headless evaluation/benchmarking, JSONL telemetry, replay visualization, Fast-F1 reference calibration, and a large evolutionary controller-search system. This is a full environment-and-experiment platform, not just a training script.

The headline result is strong and concrete: the latest large evolved-controller run evaluated `9000` candidates over `60` generations, wrote `9000` losslessly compressed all-candidate telemetry traces, produced `926` lap-complete attempts, and found a fastest valid rolling-start evolved lap of `89.983s`. The Fast-F1 reference target is `79.662s`, so the current evolved result is within `10.321s` of the real reference target while preserving full replay/debug artifacts.

The best resume-safe positioning is assertive: custom RL environment and simulation infrastructure, Fast-F1-calibrated vehicle/track reference system, large-scale evolutionary search, telemetry/replay/debugging infrastructure, and validated Python engineering. The only constraint is precision: claim the solved evolved rolling-start lap and the implemented PPO system; save "PPO solves Monza" and "`<=80s` agent" for when artifacts prove them.

## Positive Read

The project has several high-value signals for SWE, ML Engineering, RL Engineering, simulation, and robotics roles:

- You built the environment, not just a model.
- You built the simulator, not just wrappers around an existing simulator.
- You built physics, perception, reward/scoring, training, evaluation, telemetry, replay, calibration, and artifacts as one coherent system.
- You integrated real motorsport telemetry and used it as a measurable target.
- You found a valid evolved lap that is close enough to the reference to be technically interesting.
- You preserved enough telemetry to defend results instead of relying on vague training curves.
- You built tests, linting, type checking, CLI tools, checkpointing, resume, and storage management.
- You discovered and documented a real ML failure mode: PPO infrastructure alone did not solve long-horizon racing, so the system pivoted to evolution-first discovery.

That is the framing to use: "I built a full racing RL lab and used it to evolve a valid high-speed Monza lap." The unfinished part is narrower: "Next I am transferring those discovered behaviors into PPO and closing the remaining 10-second gap to the Fast-F1 target."

## Repository Map

Important root files:

- `README.md`: concise user-facing overview and commands.
- `AGENTS.md`: active agent rules and current repo mission.
- `goal.md`: current optimization goal for evolved-controller search under `80.0s`.
- `workflow.md`: repeatable artifact-analysis/change/validation/archive/run loop.
- `Documentation.md`: live status and historical validation notes.
- `EvolutionSearchPlan.md` and `EvolutionGoal.md`: active evolution-search planning/prompt files.
- `pyproject.toml`: package metadata, dependencies, console scripts, pytest/ruff/pyright config.
- `uv.lock`: locked dependency state.
- `.rgignore`: hides `archive/`, `artifacts/`, and PDFs from default search.

Important active source modules under `src/f1rl`:

- `env.py`: `MonzaEnv`, Gymnasium wrapper around the simulator.
- `sim.py`: `MonzaSim`, shared simulator for manual, scripted, PPO, eval, replay, telemetry, and evolution.
- `physics.py`: deterministic top-down bicycle-style vehicle dynamics.
- `config.py`: project paths, car parameters, reward parameters, action sets, observation profiles.
- `track_build.py`: OpenCV-based Monza track generation from source images.
- `track_model.py`: persisted `TrackSpec` loading/saving and runtime geometry access.
- `geometry.py`: polyline, projection, ray intersection, and segment intersection utilities.
- `track_sections.py`: named Monza sections and braking gates.
- `telemetry.py`: per-step telemetry schema, episode summaries, JSONL/gzip loading.
- `render.py`: Pygame renderer for manual driving and replay.
- `manual.py`: keyboard/manual driving CLI.
- `scripted.py`: pure-pursuit-style scripted baseline.
- `reference_agent.py`: Fast-F1 ghost and reference-control baseline.
- `calibration.py`: Fast-F1 reference summary and simulator calibration report.
- `train.py`: Stable-Baselines3 PPO training CLI with curriculum, checkpointing, eval callbacks, TensorBoard.
- `eval.py`: metadata-aware PPO checkpoint evaluation and telemetry export.
- `benchmark.py`: headless random/scripted/reference/PPO benchmark harness.
- `policy_io.py`: checkpoint resolution, metadata-aware eval config, VecNormalize loading, model-space validation.
- `curriculum.py`: segment/state-library curriculum sampling.
- `state_snapshot.py` and `state_library.py`: state snapshot serialization and elite state-library generation.
- `evolution_search.py`: active elitist evolutionary search engine.
- `evolution_ladder.py`: repeatable evolution ladder runner.
- `qc.py` and `section_analysis.py`: telemetry QC and section/failure analysis.
- `action_search.py` and `elite_search.py`: legacy diagnostics, not the active main path.
- `hardware.py`: PyTorch/CUDA runtime policy inspection.

Generated/static assets:

- `imgs/`: Monza track images, background image, car sprites.
- `assets/tracks/monza/track_spec.npz`: persisted Monza geometry.
- `assets/tracks/monza/track_manifest.json`: build manifest for the track.
- `assets/reference/monza_2024_Q_VER_telemetry.csv`: Fast-F1 reference telemetry.
- `assets/reference/monza_2024_Q_VER_summary.json`: reference metrics.

Artifact locations:

- Repo-local `artifacts/`: smoke runs, benchmarks, PPO artifacts, scripted/manual/reference outputs.
- `C:\f1rl-artifacts`: hot large-run control artifacts. Current large run: `C:\f1rl-artifacts\evolution-speed-150x60-20260604-0501`.
- `D:\f1-rl-artifacts\cold-telemetry`: compressed all-candidate telemetry payloads.
- `D:\f1-rl-artifacts\archives`: compressed old large-run archives.

Current archive state:

- Current hot run size: about `0.50 GB` on C:.
- Current cold telemetry size: about `5.10 GB` on D:.
- Current cold trace count: `9000` `.jsonl.gz` files.
- Verified archives on D: include prior `100x10`, `100x30`, Roggia, adaptive, and speed runs.

Codebase size:

- `30` Python source files under `src/f1rl`.
- `19` pytest files.
- `143` collected tests.
- Source/test line highlights: `evolution_search.py` has `3742` lines, `train.py` `1609`, `sim.py` `1237`, `replay.py` `712`, `config.py` `576`, `qc.py` `583`, `test_evolution_search.py` `856`, and `test_sim.py` `809`.

Active vs legacy:

- Active path: `track geometry -> car physics -> shared simulator -> manual/scripted -> telemetry -> Gymnasium/SB3 PPO -> eval/replay`, with evolution-first discovery before PPO transfer.
- Active search modules: `evolution_search.py`, `evolution_ladder.py`.
- Active learning target: evolved-controller lap under `80.0s`, then PPO transfer later.
- Legacy diagnostics: `action_search.py`, `elite_search.py`.
- Historical context lives under `archive/`, `artifacts/`, and `transcripts/`; default searches intentionally ignore them.

## Core Thesis

Plain-English thesis:

You built a custom, telemetry-calibrated 2D Formula 1 Monza simulator and learning stack that can generate, evaluate, replay, and evolve racing policies, with the long-term goal of transferring discovered behavior into PPO.

Technical thesis:

This project implements a deterministic top-down Monza racing environment with OpenCV-derived track geometry, bicycle-model vehicle dynamics, ray-cast and racing-line observations, Gymnasium/SB3 PPO integration, Fast-F1 reference calibration, full-fidelity JSONL telemetry/replay, and an elitist evolutionary controller-search engine that evaluates thousands of candidate controllers with checkpointing, lineage, multi-profile scoring, and compressed all-candidate artifacts.

Best positioning:

- Resume: "Built a custom Gymnasium F1 racing environment with calibrated physics, Fast-F1 reference telemetry, SB3 PPO infrastructure, and a large-scale evolutionary search engine that evaluated 9000 controllers and found a 89.983s valid Monza lap."
- Portfolio: "A full RL/simulation system, not a toy model: track extraction, physics, observations, reward/scoring, telemetry, replay, baselines, PPO, and evolutionary search are all implemented and validated."
- GitHub README: lead with architecture, environment contract, Fast-F1 target, latest verified results, how to reproduce benchmarks/replay, and honest limitations.
- Interview: explain the environment contract, why PPO struggled, why evolution-first discovery was added, how telemetry made failures diagnosable, and what the next speed bottleneck is.

## Environment Architecture

Environment class:

- `MonzaEnv` in `src/f1rl/env.py`.
- Inherits `gymnasium.Env[np.ndarray, Any]`.
- Metadata: `render_modes=["human", "rgb_array"]`, `render_fps=60`.
- Wraps `MonzaSim`.

Observation spaces:

- All observation spaces are `Box(-1.0, 1.0, shape=(N,), dtype=float32)`.
- Verified dimensions:
  - `base`: `18`.
  - `brake`: `21`.
  - `guidance`: `23`.
  - `racing`: `31`.
  - `racing_release`: `35`.
  - `racing_v2`: `35`.
- Current active evolutionary search uses `racing_v2`.

Base observation contract:

- Speed normalized by max speed.
- Yaw rate normalized by `2.0 rps`.
- Heading error normalized by pi.
- Absolute lateral error normalized against `30m`.
- Lap progress ratio normalized to `[-1, 1]`.
- Last action encoded as normalized discrete action id or continuous drive-brake value.
- Last steer.
- `7` ray-cast distance sensors normalized by `1300m`.
- `4` lookahead heading errors at `40m`, `90m`, `160m`, and `280m`.

Additional observation profiles:

- `brake`: adds target speed, speed error, and brake-demand features.
- `guidance`: adds target steer and steer error.
- `racing`: adds signed lateral error, last throttle/brake, lookahead target-speed features, and distance to next braking gate.
- `racing_release`: adds release-band features.
- `racing_v2`: adds section-aware target speed, surplus speed, brake-zone flag, and brake-zone phase.

Action spaces:

- Discrete mode: `gym.spaces.Discrete(action_dim)`.
- Continuous mode: `gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=float32)`, interpreted as drive/brake plus steer.
- MultiDiscrete mode: `gym.spaces.MultiDiscrete([5, 5])`, combining 5 drive levels with 5 steer levels.
- Base default action set `legacy`: 9 actions.
- Active `racing` action set: 20 actions, covering throttle, half-throttle, soft-brake, full-brake, and five steer levels across left/straight/right.
- Continuous controller search calls `step_controls()` directly, so evolutionary controller genomes are not limited to the discrete action set.

Reset behavior:

- `MonzaEnv.reset(seed=...)` calls Gymnasium seeding and `MonzaSim.reset`.
- If curriculum is enabled and no reset options are provided, the curriculum sampler chooses options.
- Reset can start from:
  - normal start pose;
  - checkpoint index;
  - explicit `start_progress_m`;
  - state snapshot;
  - noisy position/heading/speed variants.
- Current large evolution run starts at `start_progress_m=0` and `start_speed_kph=80`; it is a rolling/flying start, not a 0 kph cold start.

Step behavior:

- `MonzaEnv.step` dispatches to `sim.step`, `sim.step_continuous`, or `sim.step_multidiscrete`.
- Returns Gymnasium tuple: observation, reward, terminated, truncated, info.
- `last_telemetry` stores the most recent `StepTelemetry`.

Reward function:

- Default simulator reward includes:
  - progress reward: `progress_delta_m * 0.08`;
  - finish bonus: `100.0`;
  - collision penalty: `-60.0`;
  - off-track penalty: `-60.0`;
  - no-progress penalty: `-90.0`;
  - lateral penalty outside `4.0m` deadzone;
  - track-limit ray penalty when nearest ray is below `10m`;
  - optional heading, speed-target, overspeed, steering-target, smoothness, scaffold, and assist terms.
- Many scaffold/assist terms are structurally implemented but default to zero or disabled.
- Evolutionary search uses a separate black-box scoring system over telemetry, not the SB3 PPO reward alone. This is important: the latest search traces can have sparse simulator reward components while still being ranked by external scoring profiles such as `fast_valid_lap`, `time_attack`, `lap_pace`, and `fast_frontier`.

Termination and truncation:

- Terminated:
  - collision with boundary segment;
  - off-track based on drivable mask;
  - no progress for `180` steps;
  - assist termination when enabled.
- Truncated:
  - lap complete;
  - segment complete;
  - segment gate failure;
  - max steps.
- Valid full-lap completion requires finish crossing, checkpoint validity, no missed checkpoints, and lap-distance crossing.

Info dictionary:

- Includes position, speed, checkpoint indices, valid-lap state, finish/segment flags, curriculum stage, segment targets/gates, progress, reward components, collision/off-track flags, and termination reason.

SB3 compatibility:

- The env uses Gymnasium spaces and was verified by `gymnasium.utils.env_checker.check_env(MonzaEnv(), skip_render_check=True)`.
- PPO training uses Stable-Baselines3 with `DummyVecEnv` or `SubprocVecEnv`.

Why the environment is learnable:

- The agent receives dense progress reward, direct speed/heading/lateral state, ray-based track proximity, lookahead curvature/heading information, and target-speed/braking-demand features.
- Termination prevents reward hacking through off-track/collision/no-progress.
- Checkpoint validity prevents fake finish crossings.
- Telemetry allows diagnosing whether policy actions match observations and reward/scoring.

## Physics Simulation

Physics model:

- `physics.py` implements deterministic top-down bicycle-style dynamics.
- State: `x`, `y`, heading, speed, yaw rate, steering angle, lap/checkpoint/progress indices, elapsed steps, alive flag.
- Units:
  - speed in m/s internally and kph for telemetry;
  - track progress in meters;
  - screen/world positions in pixels;
  - scale from persisted track: `1.276394m/px`.
- Timestep: `1/60s`.

Verified car parameters:

- Mass: `798 kg`.
- Wheelbase: `3.6m`.
- Max steering: `18 deg`.
- Steering response: `6.0 rad/s` equivalent response cap in code.
- Engine acceleration: `24.5 m/s^2`.
- Brake acceleration: `38.0 m/s^2`.
- Drag coefficient: `0.0025`.
- Rolling resistance: `0.25 m/s^2`.
- Base grip: `2.2g`.
- Aero grip term: `0.00023 * speed^2`.
- Max grip: `4.2g`.
- Max drive: `2.45g`.
- Max brake: `4.6g`.
- Steering speed sensitivity: `0.0008`.
- Max speed: `110 m/s` (`396 kph`).

Dynamics details:

- Steering command is clipped to `[-1, 1]` and rate-limited toward the target steering angle.
- Effective steering decreases with speed through speed sensitivity.
- Lateral acceleration demand is capped by grip.
- Longitudinal acceleration shares grip capacity with turning.
- Drag and rolling resistance reduce speed.
- Heading integrates yaw rate.
- Position updates in pixel coordinates using meters-per-pixel scale.
- Collision tests use movement segment intersection against track boundary segments.
- Off-track tests use the drivable mask.

Calibration:

- Fast-F1 reference: 2024 Italian GP qualifying, Verstappen fastest lap.
- Reference lap time: `79.662s`.
- Reference distance in CSV summary: `5745.669m`.
- Monza track model length: `5793.0m`.
- Reference speed:
  - min `75.0 kph`;
  - mean `259.914 kph`;
  - p10 `142.0 kph`;
  - p50 `274.223 kph`;
  - p90 `336.231 kph`;
  - max `348.0 kph`.
- Reference curvature/turning targets:
  - curvature p90 `0.012834 rad/m`;
  - curvature p95 `0.020375 rad/m`;
  - radius p05 `49.099m`;
  - radius p10 `77.918m`;
  - lateral-g p90 `3.278g`;
  - lateral-g p95 `4.095g`.
- Simulator estimates:
  - terminal speed `351.139 kph`;
  - full-throttle speed after 5s `294.006 kph`;
  - full-throttle speed after 8s `336.051 kph`;
  - braking `330->150 kph`: `66.710m`;
  - braking `330->100 kph`: `78.351m`;
  - steering-limited radius `11.080m`;
  - cornering radius at 100/150/200/250/300 kph: `33.08m`, `68.08m`, `108.12m`, `148.55m`, `186.42m`.

Simplifications:

- It is not a full tire/engine/aero simulator.
- No suspension, tire temperature, drivetrain gears, fuel, tire wear, or 3D load transfer.
- It is intentionally a learnable, deterministic, top-down dynamics model calibrated to coarse F1 speed and cornering targets.

## Track Geometry

Track construction:

- `track_build.py` reads `imgs/Monza_track_extra_wide_contour.png`.
- OpenCV thresholds the contour image at `225`.
- Finds inner and outer contours.
- Builds a drivable mask by filling the outer contour and cutting out the inner contour.
- Resamples left and right boundaries to `900` points each, closed into `901` point polylines.
- Builds `120` checkpoint gates and centerline midpoints.
- Persists `TrackSpec` to compressed NPZ.

Verified track metrics:

- Track name: `monza`.
- Source image size after coordinate scale: `1894 x 956`.
- Real track length: `5793.0m`.
- Centerline length: `4538.5669px`.
- Meters per pixel: `1.276394`.
- Centerline points: `121`.
- Checkpoints: `120`.
- Left boundary points: `901`.
- Right boundary points: `901`.
- Boundary collision segments: `1800`.
- Drivable mask shape: `956 x 1894`.
- Drivable pixels: `122244`.
- Drivable ratio: `6.751%`.
- Start pose: `(1501.0, 870.0, pi radians)`.
- Finish line: `(1501.0, 888.0) -> (1501.0, 852.0)`.

Geometry support for RL:

- Progress is computed by projecting car position onto the closed centerline.
- Projection uses local windowing to avoid jumps across nearby track segments.
- Signed and absolute lateral error support reward, observations, scoring, and validity checks.
- Boundary segments support collision detection and ray-cast sensors.
- The drivable mask supports off-track termination.
- Checkpoints support valid-lap enforcement and prevent shortcut reward hacking.

Named Monza sections:

- start/finish straight: `0-450m`, target `340 kph`.
- Rettifilo: `450-1150m`, brake gate `520m`, target `115 kph`.
- Curva Grande/Roggia run: `1150-1850m`, target `315 kph`.
- Roggia: `1850-2500m`, brake gate `1950m`, target `135 kph`.
- Lesmo 1: `2500-3150m`, brake gate `2620m`, target `185 kph`.
- Lesmo 2/Serraglio: `3150-3850m`, brake gate `3250m`, target `195 kph`.
- Ascari approach: `3850-4450m`, target `320 kph`.
- Ascari: `4450-5150m`, brake gate `4560m`, target `165 kph`.
- Parabolica/finish: `5150-5793m`, brake gate `5260m`, target `205 kph`.

## Observation And Raycast System

Ray-cast perception:

- `7` rays.
- Spread: `120 deg`, from about `-60 deg` to `+60 deg`.
- Forward bias: `1.6`, producing denser forward angles: approximately `[-60.0, -31.36, -10.35, 0.0, 10.35, 31.36, 60.0]`.
- Range: `1300m`.
- Each ray is a segment from car position to max range.
- Distance is nearest intersection with boundary segments.
- Distances are normalized to `[-1, 1]`.
- A cache avoids recomputing ray distances for unchanged car pose.

Why it matters:

- The policy can detect upcoming track limits without image input.
- Rays expose wall proximity and drivable corridor width.
- Combined with heading/lateral/progress/lookahead features, the policy gets both local geometry and high-level racing context.

Observation engineering beyond rays:

- Lookahead heading errors encode upcoming curvature.
- Target-speed features convert curvature into braking/speed context.
- `racing_v2` adds section-aware brake-zone phase and target-speed surplus.
- Evolutionary controller features additionally expose future braking demand, target-speed drops, braking-gate proximity, curvature, target steer, last actions, and segment/lap progress.

## Fast-F1 Telemetry Integration

Reference data:

- Stored in `assets/reference/monza_2024_Q_VER_telemetry.csv`.
- Summary in `assets/reference/monza_2024_Q_VER_summary.json`.
- Source: Fast-F1 2024 Italian Grand Prix Qualifying, Verstappen fastest lap.
- CSV rows: `610`.
- Unique distance samples: `610`.
- Lap time: `79.662s`.
- Mean speed: `259.914 kph`.
- Max speed: `348.0 kph`.
- Brake samples: `81`.
- Throttle range: `0.0` to `1.0`.

Uses:

- Calibration target for speed, braking, curvature, radius, and lateral-g capacity.
- `reference_ghost` policy that replays the Fast-F1 speed/distance profile on the simulator centerline.
- `reference_control` pure-pursuit controller using reference speed targets.
- Benchmark baseline for comparing random/scripted/PPO/evolved behavior against the real-world target.

Why it strengthens the project:

- The project is not just arbitrary toy physics; car parameters are compared against a real F1 lap.
- The target time and speed distribution are concrete.
- Replay/benchmark can render and compare against a ghost-like reference baseline.

## RL Training And Evaluation

Algorithm and framework:

- Stable-Baselines3 PPO.
- PyTorch backend.
- Policy type: `MlpPolicy`.
- Supports CPU/CUDA through `hardware.py`.
- Current runtime check: PyTorch `2.10.0+cu128`, CUDA `12.8`, device `NVIDIA GeForce RTX 4060 Laptop GPU`.
- Compute policy: PyTorch training/inference on CUDA when requested/available; simulator, physics, geometry, rendering, telemetry, and vector workers remain CPU-bound.

Training infrastructure:

- CLI: `f1-train` / `python -m f1rl.train`.
- Supports:
  - timesteps;
  - seed;
  - `n_envs`;
  - max steps;
  - CPU/CUDA/auto device;
  - required GPU guard;
  - checkpointing;
  - TensorBoard logs;
  - eval callbacks;
  - selected/all telemetry;
  - curriculum modes;
  - state-library curriculum starts;
  - reward overrides;
  - assist overrides;
  - VecNormalize reward stats;
  - resume from checkpoint;
  - transfer initialization from checkpoint with action/observation expansion support.

Default PPO hyperparameters exposed by CLI:

- `n_steps=128`.
- `batch_size=128`.
- `n_epochs=4`.
- `learning_rate=3e-4`.
- `gamma=0.995`.
- `ent_coef=0.02`.
- Optional SDE.

Evaluation:

- CLI: `f1-eval`.
- Metadata-aware checkpoint config resolution through `policy_io.py`.
- Validates model observation/action spaces before running.
- Can disable scaffold rewards and training assists for honest eval.
- Supports deterministic or stochastic policy evaluation.
- Writes telemetry via `TelemetryWriter`.

Verified PPO result:

- PPO is structurally implemented and tested.
- There are `171` train metadata files in local artifacts.
- Best verified PPO benchmark found in artifacts:
  - run `benchmark-20260603-014242`;
  - PPO best progress `970.7748596765059m`;
  - completion rate `0.0`;
  - valid lap rate `0.0`;
  - best lap time: none;
  - termination reasons: collisions.
- Best framing: PPO training/eval infrastructure is implemented and benchmarked, while the current strongest policy result comes from evolutionary search. PPO transfer is the next research milestone, not the headline result today.

## Evolutionary Search System

Active module:

- `src/f1rl/evolution_search.py`.

Implemented genome types:

- `phase`: fixed action phases by step count.
- `progress_phase`: action phases by progress in meters.
- `controller`: observation-driven continuous controller with 23 normalized features and 3 outputs.

Controller feature names:

- bias;
- speed norm;
- target speed norm;
- speed error norm;
- brake demand;
- future brake demand;
- target speed drop norm;
- braking-gate proximity;
- braking-gate distance norm;
- lookahead absolute max;
- signed lateral error norm;
- heading error norm;
- yaw rate norm;
- curvature norm;
- target steer;
- last throttle;
- last brake;
- last steer;
- segment progress ratio;
- lookahead_0 through lookahead_3.

Scoring profiles:

- `frontier`;
- `max_progress`;
- `clean_exit`;
- `brake_zone`;
- `apex`;
- `exit_speed`;
- `full_lap_validity`;
- `risk_seeking`;
- `frontier_fast`;
- `early_pace`;
- `clean_distance`;
- `farthest_distance`;
- `frontier_recovery`;
- `frontier_novelty`;
- `fast_valid_lap`;
- `time_attack`;
- `lap_pace`;
- `fast_frontier`.

Selection and evolution features:

- Elites.
- Rank-biased parent sampling.
- Tiered parent buckets:
  - fastest valid lap;
  - fast valid lap score;
  - fast frontier score;
  - survival gate;
  - late frontier distance;
  - frontier distance;
  - farthest distance;
  - far-fast;
  - fastest pace;
  - cleanest distance.
- Smart immigrants from current/global elites.
- Pure random immigrants.
- Crossover.
- Controller masked Gaussian mutation.
- Controller weight reset mutation.
- Dynamic survival floors.
- Plateau mode.
- Frontier focus.
- Lineage logging.
- Checkpoint/resume.
- Streaming `attempts.jsonl` and `generation_summary.jsonl`.
- `best_so_far.json`.
- Top genome snapshots.
- Elite state-library and PPO bridge outputs.
- Lossless `.jsonl.gz` all-candidate telemetry on external cold storage.

Latest large evolution run:

- Hot run folder: `C:\f1rl-artifacts\evolution-speed-150x60-20260604-0501`.
- Cold telemetry folder: `D:\f1-rl-artifacts\cold-telemetry\evolution-speed-150x60-20260604-0501`.
- Population: `150`.
- Generations: `60`.
- Attempts: `9000`.
- Max steps: `18000`.
- Start: progress `0m`, speed `80 kph`.
- Target milestone: `1500m`, with no target termination.
- Telemetry selection: all.
- Telemetry compression: gzip.
- Manifest traces: `9000`.
- Cold trace files: `9000`.
- Runtime: previously recorded `5335.89s`.

Latest large-run results:

- Lap-complete attempts: `926`.
- Overall lap-complete rate: `10.289%`.
- Fastest valid evolved lap: `89.983333s`.
- Fastest valid evolved pace: `232.225 kph`.
- Best valid lap generated at generation `49`, candidate `148`.
- Best valid source: `smart_immigrant`.
- Best valid bucket: `smart_current_elite`.
- Best valid mutation: `controller_weight_reset`.
- Best valid trace: `D:\f1-rl-artifacts\cold-telemetry\evolution-speed-150x60-20260604-0501\evolution-all_candidates-gen-049-candidate-00148-steps.jsonl.gz`.

Latest run generation highlights:

- Best fastest-valid generation: generation `49`, fastest valid `89.983s`, average valid `101.560s`, valid laps `16/150`.
- Best average valid-lap generation: generation `55`, average valid `96.605s`, fastest valid `89.983s`.
- Best valid-count generation: generation `26`, `32/150` valid laps, fastest `104.233s`.
- Best average-distance generation: generation `27`, average distance `2665.415m`, valid laps `27/150`.
- Best average-pace generation: generation `57`, average pace `216.726 kph`, top-decile pace `264.743 kph`.
- Final generation `59`: average distance `1601.020m`, average pace `214.070 kph`, top-decile distance `5766.924m`, top-decile pace `258.643 kph`, valid laps `14/150`, fastest valid `89.983s`.

Latest run aggregate gate totals across `9000` attempts:

- `>=450m`: `7234`.
- `>=1000m`: `2201`.
- `>=1220m`: `1988`.
- `>=1500m`: `1983`.
- `>=2000m`: `1947`.
- `>=3000m`: `1186`.
- `>=4000m`: `1094`.
- `>=5000m`: `1074`.
- `>=5793m`: `926`.

Termination counts across latest run:

- off_track: `5376`.
- collision: `2164`.
- no_progress: `513`.
- max_steps: `21`.
- lap_complete: `926`.

Fastest evolved lap trace details:

- Steps: `5399`.
- Elapsed time: `89.983s`.
- Final progress: `5804.548m`.
- Final speed: `271.854 kph`.
- Average speed across rows: `230.846 kph`.
- Max speed: `315.868 kph`.
- Min speed: `81.028 kph`.
- Low-speed rows under `120 kph`: `39`, only through `17.193m`, so current fastest lap is not suffering from a long launch crawl.
- Important crossings:
  - `300m`: `5.067s`, `297.039 kph`.
  - `450m`: `6.833s`, `312.004 kph`.
  - `650m`: `9.367s`, `272.490 kph`.
  - `1000m`: `14.700s`, `211.818 kph`, lateral error `12.228m`, heading error `38.865 deg`.
  - `1220m`: `18.433s`, `225.889 kph`.
  - `2000m`: `29.767s`, `255.621 kph`.
  - `2400m`: `35.983s`, `165.223 kph`.
  - `3000m`: `48.650s`, `238.882 kph`.
  - `4000m`: `62.233s`, `219.512 kph`, heading error `23.403 deg`.
  - `5000m`: `75.250s`, `201.773 kph`.
  - `5500m`: `86.083s`, `247.140 kph`.
  - finish: `89.983s`, `271.854 kph`.

Main current evolution bottleneck:

- Distance is no longer the scarce signal. The system can finish laps.
- The bottleneck is lap time and line quality, especially reducing inefficient cornering and widening the fast-valid family beyond a narrow elite.

## Baselines

Fresh verified benchmark command:

```powershell
uv run --no-sync python -m f1rl.benchmark --policies random scripted reference_ghost --episodes 3 --max-steps 18000 --seed 7 --telemetry none
```

Output: `artifacts\benchmark-20260604-063725`.

Baseline table:

| Policy | Episodes | Completion | Valid lap | Best lap | Avg progress | Best progress | Avg speed/pace note | Terminations |
|---|---:|---:|---:|---:|---:|---:|---|---|
| random | 3 | 0% | 0% | none | `100.565m` | `167.717m` | very slow/unstable, avg reward `-216.204` | 2 collision, 1 off_track |
| scripted | 3 | 100% | 100% | `214.467s` | `5800.349m` | `5800.349m` | avg speed `96.042 kph`, max `107.882 kph` | 3 lap_complete |
| reference_ghost | 3 | 100% | 100% | `79.662s` | `5793.0m` | `5793.0m` | avg speed `259.914 kph`, max `348.0 kph` | 3 lap_complete |
| PPO best artifact | historical benchmark | 0% | 0% | none | `970.775m` | `970.775m` | PPO infrastructure benchmark, transfer still pending | collisions |
| evolved controller latest | 9000 attempts | 926 lap-complete attempts | lap-complete search result | `89.983s` | generation-dependent | `5812.895m` | fastest trace avg `230.846 kph` | many off_track/collision, 926 lap_complete |

Interpretation:

- Random establishes the task is nontrivial.
- Scripted proves the simulator can complete valid laps but is slow.
- Fast-F1 ghost provides a real-world target.
- PPO infrastructure exists and has produced measurable benchmark artifacts; the current solved behavior is evolutionary, with PPO transfer still pending.
- Evolutionary search has found high-quality behavior and is currently the strongest learning/search result.

## Telemetry, Replay, Visualization, And Artifacts

Telemetry schema:

- `StepTelemetry` records per step:
  - time, position, heading, speed, yaw, acceleration, longitudinal/lateral g, curvature;
  - throttle, brake, steering and deltas;
  - action id/name;
  - raw/monotonic progress and progress delta;
  - lateral error, racing-line deviation, heading error;
  - reference/ghost fields;
  - checkpoint/lap validity;
  - segment/curriculum fields;
  - ray distances;
  - collision/off-track/termination/truncation;
  - total reward and reward components.

Episode summary:

- Lap/segment validity.
- Elapsed and lap time.
- Distance, speed, crashes, off-track count.
- Reward totals.
- Sector times/speeds.
- Braking zones.
- Corner summaries.
- Smoothness and g-force metrics.
- Ghost gap metrics when available.

Replay:

- CLI: `f1-replay`.
- Supports JSONL and `.jsonl.gz`.
- Supports manifest-based directory replay.
- Supports sorting by best progress or score.
- Supports generation-by-generation swarm replay.
- Supports headless replay checks.
- Supports interactive speed controls and generation skipping.
- Current manifests can point from C: hot artifacts to D: cold telemetry paths.

Latest replay command:

```powershell
uv run --no-sync f1-replay C:\f1rl-artifacts\evolution-speed-150x60-20260604-0501\selected_telemetry --sort best-progress --by-generation --generation-limit 100 --speed 1
```

Portfolio-visible artifacts:

- Full swarms by generation.
- Fastest evolved-lap trace.
- Benchmark summaries.
- Fast-F1 ghost/reference.
- Scripted baseline.
- Telemetry-derived plots can be generated from JSONL/gzip traces.

## Tests And Validation

Validation commands run during this audit:

```powershell
uv run --no-sync ruff check .
uv run --no-sync pyright src/f1rl
uv run --no-sync pytest -q
uv run --no-sync python -m f1rl.hardware --json
python - <<env checker script>>
uv run --no-sync python -m f1rl.benchmark --policies random scripted reference_ghost --episodes 3 --max-steps 18000 --seed 7 --telemetry none
```

Results:

- Ruff: passed, `All checks passed!`.
- Pyright: passed, `0 errors, 0 warnings, 0 informations`.
- Pytest: passed with warning-only SB3 `VecMonitor` messages.
- Pytest collection: `143` tests.
- Gymnasium env checker: `gymnasium_check_env_passed`.
- Hardware: CUDA visible, RTX 4060 Laptop GPU, PyTorch `2.10.0+cu128`.
- Benchmark: completed and wrote `artifacts\benchmark-20260604-063725`.

Test coverage areas:

- Track build.
- Physics.
- Simulator step/termination/progress/checkpoints.
- Environment spaces and stepping.
- Telemetry.
- Replay.
- Scripted/reference baseline.
- Calibration.
- Benchmark.
- Policy IO.
- PPO smoke training/resume/transfer/normalization.
- Curriculum.
- State library.
- Section analysis/QC.
- Evolution search.
- Legacy action/elite search.

## Benchmarks And Performance

Verified performance metrics:

- Simple headless simulator throttle run: `377` steps in `0.631s`, about `597 steps/s`, ending off-track at `431.869m`.
- Fresh baseline benchmark:
  - random avg steps/s: `265.322`;
  - scripted avg steps/s: `259.174`;
  - scripted full-lap sim episode: `12868` steps, about `214.467s` simulated time.
- Latest large evolutionary search:
  - `9000` candidates;
  - runtime about `5335.89s`;
  - effective all-in throughput about `1.69 candidates/s`;
  - per-generation throughput varied; final generation `4.131 candidates/s`.
- Current artifact storage:
  - hot C: latest run about `0.50 GB`;
  - cold D: latest compressed telemetry about `5.10 GB`;
  - `9000` compressed step traces.

Recommended additional benchmarks:

- Dedicated raycast throughput benchmark with and without ray cache.
- Evolution evaluator candidates/sec by `workers`, `worker_chunk_size`, telemetry mode, and compression mode.
- Replay load latency for `jsonl` vs `jsonl.gz`.
- PPO training FPS on CUDA with `DummyVecEnv` vs `SubprocVecEnv`.
- Memory usage during all-candidate telemetry runs.
- Plot of candidate throughput versus max-step cap.

## Fully Implemented Vs Partial Vs Not Implemented

Fully implemented and verified:

- Custom Gymnasium environment (`MonzaEnv`).
- Shared deterministic Monza simulator (`MonzaSim`).
- Bicycle-style top-down physics.
- OpenCV track extraction and persisted track asset.
- Ray-cast observations.
- Multiple action modes and action sets.
- JSONL/gzip telemetry schema and episode summaries.
- Pygame render/replay.
- Manual driving CLI.
- Scripted baseline.
- Fast-F1 ghost/reference integration.
- Calibration report.
- Headless benchmark harness.
- SB3 PPO training/eval infrastructure.
- Curriculum/state-library infrastructure.
- Metadata-aware PPO checkpoint loading/eval.
- Evolutionary search v1 with controller genomes, scoring profiles, lineage, checkpoint/resume, compressed telemetry, and PPO bridge artifacts.
- Evolution ladder CLI.
- Tests/lint/typecheck passing.
- Latest evolved-controller valid lap at `89.983s` from rolling `80 kph` start.

Implemented but not fully benchmarked enough for aggressive claims:

- PPO curriculum transfer from evolution-discovered states.
- Long-run PPO training success.
- Continuous-control PPO result quality.
- Multi-seed robustness of evolved controllers.
- External-drive cold telemetry replay at very large scales beyond current `9000` traces.
- Full portfolio plots/videos.
- Detailed reward/scoring ablations.
- End-to-end evolution-to-PPO behavior transfer.
- Generality beyond Monza.

Next milestones / claims to hold until artifacts prove them:

- PPO policy completing a valid Monza lap.
- PPO lap near or below `80s`.
- Cold-start `0 kph` `<=80s` evolved or PPO lap.
- Realistic full F1 tire/aero/drivetrain simulation.
- Multi-track generalization.
- Production-grade distributed training cluster.
- Human-beating or real-driver-equivalent policy.
- Final solved PPO-transfer result.

## Interview Defense Sheet

Q: Why build a custom Gymnasium env?

A: The project needed full control over the observation/action/reward contract, telemetry, reset curriculum, and simulator artifacts. A custom env let me connect track geometry, car physics, reward shaping, PPO training, and replay into one debuggable loop.

Q: Why discrete actions?

A: Discrete actions made the early PPO problem simpler and easier to debug. The repo also supports continuous and multidiscrete actions. Evolutionary controller search bypasses discrete actions and uses continuous throttle/brake/steer through `step_controls()`.

Q: Why PPO?

A: PPO is a stable on-policy baseline for continuous-control-like environments and has strong SB3 tooling for vectorization, TensorBoard, callbacks, checkpointing, and deterministic evaluation. It is implemented, but the current best result is evolutionary, not PPO.

Q: Why ray-cast sensors?

A: They give compact geometric perception without image-based policies. They expose wall/track-limit distances and make the policy aware of corridor constraints while keeping training lightweight.

Q: How did reward shaping work?

A: The simulator reward uses dense progress, lateral/track-limit penalties, finish/collision/off-track/no-progress terms, and optional scaffold/assist components. Evolution uses a separate scorer with profiles for fast valid laps, time attack, lap pace, fast frontier, clean distance, and recovery.

Q: How did you avoid reward hacking?

A: Checkpoint validity prevents shortcut finish crossings; collision/off-track/no-progress terminate; segment gates can enforce target speed/lateral/heading/yaw/steering; benchmark and replay inspect whether behavior is real or gamed.

Q: How was physics calibrated?

A: Car constants and calibration reports are compared against Fast-F1 Monza telemetry: lap time, speed distribution, terminal speed, braking distances, curvature, radius, and lateral-g targets.

Q: What does Fast-F1 add?

A: It gives a real target: Verstappen's 2024 Monza qualifying lap at `79.662s`, with 610 telemetry samples, mean speed `259.914 kph`, and max speed `348 kph`.

Q: What did PPO learn?

A: PPO infrastructure is real, and historical PPO benchmarks reached about `970.775m`. The strongest current policy result is evolutionary, so the project pivoted to evolution-first trajectory discovery before PPO transfer.

Q: What did evolution learn?

A: Evolutionary controller search produced valid rolling-start evolved laps. The latest large run generated 926 lap-complete attempts and a fastest valid evolved lap of `89.983s`.

Q: What did the project learn from failed runs?

A: PPO micro-curriculum work did not transfer to a normal-start full lap. Evolution originally overvalued distance/safe completion, then scoring/selection had to shift toward speed, valid lap time, top-decile pace, and breadth.

Q: What would improve next?

A: Push fast-valid breadth, reduce elite cloning around one narrow 89.983s family, add lap-time plateau branching, tune section-time/line-quality scoring, and eventually transfer high-quality evolved states into PPO curriculum.

Q: Why not use an existing racing simulator?

A: Existing simulators hide too much of the environment contract. This project is about building and owning the full RL stack: geometry, dynamics, observations, reward, telemetry, eval, replay, and search.

Q: How do you know the environment is valid?

A: It passes Gymnasium env checker, has deterministic seeding paths, has explicit spaces, enforces lap/checkpoint validity, has unit tests across environment/sim/physics/telemetry/replay/evolution, and generated benchmarks reproduce expected baseline behavior.

Q: What was hardest technically?

A: Long-horizon credit assignment and observability. The solution was not just "train longer"; it required telemetry, generation summaries, lineage, scoring profiles, replay, checkpoints, and compressed all-candidate traces.

## Resume Bullets

20 raw technical claims:

1. Built a custom Gymnasium F1 Monza environment with `Box` observations and discrete/continuous/multidiscrete action modes. Verified.
2. Built deterministic top-down bicycle-model car physics with grip, drag, braking, steering-rate limits, and speed-sensitive steering. Verified.
3. Built OpenCV track extraction from Monza contour images into persisted centerline, boundary, checkpoint, mask, and scale artifacts. Verified.
4. Built ray-cast perception with 7 boundary sensors over a 120-degree field and 1300m range. Verified.
5. Built racing observations with lookahead heading, target speed, brake demand, target steer, lateral error, and section brake-zone features. Verified.
6. Integrated Fast-F1 Monza reference telemetry from Verstappen's 2024 qualifying lap. Verified.
7. Calibrated simulator speed/braking/cornering estimates against Fast-F1 targets. Verified.
8. Built SB3 PPO training with checkpointing, eval callbacks, TensorBoard logs, vector envs, CUDA selection, reward overrides, curricula, and resume/transfer support. Verified structurally.
9. Built metadata-faithful PPO eval and benchmark tooling. Verified.
10. Built telemetry logging with per-step kinematics, controls, rewards, checkpoints, rays, and termination state. Verified.
11. Built Pygame replay with multi-trace and generation-by-generation swarm playback. Verified.
12. Built state snapshot and state-library tooling for curriculum starts. Verified.
13. Built elitist evolutionary controller search with phase, progress-phase, and observation-driven controller genomes. Verified.
14. Built multi-profile scoring for progress, clean exit, risk, frontier, valid-lap time, lap pace, and time attack. Verified.
15. Built dynamic survival floors, parent buckets, smart immigrants, crossover, mutation, lineage, and plateau behavior. Verified.
16. Built checkpoint/resume and streaming artifact outputs for long evolutionary searches. Verified.
17. Built lossless compressed all-candidate telemetry split across C: hot manifests and D: cold traces. Verified.
18. Ran a latest large evolution search over 9000 candidates and 60 generations. Verified.
19. Latest evolution run produced 926 lap-complete attempts and a fastest valid evolved lap of 89.983s. Verified.
20. PPO transfer remains the next milestone; best verified PPO benchmark progress is about 970.775m. Verified.

Top 10 strongest claims:

1. Built a complete custom RL racing environment and simulator rather than only training an off-the-shelf environment.
2. Implemented physics, geometry, observations, rewards, PPO, evolution, telemetry, replay, and benchmarks in one coherent stack.
3. Integrated real Fast-F1 telemetry and calibrated against a `79.662s` Monza reference lap.
4. Designed rich observation features including rays, lookahead curvature, braking demand, and section-aware target speeds.
5. Built large-scale evolutionary search with controller genomes and multi-profile scoring.
6. Evaluated `9000` controllers in the latest large run with full compressed telemetry preservation.
7. Achieved a `89.983s` valid evolved rolling-start Monza lap.
8. Built reproducible benchmark tooling comparing random, scripted, PPO, reference ghost, and evolved behavior.
9. Built robust artifact/replay infrastructure for debugging policy failures visually and quantitatively.
10. Maintained a validated Python package with `143` tests, ruff, pyright, and Gymnasium env checker passing.

Top 5 metrics:

1. Fast-F1 target: `79.662s`, mean `259.914 kph`, max `348.0 kph`.
2. Latest evolved lap: `89.983s`, average trace speed `230.846 kph`, max `315.868 kph`.
3. Latest evolution scale: `9000` candidates, `60` generations, `926` lap-complete attempts.
4. Track geometry: `120` checkpoints, `1800` boundary segments, `122244` drivable pixels.
5. Validation: `143` tests collected, ruff pass, pyright pass, Gymnasium check pass.

Best 2-bullet resume version:

- Built a custom Gymnasium/SB3 Formula 1 Monza RL stack with OpenCV-derived track geometry, deterministic bicycle-model physics, ray-cast/racing-line observations, checkpoint-valid lap logic, telemetry replay, and Fast-F1 calibration against a `79.662s` reference lap.
- Designed an elitist evolutionary controller-search engine with multi-profile scoring, dynamic survival floors, lineage, checkpoint/resume, compressed all-candidate telemetry, and swarm replay; latest `150x60` run evaluated `9000` controllers and found a `89.983s` valid evolved Monza lap.

Alternate resume versions:

ML Engineering:

- Built a custom Gymnasium/SB3 racing environment with calibrated physics, dense telemetry, curriculum resets, checkpoint-aware evaluation, and PPO training/eval infrastructure.
- Implemented a large-scale evolutionary controller search that generated 926 valid lap completions from 9000 candidates and produced a fastest valid evolved lap of 89.983s against a 79.662s Fast-F1 target.

SWE:

- Engineered a Python simulation/ML platform with modular physics, geometry, telemetry, rendering, benchmarking, CLI entrypoints, compressed artifacts, checkpoint/resume, and 143-test validation.
- Built a reproducible evolutionary search pipeline with streaming JSONL artifacts, manifest-based replay, lineage tracking, and hot/cold artifact storage across local/external drives.

Robotics/Simulation:

- Built a deterministic top-down vehicle simulator with bicycle dynamics, grip-limited turning, collision/off-track detection, ray-cast perception, and track-projection progress over a persisted Monza map.
- Calibrated dynamics against Fast-F1 Monza telemetry and used controller search to discover valid high-speed trajectories through the full lap.

Quant/Algorithms:

- Designed a black-box evolutionary optimization system for long-horizon driving policies with rank-biased parent selection, quality buckets, adaptive survival floors, smart immigrants, crossover/mutation, and multi-objective scoring.
- Evaluated 9000 controller candidates with full lineage/artifact capture and optimized from early failures to 926 valid lap completions.

Research:

- Investigated long-horizon RL failure modes in a custom racing environment, showing PPO curriculum transfer plateaued around 970m while evolutionary search discovered full-lap behavior.
- Built observability tooling to analyze policy failures through telemetry, section metrics, generation summaries, replay, and artifact-preserving experiments.

## Portfolio Rewrite

Polished title:

F1RL: Custom Monza Racing Simulator, PPO Environment, and Evolutionary Controller Search

Polished summary:

F1RL is a full-stack reinforcement-learning racing project that builds the environment from first principles: Monza track geometry, deterministic vehicle physics, Gymnasium/SB3 integration, Fast-F1 calibration, telemetry/replay, baselines, and an evolutionary search engine that discovered valid high-speed laps.

Context:

- The project explores whether a simplified 2D racing simulator can support serious RL/search development.
- It uses Monza because the track has clear high-speed straights, braking zones, chicanes, and a real Fast-F1 reference lap.

Problem:

- A racing policy must learn long-horizon credit assignment: accelerate, brake, turn, survive, and finish quickly.
- PPO alone struggled to transfer local chicane skills to full-lap behavior.
- Debugging required more than reward curves; it required step-by-step telemetry and replay.

System/model architecture:

- Track images are processed into centerline, boundaries, mask, checkpoints, and scale.
- A deterministic bicycle model updates car state at 60 Hz.
- `MonzaSim` is shared by manual, scripted, PPO, benchmark, replay, telemetry, and evolution.
- `MonzaEnv` exposes the simulator as a Gymnasium environment.
- PPO training uses SB3.
- Evolutionary search optimizes controller genomes outside PPO, then writes elite state libraries for future PPO curriculum.

Core implementation:

- Track projection and checkpoint validity.
- Ray-cast sensors.
- Racing observations with lookahead and braking-demand features.
- Rich telemetry schema.
- Fast-F1 ghost and calibration report.
- PPO training/eval/checkpointing.
- Evolutionary controller search with mutation/crossover/lineage.
- Compressed telemetry and manifest-based replay.

Evaluation/benchmarks:

- Fast-F1 reference: `79.662s`.
- Scripted baseline: valid but slow `214.467s`.
- Random baseline: no completion, best `167.717m` in fresh 3-episode benchmark.
- PPO best verified: `970.775m`, no valid lap.
- Latest evolved controller: `89.983s` valid rolling-start lap.
- Latest evolution run: `9000` candidates, `926` lap completions.

Impact/outcome:

- The project now has a clear, data-driven path from simulator to search-discovered trajectories to eventual PPO transfer.
- The result is already portfolio-grade for custom environment/simulation/search engineering.
- The final learning target remains open: `<=80s` and eventual PPO transfer.

Technical stack / what made it hard:

- Python 3.11, NumPy, OpenCV, Gymnasium, Pygame, Stable-Baselines3, PyTorch/CUDA, pytest, ruff, pyright.
- Hard parts:
  - long-horizon racing credit assignment;
  - observation/reward design;
  - avoiding shortcut/reward hacking;
  - preserving enough telemetry without exhausting disk/memory;
  - making results replayable and auditable;
  - turning one-off breakthroughs into population-wide improvement.

Suggested tags:

Reinforcement Learning, Gymnasium, Stable-Baselines3, PPO, Evolutionary Algorithms, Genetic Algorithms, Simulation, Vehicle Dynamics, Control, Robotics, Autonomous Driving, Racing AI, Formula 1, Fast-F1, Telemetry, Pygame, OpenCV, NumPy, PyTorch, CUDA, Python, Ray Casting, Path Planning, Curriculum Learning, State Libraries, Benchmarking, Replay Systems, JSONL, Gzip, Data Engineering, Optimization, Search, Physics Simulation, ML Engineering, SWE, Observability, Experimentation, Technical Portfolio, Monza.

Suggested visuals:

- Architecture diagram from track image to sim to env to PPO/evolution to telemetry/replay.
- Monza track with centerline, checkpoints, boundaries, and rays.
- Fast-F1 speed profile versus evolved lap speed profile.
- Evolution generation chart: fastest valid lap, valid count, average distance, top-decile pace.
- Replay video of the latest evolved `89.983s` lap.
- Swarm replay video showing generation-by-generation improvement.
- Table comparing random/scripted/PPO/evolved/reference.

## README Recommendations

Recommended README outline:

1. Project Overview.
2. Current Verified Results.
3. Architecture Diagram.
4. Environment Contract.
5. Track Geometry.
6. Physics Model And Calibration.
7. Fast-F1 Reference Telemetry.
8. Observation And Action Spaces.
9. PPO Training.
10. Evolutionary Search.
11. Telemetry And Replay.
12. Baselines And Benchmarks.
13. Validation.
14. How To Run.
15. Artifact Storage.
16. Limitations.
17. Roadmap.

Exact Project Overview text:

F1RL is a custom reinforcement-learning racing stack for a simplified top-down Formula 1 car at Monza. It builds the environment from first principles: OpenCV-derived track geometry, deterministic bicycle-model physics, Gymnasium/SB3 PPO integration, Fast-F1 reference telemetry, telemetry/replay tooling, and an evolutionary controller-search engine. The current strongest result is a valid evolved racing controller: the latest large run evaluated `9000` controller candidates, produced `926` lap-complete attempts, and found a `89.983s` valid rolling-start evolved lap against a `79.662s` Fast-F1 reference target. PPO infrastructure is implemented and tested, and PPO transfer is the next milestone.

## Missing Work And Next Steps

Priority 1: Get evolved lap under `80s`.

- Task: improve fast-valid breadth and lap-time scoring.
- Files: `src/f1rl/evolution_search.py`, `workflow.md`, `Documentation.md`.
- Command: full `150x60` or larger evolution run with compressed telemetry.
- Artifact: new `C:\f1rl-artifacts\<run>` and D: cold telemetry.
- Metric: fastest valid lap, average valid lap, valid count, top-decile pace.
- Recruiting value: turns strong infrastructure into a headline result.

Priority 2: Produce portfolio video.

- Task: export or record replay of the fastest evolved lap and one swarm generation sequence.
- Files: `src/f1rl/replay.py`, artifact telemetry, maybe a small capture script.
- Command: `f1-replay <selected_telemetry> --sort best-progress --by-generation`.
- Artifact: MP4/GIF.
- Metric: visual proof of behavior.
- Recruiting value: makes the project instantly understandable.

Priority 3: Create benchmark plots.

- Task: generate charts from `generation_summary.jsonl` and fastest trace.
- Files: add `tools/plot_evolution_metrics.py`.
- Commands: plot latest run summaries.
- Artifacts: PNG charts for lap time, valid count, avg distance, top-decile pace, speed over distance.
- Metrics: charted trends.
- Recruiting value: makes results defensible.

Priority 4: PPO transfer from evolved states.

- Task: use `elite_state_library.json` to train PPO curriculum.
- Files: `train.py`, `curriculum.py`, `state_library.py`, latest evolution bridge files.
- Command: `f1-train --curriculum segments --curriculum-state-library <elite_state_library.json> ...`.
- Artifact: PPO run with eval summaries.
- Metric: honest normal-start PPO progress and lap completion.
- Recruiting value: closes the RL loop.

Priority 5: Reward/scoring ablations.

- Task: compare scoring profiles and parent-selection strategies.
- Files: `evolution_search.py`, new analysis tool.
- Command: small/medium repeated runs with fixed seeds.
- Artifact: ablation table.
- Metric: valid lap count, fastest lap, avg valid lap, top-decile pace.
- Recruiting value: shows experimental rigor.

Priority 6: README polish.

- Task: rewrite README around verified architecture/results.
- Files: `README.md`, `project.md`.
- Artifact: serious technical project page.
- Metric: clarity and reproducibility.
- Recruiting value: high.

Priority 7: Artifact cleanup.

- Task: keep old large runs compressed on D: and avoid context bloat.
- Files: `workflow.md`, artifact folders.
- Command: verified `tar -czf`, `tar -tzf`, delete original only after verification.
- Metric: free disk space and archive list.
- Recruiting value: indirect but important for continued experimentation.

Priority 8: Add benchmark suite for throughput.

- Task: formalize simulator/evolution/replay throughput benchmarks.
- Files: `benchmark.py` or `tools/benchmark_throughput.py`.
- Artifact: benchmark JSON/CSV.
- Metric: steps/sec, candidates/sec, replay load/sec, memory usage.
- Recruiting value: supports SWE/ML systems claims.

## Strict Claim Boundaries

Safe to claim:

- You built the simulator, Gymnasium environment, physics, track geometry, observations, telemetry, replay, PPO infrastructure, benchmark tools, Fast-F1 integration, and evolutionary search system.
- You evaluated thousands of evolved controllers and found a valid `89.983s` rolling-start Monza lap.
- The repo has passing lint/typecheck/tests and a concrete benchmark suite.

Hold these claims until the next artifacts prove them:

- PPO solved the lap.
- The final agent reaches `<=80s`.
- The current result is a cold-start lap.
- The simulator is a fully realistic F1 simulator.
- The approach generalizes to multiple tracks.
