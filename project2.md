# F1RL Project2: Full Process Reconstruction And Technical Narrative

Date: 2026-06-04.

This document reconstructs the technical development story behind the current F1RL result: a custom top-down Monza RL/simulation system whose strongest solved behavior is now an evolved controller completing a valid rolling-start Monza lap in `89.983s`.

This is not just a latest-run summary. It covers the simulator, Gymnasium environment, PPO work, telemetry/replay tooling, failed approaches, the pivot to evolutionary search, the search architecture, major runs, scoring/selection changes, the exact fastest-lap artifact, resume wording, portfolio positioning, README lead, and interview defense.

## Claim Labels

I use these labels throughout:

- Verified: confirmed from current source code, active docs, current artifact files, current archives, or direct artifact parsing.
- Documented: recorded in active repo docs such as `Documentation.md`, `goal.md`, or `workflow.md`; the original run may now be archived or removed.
- Inferred: derived from a run name or surrounding documentation when the original uncompressed run is not currently available for direct parsing.

Important boundary: the current strongest solved behavior is evolutionary controller search, not PPO. PPO infrastructure is implemented and tested, but PPO has not yet completed an honest normal-start Monza lap.

## Executive Story

Verified: F1RL is a custom Python racing AI system built around a simplified top-down Formula 1 car at Monza. The repo owns the full path:

`track geometry -> car physics -> shared simulator -> manual/scripted -> telemetry -> Gymnasium/SB3 PPO -> eval/replay`

Then it adds a second discovery layer:

`shared simulator -> evolutionary controller search -> elite telemetry/state libraries -> future PPO transfer`

The project began as a custom Gymnasium/SB3 PPO environment with Monza geometry, deterministic car physics, ray-cast observations, reward shaping, telemetry, and replay. PPO could learn local progress, and the training infrastructure became substantial, but honest normal-start PPO plateaued around `970.775m` without a valid lap. Telemetry and replay showed the core failure: the policy repeatedly optimized short-term progress and crashed instead of discovering long-horizon braking, line setup, and lap-completion behavior.

The project then pivoted, explicitly borrowing lessons from the Yosh Trackmania transcripts: spawn from many states, add missing observations, use temporary scaffolds, save successful states, run segmented search, preserve elite attempts, mutate aggressively, and keep returning to honest full-lap tests. That pivot produced `f1rl.evolution_search` and `f1rl.evolution_ladder`.

Verified total scale: counting every currently readable evolution summary plus every compressed archive under `C:\f1rl-artifacts\archives` and `D:\f1-rl-artifacts\archives`, the project currently preserves `76` unique evolution/search run artifacts representing `27650` candidate evaluations and `340` generations. The stricter subtotal for serious/named scale runs is `26640` candidates across `21` runs. This includes multiple repeated `20x4`, `100x5`, `100x10`, and `100x30` experiments, not only the latest large run.

Verified latest result: `C:\f1rl-artifacts\evolution-speed-150x60-20260604-0501` evaluated `9000` controller candidates over `60` generations. It wrote `9000` losslessly compressed all-candidate telemetry traces to `D:\f1-rl-artifacts\cold-telemetry\evolution-speed-150x60-20260604-0501`, produced `926` valid lap-complete attempts, and found a fastest valid evolved rolling-start lap of `89.983333s`. The Fast-F1 reference target is `79.662s`, so the current evolved controller is `10.321s` slower than the reference.

The honest one-line framing:

> Built a custom Gymnasium/SB3 Monza racing simulator with calibrated physics, telemetry/replay, PPO infrastructure, and an evolutionary controller-search engine with `27650` counted candidate evaluations across preserved run artifacts; the latest `150x60` run evaluated `9000` controllers and found a `89.983s` valid evolved rolling-start lap against a `79.662s` Fast-F1 reference.

## Part 1: Full Project Timeline Reconstruction

### Phase 1: Track Geometry Foundation

Verified files:

- `src/f1rl/track_build.py`
- `src/f1rl/track_model.py`
- `src/f1rl/geometry.py`
- `assets/tracks/monza/track_spec.npz`
- `assets/tracks/monza/track_manifest.json`

Verified design:

- The project starts from image-derived Monza track geometry, not an external racing simulator.
- OpenCV processes a Monza contour image into a drivable mask, inner/outer boundaries, centerline, checkpoints, and scale.
- `TrackSpec` stores the persisted geometry and is loaded by the simulator.
- Projection onto the centerline gives continuous lap progress in meters.
- Boundary segments provide collision checks and ray intersections.
- The drivable mask provides off-track checks.
- Checkpoints enforce valid-lap ordering so the system cannot fake a lap by crossing the finish from the wrong direction or skipping the course.

Verified track metrics:

- Track: Monza.
- Real track length: `5793.0m`.
- Centerline pixel length: `4538.5669px`.
- Scale: `1.276394m/px`.
- Centerline points: `121`.
- Checkpoints: `120`.
- Left boundary points: `901`.
- Right boundary points: `901`.
- Boundary collision segments: `1800`.
- Drivable pixels: `122244`.
- Start pose: `(1501.0, 870.0, pi radians)`.
- Finish line: `(1501.0, 888.0) -> (1501.0, 852.0)`.

Why this mattered:

- The learning system needed progress, off-track, collision, checkpoint validity, and replay coordinates to all agree.
- Building this geometry layer made the environment debuggable instead of opaque.
- It also made later search possible because evolutionary candidates could be scored on exact progress, gates, lateral error, heading error, and finish validity.

### Phase 2: Car Physics And Shared Simulator

Verified files:

- `src/f1rl/physics.py`
- `src/f1rl/sim.py`
- `src/f1rl/config.py`

Verified physics model:

- Deterministic top-down bicycle-style dynamics.
- Timestep: `1/60s`.
- State includes position, heading, speed, yaw rate, steering, checkpoint/progress state, alive flag, and elapsed steps.
- Controls are throttle, brake, and steering.
- Steering is clipped and rate-limited.
- Speed-sensitive steering reduces effective steering at high speed.
- Grip limits lateral acceleration.
- Longitudinal acceleration shares grip with turning.
- Drag and rolling resistance reduce speed.
- Collision is tested against track boundary segments.
- Off-track is tested against the drivable mask.

Verified car constants:

- Mass: `798kg`.
- Wheelbase: `3.6m`.
- Max steering: `18deg`.
- Steering response: `6.0`.
- Engine acceleration: `24.5m/s^2`.
- Brake acceleration: `38.0m/s^2`.
- Drag coefficient: `0.0025`.
- Rolling resistance: `0.25m/s^2`.
- Base grip: `2.2g`.
- Aero grip term: `0.00023 * speed^2`.
- Max grip: `4.2g`.
- Max drive: `2.45g`.
- Max brake: `4.6g`.
- Steering speed sensitivity: `0.0008`.
- Max speed: `110m/s` or `396kph`.

Verified shared simulator role:

- `MonzaSim` is the central runtime.
- Manual driving, scripted driving, telemetry, Gymnasium, PPO eval, replay, benchmark, and evolution all use the same simulator path.
- This avoids one common ML-sim bug: training and evaluation using subtly different dynamics.

### Phase 3: Fast-F1 Calibration And Reference Target

Verified files:

- `src/f1rl/calibration.py`
- `src/f1rl/reference_agent.py`
- `assets/reference/monza_2024_Q_VER_telemetry.csv`
- `assets/reference/monza_2024_Q_VER_summary.json`

Verified reference:

- Source: Fast-F1 2024 Italian Grand Prix qualifying, Verstappen fastest lap.
- Reference lap time: `79.662s`.
- CSV samples: `610`.
- Reference distance in CSV summary: `5745.669m`.
- Mean speed: `259.914kph`.
- Max speed: `348.0kph`.
- Min speed: `75.0kph`.
- Speed p10: `142.0kph`.
- Speed p50: `274.223kph`.
- Speed p90: `336.231kph`.
- Brake samples: `81`.

Verified calibration estimates:

- Simulator terminal speed: `351.139kph`.
- Full throttle after 5s: `294.006kph`.
- Full throttle after 8s: `336.051kph`.
- Braking `330->150kph`: `66.710m`.
- Braking `330->100kph`: `78.351m`.
- Cornering radius at `100/150/200/250/300kph`: `33.08m`, `68.08m`, `108.12m`, `148.55m`, `186.42m`.

Why this mattered:

- The project has a real external target, not only an internal reward score.
- The reference ghost gives a benchmark policy class.
- The evolved `89.983s` result can be measured against a concrete `79.662s` target.

### Phase 4: Gymnasium Environment

Verified file:

- `src/f1rl/env.py`

Verified environment:

- `MonzaEnv` inherits `gymnasium.Env`.
- Metadata includes `render_modes=["human", "rgb_array"]` and `render_fps=60`.
- Observations are float32 boxes normalized to `[-1, 1]`.
- Actions support discrete, continuous, and multidiscrete modes.
- Reset supports normal start, progress starts, checkpoint starts, state snapshots, and curriculum-driven starts.
- Step returns Gymnasium `(obs, reward, terminated, truncated, info)`.

Verified observation dimensions:

- `base`: `18`.
- `brake`: `21`.
- `guidance`: `23`.
- `racing`: `31`.
- `racing_release`: `35`.
- `racing_v2`: `35`.

Verified base observation components:

- Speed.
- Yaw rate.
- Heading error.
- Lateral error.
- Lap progress.
- Last action.
- Last steer.
- Seven ray-cast distances.
- Four lookahead heading errors at `40m`, `90m`, `160m`, and `280m`.

Verified racing-style additions:

- Target speed.
- Speed error.
- Brake demand.
- Future brake demand.
- Target speed drop.
- Brake gate proximity.
- Brake gate distance.
- Signed lateral error.
- Target steer.
- Last throttle/brake/steer.
- Section progress and lookahead curvature/heading signals.

Verified action dimensions:

- Base discrete action set: `9`.
- Active `racing` discrete action set: `20`.
- Continuous action space: `Box(shape=(2,))`.
- Multidiscrete action space: `[5, 5]`.

Important architectural point:

- PPO can use discrete, continuous, or multidiscrete modes.
- Evolutionary controller search calls `MonzaSim.step_controls()` directly and is not limited by the discrete action set.

### Phase 5: PPO Infrastructure

Verified files:

- `src/f1rl/train.py`
- `src/f1rl/eval.py`
- `src/f1rl/benchmark.py`
- `src/f1rl/policy_io.py`
- `src/f1rl/curriculum.py`
- `src/f1rl/state_snapshot.py`
- `src/f1rl/state_library.py`

Verified PPO infrastructure:

- Stable-Baselines3 PPO training.
- CUDA-aware device selection.
- `DummyVecEnv` and `SubprocVecEnv`.
- TensorBoard support.
- Checkpointing.
- Evaluation callbacks.
- Resume support.
- VecNormalize loading.
- Metadata-aware checkpoint evaluation.
- Action/observation profile metadata checks.
- Curriculum configuration.
- State-library curriculum starts.
- Segment evaluation.
- Telemetry export.
- Benchmark comparison across random, scripted, reference ghost, and PPO policies.

Verified fresh benchmark from previous audit:

- Random baseline, 3 episodes: `0%` completion, best progress `167.717m`, average progress `100.565m`.
- Scripted baseline: `100%` valid completion, lap `214.4667s`, average speed `96.042kph`, max speed `107.882kph`.
- Reference ghost: `100%` valid completion, lap `79.662s`, average speed `259.914kph`, max speed `348kph`.

Verified PPO result boundary:

- Best verified honest normal-start PPO benchmark progress is about `970.775m`.
- Verified PPO benchmarks had `0%` completion, `0%` valid laps, and no lap time.
- PPO did not solve full-lap normal-start driving.

Why PPO alone was not enough:

- The task has long-horizon credit assignment. Early throttle and brake decisions affect a crash hundreds of meters later.
- PPO repeatedly found "go far fast then crash" behavior.
- Local curriculum/rung work produced useful subskills but did not transfer to honest normal-start full-lap behavior.
- The training loop risked becoming micro-engineered around one failing section instead of discovering a complete driving strategy.

### Phase 6: Telemetry, Replay, QC, And Observability

Verified files:

- `src/f1rl/telemetry.py`
- `src/f1rl/replay.py`
- `src/f1rl/render.py`
- `src/f1rl/qc.py`
- `src/f1rl/section_analysis.py`

Verified telemetry:

- Per-step JSONL telemetry includes position, heading, speed, progress, lateral error, heading error, yaw rate, curvature, throttle, brake, steering, checkpoint state, target speed, braking demand, reward components, and termination state.
- Gzip telemetry loading is supported.
- Episode summaries include sector/corner/braking information.

Verified replay:

- Pygame replay can load telemetry traces.
- Replay supports manifest resolution.
- Replay can read `.jsonl.gz` compressed traces from cold storage.
- Replay supports generation-by-generation swarm playback.
- Replay supports skipping generations and in-replay speed controls.

Why this mattered:

- PPO failure was not diagnosed only from reward curves.
- Evolution failure was not diagnosed only from best score.
- Replay made it obvious when cars crawled, wiggled, over-braked, failed to set up, crashed at high speed, or completed valid but slow laps.
- Generation summaries showed whether best distance, average distance, top-decile pace, valid count, and fastest valid lap were improving together or diverging.

### Phase 7: Yosh Transcript Lessons Enter The Process

Verified files:

- `transcripts/README.md`
- `transcripts/05-f1rl-methods-summary.txt`
- `transcripts/01-yosh-trackmania-2023.txt`
- `transcripts/02-yosh-noseboost.txt`
- `transcripts/03-yosh-a01.txt`
- `transcripts/04-yosh-a06.txt`

Documented lessons applied to F1RL:

- Spawn from many positions, not only the start.
- Add observations when failure reveals missing information.
- Use temporary reward shaping to induce rare skills.
- Force exploration when the policy keeps choosing a safe-but-bad route.
- Save successful states and train from them.
- Use segmented search / best-state continuation.
- Use brute force / evolutionary attempts when RL does not discover rare behavior.
- Watch telemetry and hunt exploits instead of trusting reward.
- Decompose hard tasks into stages, then transfer back to full-lap evaluation.

Direct project translation:

- Add `racing` and `racing_v2` observations.
- Add state snapshots and state libraries.
- Add curriculum starts.
- Add section analysis and first-bad-event analysis.
- Build evolution search and evolution ladder.
- Keep full-lap probes as the scoreboard.
- Stop treating local segment success as final progress.

### Phase 8: Early Search Systems Before The Final Evolution Engine

Verified files:

- `src/f1rl/action_search.py`
- `src/f1rl/elite_search.py`
- `src/f1rl/evolution_search.py`

Verified current status:

- `action_search.py` and `elite_search.py` are legacy diagnostics.
- They remain useful for small probes but are not the active main path.
- Active main path is `evolution_search.py` plus `evolution_ladder.py`.

Technical progression:

1. PPO and scripted/manual workflows established the environment and telemetry.
2. Focused curriculum and state-library work tried to isolate the Rettifilo failure.
3. Smaller action/elite search scripts tested the idea of brute-force control variants.
4. Those scripts were too narrow/manual for full-lap discovery.
5. The project moved to a general evolutionary engine with populations, genomes, scoring profiles, streaming artifacts, checkpoint/resume, lineage, and replay.

### Phase 9: Evolution Search V1

Verified file:

- `src/f1rl/evolution_search.py`

Verified capabilities:

- Genome types: `phase`, `progress_phase`, `controller`.
- Candidate populations.
- Elite retention.
- Mutation.
- Crossover.
- Pure random immigrants.
- Smart immigrants.
- Parent buckets.
- Dynamic survival floors.
- Plateau mode.
- Frontier focus.
- Multiple scoring profiles.
- Multi-profile elite selection.
- Lineage tracking.
- Worker processes and chunked evaluation.
- Checkpoint/resume.
- Streaming `attempts.jsonl`.
- Streaming `generation_summary.jsonl`.
- Rolling `best_so_far.json`.
- Per-generation top genome dumps.
- Selected telemetry.
- All-candidate telemetry.
- Lossless gzip trace output.
- Hot/cold artifact split.
- PPO bridge and state-library outputs.

The design shift:

- PPO optimizes policy weights through gradient updates.
- Evolution search optimizes controller parameters directly through black-box scoring.
- This made it much easier to "spawn 100 or 150 cars, let them drive, score them, keep the best, mutate, and repeat."

### Phase 10: First Visual Swarms And Throughput Work

Verified readable artifacts:

- `artifacts\evolution-swarm-20x4-fast-20260604`
- `artifacts\evolution-swarm-20x4-visible-20260604`
- `artifacts\evolution-swarm-100x5-fast-20260604`
- `artifacts\evolution-swarm-100x5-visible-next`

Verified results from readable summaries:

- `20x4` runs: `80` candidates, best distance about `208.749m`.
- `100x5` visible run: `500` candidates, best distance `312.602m`, best generation average distance `170.753m`.

What changed:

- All-candidate telemetry became replayable.
- Swarm replay let the user see every candidate in a generation.
- Generation logs were expanded to show leader progress, generation best distance, average distance, global best distance, completion rate, and later average pace/top-decile metrics.

What was learned:

- A lot of candidates died immediately.
- Best distance could plateau while average distance improved.
- Replay exposed slow-crawl and weird wiggle behavior that metrics alone did not explain.
- Telemetry packaging could become a bottleneck and memory risk.

### Phase 11: Streaming Telemetry And No-Target Termination

Documented run:

- `C:\f1rl-artifacts\evolution-0to1220-100x10-stream-20260604`

Documented result:

- `1000` candidates.
- Best distance `1130.4m`.
- No `1220m` completion.
- Completed without the prior multiprocessing telemetry `MemoryError`.
- Wrote `1000` traces.

What changed:

- Workers stopped returning huge step traces through process serialization.
- Workers streamed replay-compatible trace files directly.
- Parent process kept compact scoring rows.
- This eliminated the earlier all-candidate telemetry `MemoryError`.

Documented open-distance change:

- `--no-target-termination` was added.
- `target_progress_m` became a milestone/reporting threshold instead of a hard stop.
- Candidates could continue beyond the target until collision, off-track, no-progress, lap complete, or max steps.

Why it mattered:

- The user explicitly wanted to know how far cars could go, not stop them at `1220m` or `1500m`.
- This turned early segment search into open full-lap brute force.

### Phase 12: Open-Distance Breakthrough Past Early Track

Documented run:

- `C:\f1rl-artifacts\evolution-open-100x10-aggressive-20260604`

Documented command shape:

- Normal start.
- `start_speed_kph=80`.
- `target_progress_m=1500`.
- `--no-target-termination`.
- `population=100`.
- `generations=10`.
- `max_steps=10000`.
- `genome_type=controller`.
- Scoring profiles included `frontier`, `risk_seeking`, `max_progress`, `clean_exit`, `exit_speed`.

Documented result:

- `1000` attempts.
- Elapsed `104.03s`.
- Best distance `2409.611m`.
- Best candidate generation `9`, candidate `72`.
- Final speed `300.68kph`.
- Final lateral error `17.34m`.
- Termination: collision.
- Counts across attempts: `88 >=1000m`, `54 >=1220m`, `14 >=1500m`, `14 >=2000m`, `1 >=2400m`.

What changed:

- Scoring became more distance/speed-biased.
- Controller mutation became more aggressive.
- Parent selection became more rank-biased.
- Generation logging exposed average and global metrics.

What was learned:

- Full-lap open search could discover much more than PPO's `~970m` plateau.
- Breakthroughs came from offspring, not just pure randoms.
- Best-distance plateaus could hide improving population quality.

### Phase 13: General 100x30 Compute Probe Reaches Final Sector

Documented and archived run:

- `C:\f1rl-artifacts\archives\evolution-open-100x30-general-20260604.tar.gz`

Documented result:

- `100x30`.
- `3000` attempts.
- Best distance `5223.047m`.
- Best candidate generation `16`, candidate `73`.
- Source: offspring from `cleanest_distance`.
- Mutation: `controller_masked_gaussian`.
- Termination: collision.
- Final speed `214.33kph`.
- No valid lap completion.
- Counts: `414 >=1500m`, `385 >=2000m`, `275 >=2200m`, `245 >=2400m`, `151 >=2600m`, `72 >=3000m`, `66 >=4000m`, `54 >=5000m`.

What changed:

- Survival floors and frontier-focused scoring were used.
- Run length scaled from 10 generations to 30 generations.
- Scoring expanded to include profiles like `frontier_fast`, `farthest_distance`, `early_pace`, `clean_distance`, `frontier_recovery`, and `frontier_novelty`.

What was learned:

- More compute solved multiple prior frontiers.
- The car could reach the final sector.
- Full-lap search worked better than the old PPO micro-rung loop.
- The new blocker became final-sector setup/finish, not Rettifilo.

Important nuance:

- At the time, the car reached about `5223m`, near Parabolica approach, but crashed.
- The user correctly reframed the problem: do not overfit to a Parabolica-specific patch; instead increase the number of candidates reaching the end, increase average distance, and increase speed.

### Phase 14: Focused Roggia Segment Search And Transfer Failure

Documented run:

- `C:\f1rl-artifacts\evolution-roggia-1900-2800-frontier-100x10-20260604`

Documented result:

- `1000` attempts.
- Best distance `2800.629m`.
- Best candidate generation `8`, candidate `85`.
- Source: offspring from `fastest_pace`.
- Ended with `segment_complete`.
- Final speed `216.92kph`.
- Completion rate improved to `17%`.

What changed:

- Added `frontier_recovery`.
- Added `frontier_novelty`.
- Added `frontier_distance` parent bucket.
- Added plateau mode.
- Added ladder rungs for Roggia/Lesmo frontier.

What was learned:

- Focused segment search can solve a local section.
- The Roggia segment result did not transfer well back to normal-start full-lap search.

Documented transfer probe:

- `C:\f1rl-artifacts\evolution-open-100x10-frontier-plateau-20260604`
- `1000` attempts.
- Best distance `2157.566m`.
- `61 >=1500m`.
- `45 >=2000m`.
- `0 >=2200m`.

Conclusion:

- Local segment recovery profiles were useful as diagnostics, but too local/recovery-biased as the leading full-lap objective.
- Full-lap probes had to remain the scoreboard.

### Phase 15: Adaptive Selection And Average-Distance Push

Documented run:

- `C:\f1rl-artifacts\evolution-open-100x10-adaptive-20260604`

Documented result:

- `1000` attempts.
- Best distance `2448.570m`.
- Best source: smart immigrant from `smart_current_elite`.
- Mutation: `controller_weight_reset`.
- Average distance improved from `71.08m` in generation 0 to `766.60m` in generation 9.

Documented run:

- `C:\f1rl-artifacts\evolution-open-100x10-adaptive-quality-v2-20260604`

Documented result:

- `1000` attempts.
- Best distance `2457.633m`.
- Best source: offspring from `survival_gate`.
- Mutation: `controller_masked_gaussian`.
- Average distance improved from `71.08m` in generation 0 to `840.55m` in generation 9.
- Counts: `589 >=100m`, `473 >=450m`, `78 >=1000m`, `41 >=1220m`, `39 >=1500m`, `35 >=2000m`, `7 >=2400m`.

What changed:

- Performance-based dynamic survival floors.
- Tiered parent buckets.
- Adaptive immigrant budget.
- Smart immigrants split from pure randoms.
- Conservative pace-sensitive scoring.
- Full lineage logging.

What was learned:

- Population average could be improved materially.
- Smart immigrants and offspring near current elites were valuable.
- Too many early deaths were still a major throughput problem.

### Phase 16: First Valid-Lap Speed Runs

Documented historical result:

- `C:\f1rl-artifacts\evolution-open-100x30-avg-speed-aggressive-20260604`
- Current compressed archive: `C:\f1rl-artifacts\archives\evolution-open-100x30-avg-speed-aggressive-20260604.tar.gz`.
- Documented historical best valid evolved lap: `148.3s`.

Verified archive:

- `D:\f1-rl-artifacts\archives\evolution-speed-100x30-fastlap-v1-20260604.tar.gz`.
- Archive size: `2797209348` bytes.
- Archive contains `attempts.jsonl`, `evolution_summary.json`, `generation_summary.jsonl`, and `selected_telemetry/manifest.json`.
- Extracted summary verifies `population=100`, `generations=30`, `3000` candidates, `39` valid laps, fastest valid lap `161.95s`, best distance `5812.908m`.

Documented active docs before latest run:

- The `100x30` fastlap-v1 run had `39` valid laps.
- Fastest valid lap was `161.95s`.
- Final generation average distance was `2733.5m`.
- Final generation average pace was `117.2kph`.
- Final generation top-decile pace was `168.4kph`.

What changed:

- The objective shifted from "can it reach the end?" to "can it finish fast?"
- Metrics shifted from best distance to fastest valid lap, average valid lap, valid count, top-decile pace, average pace, and gate pass rates.
- Storage shifted toward compressed all-candidate telemetry because raw traces were too large.

What was learned:

- The car could finish, but slowly.
- Safe completion was no longer enough.
- Scoring still overvalued slow valid laps.
- Speed had to become the main target.

### Phase 17: Hot/Cold Artifact Split And Lossless Compression

Verified docs:

- `workflow.md`
- `Documentation.md`
- `goal.md`

Verified storage contract:

- Hot artifacts stay on `C:\f1rl-artifacts\<run-name>`.
- Cold all-candidate telemetry goes to `D:\f1-rl-artifacts\cold-telemetry\<run-name>`.
- Old archives go under `C:\f1rl-artifacts\archives` or `D:\f1-rl-artifacts\archives`.
- Manifests on C: point to cold `.jsonl.gz` traces on D:.
- Telemetry fields are not dropped; compression is lossless.

Verified current storage:

- Current hot run: `C:\f1rl-artifacts\evolution-speed-150x60-20260604-0501`.
- Current hot run size: about `0.497 GiB`.
- Current cold telemetry: `D:\f1-rl-artifacts\cold-telemetry\evolution-speed-150x60-20260604-0501`.
- Current cold telemetry count: `9000` `.jsonl.gz` files.
- Current cold telemetry size: about `5.098 GiB`.
- Current C: archive directory size: about `7.464 GiB`.
- Current D: archive directory contains the verified `100x30` speed archive.

Why it mattered:

- Large runs were previously storage- and memory-risky.
- Compressed all-candidate telemetry made it possible to keep debugging value without exhausting C:.
- The storage loop became part of the experimentation workflow, not an afterthought.

### Phase 18: Latest 150x60 Speed Run Reaches 89.983s

Verified run:

- `C:\f1rl-artifacts\evolution-speed-150x60-20260604-0501`

Verified command shape:

- `population=150`.
- `generations=60`.
- `max_steps=18000`.
- `start_progress_m=0`.
- `start_speed_kph=80`.
- `target_progress_m=1500`.
- `--no-target-termination`.
- `genome_type=controller`.
- Scoring profiles included `fast_valid_lap`, `time_attack`, `fast_frontier`, `lap_pace`, `frontier_fast`, `farthest_distance`, `early_pace`, `clean_distance`, `frontier`, `risk_seeking`, `max_progress`, `clean_exit`, `exit_speed`, `frontier_recovery`, and `frontier_novelty`.
- All-candidate telemetry used gzip compression on D:.

Verified aggregate:

- Attempts: `9000`.
- Generations: `60`.
- Population: `150`.
- Lap-complete valid attempts: `926`.
- Valid-lap rate: `10.289%`.
- Fastest valid lap: `89.983333s`.
- Best distance: `5812.895m`.
- Hot artifact files: `69`.
- Hot artifact size: about `0.534 GB` decimal, about `0.497 GiB`.
- Cold telemetry traces: `9000`.
- Cold telemetry size: about `5.098 GiB`.

Verified termination counts across `9000` attempts:

- `off_track`: `5376`.
- `collision`: `2164`.
- `no_progress`: `513`.
- `max_steps`: `21`.
- `lap_complete`: `926`.

Verified gate counts across `9000` attempts:

- `>=450m`: `7234`.
- `>=1000m`: `2201`.
- `>=1220m`: `1988`.
- `>=1500m`: `1983`.
- `>=2000m`: `1947`.
- `>=3000m`: `1186`.
- `>=4000m`: `1094`.
- `>=5000m`: `1074`.
- `>=5793m`: `926`.

Verified valid-lap time distribution:

- `<=90s`: `21`.
- `<=95s`: `165`.
- `<=100s`: `257`.
- `<=110s`: `503`.
- `<=120s`: `594`.
- `<=130s`: `665`.
- `<=150s`: `762`.
- `<=170s`: `862`.
- `<=200s`: `898`.

Verified key generations:

- Best fastest-valid generation: generation `49`, fastest `89.983s`, average valid `101.560s`, valid `16/150`, average distance `1339.575m`, top-decile pace `262.665kph`.
- Best average-valid generation: generation `55`, average valid `96.605s`, fastest `89.983s`, valid `13/150`, top-decile pace `264.464kph`.
- Best valid-count generation: generation `26`, valid `32/150`, fastest `104.233s`, average valid `123.189s`, average distance `2479.489m`.
- Best average-distance generation: generation `27`, average distance `2665.415m`, valid `27/150`, fastest `100.467s`.
- Best average-pace generation: generation `57`, average pace `216.726kph`, fastest `89.983s`, valid `13/150`.
- Final generation `59`: average distance `1601.020m`, average pace `214.070kph`, top-decile distance `5766.924m`, top-decile pace `258.643kph`, valid `14/150`, fastest `89.983s`.

Interpretation:

- Distance/finish was no longer the scarce signal.
- Speed and breadth became the bottlenecks.
- The population had a real fast-valid family, not just one lucky finish.
- But the fastest family still did not close the full gap to Fast-F1.

## Part 2: Total Search Scale

### Current Verifiable Artifact Inventory

Verified from readable summaries:

- Readable `evolution_summary.json` files under repo `artifacts/` and `C:\f1rl-artifacts`: `66`.
- Candidate records represented by those readable summaries: `11650`.
- Generation summaries represented by those readable summaries: `180`.
- Current readable large run: `9000` candidates.
- Current readable replay swarms include two `100x5` runs, `500` candidates each.

Verified current compressed archive inventory:

- `C:\f1rl-artifacts\archives`: `9` `.tar.gz` archives.
- `D:\f1-rl-artifacts\archives`: `1` `.tar.gz` archive.
- Total currently visible old-run archives: `10`.

Verified archive names on C::

- `evolution-0to1220-100x10-aggressive-20260604.tar.gz`.
- `evolution-0to1220-100x10-stream-20260604.tar.gz`.
- `evolution-open-100x10-adaptive-20260604.tar.gz`.
- `evolution-open-100x10-adaptive-quality-v2-20260604.tar.gz`.
- `evolution-open-100x10-aggressive-20260604.tar.gz`.
- `evolution-open-100x10-frontier-plateau-20260604.tar.gz`.
- `evolution-open-100x30-avg-speed-aggressive-20260604.tar.gz`.
- `evolution-open-100x30-general-20260604.tar.gz`.
- `evolution-roggia-1900-2800-frontier-100x10-20260604.tar.gz`.

Verified archive name on D::

- `evolution-speed-100x30-fastlap-v1-20260604.tar.gz`.

Verified all-preserved-run total:

- Unique preserved evolution/search run artifacts counted: `76`.
- Total candidate evaluations represented by those preserved artifacts: `27650`.
- Total generations represented by those preserved artifacts: `340`.
- Live/readable candidate evaluations: `11650`.
- Archived candidate evaluations: `16000`.
- This count includes the repeated experiments. It does not collapse "a 100x10 run" into one bucket; every preserved 100x10/100x30/etc. run is counted separately.

Verified serious/named-scale subtotal:

- Serious or named scale runs counted: `21`.
- Serious/named candidate evaluations: `26640`.
- Serious/named generations: `262`.
- This subtotal includes the runs that match the actual experimental ladder the project followed: repeated `20x4`, repeated `100x5`, repeated `100x10`, repeated `100x30`, and the latest `150x60`.

Scale breakdown from preserved artifacts:

| Scale | Counted runs | Candidates represented | Notes |
| --- | ---: | ---: | --- |
| `150x60` | `1` | `9000` | Latest speed run, current strongest result. |
| `100x30` | `3` | `9000` | General probe, avg-speed/aggressive run, speed fastlap-v1 run. |
| `100x10` | `7` | `7000` | Stream/aggressive/adaptive/frontier/Roggia-style iterations. |
| `100x5` | `2` | `1000` | Visual/swarm replay development runs. |
| `20x4` | `8` | `640` | Tiny/load/perf/swarm development and throughput runs. |
| Smaller smoke/debug runs | many | `1010` | CLI, ladder, gzip, worker, resume, parsing, and profile checks. |
| Total | `76` preserved run artifacts | `27650` | Full current artifact-backed count. |

Resume-safe phrasing:

- Strongest single-run metric: latest `150x60` run evaluated `9000` controllers and produced the `89.983s` lap.
- Strongest total-scale metric: preserved artifacts account for `27650` evolutionary/search candidate evaluations across `76` runs and `340` generations.

### Major Run Table

| Run | Status | Population x generations | Candidates | Best distance | Valid laps | Fastest valid lap | Main scoring / change | What was learned |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| PPO historical benchmarks | Verified from artifact audit | n/a | n/a | `970.775m` | `0` | none | SB3 PPO with curriculum/eval infrastructure | PPO infrastructure worked, but honest normal-start PPO plateaued before full-lap behavior. |
| `evolution-throughput-audit-20260603` | Verified readable | `64x2` | `128` | `530.154m` | `0` | none | Early throughput audit | Baseline throughput was slow enough to justify evaluator/storage work. |
| `evolution-swarm-20x4-visible-20260604` | Verified readable | `20x4` | `80` | `208.749m` | `0` | none | All-candidate visual swarm | Replay exposed early deaths and non-moving/slow candidates. |
| `evolution-swarm-100x5-visible-next` | Verified readable | `100x5` | `500` | `312.602m` | `0` | none | Generation-by-generation replay | Swarm replay became usable; still far from useful driving. |
| `evolution-0to1220-100x10-aggressive-20260604` | Verified archive summary | `100x10` | `1000` | `1220.474m` | `0` | none | Early Rettifilo/1220 target search | Hit the early target but still needed open-distance full-lap pressure. |
| `evolution-0to1220-100x10-stream-20260604` | Documented + archived | `100x10` | `1000` | `1130.4m` | `0` | none | Streamed all-candidate telemetry | Fixed multiprocessing telemetry memory failure, but still short of `1220m`. |
| `evolution-open-100x10-aggressive-20260604` | Documented + archived | `100x10` | `1000` | `2409.611m` | `0` | none | Open distance, aggressive controller mutation/scoring | Big breakthrough past PPO plateau and early track. |
| `evolution-open-100x30-general-20260604` | Documented + archived | `100x30` | `3000` | `5223.047m` | `0` | none | More compute, dynamic survival/frontier scoring | Reached final sector; finish remained unsolved. |
| `evolution-roggia-1900-2800-frontier-100x10-20260604` | Documented + archived | `100x10` | `1000` | `2800.629m` segment | segment `17%` | n/a | Frontier recovery/novelty segment search | Local Roggia skill solved, but local success did not transfer. |
| `evolution-open-100x10-frontier-plateau-20260604` | Documented + archived | `100x10` | `1000` | `2157.566m` | `0` | none | Normal-start transfer probe after Roggia segment | Transfer regressed; frontier-recovery was too local as main full-lap objective. |
| `evolution-open-100x10-adaptive-20260604` | Documented + archived | `100x10` | `1000` | `2448.570m` | `0` | none | Dynamic floor, smart immigrants, parent buckets | Average distance improved, smart immigrants helped. |
| `evolution-open-100x10-adaptive-quality-v2-20260604` | Documented + archived | `100x10` | `1000` | `2457.633m` | `0` | none | Broader quality pass-rate adaptation | Average distance improved more; still no lap. |
| `evolution-open-100x30-avg-speed-aggressive-20260604` | Verified archive summary + documented fastest lap | `100x30` | `3000` | `5808.352m` | valid laps present | `148.3s` documented | First strong speed/valid-lap scoring | Valid evolved laps existed, but still much slower than target. |
| `evolution-speed-100x30-fastlap-v1-20260604` | Verified archive summary | `100x30` | `3000` | `5812.908m` | `39` | `161.95s` | Fast-lap/breadth scoring | Improved valid-lap breadth but not fastest lap relative to `148.3s`. |
| `evolution-speed-150x60-20260604-0501` | Verified current artifacts | `150x60` | `9000` | `5812.895m` | `926` | `89.983s` | `fast_valid_lap`, `time_attack`, `lap_pace`, hot/cold gzip telemetry | Major speed breakthrough; now within `10.321s` of Fast-F1 reference. |

### Progression Of Best Result

Verified/documented progression:

- PPO plateau: `~970.775m`, no valid lap.
- Early visual swarm: `208.749m` to `312.602m`.
- Streamed `0->1220` run: `1130.4m`.
- Open aggressive `100x10`: `2409.611m`.
- General `100x30`: `5223.047m`.
- First documented valid evolved speed result: `148.3s`.
- Verified archived `100x30` speed run: `39` valid laps, fastest `161.95s`.
- Latest `150x60`: `926` valid laps, fastest `89.983s`.

Dead ends and lessons:

- PPO micro-rungs: useful local work, poor normal-start transfer.
- Roggia segment search: solved local segment, failed full-lap transfer.
- Frontier-recovery as main objective: too local/recovery-biased.
- Distance-only scoring: got far but could reward terrible setup.
- Slow safe completion: solved finish, not speed.
- Oversized telemetry on C: and uncompressed all-candidate traces: slowed iteration and caused memory/storage pressure.

## Part 3: Evolutionary Search Architecture

### Entry Points

Verified files:

- `src/f1rl/evolution_search.py`
- `src/f1rl/evolution_ladder.py`

Verified CLIs:

- `python -m f1rl.evolution_search`
- `f1-evolution-search`
- `python -m f1rl.evolution_ladder`
- `f1-evolution-ladder`

### Genomes

Verified genome types:

- `phase`.
- `progress_phase`.
- `controller`.

`phase` genome:

- Fixed action schedule by elapsed step count.
- Useful for compatibility and small diagnostics.
- Weak for full-lap behavior because timing changes with speed.

`progress_phase` genome:

- Action schedule changes by progress delta in meters.
- More robust than step phases for segment search because it follows distance landmarks.
- Still limited because it is effectively an action tape.

`controller` genome:

- Observation-driven continuous controller.
- Current serious runs use this.
- It stores `69` controller weights.
- It uses `23` controller features.
- It outputs continuous throttle, brake, and steering through `step_controls()`.

Verified controller feature list from the fastest genome:

- `bias`.
- `speed_norm`.
- `target_speed_norm`.
- `speed_error_norm`.
- `brake_demand`.
- `future_brake_demand`.
- `target_speed_drop_norm`.
- `brake_gate_proximity`.
- `brake_gate_distance_norm`.
- `lookahead_abs_max`.
- `signed_lateral_error_norm`.
- `heading_error_norm`.
- `yaw_rate_norm`.
- `curvature_norm`.
- `target_steer`.
- `last_throttle`.
- `last_brake`.
- `last_steer`.
- `segment_progress_ratio`.
- `lookahead_0`.
- `lookahead_1`.
- `lookahead_2`.
- `lookahead_3`.

Controller outputs:

- Throttle in `[0, 1]`.
- Brake in `[0, 1]`.
- Steering in `[-1, 1]`.

Why the controller mattered:

- It generalized better than fixed action phases.
- It could react to braking demand, target speed, heading error, lateral error, and lookahead.
- It made "spawn many cars, let them drive, pick the best" closer to actual driving behavior than schedule enumeration.

### Mutation And Crossover

Verified mutation operators include:

- `controller_masked_gaussian`.
- `controller_full_gaussian`.
- `controller_weight_reset`.
- `none`.
- Phase/progress-phase schedule mutation for non-controller genomes.

Verified metadata:

- Mutation type.
- Mutation sigma.
- Reset count.
- Changed gene count.
- Genome distance from parent.
- Whether plateau mode was active.

Verified crossover metadata:

- Whether crossover was used.
- Crossover partner generation/candidate/seed.
- Partner source and bucket.

Interpretation:

- The fast family was not a single random miracle.
- It descended through repeated offspring, crossover, elite preservation, plateau-mode mutation, and smart-immigrant mutation.

### Selection

Verified mechanisms:

- Rank-biased parent sampling.
- Elite retention.
- Multi-profile elite union.
- Parent buckets.
- Dynamic survival floors.
- Smart immigrants.
- Pure random immigrants.
- Current elite and global elite archives.
- Plateau mode.

Parent buckets documented/verified across lineages:

- `survival_gate`.
- `farthest_distance`.
- `fastest_pace`.
- `cleanest_distance`.
- `frontier_distance`.
- `fastest_valid_lap`.
- `fast_valid_lap_score`.
- `fast_frontier_score`.
- `far_fast`.
- `clean_distance`.
- `smart_current_elite`.
- `smart_global_elite`.

Dynamic survival floors:

- Survival floors move outward as population/frontier quality improves.
- Latest run used stages including `450`, `1000`, `1220`, `1500`, `2000`, `2400`, `3000`, `4000`, `5000`.
- In the final generation of the latest run, survival floor was `5000m`.

Plateau mode:

- Detects flat best distance with population changes.
- Can reduce elite-copy pressure and increase mutation.
- Earlier versions were too disruptive late; later use became more local/frontier-focused.

### Scoring Profiles

Verified scoring profiles include:

- `max_progress`.
- `clean_exit`.
- `brake_zone`.
- `apex`.
- `exit_speed`.
- `full_lap_validity`.
- `risk_seeking`.
- `frontier`.
- `frontier_fast`.
- `early_pace`.
- `clean_distance`.
- `farthest_distance`.
- `frontier_recovery`.
- `frontier_novelty`.
- `fast_valid_lap`.
- `time_attack`.
- `fast_frontier`.
- `lap_pace`.

Scoring evolution:

1. Early search rewarded max progress and clean exits.
2. Open-distance search stopped capping progress at target milestones.
3. Frontier scoring helped push beyond repeated crash points.
4. Dynamic survival and parent buckets raised population average distance.
5. Valid-lap scoring and time-attack scoring shifted from "finish at all" to "finish fast."
6. Lap-pace scoring and fast-frontier scoring helped before valid laps were common.

Key lesson:

- Best distance was the right metric until candidates could finish.
- After lap completion became common, best distance became saturated and misleading.
- The main metrics became valid lap count, fastest valid lap, average valid lap, average pace, top-decile pace, and gate pass rates.

### Artifacts And Reproducibility

Verified evolution outputs:

- `attempts.jsonl`.
- `generation_summary.jsonl`.
- `evolution_summary.json`.
- `best_so_far.json`.
- `population_checkpoint.json`.
- `top_genomes/*.json`.
- `selected_telemetry/manifest.json`.
- `elite_state_library.json`.
- `ppo_bridge.json`.
- `next_commands.md`.
- Optional cold all-candidate `.jsonl.gz` traces.

Verified resume/checkpoint design:

- Checkpoint stores generation number, next population, RNG state, config hash, elites, best candidates, output paths, and resume command.
- Resume rejects mismatched config unless forced.
- Atomic checkpoint writing uses temp file then replace.

Verified hot/cold design:

- Hot summaries/manifests/checkpoints stay on C:.
- Cold traces go to D:.
- Replay reads manifest paths to find compressed traces.

## Part 4: What Made The 89.983s Controller Fast

### Fastest Lap Identity

Verified fastest attempt:

- Run: `C:\f1rl-artifacts\evolution-speed-150x60-20260604-0501`.
- Generation: `49`.
- Candidate: `148`.
- Seed: `49000155`.
- Lap time: `89.983333s`.
- Best progress: `5804.548m`.
- Score: `37708072.426`.
- Primary scoring profile: `fast_valid_lap`.
- Genome kind: `controller`.
- Source: `smart_immigrant`.
- Immigrant type / bucket: `smart_current_elite`.
- Mutation: `controller_weight_reset`.
- Mutation sigma: `0.82`.
- Reset count: `7`.
- Parent: generation `48`, candidate `4`.
- Selected telemetry: `D:\f1-rl-artifacts\cold-telemetry\evolution-speed-150x60-20260604-0501\evolution-all_candidates-gen-049-candidate-00148-steps.jsonl.gz`.

Verified trace:

- Steps: `5399`.
- Elapsed: `89.983333s`.
- Final progress: `5804.548m`.
- Average trace speed: `230.846kph`.
- Max speed: `315.868kph`.
- Min speed: `81.028kph`.
- Final speed: `271.854kph`.
- Final lateral error: `3.993m`.
- Final heading error: `-1.539deg`.
- Final throttle: `0.812`.
- Final brake: `0.007`.
- Final steering: `0.00046`.
- Termination: `lap_complete`.
- Valid lap: `true`.
- Finish crossed: `true`.

Important start boundary:

- This is a rolling/flying start from `80kph`, not a `0kph` cold start.

### Speed And Section Behavior

Verified progress crossings for the fastest trace:

| Progress | Time | Speed | Lateral error | Heading error | Interpretation |
| ---: | ---: | ---: | ---: | ---: | --- |
| `300m` | `5.067s` | `297.039kph` | `1.337m` | `-1.658deg` | Strong launch/straight speed from rolling start. |
| `450m` | `6.833s` | `312.004kph` | `3.347m` | `-1.492deg` | Very fast approach to Rettifilo. |
| `650m` | `9.367s` | `272.490kph` | `6.711m` | `2.401deg` | Still very fast; likely not human-like braking yet. |
| `1000m` | `14.700s` | `211.818kph` | `12.228m` | `38.865deg` | Major line/rotation inefficiency at Rettifilo exit. |
| `1220m` | `18.433s` | `225.889kph` | `2.554m` | `-1.979deg` | Recovers and exits early complex. |
| `1500m` | `22.983s` | `224.296kph` | `0.053m` | `1.254deg` | Clean centerline recovery. |
| `2000m` | `29.767s` | `255.621kph` | `2.652m` | `5.202deg` | Good high-speed progress. |
| `2400m` | `35.983s` | `165.223kph` | `7.761m` | `0.933deg` | Slow/line loss around Roggia/Lesmo. |
| `3000m` | `48.650s` | `238.882kph` | `4.379m` | `-1.203deg` | Rebuilds speed. |
| `4000m` | `62.233s` | `219.512kph` | `2.547m` | `23.403deg` | Significant heading inefficiency in later sector. |
| `5000m` | `75.250s` | `201.773kph` | `4.376m` | `-0.040deg` | Controlled final-sector entry. |
| `5500m` | `86.083s` | `247.140kph` | `8.191m` | `1.715deg` | Good acceleration toward finish, but line not perfect. |
| `5793m` | `89.983s` | `271.854kph` | `3.993m` | `-1.539deg` | Valid finish. |

What it handled well:

- Very fast rolling-start acceleration.
- Strong early straight speed.
- Enough braking/turning to survive the lap.
- Recovery from ugly early line states.
- Final-sector completion without collision/off-track.
- High finish speed.

Where it lost time:

- Rettifilo remains inefficient at around `1000m`: huge heading error and lateral error.
- Roggia/Lesmo region slows to `165kph` at `2400m`.
- Later sector has heading inefficiency at `4000m`.
- Average trace speed `230.846kph` remains below Fast-F1 reference mean `259.914kph`.
- Max speed `315.868kph` remains below Fast-F1 max `348kph`.
- Lap time gap to reference is `10.321s`.

### Was It Lucky Or A Family?

Verified: it was part of a broader family, not only one isolated finish.

Evidence:

- Latest run produced `926` valid lap-complete attempts.
- `21` valid laps were `<=90s`.
- `165` valid laps were `<=95s`.
- `257` valid laps were `<=100s`.
- `503` valid laps were `<=110s`.
- The best `89.983s` lap appeared at generation `49`.
- It was then preserved by elite copies in later generations.
- Best average-valid generation later reached `96.605s`.
- Final generation still had `14/150` valid laps and fastest `89.983s`.

Important caveat:

- The exact fastest controller family was heavily preserved through elite copies after discovery.
- That is good for retaining the breakthrough, but it also means future improvement needs broader fast-valid branching, not just preserving the same champion.

### What Likely Made It Fast

Verified inputs it had:

- Speed and target speed.
- Speed error.
- Brake demand and future brake demand.
- Target speed drop.
- Brake gate proximity/distance.
- Lookahead heading signals.
- Lateral and heading error.
- Curvature.
- Target steer.
- Last controls.
- Segment progress.

Likely mechanism:

- The controller learned a rough reactive mapping from upcoming speed-drop/brake-demand/lookahead features to throttle/brake/steer.
- It learned enough line recovery to survive bad intermediate states.
- It exploited high acceleration and high speed on straights.
- It optimized lap time enough to beat slow valid families, but not enough to match the Fast-F1 racing line.

Do not overclaim:

- There is no proof it learned a human-quality racing line.
- There is proof it learned a valid high-speed controller in this simulator.

## Part 5: PPO Vs Evolutionary Search

### What PPO Includes

Verified PPO infrastructure:

- SB3 PPO training.
- CUDA-aware configuration.
- Vectorized environments.
- Curriculum starts.
- State-library starts.
- TensorBoard.
- Checkpoints.
- Metadata-faithful eval.
- Benchmark suite.
- Replayable telemetry.
- Reward overrides and assist/scaffold controls.

### What PPO Achieved

Verified:

- PPO reached about `970.775m` in best honest normal-start benchmark artifacts.
- It did not produce a valid lap.
- It did not produce a lap time.

### Why PPO Plateaued

Evidence-based diagnosis:

- The environment rewards progress.
- Early in training, going fast and crashing later can look better than braking early and losing immediate progress.
- The useful behavior requires long-horizon coordination across braking zones, line setup, corner rotation, and exits.
- PPO needs many samples to discover rare behavior from scratch.
- Narrow local curriculum helped isolated skills but did not reliably transfer to normal-start full-lap behavior.

### What Evolution Solved

Evolution solved discovery:

- It could evaluate many whole-controller variants.
- It could keep rare breakthroughs.
- It could mutate around elite families.
- It could use multiple scoring profiles at once.
- It could directly optimize lap-time metrics once valid laps existed.
- It produced `926` valid laps in the latest run and a `89.983s` best lap.

### What Evolution Does Not Yet Solve

Evolution does not yet provide:

- A trained PPO policy.
- A scratch PPO normal-start full-lap solution.
- Generalization beyond the evolved controller distribution.
- A guaranteed robust policy under noise/seeds beyond evaluated candidates.
- The final target `<=80s`.

Best framing:

> The project includes PPO, but the current strongest solved behavior comes from evolutionary controller search. The right next research step is to use evolved states/trajectories to transfer the discovered behavior back into PPO.

### How Evolution Can Feed PPO

Verified bridge artifacts:

- `elite_state_library.json`.
- `ppo_bridge.json`.
- `next_commands.md`.

Transfer plan:

- Select fast valid and fast frontier states.
- Build curriculum starts from those states.
- Train PPO to reproduce evolved behavior from varied starts.
- Expand starts outward.
- Retest honest normal-start PPO.
- Disable scaffolds/assists for final evaluation.

## Part 6: Metrics And Proof

### Top 15 Resume/Portfolio Metrics

1. Verified latest evolved result: `89.983333s` valid rolling-start Monza lap.
2. Verified Fast-F1 reference target: `79.662s`.
3. Verified gap to reference: `10.321s`.
4. Verified preserved search scale: `27650` candidate evaluations across `76` run artifacts and `340` generations.
5. Verified serious/named search scale: `26640` candidates across `21` repeated scale runs.
6. Verified latest run scale: `9000` controller candidates over `60` generations.
7. Verified latest run valid laps: `926`.
8. Verified latest run valid-lap rate: `10.289%`.
9. Verified fast-lap family breadth: `21 <=90s`, `165 <=95s`, `257 <=100s`.
10. Verified fastest trace speed: average `230.846kph`, max `315.868kph`.
11. Verified Fast-F1 speed reference: mean `259.914kph`, max `348kph`.
12. Verified cold telemetry: `9000` `.jsonl.gz` traces, about `5.098 GiB`.
13. Verified hot artifacts: current run about `0.497 GiB`, summaries/manifests/checkpoints on C:.
14. Verified track model: `5793m`, `120` checkpoints, `1800` collision segments.
15. Verified validation from previous audit: `143` tests collected; ruff, pyright, pytest, and Gymnasium env checker passed.

### Baseline Comparison

| Policy/system | Completion | Lap time | Progress | Notes |
| --- | ---: | ---: | ---: | --- |
| Random baseline | `0%` in fresh 3-episode benchmark | none | best `167.717m` | Crashes/off-track quickly. |
| Scripted baseline | `100%` valid | `214.467s` | full lap | Safe but very slow. |
| Reference ghost | `100%` valid | `79.662s` | full lap | Fast-F1 target. |
| Best PPO benchmark | `0%` valid | none | `~970.775m` | PPO plateau. |
| Latest evolved controller | valid | `89.983s` | `5804.548m` | Rolling start at `80kph`; current strongest solved behavior. |

### Throughput And Storage Metrics

Verified/documented:

- Early throughput audit: `128` candidates in `156.69s`, about `0.82 candidates/s` with `workers=2`.
- Latest full run generation throughput varied by generation; final generation logged `4.131 candidates/s`.
- Latest cold telemetry size: `5.098 GiB` for `9000` compressed traces.
- Latest hot artifact size: about `0.497 GiB`.
- Current C: archives: about `7.464 GiB`.
- Current D: verified speed archive: `2.797GB` decimal.

## Part 7: Resume Wording

Rules satisfied:

- Two bullets per version.
- Verified metrics only.
- No claim that PPO solved the lap.
- Strong ownership language.

### A. ML Engineering Version

- Built a custom Gymnasium/SB3 Formula 1 Monza RL stack with OpenCV-derived track geometry, deterministic bicycle-model physics, ray-cast/racing-line observations, PPO training/eval infrastructure, telemetry replay, and Fast-F1 calibration against a `79.662s` reference lap.
- Engineered an evolutionary controller-search pipeline with multi-profile scoring, dynamic survival floors, lineage, checkpoint/resume, and compressed all-candidate telemetry; preserved artifacts count `27650` candidate evaluations, with the latest `150x60` run producing `926` valid laps and a `89.983s` evolved lap.

### B. SWE / Systems Version

- Built a modular Python simulation platform for F1 racing AI, connecting image-derived track geometry, deterministic vehicle physics, Gymnasium/SB3 integration, benchmark tooling, replay visualization, JSONL/gzip telemetry, and artifact-safe experiment workflows.
- Designed a reproducible evolutionary search system with controller genomes, mutation/crossover, smart immigrants, dynamic parent selection, streaming summaries, resumable checkpoints, and hot/cold storage across `27650` counted candidate evaluations, culminating in a `89.983s` valid Monza lap.

### C. Robotics / Simulation Version

- Built a deterministic top-down vehicle simulator with bicycle-model dynamics, grip-limited steering/braking, collision/off-track detection, checkpoint-valid lap logic, ray-cast perception, racing-line observations, and Fast-F1-calibrated Monza reference telemetry.
- Evolved continuous driving controllers through repeated population-search rollouts from `20x4` through `150x60`, with preserved artifacts counting `27650` candidate evaluations and the latest run discovering `926` valid laps plus a fastest `89.983s` rolling-start Monza lap.

### D. Final Best General Resume Version

- Built a custom Gymnasium/SB3 Formula 1 Monza racing stack with OpenCV track extraction, calibrated bicycle-model physics, ray-cast/racing-line observations, PPO training/evaluation infrastructure, benchmark tooling, and full-fidelity telemetry/replay.
- Designed and optimized an elitist evolutionary controller-search engine with multi-profile scoring, dynamic selection, lineage, checkpoint/resume, and compressed all-candidate artifacts; preserved runs count `27650` candidate evaluations, with the latest `150x60` run producing `926` valid laps and a `89.983s` evolved lap against a `79.662s` Fast-F1 target.

## Part 8: Portfolio Rewrite

### Polished Title

F1RL: Custom Monza Racing Simulator, PPO Environment, And Evolutionary Controller Search

### Polished Summary

Built a full-stack Formula 1 racing AI lab from scratch: Monza track geometry, deterministic vehicle physics, Gymnasium/SB3 PPO infrastructure, Fast-F1 reference calibration, telemetry/replay tooling, and an evolutionary controller-search engine that discovered a `89.983s` valid rolling-start Monza lap.

### 1. Context

- The project asks whether a simplified top-down simulator can support serious racing AI experimentation.
- Monza is used because it has fast straights, heavy braking zones, chicanes, medium-speed corners, and a real Fast-F1 reference lap.
- The goal is not just "train PPO"; it is to build a complete environment and experiment system where driving behavior can be generated, inspected, benchmarked, replayed, and improved.

### 2. Problem

- A racing policy has to coordinate long-horizon behavior: throttle, braking, turn-in, apex, exit, recovery, and final lap completion.
- PPO from scratch plateaued around `970.775m`.
- Early evolutionary runs could go far but crashed.
- Later runs could finish but were too slow.
- The current bottleneck is speed and fast-valid breadth, not basic lap completion.

### 3. System / Model Architecture

- Track layer: OpenCV-derived Monza geometry, centerline, boundaries, mask, checkpoints, and scale.
- Physics layer: deterministic bicycle model at `60Hz` with grip, drag, braking, and speed-sensitive steering.
- Simulator layer: `MonzaSim`, shared by every runtime mode.
- Environment layer: `MonzaEnv`, Gymnasium-compatible wrapper.
- PPO layer: SB3 training, eval, checkpointing, curriculum, and benchmarks.
- Search layer: evolutionary controller genomes optimized by simulator rollouts.
- Observability layer: JSONL/gzip telemetry, generation summaries, lineage, replay, QC, and section analysis.

### 4. Core Implementation

- Track projection and valid checkpoint progression.
- Ray-cast perception using 7 rays over a 120-degree field.
- Racing observations with lookahead heading, target speed, brake demand, brake gate proximity, and target steer.
- Discrete, continuous, and multidiscrete action modes.
- Continuous controller search through `step_controls()`.
- Multi-profile scoring for distance, speed, clean line, valid lap time, lap pace, and frontier behavior.
- Dynamic survival floors and parent buckets.
- Smart immigrants around current/global elites.
- Plateau mode and mutation schedules.
- Hot/cold artifact storage with lossless compressed all-candidate telemetry.

### 5. Evaluation / Benchmarks

- Fast-F1 reference: `79.662s`.
- Random baseline: no completion; best fresh progress `167.717m`.
- Scripted baseline: valid lap `214.467s`.
- PPO best verified: `~970.775m`, no valid lap.
- Latest evolved controller: valid lap `89.983s`.
- Preserved search scale: `27650` candidate evaluations across `76` run artifacts and `340` generations.
- Latest large run: `9000` candidates, `60` generations, `926` valid laps.
- Cold telemetry: `9000` compressed traces.

### 6. Impact / Outcome

- Converted the project from a PPO plateau into a working hybrid search platform.
- Demonstrated full-lap evolved behavior near the Fast-F1 target.
- Preserved enough artifacts to defend the result quantitatively and visually.
- Built a path for PPO transfer from evolved elite states.

### 7. Technical Stack / What Made It Hard

Stack:

- Python.
- NumPy.
- OpenCV.
- Gymnasium.
- Stable-Baselines3.
- PyTorch/CUDA.
- Pygame.
- Fast-F1-derived telemetry.
- pytest.
- ruff.
- pyright.
- JSONL/gzip artifacts.

Hard parts:

- Long-horizon credit assignment.
- Reward/scoring design.
- Avoiding fake progress and reward hacking.
- Making line quality matter without killing exploration.
- Scaling all-candidate telemetry without memory/storage failure.
- Debugging behavior through replay instead of just scalar reward.
- Turning rare one-off breakthroughs into a broader fast-valid population.

### Recommended Portfolio Visuals

- Architecture diagram: track geometry -> physics -> simulator -> Gym/PPO/evolution -> telemetry/replay.
- Monza track render with centerline, checkpoints, boundaries, and ray sensors.
- Fast-F1 reference speed profile vs evolved `89.983s` trace.
- Generation chart: fastest valid lap, valid count, average distance, average pace, top-decile pace.
- Swarm replay video showing many candidates per generation.
- Single-car replay video of the `89.983s` lap.

### Recommended Tags

Reinforcement Learning, Evolutionary Algorithms, Genetic Algorithms, Gymnasium, Stable-Baselines3, PPO, Vehicle Dynamics, Simulation, Control, Robotics, Racing AI, Formula 1, Fast-F1, Telemetry, Replay Systems, OpenCV, Pygame, PyTorch, CUDA, Python, Optimization, Curriculum Learning, State Libraries, Experiment Tracking.

## Part 9: README Rewrite

### Recommended README Structure

1. Project thesis and current verified result.
2. Architecture overview.
3. Environment contract.
4. Track geometry.
5. Physics and calibration.
6. Observation/action spaces.
7. PPO infrastructure.
8. Evolutionary search.
9. Telemetry/replay.
10. Benchmarks and latest results.
11. Artifact storage.
12. How to run.
13. Limitations.
14. Roadmap.

### Exact Top 3 README Paragraphs

F1RL is a custom Formula 1 racing AI stack for a simplified top-down Monza simulator. It builds the environment from first principles: OpenCV-derived track geometry, deterministic bicycle-model vehicle physics, Gymnasium/SB3 PPO integration, Fast-F1 reference telemetry, benchmark tooling, JSONL/gzip telemetry, replay visualization, and evolutionary controller search.

The current strongest verified result is evolutionary. Across preserved search artifacts, the project currently counts `27650` candidate evaluations across `76` runs and `340` generations, including repeated `20x4`, `100x5`, `100x10`, `100x30`, and `150x60` experiments. The latest large run evaluated `9000` continuous controller candidates over `60` generations, produced `926` valid lap-complete attempts, and found a `89.983s` valid rolling-start Monza lap. The reference target is Verstappen's 2024 Monza qualifying lap from Fast-F1 at `79.662s`, so the evolved controller is currently `10.321s` off the reference while preserving full replay/debug artifacts.

PPO infrastructure is implemented and tested, but PPO has not yet solved the full lap. Historical honest normal-start PPO benchmarks plateaued around `970.775m`, so the active strategy is evolution-first discovery: use population search to find viable fast trajectories and elite state libraries, then transfer those discovered behaviors into PPO curriculum training.

### README Limitations Section To Include

- The `89.983s` lap is rolling-start from `80kph`, not a `0kph` cold start.
- The strongest solved behavior is evolutionary controller search, not PPO.
- The simulator is simplified top-down physics, not full F1 vehicle dynamics.
- The result is Monza-specific.
- The current target remains `<=80s` and eventual PPO transfer.

## Part 10: Interview Defense Sheet

### What exactly did you build?

I built a custom top-down Formula 1 Monza racing AI system: image-derived track geometry, deterministic vehicle physics, a shared simulator, Gymnasium/SB3 PPO training/evaluation, Fast-F1 calibration, telemetry/replay, benchmark tools, and an evolutionary controller-search engine.

### Is this RL, evolution, or simulation?

It is a hybrid simulation and learning project. The environment and PPO infrastructure are RL. The current strongest solved behavior comes from evolutionary controller search. The project is best framed as custom RL environment engineering plus evolutionary optimization, with PPO transfer as the next step.

### Why did PPO struggle?

PPO struggled because full-lap racing has long-horizon credit assignment. Going fast and crashing later can be locally attractive. Correct braking and line setup require sacrificing immediate progress for future success. Historical PPO reached about `970.775m` but did not complete a valid normal-start lap.

### Why did evolution work better?

Evolution could evaluate whole driving controllers, preserve rare good trajectories, branch around them, and optimize explicit lap-time/validity metrics. It did not need the same dense differentiable reward path that PPO relied on. Once a rare full-lap family appeared, elitism and smart mutation could exploit it.

### How does the controller search work?

Each candidate is a genome. For controller genomes, the genome maps simulator features to continuous throttle, brake, and steering. A population is evaluated in the simulator, scored across profiles, ranked, and used to produce the next generation through elites, offspring, crossover, smart immigrants, and pure random immigrants.

### What are the genome features?

The fastest verified controller used `23` features: bias, speed, target speed, speed error, brake demand, future brake demand, target speed drop, brake gate proximity/distance, lookahead max, signed lateral error, heading error, yaw rate, curvature, target steer, last controls, segment progress, and four lookahead heading features.

### What are the mutation/crossover operators?

Controller mutations include masked Gaussian noise, full Gaussian noise, weight resets, and no-op preservation. Crossover combines parent genomes. The system logs mutation type, sigma, reset count, changed genes, parent, crossover partner, and genome distance from parent.

### How did scoring evolve?

It started with progress and clean exits. Then it added risk/distance pressure, open-distance scoring, frontier recovery/novelty, dynamic survival floors, pace pressure, valid-lap scoring, time-attack scoring, and lap-pace scoring. Metrics shifted from best distance to fastest valid lap, valid-lap count, average valid lap, and top-decile pace.

### What did telemetry reveal?

Telemetry showed early PPO and search policies often chose short-term progress over braking/setup. Replay exposed early deaths, slow crawlers, weird wiggle behavior, line instability, final-sector crashes, and later slow-but-valid finishers. Those observations drove scoring and selection changes.

### What produced the 89.983s result?

The fastest lap came from a controller genome in generation `49`, candidate `148`, source `smart_immigrant`, bucket `smart_current_elite`, with `controller_weight_reset` mutation. It completed a valid rolling-start lap in `89.983333s` with average trace speed `230.846kph` and max speed `315.868kph`.

### How close is it to real Fast-F1?

The reference lap is `79.662s`. The current evolved lap is `89.983s`, so it is `10.321s` slower. The evolved average trace speed is `230.846kph`; the reference mean speed is `259.914kph`. The evolved max is `315.868kph`; the reference max is `348kph`.

### Is the lap cold-start or rolling-start?

Rolling-start. The latest run starts at `start_progress_m=0` and `start_speed_kph=80`. Do not claim this is a `0kph` cold start.

### What is still unsolved?

- `<=80s` evolved lap is not solved yet.
- PPO has not completed a valid normal-start lap.
- PPO transfer from evolved states remains future work.
- The fast evolved family needs broader robustness and speed improvement.
- The simulator is simplified and Monza-specific.

### How would you transfer evolution into PPO?

Use elite evolved telemetry and state snapshots to create state libraries, train PPO from those starts, expand the curriculum outward, benchmark linked starts, and then test honest normal-start PPO. Evolution provides discovery; PPO should learn a reusable policy from those discovered states.

### Why is this more than a toy Gym environment?

Because the repo implements the full stack: track extraction, physics, observations, action modes, reward/scoring, PPO, benchmarks, Fast-F1 calibration, telemetry, replay, state libraries, evolutionary search, checkpoint/resume, lineage, compressed artifacts, and validation. The latest run produced `926` valid laps and a `89.983s` evolved lap with full trace artifacts.

## Strongest Final Narrative

The project started as a custom RL environment and grew into a full racing AI lab. PPO was implemented seriously, with curriculum, eval, metadata-safe checkpoints, TensorBoard, and telemetry, but it plateaued before full-lap behavior. Instead of pretending more PPO hours were progress, the project pivoted to Yosh-style search: spawn many controllers, evaluate them honestly, preserve elites, mutate/cross over, use segmented diagnostics, and keep full-lap probes as the scoreboard.

That pivot worked. The system moved from no lap, to `970m` PPO plateaus, to `2409m` open-distance breakthroughs, to `5223m` final-sector attempts, to slow valid evolved laps, and finally to a verified `89.983s` evolved rolling-start lap after `27650` counted evolutionary/search candidate evaluations across preserved artifacts. The result is not final, but it is a real technical achievement: a custom simulator, a custom learning environment, a custom evolutionary optimizer, and an auditable high-speed racing result against a real Fast-F1 target.

The next milestone is clear: use the same artifact-driven loop to reduce the evolved lap below `80s`, then transfer the discovered fast behavior into PPO.
