# RL Completion Plan

## Goal
Train an agent that visibly and measurably learns Monza over many attempts: it should fail early, survive longer, complete larger portions of the lap, eventually complete clean laps if possible, and improve lap time and driving quality.

This plan is the implementation contract for the next Codex Goal. The goal is not perfection, world-record pace, or a final polished research paper. The goal is a solid proof of concept that the simulator, benchmark harness, PPO training, curriculum spawning, telemetry, evaluation, and artifacts all work together and produce measurable learning evidence.

## Completion Standard
Do not mark the Goal complete merely because training ran. Completion must be evidence-backed.

The Goal is complete only if all of the following are true:

1. Phases 1-6 in this file are implemented pragmatically or explicitly documented as unnecessary with evidence.
2. `ruff`, `pyright`, and `pytest` pass.
3. CUDA hardware check passes on the RTX 4060 when `--require-gpu` is used.
4. CUDA-required training aborts if PyTorch resolves to CPU.
5. Scripted baseline still completes a valid slow lap.
6. Strict-enough lap validity is implemented and tested.
7. Benchmark harness exists and writes:
   - `summary.json`
   - `summary.csv`
   - `per_episode.jsonl`
   - selected full telemetry JSONL
8. PPO training writes:
   - run metadata
   - checkpoints
   - final model
   - best model or best checkpoint if available
   - TensorBoard logs
   - eval metrics
   - selected telemetry
9. Curriculum/segment spawning works and is validated by tests or smoke runs.
10. At least one PPO or curriculum-trained policy is benchmarked against random, scripted, and reference ghost.
11. Artifacts show at least one measurable learning signal:
   - improved average progress over random or early PPO baseline
   - improved best progress over early PPO baseline
   - increased checkpoints reached
   - improved segment completion rate
   - reduced crash/off-track rate
   - increased average reward
   - longer average survival time
   - telemetry showing more plausible throttle/steering/braking behavior after training
12. README and Documentation contain a clear, honest first results summary.

A clean full lap by PPO is a strong success, but it is not required for this proof-of-concept Goal. If the trained PPO policy does not complete a full lap, the Goal can still be complete only if benchmark artifacts show clear measurable learning progress.

## Current Baseline
- The active simulator, manual mode, replay, telemetry, Gymnasium env, PPO smoke training, Fast-F1 reference ghost, hardware checks, tests, linting, and type checks are implemented.
- Verified current metrics include `18` observation features, `9` discrete actions, `7` ray sensors, `120` checkpoints, `1,800` boundary segments, `60 Hz` physics/render timing, and Fast-F1 reference ghost replay at `79.662s`.
- The scripted baseline can complete a slow clean lap in no-telemetry mode.
- PPO training is currently smoke-level only. No trained RL policy has learned to complete Monza yet.

## Non-Goals For This Goal
Do not implement these unless strictly required for proof-of-concept completion:

- image observations
- offline RL
- imitation learning / supervised warm start
- continuous RL action space
- evolutionary search
- multi-track support
- distributed training
- replay video export
- full telemetry dashboard
- advanced race-control system
- world-record optimization

These are later phases after the basic PPO/curriculum loop proves learning.

## Operating Principles
- Keep the simulator as the single source of truth for manual, scripted, PPO, eval, benchmark, replay, and curriculum.
- Keep implementation pragmatic and focused on learning progress.
- Do not overengineer collision logic beyond what is needed for Monza training.
- Do not build new runtime paths for future features.
- Keep full per-step telemetry for manual/eval/replay/benchmark/selected training episodes.
- Keep PPO training telemetry lightweight by default so overnight runs are not slowed down by JSONL writes.
- Always produce machine-readable artifacts: JSON, JSONL, CSV, TensorBoard logs, checkpoints, and metadata.
- Final evaluation must be strict full-lap evaluation from normal start, even if training uses curriculum.
- Do not overclaim. If PPO improves but does not complete a lap, document that exactly.

## Compute And QC Rules
Compute placement is strict:

- CPU:
  - simulator stepping
  - physics
  - geometry
  - rendering
  - keyboard input
  - telemetry/logging
  - track preprocessing
  - vector env workers
- GPU:
  - PyTorch policy training
  - PyTorch policy inference when available

Quality-control rules:

- Any command run with `--require-gpu` must abort if `torch.cuda.is_available()` is false or if the resolved model device is not `cuda`.
- If a training command expected to use GPU resolves to CPU, stop, fix device resolution/configuration, and rerun the command.
- If `f1-hardware-check --json --require-gpu` fails, do not run long training.
- If `ruff`, `pyright`, or `pytest` fails, fix and rerun before moving to training.
- If benchmark artifacts are missing required files, fix benchmark output and rerun.
- If PPO training does not write checkpoints/metadata/eval metrics, fix training infrastructure and rerun.
- If curriculum smoke test fails, do not start a long curriculum run.
- If a long command is shortened due to wall-clock/GPU stability constraints, document it honestly and keep the proof-of-concept criteria intact.
- Update `Documentation.md` after meaningful implementation, validation, training, benchmark, or blocker events.

## Phase 1: Minimal Strict Lap Validity
Purpose: make lap completion trustworthy without building a full race-control system.

Current gap:
- `TrackSpec.finish_line` exists.
- Ordered checkpoints exist.
- `completed_lap` is still primarily monotonic distance based.
- Checkpoint index is mostly derived from monotonic progress.

Implementation:
- Keep monotonic centerline progress as the dense reward signal.
- Add explicit checkpoint/lap validity state:
  - `next_checkpoint_index`
  - `checkpoints_passed`
  - `missed_checkpoint_count`
  - `lap_valid`
  - `finish_crossed`
- Detect movement segment crossing checkpoint gates and finish line.
- Require ordered checkpoint crossing for valid lap completion.
- Require finish-line crossing near the end of the lap.
- Keep this Monza-focused and simple. Do not build advanced race steward logic.

Expected code files:
- `src/f1rl/sim.py`
- `src/f1rl/geometry.py`
- `src/f1rl/track_model.py` if the persisted gate/finish representation needs a small helper

Expected tests:
- `tests/test_lap_validity.py`
- expand `tests/test_sim.py`

Acceptance criteria:
- A car cannot complete a lap by only accumulating distance if it skipped required checkpoints.
- Crossing finish too early does not complete a valid lap.
- Scripted baseline still completes a valid lap.
- Reference ghost still completes a valid lap.
- PPO/env step contract remains valid.

Validation commands:
```powershell
uv run pytest -q tests/test_lap_validity.py tests/test_sim.py
uv run f1-scripted --steps 18000 --no-telemetry
uv run f1-reference-agent --mode ghost --no-telemetry
```

Artifacts:
- No large artifact required.
- If telemetry is enabled, valid-lap fields should appear in JSONL/summary.

## Phase 2: Minimal Collision Robustness
Purpose: make sure collision/off-track behavior is reliable enough for training without overengineering.

Current position:
- Collision/off-track mostly works from manual testing.
- Keep this lightweight unless training exposes a real bug.

Implementation:
- Verify off-track termination on Monza.
- Verify obvious boundary-hit termination on Monza.
- Keep boundary collision and drivable-mask off-track reporting distinct.
- Avoid broad synthetic geometry test matrices for now.

Expected code files:
- `src/f1rl/sim.py`
- `src/f1rl/geometry.py` only if a specific bug appears

Expected tests:
- expand `tests/test_sim.py`
- optionally add `tests/test_collision.py`

Acceptance criteria:
- Obvious boundary hit terminates.
- Off-track position terminates.
- Scripted baseline still completes clean lap.
- Reference ghost still replays cleanly.

Validation commands:
```powershell
uv run pytest -q tests/test_sim.py
uv run f1-scripted --steps 18000 --no-telemetry
uv run f1-replay artifacts/reference-ghost-20260424-064405/steps.jsonl --headless
```

## Phase 3: Lightweight Debug HUD
Purpose: expose enough live information to debug manual driving, ghost comparison, and failures without cluttering the screen.

Current position:
- HUD already shows speed, progress, checkpoint, lap, reason, and controls.
- Rays render visually.
- Full reward/ray details are not visible in HUD.

Implementation:
- Add a compact debug overlay, not a giant dashboard.
- Show:
  - sim time
  - speed
  - progress
  - progress delta
  - checkpoint/lap
  - reward total
  - key reward components
  - lateral error
  - heading error
  - min/center/max ray distance
  - ghost gap when reference overlay is active
  - termination reason
- Keep full detailed values in telemetry JSONL rather than trying to show every field on screen.

Expected code files:
- `src/f1rl/render.py`
- `src/f1rl/manual.py`

Expected tests:
- `tests/test_render_config.py`
- optional headless/manual smoke if useful

Validation commands:
```powershell
uv run pytest -q tests/test_render_config.py
uv run f1-manual --headless --ghost-reference --flying-start --max-steps 60
```

## Phase 4: Benchmark Harness
Purpose: create one canonical command that compares policies and produces machine-readable results.

This is mandatory before serious RL experiments. The benchmark harness is how we stop eyeballing videos and start reading numbers.

Policies to support:
- `random`
- `scripted`
- `ppo`
- `reference_ghost`
- optional `reference_control` diagnostic

Metrics to capture per episode:
- policy name
- seed
- episode index
- completed lap
- valid lap
- termination reason
- elapsed sim time
- lap time if completed
- total steps
- total reward
- reward component totals
- final progress
- best progress
- checkpoints passed
- checkpoint count
- crash/off-track/no-progress/max-step flags
- average speed
- max speed
- average and max lateral g
- minimum and average ray distance
- average racing-line deviation
- final ghost gap when reference is available
- wall-clock runtime
- simulator steps/sec

Telemetry policy:
- Benchmark always writes `summary.json`, `summary.csv`, and `per_episode.jsonl`.
- Benchmark may write full `steps.jsonl` for selected episodes only.
- CLI should support:
  - `--telemetry none`
  - `--telemetry selected`
  - `--telemetry all`
  - `--telemetry-every N`
  - `--save-best-replay`
  - `--save-failure-replay`
- Default should be `selected`, so we get representative step-by-step data without crushing throughput.

Expected artifact layout:
```text
artifacts/benchmark-YYYYMMDD-HHMMSS/
  config.json
  summary.json
  summary.csv
  per_episode.jsonl
  selected_telemetry/
    <policy>-episode-000-steps.jsonl
    <policy>-best-steps.jsonl
    <policy>-failure-steps.jsonl
```

Expected code files:
- new `src/f1rl/benchmark.py`
- possibly new `src/f1rl/runners.py`
- `src/f1rl/scripted.py`
- `src/f1rl/policy_io.py`
- `src/f1rl/reference_agent.py`
- `src/f1rl/telemetry.py`

Expected tests:
- `tests/test_benchmark.py`

Validation commands:
```powershell
uv run python -m f1rl.benchmark --policies random scripted reference_ghost --episodes 3 --max-steps 600 --telemetry selected
uv run python -m f1rl.benchmark --policies ppo --checkpoint latest --episodes 2 --max-steps 200 --device auto --telemetry selected
uv run pytest -q tests/test_benchmark.py
```

Acceptance criteria:
- Benchmark runs without opening a Pygame window.
- Benchmark produces JSON, CSV, and JSONL outputs.
- Benchmark records termination reasons and progress metrics.
- Benchmark can run PPO checkpoint eval if a checkpoint exists.
- Benchmark can save selected full telemetry for later replay/analysis.

## Phase 5: Real PPO Training Infrastructure
Purpose: upgrade PPO from smoke-test mode to actual experiment mode.

Current gap:
- PPO can train and save checkpoints.
- Current runs are short smoke validations.
- Training does not yet have robust eval callbacks, best-model tracking, benchmark-style metrics, or selected training telemetry.

Implementation:
- Keep Stable-Baselines3 PPO.
- Keep simulator/env workers on CPU.
- Keep PyTorch policy training/inference on CUDA when available.
- Add proper training artifacts:
  - run metadata
  - checkpoints
  - best model
  - final model
  - TensorBoard logs
  - periodic eval metrics
  - periodic selected rollout telemetry if enabled
- Use SB3 `Monitor`/`VecMonitor` for episode stats.
- Add periodic eval using a separate deterministic eval environment.
- Save best model by evaluation metric.
- Add CLI flags:
  - `--eval-every`
  - `--eval-episodes`
  - `--save-best`
  - `--telemetry none|selected|all`
  - `--telemetry-every`
  - `--run-name`
  - `--curriculum none|segments`
- Keep full per-step training telemetry off by default.

Expected artifact layout:
```text
artifacts/train-YYYYMMDD-HHMMSS/
  run_metadata.json
  checkpoints/
  tensorboard/
  eval/
    eval_metrics.jsonl
    best_eval_summary.json
    selected_telemetry/
  best_model.zip
  final_model.zip
```

Expected code files:
- `src/f1rl/train.py`
- `src/f1rl/eval.py`
- `src/f1rl/env.py`
- `src/f1rl/policy_io.py`
- possibly new `src/f1rl/callbacks.py`

Expected tests:
- update `tests/test_policy_train_smoke.py`
- optional `tests/test_eval.py`

Validation commands:
```powershell
uv run f1-train --timesteps 1024 --n-envs 1 --max-steps 120 --device auto --checkpoint-every 512 --eval-every 512 --eval-episodes 1 --telemetry selected
uv run f1-eval --checkpoint latest --steps 600 --device auto
uv run pytest -q tests/test_policy_train_smoke.py
```

Short full-lap baseline command:
```powershell
uv run f1-train --timesteps 250000 --n-envs 8 --max-steps 3600 --device auto --require-gpu --eval-every 25000 --eval-episodes 5 --telemetry selected
```

Acceptance criteria:
- Training still smoke-tests quickly.
- Training writes best/final/checkpoint artifacts.
- Eval metrics are visible without watching video.
- Selected telemetry exists for representative eval or training episodes.
- CUDA is used for PyTorch when `--require-gpu` is passed.

## Phase 6: Curriculum And Segment Spawning
Purpose: help PPO see and learn the entire track instead of crashing near the start for most of training.

Important distinction:
- Full-lap PPO uses one normal start and learns only from what it reaches.
- Curriculum PPO still trains one shared PPO policy, but changes reset/start conditions so the policy practices many parts of the track.
- The user's "spawn many agents and keep the best" idea is possible, but that is closer to evolutionary search or elite selection. For now, use vectorized PPO plus curriculum because it is simpler, standard, and more likely to learn efficiently.

Implementation:
- Add reset options:
  - `start_progress_m`
  - `start_checkpoint`
  - `start_speed_kph`
  - `segment_length_m`
  - `position_noise_m`
  - `heading_noise_deg`
  - `speed_noise_kph`
  - `curriculum_stage`
- Add simulator helpers:
  - spawn at centerline progress
  - get centerline tangent heading
  - optionally use Fast-F1 reference speed near spawn
  - mark segment target progress
- Add segment termination:
  - segment complete when target progress is reached
  - off-track/collision/no-progress still terminate
  - full-lap eval remains unchanged
- Add curriculum stage metadata to info/telemetry summaries.

Proposed curriculum stages:
- Stage A: short 250-300m segments, low speed, low noise.
- Stage B: 500-800m segments, moderate speed, low noise.
- Stage C: 1000-1500m segments, moderate/high speed, moderate noise.
- Stage D: random checkpoint starts, varied segment lengths.
- Stage E: full-lap flying-start practice.
- Stage F: full-lap normal-start training.

Promotion criteria:
- Prefer automatic promotion if metrics are available:
  - segment completion rate above `70-80%`
  - crash/off-track rate below threshold
  - average progress above threshold
- Also allow fixed timestep schedules for simplicity:
  - A for N timesteps
  - B for N timesteps
  - C for N timesteps
  - D/E/F for N timesteps

Expected code files:
- new `src/f1rl/curriculum.py`
- `src/f1rl/env.py`
- `src/f1rl/sim.py`
- `src/f1rl/train.py`
- `src/f1rl/telemetry.py`

Expected tests:
- `tests/test_curriculum.py`
- expand `tests/test_env.py`

Validation commands:
```powershell
uv run pytest -q tests/test_curriculum.py tests/test_env.py
uv run f1-train --timesteps 2048 --n-envs 2 --max-steps 600 --device auto --curriculum segments --eval-every 1024 --eval-episodes 1
uv run python -m f1rl.benchmark --policies ppo --checkpoint latest --episodes 3 --max-steps 1200 --device auto --telemetry selected
```

Acceptance criteria:
- Env can reset at specified progress/checkpoint.
- Segment episodes terminate correctly at target segment completion.
- Curriculum training smoke test runs.
- Full-lap eval still starts from normal start unless explicitly configured otherwise.

## Training Experiment Flow
Run this only after Phases 1-6 are implemented and smoke-tested.

### Step 1: Validate Everything
```powershell
uv run ruff check .
uv run pyright src/f1rl
uv run pytest -q
uv run f1-hardware-check --json --require-gpu
uv run f1-scripted --steps 18000 --no-telemetry
```

Expected result:
- Lint/type/tests pass.
- GPU is visible.
- Scripted baseline completes a slow valid lap.

QC:
- If hardware check does not report CUDA on the RTX 4060, stop and fix environment/device selection before training.
- If scripted baseline fails after implementation changes, fix simulator/lap validity before PPO.

### Step 2: Benchmark Before Training
```powershell
uv run python -m f1rl.benchmark --policies random scripted reference_ghost --episodes 20 --max-steps 3600 --telemetry selected
```

Expected result:
- Random baseline likely crashes/fails early.
- Scripted baseline completes slowly.
- Reference ghost completes at `79.662s`.
- Benchmark outputs summary JSON/CSV/JSONL plus selected telemetry.

What to inspect:
- random average progress
- scripted lap time
- termination reason distribution
- steps/sec
- selected failure telemetry

QC:
- If summary JSON/CSV/JSONL is missing, fix benchmark output and rerun.
- If selected telemetry is missing, fix telemetry sampling and rerun.
- If benchmark is too slow, reduce telemetry sampling before training.

### Step 3: Short Full-Lap PPO Baseline
```powershell
uv run f1-train --timesteps 250000 --n-envs 8 --max-steps 3600 --device auto --require-gpu --eval-every 25000 --eval-episodes 5 --telemetry selected
```

Purpose:
- Prove what full-lap PPO does before curriculum.
- Capture the failure mode.
- Avoid guessing whether curriculum is necessary.

What to inspect:
- average progress over evals
- termination reasons
- reward curve
- crash/off-track rate
- any improvement in first sector survival
- selected eval telemetry

QC:
- If training reports CPU with `--require-gpu`, abort, fix, and rerun.
- If no checkpoint/final model is written, fix training output and rerun.
- If no eval metrics are written, fix callbacks/eval runner and rerun.
- If selected telemetry is missing, fix sampling and rerun a short slice.

### Step 4: Benchmark Full-Lap PPO Baseline
```powershell
uv run python -m f1rl.benchmark --policies ppo --checkpoint latest --episodes 20 --max-steps 3600 --device auto --telemetry selected
```

Expected result:
- Probably not clean laps yet.
- Should produce hard evidence of where PPO fails or improves.

Decision point:
- If PPO already reaches far into the lap, continue full-lap training longer.
- If PPO repeatedly dies early, move to curriculum.

QC:
- If PPO checkpoint loading fails, fix checkpoint discovery/loading and rerun.
- If benchmark cannot compare PPO numerically against earlier artifacts, fix artifact metadata.

### Step 5: Mini Curriculum PPO Baseline
```powershell
uv run f1-train --timesteps 250000 --n-envs 8 --max-steps 1200 --device auto --require-gpu --curriculum segments --eval-every 25000 --eval-episodes 5 --telemetry selected
```

Purpose:
- Smoke-test the curriculum learning setup.
- Confirm the agent can learn short segments more efficiently than full-lap starts.

What to inspect:
- segment completion rate
- segment crash/off-track rate
- reward improvement by stage
- selected segment telemetry
- whether training throughput remains acceptable

QC:
- If curriculum does not actually vary spawn points, fix reset logic before continuing.
- If segment completion is not tracked, fix curriculum metrics.
- If the run is too slow, reduce selected telemetry frequency and rerun.

### Step 6: First Serious Curriculum PPO Run
```powershell
uv run f1-train --timesteps 1000000 --n-envs 8 --max-steps 3600 --device auto --require-gpu --curriculum segments --eval-every 50000 --eval-episodes 5 --telemetry selected
```

Purpose:
- Start the actual learning run.
- Review results with telemetry, benchmark summaries, and selected replays.

Follow-up benchmark:
```powershell
uv run python -m f1rl.benchmark --policies random scripted ppo reference_ghost --checkpoint latest --episodes 20 --max-steps 3600 --device auto --telemetry selected
```

QC:
- If full `1M` is infeasible due to wall-clock or GPU stability, run the largest feasible shorter run, label it clearly, and still require measurable learning evidence unless blocked.
- If there is no measurable learning signal, inspect telemetry and make the smallest defensible fix before rerunning a shorter slice.

### Step 7: First Results Pass
Do this before marking the Goal complete.

Required outputs:
- Update `README.md` with a first `Results` section.
- Update `Documentation.md` with commands, artifacts, metrics, failures, and final status.
- Include paths to key artifacts.
- Clearly separate:
  - implemented
  - partially implemented
  - not implemented
  - deferred

README results should include, when available:
- observation/action dimensions
- track/checkpoint/boundary metrics
- hardware and CUDA status
- training timesteps
- number of envs
- benchmark table for random/scripted/PPO/reference ghost
- completion rate
- crash/off-track rate
- average progress
- best progress
- reward statistics
- best PPO lap time if any PPO lap completes
- artifact paths

## Data And Telemetry Requirements
Training:
- Always log aggregate episode metrics.
- Always log TensorBoard scalars.
- Always log eval metrics.
- Do not write full per-step JSONL for every training env by default.
- Support selected telemetry sampling.
- Selected telemetry should be frequent enough to debug behavior but sparse enough to avoid training slowdown.

Eval/benchmark/replay:
- Full telemetry must be available.
- Include every existing `StepTelemetry` field:
  - time
  - position
  - heading
  - speed
  - yaw rate
  - acceleration
  - longitudinal/lateral g
  - curvature
  - throttle/brake/steering
  - control deltas
  - action id
  - raw and monotonic progress
  - progress delta
  - lateral error
  - racing-line deviation
  - heading error
  - reference progress/speed/gap when available
  - checkpoint/lap
  - all ray distances
  - collision/off-track/termination flags
  - reward total
  - every reward component

Episode summaries:
- Keep existing rich summary fields.
- Add lap-validity/curriculum fields when implemented.
- Add benchmark policy/run metadata.

Benchmark summaries:
- Include aggregate policy comparison.
- Include per-policy completion rate.
- Include per-policy crash/off-track/no-progress/max-step counts.
- Include per-policy average and best progress.
- Include per-policy average reward and reward components.
- Include simulator throughput.

## Learning-Failure Triage
If PPO fails to improve, do not blindly run longer training. Inspect benchmark summaries and selected telemetry.

Diagnose the likely cause:

- Reward design:
  - progress reward too weak/strong
  - penalties overwhelming progress
  - no segment completion signal
- Curriculum reset design:
  - starts too hard
  - starts too fast
  - too much heading/position noise
  - segment length too long
- Lap validity:
  - valid progress not credited correctly
  - checkpoint/finish logic blocking good behavior
- Collision/off-track behavior:
  - false positives
  - termination too sensitive
  - drivable mask issue
- Action space:
  - discrete controls too coarse
  - brake/steer combos insufficient
- PPO config:
  - learning rate
  - rollout length
  - batch size
  - entropy coefficient
  - reward normalization
- Observation quality:
  - ray distances not informative enough
  - missing curvature/lookahead information
  - progress/heading/lateral features wrong
- Training duration:
  - learning signal exists but run is too short

Make the smallest reasonable fix, rerun a short smoke/benchmark/training slice, and continue.

## Blocked Condition
Only mark the Goal blocked if all of the following are true:

- Phases 1-6 are implemented or the missing phase is impossible without user input.
- Validations pass or the remaining validation failure is the blocker.
- Multiple PPO/curriculum attempts produce no measurable learning signal.
- The next step requires a user decision, substantially longer training time, or a deeper redesign.

Blocked report must include:
- commands run
- artifacts produced
- metrics observed
- why learning failed or is inconclusive
- likely root causes
- exact next recommended experiment

## Later Phases
Save these until after this Goal unless needed for proof-of-concept:

- polished plots and analysis tooling
- replay video export
- advanced recruiter-facing README media
- imitation learning / supervised warm start
- continuous action RL
- evolutionary search
- advanced anti-cutting/race-control system
- multi-track support

## Definition Of Done For This Plan
- Phases 1-6 implemented pragmatically.
- Smoke tests pass for each phase.
- Benchmark harness produces useful JSON/CSV/JSONL outputs.
- PPO training has real experiment artifacts, eval callbacks, selected telemetry, best/final checkpoints, and TensorBoard logs.
- Curriculum reset/segment spawning works.
- Short full-lap PPO baseline has been run and benchmarked, unless a documented blocker prevents it.
- Mini curriculum baseline has been run and benchmarked, unless a documented blocker prevents it.
- First serious curriculum PPO run has been run at `1M` timesteps or the largest feasible shorter run is honestly documented.
- At least one measurable learning signal is shown, or the Goal is marked blocked with a complete blocked report.
- README and Documentation contain the first proof-of-concept results.
- No claim is made that the agent has mastered Monza until benchmark data proves it.
