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
- The scripted baseline completes a slow valid lap at about `214.47s`.
- Scratch PPO training is beyond smoke level now:
  - initial scratch PPO benchmark: `0.0m`
  - best trained PPO benchmark: `797.6m`
  - current best curriculum eval reached `100%` segment completion
  - no PPO clean or valid full lap yet.
- QC artifacts exist:
  - latest QC run: `artifacts/qc-20260602-130020`
  - lap-validity QC confirms huge progress jumps and wide checkpoint crossings invalidate laps
  - telemetry dashboard exists as lightweight local HTML, not a final recruiter-facing plotting package.

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
- final recruiter-facing telemetry dashboard
- advanced race-control system
- world-record optimization

These are later phases after the basic PPO/curriculum loop proves learning.

## Operating Principles
- Keep the simulator as the single source of truth for manual, scripted, PPO, eval, benchmark, replay, and curriculum.
- Keep implementation pragmatic and focused on learning progress.
- Do not overengineer collision logic beyond what is needed for Monza training.
- Do not build new runtime paths for future features.
- Treat every training idea as an experiment with inputs, artifacts, metrics, and a decision. Do not "just train" without producing comparable outputs.
- Prefer simple PPO improvements first: better reset distribution, better evaluation, better metrics, and better telemetry. Avoid algorithm sprawl until PPO/curriculum is clearly insufficient.
- Keep full per-step telemetry for manual/eval/replay/benchmark/selected training episodes.
- Keep PPO training telemetry lightweight by default so overnight runs are not slowed down by JSONL writes.
- Always produce machine-readable artifacts: JSON, JSONL, CSV, TensorBoard logs, checkpoints, and metadata.
- Final evaluation must be strict full-lap evaluation from normal start, even if training uses curriculum.
- Do not overclaim. If PPO improves but does not complete a lap, document that exactly.
- Preserve failed episodes. Failures are training/debugging signal, not noise. Benchmark outputs must make it clear whether the model fails by collision, off-track, no-progress timeout, max-step truncation, invalid checkpoint order, or simply poor progress.
- Keep the reference ghost as a calibration/evaluation baseline. Do not treat reference ghost replay as proof that the simulator has a physically controllable perfect agent.

## Compute And QC Rules
Default compute placement is strict:

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

This is the correct architecture for the current simulator because the environment is Python/NumPy/Pygame/control-flow heavy, while neural model updates are PyTorch operations. Do not move simulator stepping, raycasts, rendering, telemetry, or geometry to GPU in this pass.

Optimization caveat:
- Stable-Baselines3 PPO with `MlpPolicy` and small numeric observations may not use GPU efficiently.
- The current observation/action sizes are small, so the CPU environment can become the bottleneck while the GPU waits for rollouts.
- GPU is still required when `--require-gpu` is passed, but fastest wall-clock training must be determined by benchmark, not assumption.
- If CPU model training is faster for small MLP PPO, document the result honestly. Keep GPU-required mode available and working because future larger policies, image inputs, or heavier networks will benefit more from CUDA.
- GPU utilization is not the same thing as correct GPU placement. A correct run may show low GPU utilization if rollout collection is CPU-bound. The QC requirement is that PyTorch policy/model tensors resolve to CUDA when required, not that Task Manager shows constant high GPU usage.
- CPU-bound simulator bottlenecks should be optimized by reducing unnecessary telemetry, using the fastest stable vector env backend, and keeping rendering disabled during training. Do not port the simulator to GPU in this Goal.
- Long runs must record enough metadata to reconstruct whether the chosen configuration was CPU-model, CUDA-model, dummy vector env, or subprocess vector env.
- If a CUDA-required run fails because CUDA PyTorch is missing or broken, the correct response is to fix the environment before training, not silently downgrade to CPU.

Throughput optimization matrix:
- Before long training, benchmark short runs across:
  - model device: `cpu`
  - model device: `cuda`
  - vector env implementation: simple/Dummy vector env
  - vector env implementation: subprocess vector env
  - `n_envs`: at least `1`, `2`, `4`, `8`, and optionally logical-core count
- Capture:
  - wall-clock seconds
  - rollout/training FPS
  - env steps/sec
  - device used by model
  - vector env backend
  - `n_envs`
  - CPU/GPU policy metadata
  - whether selected telemetry was enabled
- Use the fastest proven configuration for long runs unless the command explicitly requires GPU.
- If `--require-gpu` is present, CUDA must be used even if a CPU-only benchmark is faster.

Suggested short throughput commands after phase 5 exists:
```powershell
uv run f1-train --timesteps 4096 --n-envs 8 --max-steps 600 --device cpu --vec-env dummy --telemetry none --run-name throughput-cpu-dummy
uv run f1-train --timesteps 4096 --n-envs 8 --max-steps 600 --device cuda --require-gpu --vec-env dummy --telemetry none --run-name throughput-cuda-dummy
uv run f1-train --timesteps 4096 --n-envs 8 --max-steps 600 --device cpu --vec-env subproc --telemetry none --run-name throughput-cpu-subproc
uv run f1-train --timesteps 4096 --n-envs 8 --max-steps 600 --device cuda --require-gpu --vec-env subproc --telemetry none --run-name throughput-cuda-subproc
```

Windows note:
- `SubprocVecEnv` can help if env stepping is the bottleneck, but Windows process spawning has overhead.
- Subprocess vector env code must be safe under Windows spawn semantics.
- If subprocess vector env is unstable or slower on short runs, keep dummy/simple vector env for that experiment and document the reason.

Quality-control rules:

- Any command run with `--require-gpu` must abort if `torch.cuda.is_available()` is false or if the resolved model device is not `cuda`.
- If a training command expected to use GPU resolves to CPU, stop, fix device resolution/configuration, and rerun the command.
- If `f1-hardware-check --json --require-gpu` fails, do not run long training.
- If throughput benchmarks show an unexpected device/vector-env result, document it in `Documentation.md` and use the fastest proven setup for non-`--require-gpu` runs.
- If CUDA runs are dramatically slower, still verify CUDA-required mode works, then use the best measured setup for optional exploratory runs.
- If `ruff`, `pyright`, or `pytest` fails, fix and rerun before moving to training.
- If benchmark artifacts are missing required files, fix benchmark output and rerun.
- If PPO training does not write checkpoints/metadata/eval metrics, fix training infrastructure and rerun.
- If curriculum smoke test fails, do not start a long curriculum run.
- If a long command is shortened due to wall-clock/GPU stability constraints, document it honestly and keep the proof-of-concept criteria intact.
- Update `Documentation.md` after meaningful implementation, validation, training, benchmark, or blocker events.

## Experimentation Policy
This section exists so Goal mode can test multiple reasonable ideas without turning the repo into a messy research sandbox.

### Scratch PPO Baseline Protocol
All PPO learning claims must use a fresh Stable-Baselines3 PPO initialization from the current codebase.

Required rules:
- Do not warm-start from reference ghost, scripted controller, previous PPO checkpoint, throughput checkpoint, smoke checkpoint, or any archived model when producing learning baselines.
- Every serious PPO run must save and evaluate `initial_model.zip` before training starts. This is the "knows nothing" PPO baseline for that run.
- Baseline tables must compare:
  - random policy
  - scripted controller
  - reference ghost replay
  - scratch PPO `initial_model.zip`
  - scratch PPO trained `final_model.zip`
  - scratch PPO trained `best_model.zip` when available
- Benchmark commands used for learning claims must pass explicit checkpoint paths from the intended run. Do not use `--checkpoint latest` for result tables, resume claims, or decisions about whether PPO is improving.
- `--checkpoint latest` is allowed only for local smoke checks where the output is labeled as infrastructure validation, not learning evidence.
- A run name must include `scratch` for serious full-lap and curriculum baselines unless there is a documented reason not to.
- If any benchmark accidentally evaluates the wrong checkpoint, discard that benchmark artifact, document the mistake in `Documentation.md`, and rerun with the exact checkpoint path.

Primary algorithms for this Goal:
- Full-lap PPO baseline.
- Curriculum/segment PPO.

Allowed diagnostics for this Goal:
- Random policy baseline.
- Scripted baseline.
- Fast-F1 reference ghost replay.
- Reference control-chase diagnostic if it already exists and is useful.
- Best-episode replay extraction from PPO runs.
- Failure replay extraction from PPO runs.
- Throughput experiments across device/vector-env choices.

Deferred algorithms:
- Evolutionary search.
- Elite-selection evolutionary training.
- Imitation learning from the reference ghost.
- Continuous-control PPO/SAC/TD3.
- Offline RL.

Important clarification about "100 agents, keep the best":
- Vectorized PPO already runs many environments in parallel. Those environments are the practical "many cars trying things" mechanism for this Goal.
- PPO does not literally keep only the best agents. It updates one shared policy using rollout data and advantage estimates from many successes and failures.
- Keeping only the few survivors from each batch is closer to evolutionary search, cross-entropy method, or elite imitation. That can be fun and useful later, but it is not the cleanest first path because it discards negative examples and creates a second training system.
- For this Goal, imitate the useful part of that idea by:
  - running many vectorized PPO environments
  - saving best and failure episode telemetry
  - benchmarking survival/progress distributions
  - using curriculum starts so the policy practices beyond the first crash point
  - promoting or adjusting curriculum based on measured segment completion/progress
- Do not implement a separate elite-selection algorithm unless PPO/curriculum infrastructure is complete and the user explicitly starts the later evolutionary phase.

Experiment branches to compare:
- Branch A: strict full-lap PPO from normal start.
- Branch B: curriculum PPO with segment spawning.
- Branch C: curriculum PPO followed by full-lap fine-tuning from normal start, only if Branch B shows learning signal.
- Branch D: smaller/faster CPU model training, only as a throughput comparison or fallback exploratory run.
- Branch E: CUDA-required training, used for final GPU-validated proof that the project can train on the RTX 4060.

Decision rules:
- If full-lap PPO improves progress meaningfully, continue full-lap PPO longer before adding more curriculum complexity.
- If full-lap PPO repeatedly crashes early with no average/best progress improvement, move to curriculum.
- If curriculum improves segment completion but full-lap eval remains poor, add a full-lap fine-tuning stage after curriculum.
- If curriculum segments are too easy, increase segment length, speed, or noise.
- If curriculum segments are too hard, shorten segments, lower spawn speed, reduce heading/position noise, or start from easier checkpoints.
- If training throughput is bad, disable unnecessary telemetry, reduce eval frequency, benchmark vector env choices, and tune `n_envs` before changing the learning algorithm.
- If no branch shows learning signal after several short runs, use selected telemetry to identify the smallest fix before running overnight.

Minimum comparable metrics for every branch:
- timesteps trained
- wall-clock training time
- device and vector env backend
- training FPS/env steps per second
- eval episode count
- completion rate
- valid lap rate
- average progress
- best progress
- checkpoints passed
- crash/off-track/no-progress/max-step rates
- average reward and reward components
- average survival time
- best selected replay path
- representative failure replay path

The benchmark harness must make these branch comparisons possible without manual video inspection.

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
- Make training reproducible enough for comparison:
  - record seed
  - record command-line args
  - record git commit or dirty-worktree status if available
  - record hardware/device metadata
  - record dependency versions if cheap to collect
  - record resolved artifact directory
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
- Best model selection should prefer learning evidence, not just raw reward if reward is unstable. Suggested ranking:
  1. valid full-lap completion rate
  2. average monotonic progress
  3. best progress
  4. checkpoints passed
  5. crash/off-track rate
  6. average reward
  7. lap time only after clean laps exist
- Add CLI flags:
  - `--eval-every`
  - `--eval-episodes`
  - `--save-best`
  - `--telemetry none|selected|all`
  - `--telemetry-every`
  - `--run-name`
  - `--curriculum none|segments`
  - `--vec-env dummy|subproc`
  - `--benchmark-throughput`
- Keep full per-step training telemetry off by default.
- Record the resolved device, vector env backend, `n_envs`, total wall-clock time, and measured training FPS/steps-per-second in `run_metadata.json`.
- Add a small throughput mode or metadata parser so CPU/CUDA and dummy/subproc comparisons are easy to read.

PPO settings to expose or document:
- `learning_rate`
- `n_steps`
- `batch_size`
- `n_epochs`
- `gamma`
- `gae_lambda`
- `ent_coef`
- `clip_range`
- `vf_coef`
- `max_grad_norm`
- `normalize_obs` if VecNormalize is added later
- `normalize_reward` if VecNormalize is added later

Default PPO stance:
- Start conservative and stable.
- Do not tune ten hyperparameters at once.
- First make sure reset logic, reward accounting, telemetry, and benchmark metrics are correct.
- If there is no learning signal, first inspect telemetry for simulator/reward bugs before changing PPO hyperparameters.
- Use entropy only enough to keep exploration alive; do not hide bad reset/reward design behind noisy policies.
- Keep `MlpPolicy` for v1 numeric observations.

Training telemetry policy:
- Always keep aggregate episode metrics.
- Always keep eval metrics.
- Always keep TensorBoard.
- Keep selected full telemetry for periodic deterministic eval episodes.
- Do not write per-step telemetry from every vector worker during long training unless intentionally debugging a short run.
- If selected telemetry slows training too much, increase `--telemetry-every` or temporarily use `--telemetry none`, but then run a separate selected-telemetry eval before judging behavior.

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
- Device/vector-env throughput metadata is recorded.
- CPU vs CUDA and dummy vs subproc short-run comparison can be executed before long runs.
- Training commands can be rerun with clear run names and produce separate artifact directories.
- Checkpoint discovery/loading is unambiguous enough for benchmark and eval commands.
- A failed/incomplete PPO run still leaves enough metadata to diagnose what happened.

## Phase 6: Curriculum And Segment Spawning
Purpose: help PPO see and learn the entire track instead of crashing near the start for most of training.

Important distinction:
- Full-lap PPO uses one normal start and learns only from what it reaches.
- Curriculum PPO still trains one shared PPO policy, but changes reset/start conditions so the policy practices many parts of the track.
- The user's "spawn many agents and keep the best" idea is possible, but that is closer to evolutionary search or elite selection. For now, use vectorized PPO plus curriculum because it is simpler, standard, and more likely to learn efficiently.
- Curriculum is not a shortcut around final full-lap eval. It is a way to teach useful behavior across the track before judging the policy from the real start.
- Curriculum success means segment completion and better full-lap progress. It does not count as "agent learned Monza" unless strict full-lap evaluation also improves.

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

Curriculum reset policy:
- Spawn exactly on or near the centerline.
- Use centerline tangent as base heading.
- Add small heading/position noise only after zero-noise segment starts work.
- Keep spawn speeds realistic but learnable.
- Prefer lower speeds in early stages so the agent can learn steering/off-track avoidance before high-speed braking.
- Later stages should include faster flying starts to expose braking/corner-entry behavior.
- Do not spawn inside walls or outside the drivable mask.
- Do not let segment completion bypass full-lap validity logic.

Curriculum reward policy:
- Keep the same global reward schema.
- Add segment-complete bonus only if needed and keep it explicitly tracked as a reward component.
- Do not reward raw speed directly.
- Avoid huge penalties that make early exploration all look equally bad.
- If the agent learns to stop or crawl, increase no-progress pressure or segment progress reward carefully.
- If the agent learns to launch off-track at high speed, reduce spawn speed/noise or increase collision penalty only after checking that the spawn is valid.

Proposed curriculum stages:
- Stage A: short 250-300m segments, low speed, low noise.
- Stage B: 500-800m segments, moderate speed, low noise.
- Stage C: 1000-1500m segments, moderate/high speed, moderate noise.
- Stage D: random checkpoint starts, varied segment lengths.
- Stage E: full-lap flying-start practice.
- Stage F: full-lap normal-start training.

Possible expanded stages to test if needed:
- Stage A0: stationary or very-low-speed centerline starts for 100-200m, used only if Stage A is too hard.
- Stage A1: first-straight and first-chicane practice.
- Stage B1: medium segments starting before heavy braking zones.
- Stage B2: medium segments starting before high-speed corners.
- Stage C1: sector-length segments.
- Stage D1: random checkpoint starts with reference-speed initialization.
- Stage E1: flying-start full lap with reduced spawn noise.
- Stage F1: normal-start full lap fine-tuning after curriculum.

Do not add all expanded stages upfront. Add them only if benchmark/telemetry shows the default stages are too coarse.

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

Promotion caveats:
- Fixed schedules are simpler and acceptable for the first proof of concept.
- Automatic promotion is better later, but only if metrics are reliable.
- Never promote based on reward alone.
- For early experiments, logging stage-level completion/progress is more important than building a perfect promotion manager.
- If automatic promotion adds too much complexity, implement staged reset sampling plus stage metrics first.

Stage-level metrics:
- episodes per stage
- segment completion rate per stage
- average progress per stage
- best progress per stage
- average reward per stage
- crash/off-track/no-progress rates per stage
- average speed per stage
- average/min ray distance per stage
- representative best/failure telemetry path per stage

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

### Step 2.5: Training Throughput Optimization Check
Run this after phase 5 exists and before serious PPO runs.

Purpose:
- Determine whether CPU or CUDA is faster for the current small `MlpPolicy`.
- Determine whether simple/Dummy vector env or subprocess vector env is faster on this Windows laptop.
- Keep GPU-required mode working even if CPU is faster for optional short runs.

Commands:
```powershell
uv run f1-train --timesteps 4096 --n-envs 8 --max-steps 600 --device cpu --vec-env dummy --telemetry none --run-name throughput-cpu-dummy
uv run f1-train --timesteps 4096 --n-envs 8 --max-steps 600 --device cuda --require-gpu --vec-env dummy --telemetry none --run-name throughput-cuda-dummy
uv run f1-train --timesteps 4096 --n-envs 8 --max-steps 600 --device cpu --vec-env subproc --telemetry none --run-name throughput-cpu-subproc
uv run f1-train --timesteps 4096 --n-envs 8 --max-steps 600 --device cuda --require-gpu --vec-env subproc --telemetry none --run-name throughput-cuda-subproc
```

Metrics to inspect:
- wall-clock seconds
- training FPS
- env steps/sec
- model device
- vector env backend
- process stability
- whether CUDA was actually used when required

QC:
- If any CUDA-required command resolves to CPU, abort and fix.
- If subprocess env fails under Windows spawn behavior, document it and continue with dummy vector env.
- If CPU is faster for MLP PPO, document the caveat. Do not remove GPU support.
- Use the fastest stable measured configuration for non-required-GPU training. For commands that include `--require-gpu`, use the fastest stable CUDA configuration.

### Step 3: Short Full-Lap PPO Baseline
```powershell
uv run f1-train --timesteps 250000 --n-envs 8 --max-steps 3600 --device auto --require-gpu --vec-env subproc --checkpoint-every 25000 --eval-every 25000 --eval-episodes 5 --telemetry selected --run-name ppo-full-lap-baseline-scratch
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
- Use the fastest stable CUDA configuration from Step 2.5 for this required-GPU run.
- Confirm the run wrote `initial_model.zip`, `final_model.zip`, and `run_metadata.json`.
- Confirm `eval/eval_metrics.jsonl` begins with a timestep-0 `initial_scratch` row.

### Step 4: Benchmark Full-Lap PPO Baseline
```powershell
uv run f1-benchmark --policies ppo --checkpoint artifacts\<ppo-full-lap-baseline-scratch-run>\initial_model.zip --episodes 20 --max-steps 3600 --device auto --telemetry selected
uv run f1-benchmark --policies ppo --checkpoint artifacts\<ppo-full-lap-baseline-scratch-run>\final_model.zip --episodes 20 --max-steps 3600 --device auto --telemetry selected
```

Expected result:
- Probably not clean laps yet.
- Should produce hard evidence of how the untrained PPO policy behaves and whether the trained-from-scratch full-lap policy improved.

Decision point:
- If PPO already reaches far into the lap, continue full-lap training longer.
- If PPO repeatedly dies early, move to curriculum.

QC:
- If PPO checkpoint loading fails, fix checkpoint discovery/loading and rerun.
- If benchmark cannot compare PPO numerically against earlier artifacts, fix artifact metadata.
- Do not substitute `latest` for the explicit scratch run checkpoint.

### Step 5: Mini Curriculum PPO Baseline
```powershell
uv run f1-train --timesteps 250000 --n-envs 8 --max-steps 1200 --device auto --require-gpu --vec-env subproc --curriculum segments --checkpoint-every 25000 --eval-every 25000 --eval-episodes 5 --telemetry selected --run-name ppo-curriculum-baseline-scratch
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
- Confirm the run wrote an untrained `initial_model.zip` and timestep-0 `initial_scratch` eval before training.

### Step 6: First Serious Curriculum PPO Run
```powershell
uv run f1-train --timesteps 1000000 --n-envs 8 --max-steps 3600 --device auto --require-gpu --vec-env subproc --curriculum segments --checkpoint-every 50000 --eval-every 50000 --eval-episodes 5 --telemetry selected --run-name ppo-curriculum-serious-scratch
```

Purpose:
- Start the actual learning run.
- Review results with telemetry, benchmark summaries, and selected replays.

Follow-up benchmark:
```powershell
uv run f1-benchmark --policies random scripted reference_ghost --episodes 20 --max-steps 18000 --device auto --telemetry selected
uv run f1-benchmark --policies ppo --checkpoint artifacts\<ppo-curriculum-serious-scratch-run>\initial_model.zip --episodes 20 --max-steps 3600 --device auto --telemetry selected
uv run f1-benchmark --policies ppo --checkpoint artifacts\<ppo-curriculum-serious-scratch-run>\final_model.zip --episodes 20 --max-steps 3600 --device auto --telemetry selected
uv run f1-benchmark --policies ppo --checkpoint artifacts\<ppo-curriculum-serious-scratch-run>\best_model.zip --episodes 20 --max-steps 3600 --device auto --telemetry selected
```

QC:
- If full `1M` is infeasible due to wall-clock or GPU stability, run the largest feasible shorter run, label it clearly, and still require measurable learning evidence unless blocked.
- If there is no measurable learning signal, inspect telemetry and make the smallest defensible fix before rerunning a shorter slice.

### Step 6.5: Compare Branches And Choose Next Experiment
Do this before deciding whether to run overnight or mark the proof of concept complete.

Required comparison:
- random baseline
- scripted baseline
- reference ghost
- short full-lap PPO baseline
- mini curriculum PPO baseline
- first serious curriculum PPO run

Comparison table columns:
- policy/run name
- training timesteps
- curriculum mode
- device
- vector env backend
- wall-clock training time
- eval episodes
- valid completion rate
- average progress
- best progress
- average checkpoints passed
- average reward
- crash/off-track rate
- no-progress rate
- max-step truncation rate
- average survival time
- best artifact path
- representative failure artifact path

Decision branches:
- If full-lap PPO is already improving strongly:
  - run a longer full-lap PPO continuation
  - keep curriculum as optional fallback
- If curriculum PPO improves segment completion and full-lap progress:
  - continue curriculum longer
  - then run full-lap normal-start fine-tuning
- If curriculum PPO learns segments but fails badly from normal start:
  - add more Stage E/F full-lap practice
  - reduce segment spawn bias over time
  - benchmark again from strict normal start
- If both full-lap and curriculum PPO show no progress:
  - inspect selected telemetry first
  - check observation scaling and ray distances
  - check reward component magnitudes
  - check no-progress timeout behavior
  - check collision false positives
  - run one smaller fix at a time
- If throughput is the blocker:
  - reduce per-step training telemetry
  - reduce eval frequency
  - use fastest measured vector env backend
  - tune `n_envs`
  - keep rendering off
  - do not change the algorithm just because training is slow

Acceptable proof-of-concept outcomes:
- Best outcome: PPO completes at least one strict valid lap.
- Strong outcome: PPO does not complete a lap but clearly improves average/best progress, checkpoints reached, survival time, reward, and crash rate against random/early PPO.
- Minimum acceptable outcome: PPO/curriculum shows a measurable learning signal on segments, selected telemetry proves it is learning sensible steering/throttle behavior, and the full-lap blocker is clearly diagnosed.

Not acceptable:
- Training ran but no benchmark artifacts exist.
- Reward improved while progress/survival got worse and no explanation exists.
- Curriculum segment metrics improved but strict full-lap eval was never run.
- CUDA-required mode silently used CPU.
- The final report relies on eyeballing video without JSON/CSV/JSONL metrics.

### Step 6.6: Optional Overnight Scale Run
Run this only after the branch comparison shows a real learning signal and the throughput configuration is known.

Example commands:
```powershell
uv run f1-train --timesteps 3000000 --n-envs 8 --max-steps 3600 --device auto --require-gpu --curriculum segments --eval-every 100000 --eval-episodes 5 --telemetry selected --run-name overnight-curriculum
uv run f1-train --timesteps 1000000 --n-envs 8 --max-steps 3600 --device auto --require-gpu --curriculum none --eval-every 50000 --eval-episodes 5 --telemetry selected --run-name full-lap-finetune
```

Overnight run rules:
- Do not start overnight training before short runs prove infrastructure and learning signal.
- Use the fastest stable CUDA-required configuration from Step 2.5.
- Keep selected telemetry sparse enough that training speed is not crushed.
- Save checkpoints frequently enough that partial progress is usable if the run is interrupted.
- On resume, benchmark the latest, best, and final checkpoints if they differ.
- If the overnight run crashes, preserve logs and metadata before rerunning.

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

Recruiter-facing artifacts to generate after the proof-of-concept run, if time permits:
- progress-over-training plot
- reward-over-training plot
- completion-rate or segment-completion-rate plot
- crash/off-track-rate plot
- benchmark comparison table
- best PPO replay clip or replay command
- representative failure replay clip or replay command
- telemetry snippet showing speed/progress/steering/ray/reward behavior

Media caveat:
- Plots and videos are high-value for recruiting, but they are downstream of the RL proof of concept.
- Do not spend hours polishing media before the benchmark/training numbers exist.
- If video export is not implemented yet, provide replay commands and artifact paths instead.
- If plots are not implemented yet, provide CSV/JSON metrics and document exactly which plot script should be added next.

Strong README result shape:
```text
Policy              Completion   Avg progress   Best progress   Crash/off-track   Avg reward   Notes
random              ...
scripted            ...
reference_ghost     ...
ppo_full_lap         ...
ppo_curriculum       ...
```

Resume-grade evidence requires:
- measured training timesteps
- measured simulator/training throughput
- clear baseline comparison
- honest PPO performance
- artifact-backed replay/telemetry
- no invented lap-completion claim

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
- Training throughput has been benchmarked across CPU/CUDA and dummy/subproc vector env options, or an option was skipped with a documented reason.
- Required-GPU training mode is verified to abort on CPU fallback and run on CUDA when available.
- Curriculum reset/segment spawning works.
- Short full-lap PPO baseline has been run and benchmarked, unless a documented blocker prevents it.
- Mini curriculum baseline has been run and benchmarked, unless a documented blocker prevents it.
- First serious curriculum PPO run has been run at `1M` timesteps or the largest feasible shorter run is honestly documented.
- At least one measurable learning signal is shown, or the Goal is marked blocked with a complete blocked report.
- README and Documentation contain the first proof-of-concept results.
- No claim is made that the agent has mastered Monza until benchmark data proves it.
