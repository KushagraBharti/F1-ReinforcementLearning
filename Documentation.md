# Documentation

## Session Date
- 2026-04-08 to 2026-04-09

## Objective
- Push the repo from a working RL baseline to a GPU-enabled, benchmark-driven optimization workflow.
- Improve environment realism, observations, reward shaping, training profiles, and evaluation rigor.
- Keep only benchmark-positive changes and record the result.

## Milestone A - Baseline and GPU Runtime
### Findings
- `nvidia-smi` detected the RTX 4060 Laptop GPU successfully.
- The project environment initially used `torch 2.10.0+cpu`; CUDA was unavailable in Python.
- Existing smoke tests and validation passed on CPU before the runtime upgrade.

### Changes
- Added CUDA-aware Torch installation through `pyproject.toml` using the PyTorch CU128 index.
- Added `gputil` to the training dependency set.
- Added `f1rl.hardware_check` / `f1-hardware-check`.
- Updated PPO config builder and train entrypoint to support `--device auto|cpu|gpu` and `--require-gpu`.
- Added train profiles: `smoke`, `benchmark`, `performance`.

### Validation
1. `nvidia-smi` -> pass
2. `uv run python -c "import torch; print({'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'cuda_version': torch.version.cuda, 'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})"` -> pass (`2.10.0+cu128`, CUDA true)
3. `uv run python -m f1rl.hardware_check --json --skip-train-smoke` -> pass
4. `uv run python -m f1rl.train --mode smoke --iterations 1 --device gpu --require-gpu` -> pass

### Notes
- On this OneDrive-backed path, repeated `uv run ...` commands required `UV_LINK_MODE=copy` to avoid Windows hardlink failures.

## Milestone B - Benchmark and Champion Workflow
### Changes
- Added `f1rl.benchmark` / `f1-benchmark`.
- Added quick/standard benchmark profiles.
- Added summary metrics:
  - completion rate
  - collision rate
  - average progress ratio
  - average reward
  - average speed
  - average stall fraction
  - average steering oscillation
  - average/best lap time when available
- Added sampled GIF clip generation for early/mid/late windows.
- Added champion storage under `artifacts/champions/current.json`.
- Added promotion logic and `.pin` protection for promoted runs.

### Validation
1. `uv run pytest -q tests/test_benchmark.py tests/test_checkpoint_roundtrip.py` -> pass
2. `uv run python -m f1rl.benchmark --checkpoint latest --profile quick --no-clips` -> pass
3. `uv run python -m f1rl.benchmark --checkpoint latest --profile quick --promote-if-best --no-clips` -> pass, first champion established

### Observed Baseline
- Champion benchmark summary:
  - checkpoint: `artifacts/train-smoke-20260409-001939/checkpoints`
  - completion_rate: `0.000`
  - collision_rate: `1.000`

## Milestone C - Environment, Observation, and Reward Upgrades
### Changes
- Replaced the previous minimal heading/speed update with a more realistic kinematic model:
  - signed speed
  - steering angle response
  - wheelbase-based yaw update
  - coast deceleration and drag
  - high-speed steering reduction and grip term
- Expanded observations:
  - forward-biased sensor rays
  - signed speed
  - heading error to next checkpoint
  - goal distance
  - centerline offset
  - previous steering/throttle
- Added track centerline metadata and average track width.
- Reworked reward shaping:
  - progress reward
  - lap bonus
  - forward-speed reward gated by heading
  - alignment reward
  - centerline penalty
  - steering change penalty
  - stall/no-progress penalties
  - collision penalty
- Added no-progress termination and reward component logging in `info`.
- Updated the scripted controller to understand the richer observation layout.
- Kept old checkpoint compatibility by translating legacy env config keys on load.

### Validation
1. `uv run pytest -q tests/test_dynamics.py tests/test_reward_logic.py tests/test_controllers.py tests/test_env_api.py tests/test_geometry.py tests/test_spawn_heading.py` -> pass (`13 passed`)
2. `uv run python -m f1rl.manual --headless --controller scripted --max-steps 120` -> pass (`manual_run_complete ... reward=15.220`)
3. `uv run python -m f1rl.train --mode smoke --iterations 1 --device gpu --require-gpu` -> pass on upgraded env

## Milestone D - PPO Profiles, Eval, and Rollout Consistency
### Changes
- Added structured PPO profiles in `rllib_utils.py`.
- Added built-in evaluation metrics during training.
- Updated `eval.py` and `rollout.py` to reuse per-checkpoint training env config from `run_metadata.json`.
- Added optional env override input to `train.py` via `--env-config-json` and run tagging via `--run-tag`.

### Validation
1. `uv run pyright src/f1rl` -> pass
2. `uv run ruff check src/f1rl tests` -> pass
3. repo test suite -> pass (`19 passed`)

## Milestone E - Recursive Candidate Loop
### Changes
- Added `f1rl.optimize` / `f1-optimize`.
- Implemented a first candidate catalog for environment/reward/dynamics hypotheses.
- Candidate loop behavior:
  - ensure champion exists
  - train candidate
  - benchmark candidate
  - promote only if benchmark beats champion
  - record result in `optimize_manifest.json`

### Validation
1. `uv run python -m f1rl.optimize --max-candidates 1 --iterations 2 --device gpu --profile quick --no-clips` -> pass
2. Result: `kept=0 rejected=1`

### Additional Benchmark Check
1. `uv run python -m f1rl.train --mode benchmark --iterations 3 --device gpu --require-gpu` -> pass
2. `uv run python -m f1rl.benchmark --checkpoint artifacts/train-benchmark-20260409-004057/checkpoints --profile quick --promote-if-best --no-clips` -> pass, not promoted

### Interpretation
- The optimization loop is functioning correctly.
- First tested candidate hypothesis was rejected automatically.
- A stronger GPU benchmark-profile challenger also failed to beat the pinned champion.
- The repo now supports evidence-based keep/reject iteration instead of manual guesswork.

## Milestone F - RLlib Modernization and Artifact-First Inference
### Changes
- Migrated PPO config to explicit RLlib new-stack usage:
  - `api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)`
  - learner-centric batching via `train_batch_size_per_learner`
  - hybrid vectorized env-runner defaults per profile
- Added default parallelism profiles:
  - smoke: `1 env runner x 2 envs`
  - benchmark: `2 env runners x 4 envs`
  - performance: `4 env runners x 4 envs`
- Added train CLI overrides:
  - `--num-env-runners`
  - `--num-envs-per-env-runner`
  - `--vector-mode`
- Added lightweight inference artifact export under each training run:
  - saved RLModule weights
  - env config
  - model config
  - observation/action metadata
- Moved `eval`, `rollout`, and `benchmark` to artifact-backed inference by default.
- Added `--legacy-algorithm-restore` compatibility mode to `eval` and `rollout`.
- Tightened reverse-driving behavior:
  - braking now slows forward motion before entering reverse
  - added direct negative-speed penalty
  - added reverse-speed and reverse-progress termination
- Expanded benchmark metrics:
  - negative speed fraction
  - reverse event count
  - longest reverse streak
- Added champion floor logic so a negative-speed or zero-progress candidate cannot displace a valid forward-driving champion.

### Validation
1. `uv run ruff check src/f1rl` -> pass
2. `uv run pyright src/f1rl` -> pass
3. `uv run pytest -q tests/test_benchmark.py tests/test_reward_logic.py tests/test_dynamics.py tests/test_checkpoint_roundtrip.py` -> pass
4. `uv run python -m f1rl.train --mode smoke --iterations 1 --device cpu` -> pass
5. `uv run python -m f1rl.train --mode smoke --iterations 1 --device gpu --require-gpu` -> pass
6. `uv run python -m f1rl.eval --checkpoint latest --headless --steps 120` -> pass, `backend=artifact`
7. `uv run python -m f1rl.benchmark --checkpoint latest --profile quick --no-clips` -> pass, `backend=artifact`
8. `uv run python -m f1rl.benchmark --checkpoint latest --profile quick --promote-if-best --no-clips` -> pass, promoted new champion

### Post-Migration Behavior
- New promoted champion:
  - checkpoint: `artifacts/train-smoke-20260409-033050/checkpoints`
  - backend: `artifact`
  - completion_rate: `0.000`
  - collision_rate: `1.000`
  - avg_progress_ratio: `0.033`
  - avg_speed: `1.525`
  - avg_negative_speed_fraction: `0.000`
  - avg_longest_reverse_streak: `0.000`
- Interpretation:
  - the policy still crashes before finishing a lap
  - reverse-driving is no longer the dominant failure mode
  - forward progress and positive average speed are now established
  - the optimization loop now has a sane champion floor to build from

## Milestone G - Benchmark Startup Tuning on Windows
### Changes
- Tuned the benchmark profile to keep the hybrid vec+actors structure while reducing startup cost:
  - `2 env runners x 3 envs`
  - `rollout_fragment_length=96`
  - `train_batch_size_per_learner=576`
  - `minibatch_size=192`
  - `evaluation_num_env_runners=0` so training uses local evaluation instead of launching a separate remote evaluation runner group
- Added disk-backed track geometry caching under `artifacts/track-cache/` so new worker processes reuse preprocessed track data instead of rebuilding contours from the source image every time.
- Increased env-runner sample timeout to avoid false “no samples returned” failures during short Windows benchmark runs.

### Validation
1. `uv run ruff check src/f1rl tests` -> pass
2. `uv run pyright src/f1rl` -> pass
3. `uv run pytest -q tests/test_dynamics.py tests/test_reward_logic.py tests/test_checkpoint_roundtrip.py tests/test_benchmark.py` -> pass
4. `uv run python -m f1rl.train --mode benchmark --iterations 1 --device gpu --require-gpu` -> pass
5. `uv run python -m f1rl.benchmark --checkpoint artifacts/train-benchmark-20260409-040352/checkpoints --profile quick --no-clips` -> pass

### Observed Outcome
- Track cache file created:
  - `artifacts/track-cache/track-84745ee0614b1c44.npz`
- Benchmark-profile setup time improved materially on the validation runs:
  - earlier observed setup: about `32s`
  - tuned profile observed setup: about `14s`
- The benchmark profile is still heavier than smoke on Windows, but startup is now materially lower and the 1-iteration benchmark run completes reliably.

## Artifacts Produced
- Champion summary:
  - `artifacts/champions/current.json`
- Example benchmark summaries:
  - `artifacts/benchmark-quick-20260409-002605/benchmark_summary.json`
  - `artifacts/benchmark-quick-20260409-005458/benchmark_summary.json`
  - `artifacts/benchmark-quick-20260409-005627/benchmark_summary.json`
- Example sampled clips:
  - `artifacts/benchmark-quick-20260409-005627/renders/early.gif`
  - `artifacts/benchmark-quick-20260409-005627/renders/mid.gif`
  - `artifacts/benchmark-quick-20260409-005627/renders/late.gif`
- Example candidate manifest:
  - `artifacts/optimize-20260409-003338/optimize_manifest.json`
- Example stronger challenger checkpoint:
  - `artifacts/train-benchmark-20260409-004057/checkpoints`

## Final Validation
### Commands and Results
1. `uv run ruff check src/f1rl tests` -> pass
2. `uv run pyright src/f1rl` -> pass
3. `uv run pytest -q` -> pass (`17 passed`)
4. `uv run python -m f1rl.validate` -> pass

### Validate Highlights
- hardware check passed with CUDA-enabled Torch on the RTX 4060 Laptop GPU
- manual scripted smoke passed
- random rollout smoke passed
- GPU smoke training passed
- artifact-backed checkpoint eval passed
- artifact-backed quick benchmark generation passed
- final output: `[validate] all checks passed`

## Remaining Limits
1. RLlib still emits internal deprecation warnings from Ray 2.54.x internals during training, and RLModule construction still warns once during artifact-backed inference in this Ray build.
2. The smoke profile now learns forward-driving behavior, and benchmark-profile startup is materially better after tuning, but Windows Ray startup still carries noticeable overhead compared to smoke.
3. Benchmark performance is still poor in absolute driving terms; infrastructure is now in place, but more candidate waves are still needed to reach reliable lap completion.
4. `UV_LINK_MODE=copy` is needed on this OneDrive-backed path for repeatable `uv run` operations.
