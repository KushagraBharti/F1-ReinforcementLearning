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

## Milestone H - Benchmark Clip Reliability and Checkpoint Progress Metrics
### Changes
- Fixed benchmark clip export to derive `early`, `mid`, and `late` windows from the actual first-episode frame stream instead of assuming the policy survives to the full benchmark horizon.
- Tightened clip-path reporting so only GIFs that were actually written are listed in `benchmark_summary.json`.
- Added explicit checkpoint-progress metrics to benchmark summaries:
  - `first_checkpoint_rate`
  - `three_checkpoint_rate`
  - `avg_checkpoints_reached`
  - `max_checkpoints_reached`

### Validation
1. `uv run pytest -q tests/test_benchmark.py` -> pass (`6 passed`)
2. `uv run python -m f1rl.train --mode smoke --iterations 2 --device auto` -> pass
3. `uv run python -m f1rl.benchmark --checkpoint latest --profile quick` -> pass

### Observed Outcome
- Latest benchmark with the 2-iteration smoke checkpoint:
  - checkpoint: `artifacts/train-smoke-20260409-043338/checkpoints`
  - summary: `artifacts/benchmark-quick-20260409-043556/benchmark_summary.json`
  - all three GIFs were written:
    - `early.gif`
    - `mid.gif`
    - `late.gif`
- Current policy quality from that run:
  - completion_rate: `0.000`
  - collision_rate: `1.000`
  - avg_checkpoints_reached: `3.0`
  - max_checkpoints_reached: `3`
  - avg_speed: `1.782`
- Interpretation:
  - clip export is now reliable for short-lived policies
  - the 2-iteration policy still fails early, and it slightly regressed on checkpoint reach vs the previous promoted smoke champion

## Milestone I - Swarm Diagnostics and Autonomous Campaign Scaffolding
### Changes
- Added stage/seed/promotion policy helpers in `src/f1rl/stages.py`:
  - stage configs for `competence`, `lap`, `stability`, `performance`
  - fixed promotion seeds
  - rotating diagnostic seed pools
  - periodic holdout seed pools
  - stage-gate helpers and stage-specific aggregate comparison helpers
- Added stage-aware environment termination and curriculum-ready spawn controls:
  - checkpoint deadlines
  - heading-away streak limits
  - wall-hugging streak limits
  - optional goal-range spawn curriculum
  - spawn position / heading jitter knobs
- Extended training CLI and PPO builder:
  - `--swarm-stage`
  - `--logical-cars`
  - `--ppo-config-json`
  - logical-car aware env-runner sizing
  - rollout fragment length derived from actual swarm geometry
  - entropy/value-loss/KL/sample-throughput fields in metrics
- Extended benchmark summaries:
  - seed tier and seed list recording
  - clean `100-step` / `300-step` survival
  - checkpoint streak metrics
  - failure histogram
  - wall-margin, heading-error, and steering-saturation diagnostics
- Added `f1rl.swarm` / `f1-swarm`:
  - many independent cars on one track
  - no car-car collisions
  - checkpoint/scripted/random policy support
  - early/mid/late clip export
  - top-k furthest traces
  - per-car swarm summary JSON
- Added `f1rl.campaign` / `f1-campaign`:
  - bounded stage-aware candidate waves
  - one family per wave
  - keep/reject manifests
  - fixed promotion seeds + rotating diagnostics + holdout checks
  - single-car benchmark remains the champion authority
  - immediate manifest persistence at startup
  - per-candidate training throughput extraction from `metrics.csv`
- Added tests for:
  - stage seed rotation / stage gates
  - campaign candidate library and diagnostic collapse guard
  - random swarm CLI smoke

### Validation
1. `uv run ruff check src/f1rl tests` -> pass
2. `uv run pyright src/f1rl` -> pass
3. `uv run pytest -q tests/test_benchmark.py tests/test_stages.py tests/test_campaign.py tests/test_cli_smoke.py` -> pass
4. `uv run python -m f1rl.swarm --policy random --headless --cars 8 --steps 40` -> pass
5. `uv run python -m f1rl.benchmark --checkpoint latest --profile quick --seed-tier promotion --rotation-index 1 --no-clips` -> pass

### Observed Outcome
- Example swarm artifact:
  - `artifacts/swarm-competence-20260409-060715/swarm_summary.json`
- Example seed-tiered benchmark artifact:
  - `artifacts/benchmark-quick-20260409-060841/benchmark_summary.json`
- Stage-aware `32 logical cars` smoke training now passes config validation and writes stage-aware run metadata under:
  - `artifacts/train-smoke-20260409-061011/run_metadata.json`
- Short bounded campaign startup now records manifests without the old subprocess-based git hash failure.

### Remaining Runtime Limits
- Campaign-scale training on Windows is still materially heavier than the old smoke loop.
- A short bounded campaign smoke can spend most of its wall-clock budget inside RLlib startup/training, so it is not a representative throughput benchmark for overnight runs.
- The orchestration path is implemented and unit/focused-smoke validated, but overnight throughput tuning is still an open optimization problem.

## Milestone J - Windows Ray Import Recovery and End-to-End Campaign Smoke
### Changes
- Added `src/f1rl/ray_compat.py` to patch Ray's Windows import path away from the WMI-backed `platform._wmi_query()` call that was hanging before any repo code executed.
- Applied the Ray import compatibility shim before all repo-controlled `ray` / `ray.rllib` imports:
  - `src/f1rl/train.py`
  - `src/f1rl/rllib_utils.py`
  - `src/f1rl/inference.py`
  - `src/f1rl/eval.py`
  - `src/f1rl/rollout.py`
- Fixed campaign checkpoint parsing in `src/f1rl/campaign.py` so Windows paths containing spaces are parsed correctly from `training_complete ... checkpoint=... use_gpu=...` output.
- Added budget-aware swarm step sizing in `src/f1rl/campaign.py` so tiny bounded campaigns do not spend full stage-length rollouts on swarm diagnostics.
- Recovered the local runtime state by clearing stale Python/Ray processes after the host accumulated more than 100 Python processes and Ray imports started hanging again.

### Root Cause
- The apparent training/campaign stall was not caused by PPO logic.
- `import ray` was hanging inside:
  - `platform._wmi_query()`
  - called from `platform.system()` / `platform.uname()`
  - triggered by `ray.__init__._configure_system()`
- On this Windows host, that WMI path became unreliable under process/page-file pressure.
- The repo now avoids that import-time WMI dependency.

### Validation
1. Direct temp-script check of `import ray` with the compatibility shim -> pass
2. Direct temp-script check of `import f1rl.train` -> pass
3. Direct compile check:
   - `.venv\Scripts\python.exe` + `py_compile` for:
     - `src/f1rl/ray_compat.py`
     - `src/f1rl/train.py`
     - `src/f1rl/rllib_utils.py`
     - `src/f1rl/inference.py`
     - `src/f1rl/eval.py`
     - `src/f1rl/rollout.py`
     - `src/f1rl/campaign.py`
   -> pass
4. Real train smoke:
   - `.venv\Scripts\python.exe -m f1rl.train --mode smoke --iterations 1 --device cpu --swarm-stage competence --logical-cars 8 --run-tag live-validate`
   -> pass
5. Direct checkpoint-backed swarm smoke:
   - `.venv\Scripts\python.exe -m f1rl.swarm --policy checkpoint --checkpoint latest --cars 8 --steps 20 --headless`
   -> pass
6. Focused tests:
   - `.venv\Scripts\python.exe -m pytest -q tests/test_stages.py tests/test_campaign.py tests/test_benchmark.py`
   -> pass (`11 passed`)
   - `.venv\Scripts\python.exe -m pytest -q tests/test_cli_smoke.py`
   -> pass
7. Real bounded autonomous campaign smoke:
   - `.venv\Scripts\python.exe -m f1rl.campaign --hours 0.005 --max-waves 1 --max-candidates-per-wave 1 --start-cars 4 --max-cars 4 --device cpu`
   -> pass

### Observed Outcome
- The bounded campaign now completes a full wave and writes a populated manifest:
  - `artifacts/campaign-20260409-070649/campaign_manifest.json`
- That run:
  - trained `reward_progress_bias`
  - produced promotion and diagnostic benchmarks
  - produced a swarm diagnostic
  - updated the incumbent checkpoint inside the manifest
- Example artifacts from the completed bounded campaign:
  - train candidate:
    - `artifacts/train-smoke-reward_progress_bias-20260409-070653/checkpoints`
  - promotion benchmark:
    - `artifacts/benchmark-quick-20260409-070727/benchmark_summary.json`
  - diagnostic benchmark:
    - `artifacts/benchmark-quick-20260409-070753/benchmark_summary.json`
  - swarm diagnostic:
    - `artifacts/swarm-competence-20260409-070819/swarm_summary.json`

### Practical Notes
- Direct `.venv\Scripts\python.exe` invocation is currently more reliable than repeated `uv run` wrappers for heavy RLlib loops on this Windows host.
- Tiny campaign budgets are now usable as smoke tests, but full overnight improvement still needs longer real runs to produce meaningful RL progress.

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
- Example swarm summary:
  - `artifacts/swarm-competence-20260409-060715/swarm_summary.json`
- Example candidate manifest:
  - `artifacts/optimize-20260409-003338/optimize_manifest.json`
- Example campaign manifests:
  - `artifacts/campaign-20260409-060715/campaign_manifest.json`
  - `artifacts/campaign-20260409-061006/campaign_manifest.json`
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
5. Large logical-car RLlib runs and short bounded campaign smokes are still expensive on Windows; the orchestration is implemented, but throughput tuning remains ongoing.
6. Ray import on this Windows host needed an explicit compatibility shim because the default import path could hang inside WMI-backed platform detection.

## Milestone K - Competence Search Exhaustion and Hybrid Warm-Start Probe (2026-04-09)

### Swarm checkpoint-config fix
- Fixed the remaining observation-geometry mismatch in the swarm diagnostic path.
- Added `training_env_config_for_checkpoint()` in `src/f1rl/inference.py`.
- `src/f1rl/benchmark.py` and `src/f1rl/swarm.py` now merge sparse eval/swarm overrides on top of the checkpoint training env config instead of silently rebuilding default 9-sensor envs.
- Added regression coverage:
  - `tests/test_benchmark.py::test_swarm_env_config_does_not_overwrite_training_sensor_count`
- Validation:
  - `.venv\Scripts\python.exe -m pytest -q tests/test_benchmark.py tests/test_campaign.py tests/test_stages.py tests/test_cli_smoke.py`
  - observation-family campaign that previously crashed now completes.

### Competence family search results
Using incumbent checkpoint:
- `artifacts/train-smoke-reward_progress_bias-20260409-065951/checkpoints`

Completed family runs:
- observations:
  - manifest: `artifacts/campaign-20260409-074455/campaign_manifest.json`
  - result: all 3 candidates rejected, family advanced to `action_control`
- action_control:
  - manifest: `artifacts/campaign-20260409-074859/campaign_manifest.json`
  - result: all 3 candidates rejected, family advanced to `curriculum`
- curriculum:
  - manifest: `artifacts/campaign-20260409-075234/campaign_manifest.json`
  - result: all 3 candidates rejected, family advanced to `ppo`
- ppo:
  - manifest: `artifacts/campaign-20260409-075607/campaign_manifest.json`
  - result: all 3 candidates rejected, family advanced to `performance`
- performance:
  - manifest: `artifacts/campaign-20260409-075948/campaign_manifest.json`
  - result: all 3 candidates rejected, family advanced back to `reward`

This completes one full competence-family rotation with no better incumbent than `reward_progress_bias`.

### Deeper pure-PPO probes
1. Higher logical-car budget:
- train: `artifacts/train-smoke-reward_progress_bias_deep-20260409-080519/checkpoints`
- benchmark: `artifacts/benchmark-quick-20260409-080624/benchmark_summary.json`
- aggregate:
  - `collision_rate = 0.0`
  - `avg_checkpoints_reached = 2.0`
  - `first_checkpoint_rate = 1.0`
- result: safer but far less progress than incumbent (`avg_checkpoints_reached = 21.0`), rejected.

2. Same competence geometry, deeper budget:
- train: `artifacts/train-smoke-reward_progress_bias_deeper4-20260409-080713/checkpoints`
- benchmark: `artifacts/benchmark-quick-20260409-080853/benchmark_summary.json`
- aggregate:
  - `collision_rate = 1.0`
  - `avg_checkpoints_reached = 2.0`
- result: regressed versus incumbent, rejected.

### Hybrid scripted warm-start implementation
- Added `scripted_warm_start_steps` to `EnvConfig` in `src/f1rl/config.py`.
- `F1RaceEnv.reset()` can now execute scripted-controller bootstrap steps, preserve the warmed physical state/goal index, then zero episode counters before returning the training start state.
- Added validation test:
  - `tests/test_env_reset.py::test_scripted_warm_start_resets_episode_counters_after_bootstrap`
- Validation:
  - `.venv\Scripts\python.exe -m pytest -q tests/test_env_reset.py tests/test_benchmark.py tests/test_campaign.py tests/test_stages.py tests/test_cli_smoke.py`

### Hybrid warm-start probes
1. Warm-start 20 steps:
- train: `artifacts/train-smoke-reward_progress_warm20-20260409-081308/checkpoints`
- benchmark: `artifacts/benchmark-quick-20260409-081529/benchmark_summary.json`
- aggregate:
  - `collision_rate = 1.0`
  - `avg_checkpoints_reached = 2.0`
- result: no transfer benefit on full benchmark, rejected.

2. Warm-start 40 steps:
- train: `artifacts/train-smoke-reward_progress_warm40-20260409-081623/checkpoints`
- benchmark: `artifacts/benchmark-quick-20260409-081928/benchmark_summary.json`
- aggregate:
  - `collision_rate = 1.0`
  - `avg_checkpoints_reached = 5.0`
  - `first_checkpoint_rate = 1.0`
  - `three_checkpoint_rate = 1.0`
- result: better than warm20, but still materially worse than incumbent `reward_progress_bias` (`avg_checkpoints_reached = 21.0`), rejected.

### Current best competence incumbent
- checkpoint: `artifacts/train-smoke-reward_progress_bias-20260409-065951/checkpoints`
- key quick-benchmark aggregate:
  - `completion_rate = 0.0`
  - `collision_rate = 1.0`
  - `first_checkpoint_rate = 1.0`
  - `three_checkpoint_rate = 1.0`
  - `avg_checkpoints_reached = 21.0`
  - `avg_progress_ratio = 0.175`

### Working conclusion
- The repo/runtime is stable enough for repeated autonomous campaigns.
- The current competence program appears plateaued under:
  - one full candidate-family rotation,
  - deeper pure-PPO budget probes,
  - first scripted warm-start hybrid probes.
- The next meaningful improvement path is no longer another small reward/control tweak. It likely requires either:
  - a broader hybrid/bootstrap strategy, or
  - a materially different training regime / campaign budget.

## Milestone L - Scripted Oracle Probe and Reversion (2026-04-09)

### Direct oracle check
- Re-measured the existing scripted controller directly on the real environment.
- Single-seed probe (`seed=101`, `max_steps=500`) result:
  - `positive_progress_total = 17`
  - `step_count = 462`
  - `terminated_reason = no_progress`
  - `reward = 50.22546616217471`
- This is a useful oracle, but it still does not clearly exceed the current PPO competence incumbent (`avg_checkpoints_reached = 21.0` on quick benchmark).

### Stronger oracle experiment attempted
- Tried a more centerline-dominant / turn-aware scripted-controller variant intended to act as a stronger oracle for hybrid bootstrapping.
- Result:
  - manual scripted smoke regressed from `462` steps to `411` steps on the same headless manual run.
  - direct seed-101 probe still only reached `17` checkpoints before `no_progress`.
- Decision:
  - reverted the controller changes immediately.
  - repo is restored to the previous scripted-controller baseline.

### Post-reversion validation
- `.venv\Scripts\python.exe -m pytest -q tests/test_env_reset.py tests/test_cli_smoke.py` -> pass
- `.venv\Scripts\python.exe -m f1rl.manual --headless --controller scripted --max-steps 1000` -> `manual_run_complete steps=462 laps=0 reward=50.225`
- direct seed-101 scripted probe restored to:
  - `17 462 no_progress 50.22546616217471`

### Updated conclusion
- The current competence search is now exhausted across:
  - one full PPO candidate-family rotation,
  - deeper pure-PPO budget probes,
  - hybrid scripted warm-start reset probes,
  - a direct stronger-oracle controller attempt that regressed and was reverted.
- No unresolved regression remains in the repository from this branch of work.

## Milestone M - Behavior Cloning and PPO Warm-Start Probe (2026-04-09)

### What was added
- Added a new imitation path in `src/f1rl/imitation.py`.
- The new path now does two things:
  - trains a standalone behavior-cloned Torch policy from scripted trajectories,
  - trains and exports a PPO-compatible RLModule warm-start artifact using the current RLlib module stack.
- Added regression coverage in `tests/test_imitation.py`.
- Added PPO warm-start support to training:
  - `src/f1rl/rllib_utils.py` now accepts `rl_module_load_state_path` and builds an `RLModuleSpec(..., load_state_path=...)`.
  - `src/f1rl/train.py` now exposes `--rlmodule-load-state-path`.

### Validation
- `.venv\Scripts\python.exe -m pytest -q tests/test_imitation.py tests/test_env_reset.py tests/test_cli_smoke.py` -> pass
- Confirmed `forward_inference()` on RLlib PPO modules is detached, so RLModule behavior cloning was switched to `forward_train()` for gradient-bearing logits.

### Behavior cloning result
- Command:
  - `.venv\Scripts\python.exe -m f1rl.imitation --episodes 8 --epochs 5 --max-steps 500 --device cpu --ppo-mode smoke --run-tag ppo-warm`
- Artifact:
  - `artifacts/bc-ppo-warm-20260409-085436/bc_summary.json`
- Result:
  - standalone BC eval `avg_progress_total = 17.0`
  - PPO-RLModule pretrain eval `avg_progress_total = 17.0`
  - both faithfully reproduced the scripted controller ceiling, but did not exceed it.

### PPO warm-start result
- 2-iteration probe:
  - train: `artifacts/train-smoke-bcwarm-20260409-090140/checkpoints`
  - benchmark: `artifacts/benchmark-quick-20260409-090319/benchmark_summary.json`
  - aggregate:
    - `completion_rate = 0.0`
    - `collision_rate = 0.0`
    - `avg_checkpoints_reached = 2.0`
    - `termination_reason = wall_hugging`
- 6-iteration probe:
  - train: `artifacts/train-smoke-bcwarm6-20260409-090431/checkpoints`
  - benchmark: `artifacts/benchmark-quick-20260409-090820/benchmark_summary.json`
  - aggregate:
    - `completion_rate = 0.0`
    - `collision_rate = 0.0`
    - `avg_checkpoints_reached = 3.0`
    - `three_checkpoint_rate = 1.0`
    - `termination_reason = wall_hugging`

### Decision
- Rejected the behavior-cloning-to-PPO warm-start branch for the current competence target.
- It is materially worse than the incumbent `reward_progress_bias` checkpoint (`avg_checkpoints_reached = 21.0`).
- The branch was given a fairer budget than a trivial smoke:
  - real scripted dataset collection,
  - PPO-compatible RLModule pretrain export,
  - warm-start PPO at 2 iterations,
  - warm-start PPO at 6 iterations.

### Updated plateau assessment
- The current competence program is now exhausted across:
  - one full PPO candidate-family rotation,
  - deeper pure-PPO budget probes,
  - hybrid scripted warm-start reset probes,
  - direct scripted-oracle probing,
  - stronger-oracle controller tuning and reversion,
  - behavior cloning,
  - PPO warm-start from cloned RLModule state.
- At this point, further meaningful progress would require a genuinely new training program, not another nearby tweak.

## Milestone N - Improved Scripted Baseline and Oracle Recheck (2026-04-09)

### Scripted-controller improvement
- Promoted a measured scripted-controller improvement into defaults in `src/f1rl/controllers.py`:
  - `caution_throttle: 0.08 -> 0.09`
  - `caution_front: 0.38 -> 0.34`
  - `brake_front: 0.22 -> 0.18`
  - `lookahead_goals: 3 -> 4`
  - `heading_gain: 1.6 -> 1.35`
  - `heading_d_gain: 0.45 -> 0.2`
- Validation:
  - `.venv\Scripts\python.exe -m pytest -q tests/test_env_reset.py tests/test_cli_smoke.py` -> pass
  - `.venv\Scripts\python.exe -m f1rl.manual --headless --controller scripted --max-steps 1000` -> `manual_run_complete steps=475 laps=0 reward=52.098`
  - direct seed-101 probe -> `18 475 no_progress 52.09840796067402`
- Result:
  - scripted oracle improved from `17 / 462 / 50.225` to `18 / 475 / 52.098`
  - still below the incumbent PPO competence checkpoint (`avg_checkpoints_reached = 21.0`)

### Oracle-to-imitation recheck
- Re-ran imitation from the stronger oracle:
  - command:
    - `.venv\Scripts\python.exe -m f1rl.imitation --episodes 4 --epochs 3 --max-steps 500 --device cpu --ppo-mode smoke --run-tag ppo-warm18`
  - artifact:
    - `artifacts/bc-ppo-warm18-20260409-092421/bc_summary.json`
- Result:
  - dataset oracle average: `18.0`
  - standalone BC eval average: `12.0`
  - PPO-compatible RLModule pretrain eval average: `8.0`
- Interpretation:
  - the slightly better oracle does not transfer cleanly through the current imitation path
  - this branch regresses before PPO even enters the loop

### Updated conclusion
- There was one more real improvement available:
  - the scripted baseline itself is now stronger.
- But the broader competence campaign remains plateaued:
  - the improved oracle still does not exceed the incumbent PPO benchmark,
  - and the imitation/warm-start bridge from that oracle regresses materially.

## Milestone O - Swarm Observability Expansion (2026-04-09)

### Why this milestone happened
- The earlier swarm view was not trustworthy enough for diagnosis:
  - identical cars overlapped visually
  - movement was difficult to perceive at whole-track scale
  - telemetry was too thin to distinguish weak policy outputs from over-aggressive terminations
- The previous plateau assessment therefore had to be treated as provisional until the swarm path exposed real movement, checkpoint, and failure data.

### Changes
- Expanded `src/f1rl/swarm.py` telemetry output substantially.
- Added aggregate per-step telemetry in:
  - `metrics/telemetry.json`
  - `metrics/telemetry.csv`
- Aggregate telemetry now records:
  - `alive_cars`
  - `moving_cars`
  - `avg_current_speed`
  - `avg_distance_travelled`
  - `max_distance_travelled`
  - `avg_checkpoints_reached`
  - `max_checkpoints_reached`
  - `dominant_termination_reason`
  - best-car index, speed, distance, checkpoints, reward, and position
- Added per-car telemetry in:
  - `metrics/car_telemetry.json`
  - `metrics/car_telemetry.csv`
- Per-car telemetry now records:
  - `x`, `y`
  - `speed`
  - `distance_travelled`
  - `total_reward`
  - `lap_count`
  - `next_goal_idx`
  - `progress_delta`
  - `checkpoints_reached`
  - `checkpoint_streak`
  - `heading_error_norm`
  - `goal_distance_norm`
  - `lateral_offset_norm`
  - `no_progress_steps`
  - `reverse_speed_steps`
  - `reverse_progress_events`
  - `heading_away_steps`
  - `wall_hugging_steps`
  - `done`
  - `collided`
  - `terminated_reason`
- Added checkpoint transition event logging in:
  - `metrics/checkpoint_events.json`
- Added top-k trace export in:
  - `metrics/top_k_traces.json`
- Added richer swarm render overlays:
  - visible checkpoint lines across the track
  - numbered checkpoint labels
  - highlighted next checkpoint for the best car
  - top-k trace lines
  - top-k car labels
  - richer HUD with alive/moving counts, distance, checkpoint, reward, position, speed, and dominant failure context

### Interpretation
- The swarm diagnostic path is now capable of answering questions that were previously guesswork:
  - is the learned policy outputting near-zero actions?
  - is it moving but being terminated too early?
  - is it reaching checkpoints and then failing on a specific termination family?
  - is failure dominated by collision, no progress, wall hugging, reverse behavior, or another constraint?

### Validation status
- Code inspection confirms the observability paths are present in `src/f1rl/swarm.py`.
- A clean fresh runtime validation of the new telemetry outputs is still required in the next session loop.
- Because the prior Codex thread became unstable during runtime work, this milestone should be treated as:
  - code landed
  - audit log now recorded
  - runtime behavior still pending revalidation

## Milestone P - Competence Relaxation and Swarm Render Responsiveness Patch (2026-04-09)

### Why this milestone happened
- The competence stage had become too punitive for weak learned policies.
- Early policies were being terminated before they could generate useful motion traces or checkpoint telemetry.
- The live swarm renderer also needed direct UX fixes after reports that:
  - the window could appear `Not Responding`
  - cars visually collapsed into nearly one point
  - tiny motion was hard to perceive at whole-track scale

### Changes
- Relaxed the `competence` stage in `src/f1rl/stages.py`:
  - `max_steps: 240 -> 420`
  - `no_progress_limit_steps: 60 -> 140`
  - `reverse_speed_limit_steps: 32 -> 48`
  - `reverse_progress_limit: 2 -> 4`
  - `heading_away_limit_steps: 45 -> 0`
  - `wall_hugging_limit_steps: 30 -> 0`
  - removed competence checkpoint deadlines entirely
- Updated swarm rendering in `src/f1rl/swarm.py`:
  - explicit `pygame.event.get()` pumping each frame to keep the window responsive
  - best-car zoom panel so small early movements remain visible
  - default render-time spawn jitter when `--render` is used without explicit jitter:
    - `spawn_position_jitter_px = 10.0`
    - `spawn_heading_jitter_deg = 6.0`

### Intended effect
- Weak policies should survive long enough in competence to produce meaningful motion and checkpoint traces.
- The live swarm window should remain responsive during short inspection runs.
- Independent cars should be visually separated enough to avoid misleading overlap.
- Small but real motion should be obvious through the zoom panel even when whole-track movement is limited.

### Validation status
- Code inspection confirms the relaxed competence settings in `src/f1rl/stages.py`.
- Code inspection confirms the render responsiveness and visibility patch in `src/f1rl/swarm.py`.
- The previous thread did not complete a clean post-patch validation of:
  - live scripted swarm responsiveness
  - live learned-policy swarm responsiveness
  - visibility of the zoom panel and checkpoint overlays
- This is therefore a pending validation milestone, not a fully closed one.

## Current Next Actions (as of 2026-04-09)

### Highest-priority validation sequence
1. Verify live scripted swarm render after restart:
   - `.venv\Scripts\python.exe -m f1rl.swarm --policy scripted --cars 8 --swarm-stage competence --render --steps 120 --top-k 4`
2. Verify live learned-policy swarm render using the current best competence checkpoint:
   - `.venv\Scripts\python.exe -m f1rl.swarm --policy checkpoint --checkpoint "C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\train-smoke-reward_progress_bias-20260409-065951\checkpoints" --cars 8 --swarm-stage competence --render --steps 120 --top-k 4`
3. Run a headless learned-policy telemetry audit and inspect:
   - `swarm_summary.json`
   - `metrics/telemetry.csv`
   - `metrics/car_telemetry.csv`
   - `metrics/checkpoint_events.json`
4. Reproduce and fix the headless clip-export memory issue if `--export-clips` still fails.
5. Only after those validations are complete, resume recursive learning experiments.

### Current working assumption
- The previous competence plateau conclusion should not yet be treated as final.
- The system is materially better instrumented now than it was when that conclusion was reached.
- The next iteration should be driven by real swarm telemetry and verified visual diagnostics, not by more blind reward or PPO tweaks.

## Milestone Q - Swarm Validation and Clip-Export Stabilization (2026-04-09)

### Live scripted swarm render validation
- Command:
  - `.venv\Scripts\python.exe -m f1rl.swarm --policy scripted --cars 8 --swarm-stage competence --render --steps 120 --top-k 4`
- Validation result:
  - window opened and remained responsive
  - checkpoint overlays and numbered labels were visible
  - best-car zoom panel was visible and useful
  - cars were visually separated instead of collapsing into one point
  - movement was clearly visible across time, including through the zoom panel
- Runtime evidence captured under:
  - `artifacts/runtime-checks/scripted-swarm-window.png`
  - `artifacts/runtime-checks/scripted-swarm-window-2.png`
- Observed behavior from the captures:
  - HUD values changed across time
  - `best checkpoints` increased from `1` to `3`
  - `avg dist` increased from `33.9` to `70.9`
  - `best dist` increased from `52.9` to `110.8`
- Interpretation:
  - the render responsiveness patch is working
  - the default render jitter and zoom panel materially improve early-motion visibility

### Live learned-policy swarm render validation
- Command:
  - `.venv\Scripts\python.exe -m f1rl.swarm --policy checkpoint --checkpoint "C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\train-smoke-reward_progress_bias-20260409-065951\checkpoints" --cars 8 --swarm-stage competence --render --steps 120 --top-k 4`
- Validation result:
  - checkpoint-backed render did start successfully
  - startup was slower than scripted render because of artifact/module load
  - the window was responsive once visible
  - cars were moving; this is not a zero-action / frozen-policy failure mode
- Runtime evidence captured under:
  - `artifacts/runtime-checks/learned-swarm-window.png`
- Observed live-window snapshot:
  - `cars: 8 alive: 6`
  - `moving: 8`
  - `avg speed: 0.61`
  - `avg dist: 70.2`
  - `best dist: 116.6`
  - `best checkpoints: 2`
- Interpretation:
  - the incumbent competence checkpoint does produce motion and some checkpoint progress
  - the problem is weak policy quality, not total motor collapse

### Headless learned-policy telemetry audit
- Command:
  - `.venv\Scripts\python.exe -m f1rl.swarm --policy checkpoint --checkpoint "C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\train-smoke-reward_progress_bias-20260409-065951\checkpoints" --cars 8 --swarm-stage competence --steps 240 --headless --top-k 4`
- Artifact:
  - `artifacts/swarm-competence-20260409-144216/swarm_summary.json`
- Aggregate outcome:
  - `completion_rate = 0.0`
  - `collision_rate = 0.375`
  - `avg_checkpoints_reached = 2.0`
  - `max_checkpoints_reached = 3`
  - `avg_speed = 0.576`
  - `avg_current_speed = 0.925`
  - `avg_distance_travelled = 115.10`
  - `max_distance_travelled = 182.16`
  - `moving_car_fraction = 1.0`
- Failure histogram:
  - `collision = 3`
  - `max_steps = 5`
- Telemetry inspection:
  - early `telemetry.csv` rows show immediate non-zero motion from the first few steps
  - late `telemetry.csv` rows show continued motion and checkpoint progress through step `240`
  - `checkpoint_events.json` contains `16` checkpoint events spanning steps `35` through `205`
  - car summaries show several cars surviving to `max_steps` with `1` to `3` checkpoints reached
- Interpretation:
  - the incumbent policy is weak but not inert
  - competence relaxation is doing useful work because many cars now survive to `max_steps`
  - the main near-term issue is not overtermination from heading-away/wall-hugging in competence
  - the current failure mix is:
    - partial early collision
    - partial timeout at `max_steps`
    - limited checkpoint reach

### Headless clip-export reproduction and stabilization
- Reproduced the clip-export path with:
  - `.venv\Scripts\python.exe -m f1rl.swarm --policy checkpoint --checkpoint "C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\train-smoke-reward_progress_bias-20260409-065951\checkpoints" --cars 8 --swarm-stage competence --steps 240 --headless --top-k 4 --export-clips`
  - `.venv\Scripts\python.exe -m f1rl.swarm --policy checkpoint --checkpoint "C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\train-smoke-reward_progress_bias-20260409-065951\checkpoints" --cars 32 --swarm-stage competence --steps 240 --headless --top-k 4 --export-clips`
- The original hard crash did not reproduce on this fresh session, but the heavier clip path still showed the same class of problem:
  - very high memory pressure during headless export
  - long wall-clock completion times
- Implemented mitigation in `src/f1rl/swarm.py`:
  - headless clip capture now downsamples rendered frames before `pygame.surfarray.array3d(...)`
  - headless clip capture now samples only every Nth frame instead of materializing every frame
  - clip durations are adjusted to preserve approximate timing after frame subsampling
  - frame-array capture is now skipped entirely on non-sampled export steps
- Post-fix validations:
  - `.venv\Scripts\python.exe -m pytest -q tests/test_cli_smoke.py tests/test_stages.py` -> pass
  - `8-car / 240-step / headless / export-clips` -> pass
    - artifact: `artifacts/swarm-competence-20260409-150453/swarm_summary.json`
    - rendered clips written:
      - `early.gif`
      - `mid.gif`
      - `late.gif`
  - `32-car / 240-step / headless / export-clips` -> pass
    - artifact: `artifacts/swarm-competence-20260409-150817/swarm_summary.json`
    - rendered clips written:
      - `early.gif`
      - `mid.gif`
      - `late.gif`
- Current status:
  - the headless clip-export path is working again on a clean session
  - the failure mode appears mitigated rather than fully eliminated, because large exports are still expensive on Windows
  - the path is now stable enough for continued diagnostics without blocking the next experiment loop

### Updated conclusion after fresh validation
- The late swarm observability work is real and useful.
- The render responsiveness patch is effective.
- The incumbent learned policy is moving and reaching a few checkpoints; it is not outputting near-zero actions.
- Competence relaxation is no longer the dominant blocker; the policy is now living long enough to expose its true weakness.
- The next recursive training loop should focus on improving checkpoint reach and forward pace under the now-validated telemetry path, not on more blind renderer or termination debugging.

## Milestone R - CUDA Inference and GPU-Backed Swarm Renderer (2026-04-09)

### Checkpoint inference moved onto CUDA
- Updated lightweight artifact inference in `src/f1rl/inference.py` so loaded RLModules move onto CUDA when `torch.cuda.is_available()`.
- Validation:
  - `.venv\Scripts\python.exe -c "from f1rl.inference import load_inference_policy; p=load_inference_policy(r'C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\train-smoke-reward_progress_bias-20260409-065951\checkpoints'); import torch; print({'policy_device': p.device, 'module_param_device': str(next(p.module.parameters()).device), 'cuda_available': torch.cuda.is_available()})"`
- Result:
  - artifact inference reported `policy_device='cuda'`
  - module parameters reported `cuda:0`

### Swarm live render moved off software surfaces onto SDL2 hardware renderer
- Refactored `src/f1rl/swarm.py` so:
  - live human render uses `pygame._sdl2.Renderer`
  - headless/export capture stays on CPU surfaces
  - static layers and text textures are cached for the hardware path
  - HUD now explicitly shows `renderer: ...` and `fps: ...`
  - swarm summaries now record `renderer_backend` and `render_scale`
- Live renderer now actively prefers hardware backends in this order:
  - `direct3d11`
  - `direct3d12`
  - `opengl`
  - `direct3d`
  - then auto / software fallback only if needed
- Fresh probe:
  - `.venv\Scripts\python.exe -c "from f1rl.config import EnvConfig; from f1rl.track import build_track; from f1rl.swarm import SwarmRenderer; cfg=EnvConfig.from_dict({'render_mode':'human','headless':False,'render':{'enabled':True,'window_size':[640,360]}}); track=build_track(cfg.track); r=SwarmRenderer(cfg, track, top_k=4, enabled=True, display_scale=0.6); r._ensure(); print({'backend': r.renderer_backend, 'display_size': r.display_size}); r.close()"`
- Result:
  - renderer backend bound as `direct3d11`
  - debug display size at default live scale is now `384x216`

### Windows hybrid-GPU routing work
- On this laptop, aggregate `nvidia-smi` stayed at `0` even after the renderer moved to SDL2 hardware mode.
- That was not enough evidence to conclude “still CPU”; Windows hybrid laptops often route tiny D3D loads in ways `nvidia-smi` does not report well.
- Switched to per-process Windows GPU-engine counters against the actual live `F1RL Swarm` process.
- Validation:
  - interactive scripted swarm process opened with `MainWindowTitle = F1RL Swarm`
  - `Get-Counter '\GPU Engine(*)\Utilization Percentage'` for the live process showed:
    - `pid_35864_luid_..._phys_0_eng_0_engtype_3d = 0.12`
- Interpretation:
  - the live swarm process is now registering real `3D` engine activity
  - the repo-side renderer is no longer pure CPU software blitting
  - `nvidia-smi` remains a weak validator for this low-load Windows D3D path

### OS graphics preference pin
- Backed up the current user GPU preference key to:
  - `artifacts/runtime-checks/user-gpu-preferences-backup-20260409.reg`
- Then pinned the repo interpreter paths to high-performance graphics:
  - `.venv\Scripts\python.exe -> GpuPreference=2;`
  - `.venv\Scripts\pythonw.exe -> GpuPreference=2;`
- This is intended to push future live swarm launches toward the high-performance adapter on the user’s hybrid laptop.

### Speed-model changes
- Began moving the project off arbitrary “units per frame” toward a more physical speed model:
  - `dt` now defaults to `1/60`
  - track config now includes `track_length_meters`
  - track build computes `meters_per_world_unit`
  - dynamics now advance position using physical speed scaled by `meters_per_world_unit`
  - env info and HUD now expose `speed_kph`
- Important caveat:
  - this is a calibration pass, not a claim of full F1-grade realism yet
  - reward, controller, and PPO tuning still need to catch up to the new speed model

### Current status after Milestone R
- Live checkpoint inference is on CUDA.
- Live swarm render now uses an SDL2 hardware renderer and binds to `direct3d11` on this machine.
- Windows per-process GPU-engine counters now show non-zero `3D` activity for the swarm render process.
- The default live debug view is smaller and should run materially faster than the earlier surface-only path.
- Further performance work should focus on:
  - policy quality under the new speed model
  - confirming whether the high-performance adapter pin moves future user-launched runs fully onto the RTX path in Task Manager

## Milestone S - Torch-Native GPU Runtime Rewrite and Telemetry Gates (2026-04-09)

### Scope
- Replaced the primary hot path with a torch-native runtime centered on `TorchSimBatch`.
- Replaced RLlib as the primary training/eval/inference path with an in-repo PyTorch PPO stack.
- Added telemetry-first simulator outputs and validation gates so future campaign/training work is blocked until simulator parity and telemetry checks pass.
- Added a new GPU-backed live renderer module using `pyglet` for the interactive path.

### Primary code changes
- Added `src/f1rl/torch_runtime.py`
  - batched simulator on CPU or CUDA
  - batched observations, rewards, terminations, checkpoint crossings, wall-distance queries, telemetry thresholds, runtime metadata
- Added `src/f1rl/torch_agent.py`
  - torch-native actor-critic policy
  - torch-native checkpoint save/load helpers
- Added `src/f1rl/torch_ppo.py`
  - CUDA-friendly rollout collection and PPO update path
- Rewrote the primary entrypoints onto the torch-native stack:
  - `src/f1rl/train.py`
  - `src/f1rl/eval.py`
  - `src/f1rl/rollout.py`
  - `src/f1rl/swarm.py`
  - `src/f1rl/manual.py`
  - `src/f1rl/validate.py`
- Added `src/f1rl/gpu_renderer.py`
  - pyglet/OpenGL live renderer
  - live HUD with renderer backend, FPS, movement/progress telemetry, and checkpoint overlays
- Added `tests/test_torch_runtime.py`
  - step telemetry surface
  - moving-car threshold persistence
  - CPU vs CUDA fixed-action parity

### Telemetry-first acceptance surface now implemented
- Per-step/per-car telemetry in the new simulator includes:
  - `speed`, `speed_kph`
  - `distance_travelled`
  - `current_checkpoint_index`
  - checkpoint crossing events
  - `time_since_last_checkpoint`
  - `termination_code` / reason mapping
  - `throttle`, `brake`, `steering`
  - `min_wall_distance_ratio`
  - `moving`
- Runtime metadata now persists:
  - `sim_device`
  - `policy_device`
  - `renderer_backend`
  - `telemetry_thresholds`
  - selected CUDA device name when applicable

### Validation results
- Passed targeted torch-runtime and CLI tests:
  - `.venv\Scripts\python.exe -m pytest -q tests/test_torch_runtime.py tests/test_cli_smoke.py tests/test_checkpoint_roundtrip.py tests/test_benchmark.py`
- Passed lint for the touched rewrite files:
  - `.venv\Scripts\python.exe -m ruff check src/f1rl/gpu_renderer.py src/f1rl/swarm.py src/f1rl/manual.py src/f1rl/validate.py src/f1rl/hardware.py tests/test_torch_runtime.py`
- Passed end-to-end validation:
  - `.venv\Scripts\python.exe -m f1rl.validate`
- `f1rl.validate` now confirms:
  - CPU vs CUDA simulator parity on fixed action batches
  - telemetry key coverage and explicit moving-car thresholds
  - CUDA visibility in `nvidia-smi`
  - torch-native train/eval/benchmark/sim runtime metadata on CUDA where a policy exists

### Runtime observations
- Fresh smoke train after rewrite:
  - `.venv\Scripts\python.exe -m f1rl.train --mode smoke --iterations 1 --device auto`
  - output reported `sim_device=cuda policy_device=cuda`
- Fresh `f1rl.validate` captured live CUDA visibility:
  - `nvidia-smi` showed the validation Python process on the RTX 4060 with non-zero memory usage during the CUDA gate
- Headless torch-native workflows now pass on the new stack:
  - manual scripted headless
  - random swarm headless
  - smoke train
  - checkpoint eval
  - quick benchmark

### Dependency / environment notes
- Added `pyglet` and `moderngl` under the `viz` extra in `pyproject.toml`.
- `uv sync --active --extra train --extra viz --extra dev` hit repeated Windows/OneDrive `.dist-info` cleanup failures inside `.venv`.
- As a pragmatic workaround, the missing live-render dependencies were installed with:
  - `uv pip install --python .venv\Scripts\python.exe pyglet>=2.1.0 moderngl>=5.12.0`
- Renderer availability probe now succeeds:
  - `.venv\Scripts\python.exe -c "from f1rl.gpu_renderer import renderer_available; print({'renderer_available': renderer_available()})"`
  - result: `{'renderer_available': True}`

### Important caveat
- The live `pyglet` renderer is now integrated into `swarm` and `manual`, but this Codex desktop tool session was not trustworthy for interactive visual validation:
  - short live render commands launched but did not complete cleanly in-tool
  - the renderer module itself imports correctly and the headless/runtime validations passed
- Because of that, the interactive live-window path still needs one direct desktop confirmation on the user machine before treating live renderer parity as fully closed.

### Current status after Milestone S
- Primary training/inference/benchmark/swarm/manual logic now runs on the torch-native runtime instead of RLlib as the default path.
- CUDA is actively used for simulator/training/eval compute on the RTX 4060 when available.
- Telemetry correctness and simulator parity are now enforced before future campaign/autonomous work should resume.
- The next milestone should be:
  - one direct desktop validation of the new live pyglet renderer
  - then recursive policy improvement on top of the validated torch-native stack
