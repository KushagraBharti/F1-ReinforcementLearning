# Plan

## Status
- Milestone A: Completed
- Milestone B: Completed
- Milestone C: Completed
- Milestone D: Completed
- Milestone E: Completed
- Milestone F: Completed
- Milestone G: Completed
- Milestone H: Completed
- Milestone I: Completed

## Milestone A: Baseline and GPU Runtime
### Acceptance Criteria
- Baseline validations and current hardware/runtime facts recorded
- CUDA-enabled Torch configured for the project environment
- GPU validation command exists and confirms CUDA + RLlib smoke training readiness

### Validation Commands
1. `nvidia-smi`
2. `uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`
3. `uv run python -m f1rl.hardware_check`
4. `uv run python -m f1rl.train --mode smoke --iterations 1`

## Milestone B: Benchmark and Champion Workflow
### Acceptance Criteria
- Benchmark evaluation command emits machine-readable metrics
- Short sampled clips are exported for candidate review
- Champion comparison logic exists and can protect promoted runs

### Validation Commands
1. `uv run python -m f1rl.benchmark --checkpoint latest --profile quick`
2. `uv run pytest -q tests/test_benchmark.py`

## Milestone C: Environment, Observation, and Reward Upgrades
### Acceptance Criteria
- Dynamics model is more realistic than the current minimal heading/speed update
- Observation includes track-relative signals beyond raw wall rays
- Reward logging explains major reward components
- Termination logic handles no-progress and unstable behavior more robustly

### Validation Commands
1. `uv run pytest -q tests/test_dynamics.py tests/test_env_api.py tests/test_reward_logic.py`
2. `uv run python -m f1rl.validate --skip-train`

## Milestone D: PPO Profiles and Evaluation During Training
### Acceptance Criteria
- Structured train profiles exist for `smoke`, `benchmark`, and `performance`
- Training writes evaluation metrics during training
- GPU-aware RLlib resource configuration is the default local path

### Validation Commands
1. `uv run python -m f1rl.train --mode benchmark --iterations 1`
2. `uv run pytest -q tests/test_checkpoint_roundtrip.py tests/test_train_profiles.py`

## Milestone E: Recursive Candidate Loop
### Acceptance Criteria
- Baseline benchmark is recorded
- At least one candidate change set is benchmarked against the champion
- Champion promotion follows completion-rate-first comparison rules

### Validation Commands
1. `uv run python -m f1rl.optimize --max-candidates 1 --profile quick`
2. `uv run python -m f1rl.benchmark --checkpoint latest --profile quick`

## Milestone F: Final Validation and Documentation
### Acceptance Criteria
- README reflects GPU, training, benchmark, and optimization-loop workflows
- `Documentation.md` records commands, outcomes, benchmark comparisons, and kept/rejected hypotheses
- Full validation suite passes

### Validation Commands
1. `uv run ruff check .`
2. `uv run pyright src/f1rl`
3. `uv run pytest -q`
4. `uv run python -m f1rl.validate`

## Milestone G: RLlib Modernization and Artifact-First Inference
### Acceptance Criteria
- RLlib config uses explicit new-stack surfaces and learner-centric batching
- Default train profiles use multi-env parallelism
- Train exports a lightweight inference artifact
- Eval/benchmark/rollout use artifact-backed inference by default
- Reverse-driving is no longer the dominant learned failure mode

### Validation Commands
1. `uv run python -m f1rl.train --mode smoke --iterations 1 --device gpu --require-gpu`
2. `uv run python -m f1rl.eval --checkpoint latest --headless --steps 120`
3. `uv run python -m f1rl.benchmark --checkpoint latest --profile quick --promote-if-best --no-clips`
4. `uv run pytest -q tests/test_benchmark.py tests/test_reward_logic.py tests/test_dynamics.py tests/test_checkpoint_roundtrip.py`

## Milestone H: Benchmark Reliability and Stage Metrics
### Acceptance Criteria
- Benchmark clip export is based on actual rendered episode frames
- Benchmark summaries include checkpoint-progress metrics and stage-gating metrics
- Benchmark supports fixed promotion seeds, rotating diagnostic seeds, and holdout seeds

### Validation Commands
1. `uv run pytest -q tests/test_benchmark.py tests/test_stages.py`
2. `uv run python -m f1rl.benchmark --checkpoint latest --profile quick --seed-tier promotion --no-clips`
3. `uv run python -m f1rl.benchmark --checkpoint latest --profile quick --seed-tier diagnostic --rotation-index 1 --no-clips`

## Milestone I: Swarm Diagnostics and Autonomous Campaign
### Acceptance Criteria
- Swarm command exists for headless and visual many-car diagnostics
- Stage-aware train CLI supports logical swarm sizing
- Autonomous campaign runner records wave-by-wave keep/reject decisions
- Campaign uses single-car benchmark promotion authority and swarm diagnostics only

### Validation Commands
1. `uv run python -m f1rl.swarm --policy random --headless --cars 8 --steps 40`
2. `uv run pytest -q tests/test_campaign.py tests/test_cli_smoke.py`
3. `uv run python -m f1rl.campaign --hours 0.02 --max-waves 1 --start-cars 8 --max-cars 8 --device auto`

## Milestone J: Torch-Native GPU Runtime and Telemetry Gates
### Acceptance Criteria
- Batched torch-native simulator exists and runs on CPU or CUDA
- Primary train/eval/swarm/manual/benchmark workflows use the torch-native path by default
- Telemetry thresholds and runtime metadata are exported with every runtime summary
- CPU vs CUDA simulator parity checks pass before campaign/training claims resume
- Validation proves CUDA visibility and runtime metadata on the primary torch-native flows

### Validation Commands
1. `.venv\Scripts\python.exe -m pytest -q tests/test_torch_runtime.py tests/test_cli_smoke.py tests/test_checkpoint_roundtrip.py tests/test_benchmark.py`
2. `.venv\Scripts\python.exe -m f1rl.train --mode smoke --iterations 1 --device auto`
3. `.venv\Scripts\python.exe -m f1rl.validate`
