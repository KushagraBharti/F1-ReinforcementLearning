# Plan

## Status
- Milestone A: Completed
- Milestone B: Completed
- Milestone C: Completed
- Milestone D: Completed
- Milestone E: Completed
- Milestone F: Completed
- Milestone G: Completed

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
