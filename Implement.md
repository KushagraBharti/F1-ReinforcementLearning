# Implement

## Execution Runbook
1. Work in milestone loops:
   - inspect current state
   - update acceptance criteria and commands if scope changed
   - implement one coherent subsystem change
   - run smoke validations immediately
   - run benchmark comparison when the subsystem is stable
   - keep or reject based on benchmark evidence
   - log outcome in `Documentation.md`
2. Do not keep speculative changes without benchmark improvement.
3. Prefer explicit config and artifact outputs over hidden defaults.
4. Preserve headless determinism for smoke and benchmark workflows.

## Technical Defaults
- Python 3.12.x on Windows
- Ray/RLlib + PyTorch with CUDA-enabled local training by default
- Artifact-first inference for eval/rollout/benchmark
- Gymnasium custom environment API
- PPO is the baseline algorithm family
- Candidate acceptance is ordered by:
  1. completion rate
  2. lap time
  3. collision/instability reduction

## Artifact Conventions
- `artifacts/train-*/` for training runs
- `artifacts/benchmark-*/` for evaluation bundles
- `artifacts/champions/` for promoted champion metadata
- `artifacts/renders/` for exported clips/frames
- protected runs use `.pin` or equivalent marker files

## Command Conventions
- Setup: `uv sync --active --all-extras`
- Hardware validation: `uv run python -m f1rl.hardware_check`
- Manual: `uv run python -m f1rl.manual`
- Train: `uv run python -m f1rl.train --mode smoke|benchmark|performance`
- Train overrides: `--num-env-runners`, `--num-envs-per-env-runner`, `--vector-mode sync|async`
- Benchmark: `uv run python -m f1rl.benchmark --checkpoint latest --profile quick`
- Compatibility restore: `uv run python -m f1rl.eval --legacy-algorithm-restore`
- Optimize loop: `uv run python -m f1rl.optimize`
- Cleanup: `uv run python -m f1rl.clean_artifacts`
- Full validation: `uv run python -m f1rl.validate`

## Local Path Note
- On this OneDrive-backed Windows path, repeated `uv run ...` commands may require `UV_LINK_MODE=copy`.
