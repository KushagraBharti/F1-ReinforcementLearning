# Plan

## Status
- Milestone 0: Completed
- Milestone 1: Completed
- Milestone 2: Completed
- Milestone 3: Completed
- Milestone 4: Completed

## Milestone 0: Baseline and Planning
### Acceptance Criteria
- Repository inventory complete.
- Initial constraints, goals, and validation workflow captured.
- `Prompt.md`, `Plan.md`, `Implement.md`, `Documentation.md` created.

### Validation Commands
1. `git status --short`
2. `rg --files`

## Milestone 1: Packaging and Project Layout
### Acceptance Criteria
- `pyproject.toml` created for uv-managed workflow.
- Source refactored into package structure under `src/`.
- Backward-compatible script shims added for legacy file names where practical.
- `uv.lock` generated.

### Validation Commands
1. `uv sync --active --all-extras`
2. `uv run python -c "import f1rl; print(f1rl.__version__)"`
3. `uv run pytest -q tests/test_imports.py`

## Milestone 2: Environment + Manual Mode Modernization
### Acceptance Criteria
- Gymnasium-compatible environment implemented.
- Manual mode command works (keyboard control, reset, quit).
- Headless-compatible rendering path exists.
- Core constants configurable.

### Validation Commands
1. `uv run pytest -q tests/test_env_api.py`
2. `uv run python -m f1rl.manual --headless --max-steps 60 --autodrive`
3. `uv run python -m f1rl.rollout --steps 120 --headless --policy random`

## Milestone 3: RLlib Training + Checkpoints + Inference
### Acceptance Criteria
- RL training entrypoint uses modern RLlib config flow (PyTorch).
- Smoke training produces checkpoint + metrics artifact.
- Inference/eval entrypoint loads checkpoint and executes rollout.
- Optional render artifact (frames/gif) supported.

### Validation Commands
1. `uv run python -m f1rl.train --mode smoke`
2. `uv run python -m f1rl.train --mode smoke --resume latest`
3. `uv run python -m f1rl.eval --checkpoint latest --steps 150 --headless`
4. `uv run pytest -q tests/test_checkpoint_roundtrip.py`

## Milestone 4: Test/Lint/Type and Docs Finalization
### Acceptance Criteria
- Unit tests for geometry/dynamics/reward behavior included.
- Smoke validations automated in scripts.
- README rewritten with setup/run/train/eval/troubleshooting.
- `Documentation.md` contains commands, outcomes, and migration notes.

### Validation Commands
1. `uv run ruff check .`
2. `uv run pyright src/f1rl`
3. `uv run pytest -q`
4. `uv run python -m f1rl.validate`
