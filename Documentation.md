# Documentation

## Session Date
- 2026-02-25

## Scope
- Upgrade runtime/dependencies to latest stable targets that keep full training/inference working on Windows.
- Replace legacy `--autodrive` with deterministic scripted controller.
- Add robust artifact cleanup command with retention policy.
- Revalidate full repo end-to-end and update docs.

## Milestone 0 - Inventory and Baseline
### Findings
- Legacy scripts and deprecated RLlib APIs were present.
- No modern packaging/lockfile existed.
- Existing RL stack was tied to old `tune.run` and legacy trainer APIs.

### Baseline commands
1. `git status --short` -> pass
2. `rg --files` -> pass

## Milestone 1 - Packaging and Structure
### Changes
- Added modern packaging:
  - `pyproject.toml`
  - `uv.lock`
- Refactored code into `src/f1rl`.
- Added CLI entrypoints for manual/rollout/train/eval/validate.
- Added compatibility wrappers for legacy script names.

### Validation
1. `uv sync --active --all-extras` -> pass
2. `uv run python -c "import f1rl; print(f1rl.__version__)"` -> pass (`0.1.0`)

## Milestone 2 - Gymnasium Env + Manual Mode
### Changes
- Implemented Gymnasium-compatible environment.
- Decoupled dynamics/track/renderer/env modules.
- Added headless-safe manual mode and rendering.

### Validation
1. `uv run pytest -q tests/test_env_api.py` -> pass
2. `uv run python -m f1rl.manual --headless --controller scripted --max-steps 80` -> pass (`steps=80`)
3. `uv run python -m f1rl.rollout --policy random --headless --steps 120` -> pass

## Milestone 3 - RLlib Modernization
### Changes
- Migrated to modern `PPOConfig` workflow.
- Added checkpoint save/load + eval + rollout entrypoints.
- Implemented RLModule-based inference path (`forward_inference`) for new API stack compatibility.
- Added structured metrics/log artifacts.

### Validation
1. `uv run python -m f1rl.train --mode smoke` -> pass
2. `uv run python -m f1rl.train --mode smoke --resume latest --iterations 1` -> pass
3. `uv run python -m f1rl.eval --checkpoint latest --headless --steps 150` -> pass
4. `uv run python -m f1rl.rollout --policy checkpoint --checkpoint latest --headless --steps 120` -> pass

## Milestone 4 - Tests and Quality Gates
### Changes
- Added tests:
  - imports
  - geometry
  - dynamics
  - env API checker
  - CLI smoke
  - checkpoint roundtrip
- Added full validation script `f1rl.validate`.

### Validation
1. `uv run ruff check .` -> pass
2. `uv run pyright src/f1rl` -> pass
3. `uv run pytest` -> pass (`13 passed`)
4. `uv run python -m f1rl.validate` -> pass (`all checks passed`)

## Milestone 5 - Python/Ray Upgrade + Controller + Artifact Retention
### Online verification and decisions
1. Latest Python stable:
- Source: Python downloads page shows `Python 3.14.3`.
- URL: https://www.python.org/downloads/

2. Latest Ray stable:
- Source: PyPI reports latest `ray` as `2.54.0`.
- URL: https://pypi.org/project/ray/
- Wheel support check (live PyPI JSON query) showed:
  - Windows wheels available for `cp310`, `cp311`, `cp312`
  - No Windows wheel for `cp313`/`cp314` on `2.54.0`
  - Command used:
    - `uv run python -c "import json, urllib.request; data=json.load(urllib.request.urlopen('https://pypi.org/pypi/ray/json')); ..."`

### Applied upgrade strategy
- Upgraded repo runtime target to **Python 3.12.x** (latest Ray-2.54-compatible on Windows).
- Upgraded RL dependency to **Ray/RLlib 2.54.0**.
- Recreated `.venv` on Python 3.12 and resynced dependencies.
- Aligned static tooling target to Python 3.12 (`[tool.ruff] target-version = "py312"`).

### Commands and results
1. `uv venv --clear --python 3.12 .venv` -> pass (`Python 3.12.12`)
2. `uv sync --active --all-extras` -> pass
3. `uv run python -c "import ray,sys; print(ray.__version__, sys.version.split()[0])"` -> pass (`2.54.0`, `3.12.12`)

### Scripted controller replacement
- Replaced `--autodrive` with `--controller scripted`.
- Added deterministic controller implementation in `src/f1rl/controllers.py`.
- Controller is deterministic and stable for longer smoke horizons.

### Validation
1. `uv run python -m f1rl.manual --headless --controller scripted --max-steps 80` -> pass (`steps=80`)
2. `uv run python -m f1rl.manual --headless --controller scripted --max-steps 120` -> pass (`steps=120`)

### Artifact cleanup command
- Added `src/f1rl/clean_artifacts.py`
- Added CLI script:
  - `f1-clean-artifacts`
  - `python -m f1rl.clean_artifacts`
- Retention policy features:
  - keep latest N runs per prefix (`--keep-runs-per-prefix`)
  - keep recent runs/files by age (`--keep-days`)
  - keep recent render file count (`--keep-render-files`)
  - optional prefix scoping (`--prefix`)
  - safe default dry-run; execute with `--apply`
  - optional JSON output (`--json`)
  - protected runs via marker files (`.pin`, `KEEP`, `.keep`)

### Cleanup validation
1. `uv run python -m f1rl.clean_artifacts --keep-runs-per-prefix 3 --keep-days 14 --json` -> pass
2. Apply-mode smoke (temp dir via Python snippet) -> pass (deleted only eligible dirs)

## Current Working Commands
- Setup: `uv sync --active --all-extras`
- Manual: `uv run python -m f1rl.manual`
- Scripted smoke: `uv run python -m f1rl.manual --headless --controller scripted --max-steps 80`
- Train smoke: `uv run python -m f1rl.train --mode smoke`
- Resume smoke: `uv run python -m f1rl.train --mode smoke --resume latest --iterations 1`
- Eval: `uv run python -m f1rl.eval --checkpoint latest --headless --steps 150`
- Rollout checkpoint: `uv run python -m f1rl.rollout --policy checkpoint --checkpoint latest --headless --steps 120`
- Validate: `uv run python -m f1rl.validate`
- Cleanup dry-run: `uv run python -m f1rl.clean_artifacts --keep-runs-per-prefix 3 --keep-days 14`

## Known Limitations
1. Ray 2.54.0 still emits internal deprecation warnings for some logger/trainable internals; workflows are functional.
2. `python 3.14.x + ray 2.54.0` is not currently reproducible on Windows due missing wheels for this combination.

## Post-upgrade Revalidation (Current Session)
1. `uv run python -c "import sys; print(sys.version)"` -> pass (`3.12.12`)
2. `uv run python -c "import ray; print(ray.__version__)"` -> pass (`2.54.0`)
3. `uv run python -m f1rl.validate` -> pass (`[validate] all checks passed`)
4. `uv run ruff check .` -> pass
5. `uv run pyright src/f1rl` -> pass
6. `uv run pytest` -> pass (`13 passed`)
7. `uv run python manual_mode.py --headless --controller scripted --max-steps 20` -> pass (`steps=20`)
8. `uv run f1-clean-artifacts --keep-runs-per-prefix 2 --keep-days 7 --json` -> pass
