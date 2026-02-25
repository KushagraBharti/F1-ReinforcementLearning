# Documentation

## Run Date
- 2026-02-25

## Milestone 0 - Inspect and Plan
### Actions
- Inventoried legacy repository (`car.py`, `track.py`, `race.py`, `manual_mode.py`, `ray_training.py`, `ray_rollout.py`, `use_agent.py`, `render_agents.py`, `utils.py`, `imgs/*`).
- Captured baseline environment/tooling status.
- Created required durable-memory docs:
  - `Prompt.md`
  - `Plan.md`
  - `Implement.md`
  - `Documentation.md`

### Commands and Results
1. `git status --short` -> pass (`?? AGENTS.md`)
2. `rg --files` -> pass
3. `python --version` -> `Python 3.13.9` (host interpreter)
4. `uv --version` -> `uv 0.9.26`
5. Dependency probe (`ray/gymnasium/pygame/torch/cv2`) -> `ray=False, gymnasium=False, pygame=False, torch=True, cv2=True`

## Milestone 1 - Packaging and Structure Modernization
### Actions
- Added modern project packaging with `pyproject.toml` and generated `uv.lock`.
- Created package layout under `src/f1rl`.
- Added CLI entrypoints:
  - `f1-manual`
  - `f1-rollout`
  - `f1-train`
  - `f1-eval`
  - `f1-validate`
- Added lint/type/test tooling configuration (`ruff`, `pyright`, `pytest`).
- Added compatibility wrappers for legacy top-level scripts.

### Important Fixes During Milestone
- `ray` installation initially failed on Python 3.14 and on newer Ray builds without Windows wheels.
- Recreated `.venv` on Python 3.11 and pinned train extra to:
  - `ray[rllib]>=2.49.2,<2.50.0`
- Updated `requires-python` to `>=3.11,<3.14` for reproducibility in this repo.

### Commands and Results
1. `uv venv --clear --python 3.11 .venv` -> pass
2. `uv sync --active --all-extras` -> pass
3. `uv run python -c "import f1rl; print(f1rl.__version__)"` -> pass (`0.1.0`)
4. `uv run pytest -q tests/test_imports.py` -> pass
5. `Test-Path uv.lock` -> pass (`True`)

## Milestone 2 - Environment and Manual Mode
### Actions
- Replaced deprecated Gym API usage with Gymnasium-compatible API:
  - `reset(...) -> (obs, info)`
  - `step(...) -> (obs, reward, terminated, truncated, info)`
- Refactored core modules:
  - `config.py` (typed configs)
  - `geometry.py` (intersections/segments)
  - `track.py` (contour extraction + goal lines + collision vectors)
  - `dynamics.py` (physics + sensor observations)
  - `renderer.py` (pygame rendering, headless-safe)
  - `env.py` (new `F1RaceEnv`)
  - `manual.py` (interactive and headless smoke mode)
- Added deterministic artifact folder helpers and structured logging.

### Runtime Fixes During Milestone
- Fixed headless render crash: `pygame.error: No video mode has been set` by avoiding `convert_alpha()` when no display surface exists.

### Commands and Results
1. `uv run pytest -q tests/test_env_api.py` -> pass
2. `uv run python -m f1rl.manual --headless --max-steps 60 --autodrive` -> pass (`manual_run_complete ...`)
3. `uv run python -m f1rl.rollout --steps 120 --headless --policy random` -> pass

## Milestone 3 - RLlib Training, Checkpoints, Inference
### Actions
- Replaced deprecated legacy training/eval scripts with modern entrypoints:
  - `src/f1rl/train.py`
  - `src/f1rl/eval.py`
  - `src/f1rl/rollout.py`
- Migrated to `PPOConfig` and new RLlib API stack defaults.
- Removed hard dependency on `ray.tune.registry.register_env` path to avoid blocked pandas import route.
- Added checkpoint discovery and `latest` resolution for modern checkpoint layout.
- Added modern inference path:
  - RLModule `forward_inference` + distribution sampling
  - fallback to legacy compute API if needed
- Added smoke metrics artifacts (`metrics.csv`, `metrics.jsonl`) and run metadata.
- Produced frame artifacts from eval (`artifacts/renders/eval_*.png`).

### Runtime Fixes During Milestone
- Fixed training metadata serialization error (`Path` not JSON-serializable) via config sanitizer (`to_dict`).
- Fixed eval failure on new API stack (`SingleAgentEnvRunner has no get_policy`) by switching inference to RLModule API.

### Commands and Results
1. `uv run python -m f1rl.train --mode smoke` -> pass
2. `uv run python -m f1rl.train --mode smoke --resume latest --iterations 1` -> pass
3. `uv run python -m f1rl.eval --checkpoint latest --steps 150 --headless` -> pass
4. `uv run python -m f1rl.rollout --policy checkpoint --checkpoint latest --headless --steps 120` -> pass
5. `uv run python -m f1rl.eval --checkpoint latest --headless --export-frames --steps 30` -> pass

## Milestone 4 - Tests, Lint, Type Checks, Validation, Docs
### Actions
- Added tests:
  - `tests/test_imports.py`
  - `tests/test_geometry.py`
  - `tests/test_dynamics.py`
  - `tests/test_env_api.py`
  - `tests/test_cli_smoke.py`
  - `tests/test_checkpoint_roundtrip.py`
- Added end-to-end validator (`python -m f1rl.validate`).
- Rewrote `README.md` with setup/run/train/eval/troubleshooting/architecture.

### Commands and Results
1. `uv run ruff check .` -> pass
2. `uv run pyright src/f1rl` -> pass (`0 errors`)
3. `uv run pytest -q` -> pass (`9 passed`)
4. `uv run python -m f1rl.validate` -> pass (`[validate] all checks passed`)

## API Migration Notes (with sources)
1. Gymnasium API migration
- Change: moved to Gymnasium reset/step signatures and env checker workflow.
- Source: https://gymnasium.farama.org/api/env/
- Source: https://gymnasium.farama.org/api/utils/#gymnasium.utils.env_checker.check_env

2. RLlib modern config and new API stack
- Change: use `PPOConfig`-based algorithm construction and new API stack behavior.
- Source: https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo
- Source: https://docs.ray.io/en/latest/rllib/algorithm-config.html
- Source: https://docs.ray.io/en/master/rllib/new-api-stack-migration-guide.html

3. uv-managed setup/lock workflow
- Change: standardized install workflow around `uv sync --active --all-extras` and lockfile.
- Source: https://docs.astral.sh/uv/concepts/projects/sync/
- Source: https://docs.astral.sh/uv/concepts/projects/layout/

4. Pygame rendering/runtime
- Change: headless-safe renderer initialization and frame export support.
- Source: https://www.pygame.org/docs/

## Current Artifacts (successful runs)
- Smoke training run example:
  - `artifacts/train-smoke-20260225-045653/`
  - contains checkpoint + metrics + logs + metadata
- Eval summary example:
  - `artifacts/eval-20260225-045535/eval_summary.json`
- Frame artifact example:
  - `artifacts/renders/eval_000000.png` (and subsequent frames)

## Known Limitations
1. RLlib emits deprecation warnings from internal logger/checkpoint internals on 2.49.x; functionality is intact.
2. Multiple smoke artifacts can accumulate over repeated runs; cleanup policy is currently manual.
3. Manual mode smoke (`--autodrive`) is intentionally simple and may collide early depending on policy/track state.
