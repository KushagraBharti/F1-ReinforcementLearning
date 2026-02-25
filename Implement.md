# Implement

## Execution Runbook
1. Work in milestone loops:
   - inspect
   - plan/update acceptance checks
   - implement small scoped diff
   - run validations
   - fix failures
   - log status in `Documentation.md`
2. Keep rendering/gameplay decoupled from RL training code.
3. Keep all primary commands reproducible through `uv run ...`.
4. Prefer deterministic output locations:
   - `artifacts/checkpoints/`
   - `artifacts/metrics/`
   - `artifacts/renders/`
   - `artifacts/logs/`
5. Maintain compatibility shims for legacy script names when straightforward.
6. Avoid broad rewrites when a focused migration can preserve behavior.

## Technical Decisions (Current)
- Use package layout: `src/f1rl/`.
- Use Gymnasium API and environment checker.
- Use RLlib PPO with PyTorch for smoke and baseline training flows.
- Python baseline: `3.12.x` (latest Ray 2.54.0-compatible on Windows).
- Use standard-library logging with JSONL metrics exports.
- Use pytest + ruff + pyright for practical quality gates.

## Command Conventions
- Setup: `uv sync --active --all-extras`
- Manual: `uv run python -m f1rl.manual`
- Smoke train: `uv run python -m f1rl.train --mode smoke`
- Eval: `uv run python -m f1rl.eval --checkpoint latest`
- Cleanup: `uv run python -m f1rl.clean_artifacts`
- Full validation: `uv run python -m f1rl.validate`
