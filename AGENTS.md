# AGENTS.md

## Mission
This repository is a top-down 2D Pygame + RL project for an F1-style driving environment. The codebase is deprecated and must be fully modernized, refactored, tested, and made reproducible.

## Non-negotiable working style
- Operate autonomously in milestone loops:
  1) inspect
  2) plan
  3) implement
  4) run validations
  5) fix failures
  6) update docs/status
  7) repeat
- Do not leave placeholders or TODOs for core functionality.
- If a validation fails, fix it before moving on.
- Prefer small, scoped diffs with explicit validation after each milestone.
- Keep a running audit log in `Documentation.md`.

## Required deliverables
- Modern Python project structure with `pyproject.toml` (uv-managed) and `uv.lock`
- Working manual gameplay mode (keyboard control)
- Working RL training entrypoint (modern APIs; no deprecated RLlib/Ray usage)
- Working agent inference/render entrypoint
- Smoke tests and unit tests
- Clear README with setup, run, train, and troubleshooting
- Logging + checkpoints + basic artifacts from a smoke training run
- Reproducible commands for all workflows

## Tech expectations
- Use Python 3.11+
- Use `uv` for dependency management (`uv sync --active --all-extras --all-packages`)
- Prefer Gymnasium-compatible environment APIs
- Prefer modern RLlib API stack if using Ray RLlib
- Use PyTorch (not TensorFlow) for RLlib
- Keep rendering and training logic decoupled
- Add structured logging

## Validation requirements
- Environment API checks (Gymnasium env checker)
- Manual play smoke test
- Fast headless smoke rollout
- RL training smoke test (very short)
- Checkpoint save/load smoke test
- Inference rollout smoke test
- Lint + tests + type checks where practical

## Documentation requirements
Maintain these files during the run:
- `Prompt.md` (spec)
- `Plan.md` (milestones + acceptance criteria + commands)
- `Implement.md` (runbook)
- `Documentation.md` (live status, decisions, validations, commands, issues)

## Internet use
Use live web search to verify current APIs/versions for:
- Ray RLlib
- Gymnasium
- Pygame
- uv
- Any replaced/deprecated packages
Document major API migrations in `Documentation.md`.