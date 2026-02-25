# Prompt

## Mission
Modernize and stabilize this repository into a reproducible Python project for:
- manual F1-style top-down driving (Pygame),
- Gymnasium-compatible RL environment,
- RLlib-based training (PyTorch),
- checkpointed inference and rendering,
- validated test and smoke workflows.

## Goals
1. Restore working manual gameplay with responsive keyboard control.
2. Refactor into a package-safe, maintainable project layout.
3. Replace deprecated Ray/RLlib usage with modern API patterns.
4. Add deterministic setup and execution via `uv` (`pyproject.toml` + `uv.lock`).
5. Add smoke/unit tests and validation scripts.
6. Add logging, checkpoints, and basic smoke-run artifacts.
7. Rewrite README for clone-to-run developer UX.

## Constraints
- Python 3.11+.
- Use `uv` for dependency management.
- Gymnasium API compliance (`reset -> (obs, info)`, `step -> (obs, reward, terminated, truncated, info)`).
- RL stack: RLlib + PyTorch.
- Rendering and training must stay decoupled.
- No placeholder TODOs for core features.
- Validation failures must be fixed before milestone completion.

## Done Criteria
- Manual mode launches and controls work.
- RL smoke training runs end-to-end and saves checkpoint.
- Checkpoint load and inference rollout work.
- Headless rollout and render smoke commands run.
- Tests and practical lint/type checks pass.
- README and required repo docs are updated and accurate.

