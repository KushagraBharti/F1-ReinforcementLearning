# F1 Reinforcement Learning (Modernized)

Top-down 2D F1-style driving simulator with:
- manual keyboard gameplay (Pygame),
- Gymnasium-compatible RL environment,
- modern RLlib (PyTorch) training/inference,
- reproducible `uv` workflow,
- smoke/unit tests and validation script.

## Requirements
- OS: Windows/Linux/macOS (project developed on Windows)
- Python: `3.11 - 3.13`  
  `Ray` currently does not provide reliable wheels for Python `3.14+` in this setup.
- Conda (recommended)
- `uv` package manager

## Setup
1. Create and activate a conda env:
```powershell
conda create -n f1rl python=3.11 -y
conda activate f1rl
```

2. Install `uv` (if needed):
```powershell
pip install uv
```

3. Sync dependencies:
```powershell
uv sync --active --all-extras
```

## Run Commands

### Manual Mode (keyboard)
```powershell
uv run python -m f1rl.manual
```
Controls:
- accelerate: `Up` / `W`
- brake: `Down` / `S`
- steer: `Left` / `A`, `Right` / `D`
- quit: window close or `Esc`

Headless manual smoke:
```powershell
uv run python -m f1rl.manual --headless --autodrive --max-steps 80
```

### Smoke Training
```powershell
uv run python -m f1rl.train --mode smoke
```

Resume from latest checkpoint:
```powershell
uv run python -m f1rl.train --mode smoke --resume latest
```

### Full Training Template
```powershell
uv run python -m f1rl.train --mode full --iterations 60 --num-env-runners 1
```

### Evaluate / Render a Trained Agent
Headless eval:
```powershell
uv run python -m f1rl.eval --checkpoint latest --headless --steps 500
```

Rendered eval:
```powershell
uv run python -m f1rl.eval --checkpoint latest --render --steps 500
```

Checkpoint rollout:
```powershell
uv run python -m f1rl.rollout --policy checkpoint --checkpoint latest --headless --steps 300
```

Random rollout smoke:
```powershell
uv run python -m f1rl.rollout --policy random --headless --steps 120
```

## Validation and Quality Gates

Lint:
```powershell
uv run ruff check .
```

Type checks:
```powershell
uv run pyright src/f1rl
```

Tests:
```powershell
uv run pytest -q
```

End-to-end validation:
```powershell
uv run python -m f1rl.validate
```

## Artifacts and Outputs
Artifacts are written under:
- `artifacts/train-smoke-*/`  
  `checkpoints/`, `metrics/metrics.csv`, `metrics/metrics.jsonl`, `logs/train.log`
- `artifacts/eval-*/eval_summary.json`
- `artifacts/renders/` (if frame export enabled)

## Architecture

### Package Layout
```text
src/f1rl/
  env.py             # Gymnasium environment
  dynamics.py        # car physics + observations
  track.py           # contour extraction + checkpoints + collision geometry
  renderer.py        # pygame renderer
  train.py           # RLlib PPO training entrypoint
  eval.py            # checkpoint evaluation entrypoint
  rollout.py         # random/checkpoint rollout runner
  validate.py        # end-to-end validation script
  artifacts.py       # artifact/checkpoint path helpers
  logging_utils.py   # structured logging
  legacy.py          # compatibility wrapper for old Race API
```

### Legacy Wrappers
Old script names still work and delegate to modern modules:
- `manual_mode.py`
- `ray_training.py`
- `ray_rollout.py`
- `use_agent.py`
- `render_agents.py`
- `race.py`

## Troubleshooting

### `ray` install fails with Python 3.14
Use Python 3.11-3.13. Recreate env and rerun:
```powershell
conda create -n f1rl python=3.11 -y
conda activate f1rl
uv sync --active --all-extras
```

### No window in manual mode
- Ensure you are not using `--headless`.
- Check local graphics/display access.

### RLlib warning spam about deprecations
Ray emits deprecation warnings for some internal loggers/checkpoints in 2.49.x.  
Training and evaluation still work; warnings are expected for this version line.

## Required Project Memory Docs
See:
- `Prompt.md`
- `Plan.md`
- `Implement.md`
- `Documentation.md`
