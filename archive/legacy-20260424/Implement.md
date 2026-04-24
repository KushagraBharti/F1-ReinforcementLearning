# Implement

## Runbook

### 1. Environment Setup
```powershell
conda activate f1rl
$env:UV_LINK_MODE='copy'
uv sync --active --all-extras
```

### 2. Baseline Validation
```powershell
uv run ruff check src/f1rl tests
uv run pyright src/f1rl
uv run pytest -q
uv run python -m f1rl.validate
```

### 3. Manual / Rollout Checks
```powershell
uv run python -m f1rl.manual --headless --controller scripted --max-steps 120
uv run python -m f1rl.rollout --policy random --headless --steps 120
uv run python -m f1rl.eval --checkpoint latest --headless --steps 300
```

### 4. Single-Car Training / Benchmark
```powershell
uv run python -m f1rl.train --mode smoke --device gpu --require-gpu
uv run python -m f1rl.benchmark --checkpoint latest --profile quick --promote-if-best
```

### 5. Stage-Aware Swarm Collection
```powershell
uv run python -m f1rl.train --mode smoke --swarm-stage competence --logical-cars 32 --device gpu --require-gpu
uv run python -m f1rl.train --mode benchmark --swarm-stage lap --logical-cars 64 --device gpu --require-gpu
```

### 6. Seed-Tiered Benchmarks
```powershell
uv run python -m f1rl.benchmark --checkpoint latest --profile quick --seed-tier promotion --no-clips
uv run python -m f1rl.benchmark --checkpoint latest --profile quick --seed-tier diagnostic --rotation-index 1 --no-clips
uv run python -m f1rl.benchmark --checkpoint latest --profile quick --seed-tier holdout --no-clips
```

### 7. Swarm Diagnostics
```powershell
uv run python -m f1rl.swarm --policy random --headless --cars 8 --steps 40
uv run python -m f1rl.swarm --policy checkpoint --checkpoint latest --cars 32 --swarm-stage competence --headless --export-clips
```

### 8. Autonomous Campaign
Short bounded smoke:
```powershell
uv run python -m f1rl.campaign --hours 0.02 --max-waves 1 --start-cars 8 --max-cars 8 --device auto
```

Overnight run:
```powershell
uv run python -m f1rl.campaign --hours 7 --device gpu
```

### 9. Cleanup
```powershell
uv run python -m f1rl.clean_artifacts --keep-runs-per-prefix 3 --keep-days 7 --apply
```

## Operational Notes
- Single-car benchmark remains the only champion promotion authority.
- Swarm traces and top-k trajectories are diagnostics only.
- Reward changes must stay isolated to reward-only waves in the campaign.
- Stage-aware training with large logical car counts is materially heavier on Windows than the old smoke path.

## Torch-Native Runtime Runbook

### Focused rewrite validation
```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_torch_runtime.py tests/test_cli_smoke.py tests/test_checkpoint_roundtrip.py tests/test_benchmark.py
.venv\Scripts\python.exe -m f1rl.train --mode smoke --iterations 1 --device auto
.venv\Scripts\python.exe -m f1rl.validate
```

### Live GPU renderer dependencies
Preferred path:
```powershell
$env:UV_LINK_MODE='copy'
uv sync --active --extra train --extra viz --extra dev
```

If OneDrive / Windows dist-info cleanup blocks `uv sync`, install only the missing live-render deps with:
```powershell
uv pip install --python .venv\Scripts\python.exe pyglet>=2.1.0 moderngl>=5.12.0
```

### Desktop-only live render smoke
Run these from the user desktop session, not a fragile remote/tooling shell:
```powershell
.venv\Scripts\python.exe -m f1rl.swarm --policy scripted --render --cars 8 --steps 120 --swarm-stage competence
.venv\Scripts\python.exe -m f1rl.manual --controller scripted --max-steps 120
```
