# Implement

Run all commands from:

```powershell
cd "C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning"
```

Use `uv run --no-sync ...` for repeated local commands after the environment is installed.

## Setup

```powershell
uv sync --active --all-extras --all-packages
```

If the editable project package is stale:

```powershell
uv pip install --no-deps -e .
```

The project targets Python `>=3.11,<3.13`. The validated local runtime uses Python `3.12.12`.

## Environment Checks

```powershell
uv run --no-sync python -c "import f1rl, pygame, gymnasium, torch; print('ok', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
uv run --no-sync python -m f1rl.hardware --json
```

Expected:

- imports succeed
- CUDA is visible on the RTX 4060 Laptop GPU
- hardware policy reports PyTorch training/inference on CUDA and sim/render/telemetry on CPU

## Validation

```powershell
uv run --no-sync ruff check .
uv run --no-sync pyright src/f1rl
uv run --no-sync pytest -q
```

Latest verified result:

- `ruff`: passed
- `pyright`: `0` errors
- `pytest`: `29` passed

## Track Build

```powershell
uv run --no-sync python -m f1rl.track_build
```

Outputs:

- `assets/tracks/monza/track_spec.npz`
- `assets/tracks/monza/track_manifest.json`

## Manual Mode

```powershell
uv run --no-sync python -m f1rl.manual
uv run --no-sync python -m f1rl.manual --ghost-reference
uv run --no-sync python -m f1rl.manual --ghost-reference --flying-start
```

Controls:

- `W` / Up: throttle
- `S` / Down: brake
- `A` / Left: steer left
- `D` / Right: steer right
- `R`: reset
- `Esc`: quit

Headless smoke:

```powershell
uv run --no-sync python -m f1rl.manual --headless --max-steps 60
```

Manual mode currently works visually. Known issue: FPS can be improved by optimizing render frame capture, sprite caching, static surface caching, and telemetry buffering.

## Scripted Baseline

```powershell
uv run --no-sync python -m f1rl.scripted --steps 18000 --no-telemetry
```

Expected:

- `reason=lap_complete`
- valid slow lap around `214.47s`

Generate telemetry and replay:

```powershell
uv run --no-sync python -m f1rl.scripted --steps 18000
uv run --no-sync python -m f1rl.replay "artifacts\<scripted-run>\steps.jsonl"
```

## Fast-F1 Reference Ghost

```powershell
uv run --no-sync python -m f1rl.reference_agent --mode ghost
```

Expected:

- target lap: `79.662s`
- average speed: `259.9 kph`
- max speed: `348.0 kph`

Replay:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\<reference-ghost-run>\steps.jsonl"
```

Diagnostic control chase:

```powershell
uv run --no-sync python -m f1rl.reference_agent --mode control --steps 7200
```

## Benchmark

Reference baselines:

```powershell
uv run --no-sync python -m f1rl.benchmark --policies random --episodes 20 --max-steps 3600 --telemetry selected --telemetry-every 5
uv run --no-sync python -m f1rl.benchmark --policies scripted --episodes 1 --max-steps 18000 --telemetry selected --telemetry-every 1
uv run --no-sync python -m f1rl.benchmark --policies reference_ghost --episodes 1 --max-steps 18000 --telemetry selected --telemetry-every 1
```

Current PPO checkpoints:

```powershell
uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "artifacts\ppo-curriculum-balanced-scratch-20260602-085522\initial_model.zip" --episodes 20 --max-steps 3600 --device auto --telemetry selected
uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "artifacts\ppo-curriculum-balanced-scratch-20260602-085522\final_model.zip" --episodes 20 --max-steps 3600 --device auto --telemetry selected
uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "artifacts\ppo-curriculum-balanced-scratch-20260602-085522\best_model.zip" --episodes 20 --max-steps 3600 --device auto --telemetry selected
```

Artifacts:

- `summary.json`
- `summary.csv`
- `per_episode.jsonl`
- selected telemetry JSONL

## Replay PPO Telemetry

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\benchmark-20260602-154222\selected_telemetry\ppo-episode-000-steps.jsonl"
uv run --no-sync python -m f1rl.replay "artifacts\benchmark-20260602-154253\selected_telemetry\ppo-episode-000-steps.jsonl"
uv run --no-sync python -m f1rl.replay "artifacts\benchmark-20260602-154555\selected_telemetry\ppo-episode-000-steps.jsonl"
```

Expected:

- initial PPO: no movement
- final PPO: drives straight, exits track
- best PPO: drives straight, begins correcting, collides later

## Train PPO

CUDA smoke:

```powershell
uv run --no-sync python -m f1rl.train --timesteps 2048 --n-envs 2 --max-steps 600 --device auto --require-gpu --vec-env subproc --curriculum segments --checkpoint-every 1024 --eval-every 1024 --eval-episodes 1 --telemetry selected --run-name manual-qc-smoke
```

Serious curriculum:

```powershell
uv run --no-sync python -m f1rl.train --timesteps 1000000 --n-envs 8 --max-steps 3600 --device auto --require-gpu --vec-env subproc --curriculum segments --checkpoint-every 50000 --eval-every 50000 --eval-episodes 5 --telemetry selected --run-name ppo-curriculum-serious-scratch
```

Training writes:

- `run_metadata.json`
- `initial_model.zip`
- `final_model.zip`
- checkpoint files
- TensorBoard event files
- `eval/eval_metrics.jsonl`
- `eval/best_eval_summary.json`
- selected telemetry

## Evaluate

```powershell
uv run --no-sync python -m f1rl.eval --checkpoint latest --steps 600 --device auto
```

For reported metrics, prefer explicit checkpoint paths over `latest`.

## QC Report

```powershell
uv run --no-sync python -m f1rl.qc --telemetry "artifacts\benchmark-20260602-154555\selected_telemetry\ppo-episode-000-steps.jsonl" --run-scripted --scripted-steps 18000
```

Outputs:

- `qc_report.json`
- `qc_report.md`
- `telemetry_dashboard.html`
- `manual_qc_checklist.md`

## TensorBoard

```powershell
uv run --no-sync tensorboard --logdir "C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts" --host 127.0.0.1 --port 6006
Start-Process "http://127.0.0.1:6006"
```

If TensorBoard fails with missing `pkg_resources`, repair setuptools:

```powershell
uv pip install --no-deps setuptools==80.9.0
```

## Documentation

After any meaningful implementation, validation, benchmark, training, or blocker event, update:

- `README.md`
- `Plan.md`
- `Implement.md`
- `Documentation.md`

Archived plans live under:

- `archive/plans/`
