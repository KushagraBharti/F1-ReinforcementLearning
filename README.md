# F1 Reinforcement Learning

Simplified top-down 2D Monza driving simulator with manual Pygame driving, Gymnasium, telemetry, and PPO.

The old complex implementation has been archived in `archive/legacy-20260424/`. The active project is intentionally small:

```text
track geometry -> car physics -> simulator -> manual/scripted -> telemetry -> Gymnasium -> PPO -> eval/replay
```

## Requirements

- Python `>=3.11,<3.13`
- `uv`
- Windows is the primary local target
- NVIDIA CUDA is used for PyTorch training/inference when available

## Setup

```powershell
$env:UV_LINK_MODE='copy'
uv sync --active --all-extras --all-packages
```

## Build Track

```powershell
uv run python -m f1rl.track_build
```

This writes the persisted Monza `TrackSpec` under `assets/tracks/monza/`.

## Manual Driving

```powershell
uv run python -m f1rl.manual
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
uv run python -m f1rl.manual --headless --max-steps 60
```

## Scripted Baseline

```powershell
uv run python -m f1rl.scripted --steps 600
```

## Fast-F1 Reference Agent

Exact ghost lap from the 2024 Italian GP qualifying VER telemetry:

```powershell
uv run f1-reference-agent --mode ghost
```

Physics-control chase of the same reference profile:

```powershell
uv run f1-reference-agent --mode control --steps 7200
```

The ghost mode writes normal telemetry and can be replayed with `f1-replay`. It is the "perfect reference" baseline; control mode is a simulator-controller diagnostic.

## Train

```powershell
uv run python -m f1rl.train --timesteps 512 --n-envs 2 --device auto
```

Training uses Stable-Baselines3 PPO. The simulator runs on CPU; the policy trains on CUDA when PyTorch can see the GPU.

## Evaluate

```powershell
uv run python -m f1rl.eval --checkpoint latest --steps 600 --device auto
```

## Replay

```powershell
uv run python -m f1rl.replay artifacts\<run>\steps.jsonl --headless
uv run python -m f1rl.replay artifacts\<run>\steps.jsonl
uv run python -m f1rl.replay artifacts\<run>\steps.jsonl --speed 2
```

Replay respects telemetry timestamps by default. Use `--speed` for intentional faster/slower playback, or `--no-timing` to scrub one telemetry row per frame.

## Validate

```powershell
uv run ruff check .
uv run pyright src/f1rl
uv run pytest -q
uv run python -m f1rl.hardware --json
uv run python -m f1rl.calibration
```

## Active CLI

- `f1-build-track`
- `f1-manual`
- `f1-scripted`
- `f1-train`
- `f1-eval`
- `f1-replay`
- `f1-hardware-check`
- `f1-calibration`
- `f1-reference-agent`
