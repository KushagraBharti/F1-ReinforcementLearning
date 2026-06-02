# Implement

## Setup

PowerShell:

```powershell
$env:UV_LINK_MODE='copy'
uv sync --active --all-extras --all-packages
```

The project targets Python `>=3.11,<3.13`. The local `.venv` may use Python 3.12 while the code stays compatible with 3.11.

## Implemented Extensions

These features are intentional additions to the simplified rebuild, not unwanted drift:

- `f1-calibration` and `f1-reference-agent`
- Fast-F1 Monza reference ghost and physics-control diagnostic mode
- dynamic/aero grip, traction limits, and speed-sensitive steering
- expanded step telemetry and episode summaries
- manual reference ghost overlay and flying-start comparison
- timestamp-interpolated replay with playback speed controls
- explicit CPU/GPU compute policy reporting

Future evolutionary search remains deferred. It should reuse the same simulator, observation/action space, telemetry, replay, and policy boundaries instead of creating a separate runtime.

## Build Track

```powershell
uv run python -m f1rl.track_build
```

Outputs:

- `assets/tracks/monza/track_spec.npz`
- `assets/tracks/monza/track_manifest.json`

## Manual Mode

```powershell
uv run python -m f1rl.manual
uv run f1-manual --ghost-reference
uv run f1-manual --ghost-reference --flying-start
```

`--ghost-reference` overlays the Fast-F1 VER Monza reference lap in the same renderer. `--flying-start` starts the manual car at the reference lap entry speed so the comparison is not a standing start against a flying lap.

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
uv run python -m f1rl.scripted --steps 18000 --no-telemetry
```

The long no-telemetry run verifies the conservative baseline can finish a slow clean lap without paying full JSONL/raycast telemetry cost.

## Fast-F1 Reference Agent

Replay the real 2024 Italian GP qualifying VER profile as a perfect ghost lap on the simulator centerline:

```powershell
uv run f1-reference-agent --mode ghost
```

Try to chase the same speed profile through simulator physics and continuous controls:

```powershell
uv run f1-reference-agent --mode control --steps 7200
```

Replay either output:

```powershell
uv run f1-replay artifacts\<reference-run>\steps.jsonl
```

## Train PPO

```powershell
uv run python -m f1rl.train --timesteps 512 --n-envs 2 --device auto
uv run python -m f1rl.train --timesteps 512 --n-envs 2 --device auto --require-gpu
```

PyTorch uses CUDA when available. Environment stepping, geometry, rendering, and telemetry stay on CPU.

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

Replay uses each row's `sim_time_s` by default. `--speed 2` plays at 2x; `--no-timing` advances one row per rendered frame.

## Validate

```powershell
uv run ruff check .
uv run pyright src/f1rl
uv run pytest -q
uv run python -m f1rl.hardware --json
uv run python -m f1rl.calibration
uv run f1-scripted --steps 18000 --no-telemetry
```
