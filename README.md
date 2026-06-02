# F1 Reinforcement Learning

Top-down 2D Monza driving simulator and reinforcement-learning project with manual driving, telemetry, replay, Gymnasium, Stable-Baselines3 PPO, Fast-F1 reference data, and benchmark/QC tooling.

The old complex implementation is archived in `archive/legacy-20260424/`. The active runtime is intentionally explicit:

```text
track geometry -> car physics -> simulator -> manual/scripted -> telemetry -> Gymnasium -> PPO -> eval/replay
```

## Current State

Implemented and verified:

- Monza track preprocessing and persisted `TrackSpec`
- 2D bicycle-style physics with dynamic grip, aero grip, traction/braking limits, drag, rolling resistance, and speed-sensitive steering
- strict-enough Monza checkpoint/lap validity
- collision and off-track termination
- forward ray-cast sensors
- manual Pygame driving
- Fast-F1 reference ghost overlay and flying-start comparison
- deterministic scripted pure-pursuit baseline
- per-step JSONL telemetry and rich episode summaries
- replay with timestamp-based playback, speed controls, and untimed playback
- Gymnasium-compatible env
- Stable-Baselines3 PPO training/eval/checkpoints
- curriculum/segment spawning
- benchmark harness
- QC report/dashboard generator
- CUDA hardware policy reporting
- TensorBoard scalar logging

Not complete yet:

- PPO clean/valid full lap
- overnight `1M+` / `3M+` serious PPO training
- final learning-curve plots
- replay video export
- final polished recruiting results section
- evolutionary search

## Requirements

- Python `>=3.11,<3.13`
- `uv`
- Windows primary local target
- NVIDIA CUDA is used for PyTorch policy training/inference when available

The local validated runtime used:

- Python `3.12.12`
- PyTorch `2.10.0+cu128`
- CUDA `12.8`
- NVIDIA GeForce RTX 4060 Laptop GPU

## Setup

From PowerShell:

```powershell
cd "C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning"
uv sync --active --all-extras --all-packages
```

For repeated commands after the environment is already synced, prefer:

```powershell
uv run --no-sync python -m f1rl.hardware --json
```

`--no-sync` avoids Windows venv reinstall churn.

If console scripts ever get stale, use `python -m f1rl.<module>` commands.

## Compute Policy

```powershell
uv run --no-sync python -m f1rl.hardware --json
```

Policy:

- CPU: simulator stepping, physics, geometry, rendering, keyboard input, telemetry/logging, track preprocessing, vector env workers
- GPU: PyTorch policy training and inference when CUDA is available

Stable-Baselines3 PPO with small `MlpPolicy` observations may not saturate the GPU. That is expected. CUDA-required runs must still resolve model training/inference to CUDA.

## Track Build

```powershell
uv run --no-sync python -m f1rl.track_build
```

Outputs:

- `assets/tracks/monza/track_spec.npz`
- `assets/tracks/monza/track_manifest.json`

Current track metrics:

- checkpoints: `120`
- boundary segments: `1800`
- source image size after active scaling: `1894x956`
- real length anchor: `5793m`

## Manual Driving

```powershell
uv run --no-sync python -m f1rl.manual
```

Controls:

- `W` / Up: throttle
- `S` / Down: brake
- `A` / Left: steer left
- `D` / Right: steer right
- `R`: reset
- `Esc`: quit

Reference ghost overlay:

```powershell
uv run --no-sync python -m f1rl.manual --ghost-reference
uv run --no-sync python -m f1rl.manual --ghost-reference --flying-start
```

The Fast-F1 reference is a flying qualifying lap and starts near `322 kph`. Use `--flying-start` for fair visual comparison.

Headless manual smoke:

```powershell
uv run --no-sync python -m f1rl.manual --headless --max-steps 60
```

## Scripted Baseline

```powershell
uv run --no-sync python -m f1rl.scripted --steps 18000 --no-telemetry
```

Validated result:

- `lap_complete`
- valid lap: true
- lap time: `214.47s`
- average speed: `96.0 kph`
- max speed: `107.9 kph`

The scripted controller is a conservative centerline/pure-pursuit baseline. It is intentionally slow and safe; it is not a race-pace policy.

## Fast-F1 Reference Ghost

Generate a reference ghost from local Fast-F1 telemetry:

```powershell
uv run --no-sync python -m f1rl.reference_agent --mode ghost
```

Validated reference:

- source: 2024 Italian GP qualifying, VER fastest lap
- lap time: `79.662s`
- average speed: `259.9 kph`
- max speed: `348.0 kph`

Diagnostic physical chase mode:

```powershell
uv run --no-sync python -m f1rl.reference_agent --mode control --steps 7200
```

Ghost mode is the perfect telemetry replay baseline. Control mode is only a diagnostic physical controller.

## Replay

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\<run>\steps.jsonl"
uv run --no-sync python -m f1rl.replay "artifacts\<run>\steps.jsonl" --speed 2
uv run --no-sync python -m f1rl.replay "artifacts\<run>\steps.jsonl" --no-timing
uv run --no-sync python -m f1rl.replay "artifacts\<run>\steps.jsonl" --headless
```

Replay respects telemetry timestamps by default.

## Train PPO

Short CUDA smoke:

```powershell
uv run --no-sync python -m f1rl.train --timesteps 2048 --n-envs 2 --max-steps 600 --device auto --require-gpu --vec-env subproc --curriculum segments --checkpoint-every 1024 --eval-every 1024 --eval-episodes 1 --telemetry selected --run-name manual-qc-smoke
```

Validated smoke artifact:

- `artifacts/manual-qc-smoke-20260602-155050`
- `device=cuda`
- `vec_env=subproc`
- training FPS: `51.15`
- env steps/sec: `102.31`
- wrote `initial_model.zip`, `final_model.zip`, checkpoints, eval metrics, selected telemetry, and TensorBoard logs

Serious curriculum run:

```powershell
uv run --no-sync python -m f1rl.train --timesteps 1000000 --n-envs 8 --max-steps 3600 --device auto --require-gpu --vec-env subproc --curriculum segments --checkpoint-every 50000 --eval-every 50000 --eval-episodes 5 --telemetry selected --run-name ppo-curriculum-serious-scratch
```

## Evaluate

```powershell
uv run --no-sync python -m f1rl.eval --checkpoint latest --steps 600 --device auto
```

Use explicit checkpoint paths for reported results.

## Benchmark

Random/scripted/reference:

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

## QC Dashboard

```powershell
uv run --no-sync python -m f1rl.qc --telemetry "artifacts\benchmark-20260602-154555\selected_telemetry\ppo-episode-000-steps.jsonl" --run-scripted --scripted-steps 18000
```

Writes:

- `qc_report.json`
- `qc_report.md`
- `telemetry_dashboard.html`
- `manual_qc_checklist.md`

The dashboard plots speed, progress, reward, racing-line deviation, and minimum ray distance.

## TensorBoard

```powershell
uv run --no-sync tensorboard --logdir "C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts" --host 127.0.0.1 --port 6006
```

Open:

```powershell
Start-Process "http://127.0.0.1:6006"
```

Validated screenshot:

- `artifacts/tensorboard-qc-20260602-155050.png`

## Validation

```powershell
uv run --no-sync ruff check .
uv run --no-sync pyright src/f1rl
uv run --no-sync pytest -q
uv run --no-sync python -m f1rl.hardware --json
```

Latest validation:

- `ruff`: passed
- `pyright`: `0` errors
- `pytest`: `29` passed

Known benign warning:

- Stable-Baselines3 `VecMonitor` warning in PPO smoke tests.

## Current Results

| Policy / checkpoint | Episodes | Valid lap | Avg progress | Best progress | Checkpoints | Termination | Avg reward | Artifact |
|---|---:|---:|---:|---:|---:|---|---:|---|
| PPO scratch initial | 20 | 0.0 | `0.0m` | `0.0m` | 0 | no-progress | -90.0 | `artifacts/benchmark-20260602-154222` |
| PPO final | 20 | 0.0 | `431.4m` | `431.4m` | 8 | off-track | -25.49 | `artifacts/benchmark-20260602-154253` |
| PPO best | 20 | 0.0 | `797.6m` | `797.6m` | 16 | collision | 3.81 | `artifacts/benchmark-20260602-154555` |
| Scripted baseline | 1 | 1.0 | `5800.3m` | `5800.3m` | 119 | lap-complete | 564.0 | `artifacts/benchmark-20260602-151142` |
| Fast-F1 reference ghost | 1 | 1.0 | `5793.0m` | `5793.0m` | 119 | lap-complete | 563.44 | `artifacts/benchmark-20260602-153332` |

Interpretation:

- The current PPO pipeline has a real learning signal.
- The best current PPO checkpoint visibly drives farther than scratch/final and begins correcting before collision.
- PPO still has not completed a clean or valid lap.

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
- `f1-benchmark`
- `f1-qc`
