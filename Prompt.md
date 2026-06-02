# Prompt

Build and iterate on this repository as a simplified top-down 2D Monza simulator and reinforcement-learning project.

The active implementation path is:

```text
track geometry -> car physics -> shared simulator -> manual/scripted -> telemetry -> Gymnasium -> PPO -> eval/replay
```

The old complex implementation is archived under `archive/legacy-20260424/` and is reference-only. Do not reintroduce active Ray/RLlib, imitation learning, campaign orchestration, swarm tooling, image observations, distributed training, or custom torch-native multi-car simulation unless the user explicitly starts that later phase.

## Implemented State

The simulator/RL proof of concept is implemented and verified:

- explicit Monza track geometry and persisted `TrackSpec`
- 2D car physics with dynamic/aero grip and traction/braking limits
- shared simulator used by manual, scripted, replay, Gymnasium, PPO, eval, benchmark, and QC
- manual Pygame driving with reference ghost overlay and flying-start comparison
- deterministic scripted baseline
- Fast-F1 reference ghost baseline
- ray sensors and numeric observation space
- discrete action space
- stable reward schema
- checkpoint/lap validity fields
- JSONL telemetry and episode summaries
- benchmark harness
- curriculum/segment spawning
- Stable-Baselines3 PPO training/eval
- CUDA-required smoke training
- TensorBoard scalar logging
- QC dashboard/report generation

The implemented extensions are intentional improvements over the original minimal rebuild, not scope drift:

- `f1rl.calibration`
- `f1rl.reference_agent`
- `f1rl.benchmark`
- `f1rl.curriculum`
- `f1rl.qc`
- richer telemetry and episode summaries
- replay timestamp interpolation and speed controls
- explicit CPU/GPU compute policy reporting

## Current Gap

The current PPO agent has a measurable learning signal but no clean/valid full lap yet.

Verified PPO progression:

- scratch initial PPO: `0.0m`, no-progress
- final checkpoint: `431.4m`, off-track
- best checkpoint: `797.6m`, collision

The next major goal is serious PPO training, curriculum-to-full-lap transfer, and artifact-backed results.

## Rules

- Keep simulator, renderer, telemetry, and trainers decoupled.
- Keep manual/scripted/PPO/eval/replay on the same simulator path.
- Use CUDA for PyTorch policy training/inference when required and available.
- Keep simulator stepping, rendering, geometry, and telemetry CPU-bound.
- Preserve artifacts and logs when running experiments.
- Do not claim PPO completed a lap unless benchmark data proves it.
- Update `Documentation.md` after meaningful implementation, validation, training, benchmark, or blocker events.
