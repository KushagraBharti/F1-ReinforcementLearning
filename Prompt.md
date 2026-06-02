# Prompt

Rebuild this repository as a simplified top-down 2D Monza driving simulator and reinforcement learning project.

The active implementation must be small and explicit:

1. Track geometry
2. Car physics
3. Shared simulator
4. Manual mode
5. Telemetry
6. Gymnasium environment
7. PPO training/eval
8. Replay

The old codebase is archived under `archive/legacy-20260424/` and is reference-only. Active v1 excludes Ray, RLlib, imitation learning, campaign orchestration, swarm tooling, image observations, custom torch-native multi-car simulation, distributed training, and evolutionary search.

Future evolutionary search should reuse the same simulator, observation/action contract, telemetry schema, checkpoint/eval boundary, and replay artifacts.

Active extras beyond the original minimal v1 are allowed when they serve calibration or debugging without reintroducing orchestration complexity; current examples are Fast-F1 calibration/reference ghost tooling and hardware policy reporting.

The implemented extensions are intentional improvements over the baseline plan, not items to remove. They include `f1rl.calibration`, `f1rl.reference_agent`, `f1-calibration`, `f1-reference-agent`, richer physics, richer telemetry/episode summaries, continuous-control scripted driving through the same simulator, manual reference ghost overlay, flying-start comparison, timestamp-interpolated replay, and explicit CPU/GPU policy reporting.

Current state: the simulator/manual/replay/telemetry/Gymnasium/PPO smoke pipeline is implemented and validated. The next major product gap is RL learning quality: curriculum training, segment/checkpoint spawning, and a policy that completes clean laps are not implemented yet.
