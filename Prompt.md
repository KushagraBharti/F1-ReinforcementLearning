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
