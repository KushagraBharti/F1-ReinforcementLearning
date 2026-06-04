# Evolution Goal Prompt

This file is retained for historical context. The active goal prompt is now `goal.md`, and the active implementation loop is `workflow.md`.

Read those first. The older Rettifilo-first language below is superseded by the current speed-focused objective: use the evolutionary loop to produce a valid normal-start Monza lap under `80.0s`.

Stop the old PPO micro-engineering loop and use `EvolutionSearchPlan.md` as the active operating plan.

The original final target still matters: train a scratch/random SB3 PPO agent that completes a valid normal-start Monza lap near `<=80.0s`. But the immediate method changes. PPO has not made honest normal-start progress beyond the old `~970.775m` failure after hours of narrow curriculum work, so do not keep adding one-off micro-rungs and schedule presets.

Use Yosh-style elitist evolutionary search as the discovery engine. Spawn a large population of candidate driving controllers/action schedules, run them through the simulator, score them with strict progress/speed/line/yaw/steering/no-collision gates, keep elites, mutate/cross over action phases or controller weights, add random immigrants, repeat for generations, and save elite telemetry/state libraries.

Use the implemented tools:

```powershell
uv run --no-sync python -m f1rl.evolution_search --help
uv run --no-sync python -m f1rl.evolution_ladder --help
```

Prefer the ladder for repeatable work. Start at `smoke`, then `small`, then `medium`, then long `large` runs only after artifacts and replay behavior make sense. This is different from the old fixed `action_search.py`; do not fall back into manually enumerating tiny schedule presets unless it is a short diagnostic. The main loop should be population search, elite retention, mutation, checkpoint/resume, and repeated generations.

First solve Rettifilo transfer by search: `520->650`, `650->900`, `900->1000`, `1000->1220`, `520->1220`, then `0->1220`. Keep returning to honest normal-start/full-lap probes so the main objective stays visible. After search finds recoverable elites, train PPO from those curriculum starts and test transfer back to honest normal-start eval. If search cannot find viable elites, change the genome type, controller representation, scoring profile, starts, or gates instead of spending more time on tiny PPO guesses.

Do not count evolutionary/scripted/search trajectories as final success and do not initialize final scratch PPO policy weights from them. Use search to discover behavior, states, missing primitives, and curriculum targets. Final success still requires honest metadata-faithful PPO normal-start completion with scaffolds and assists disabled.

Be direct and progress-focused. Do not micro-engineer. Do not polish isolated completions. Do not treat runtime or artifacts as progress. The next proof must be search-discovered linked section transfer, then PPO reproduction, then honest normal-start improvement.
