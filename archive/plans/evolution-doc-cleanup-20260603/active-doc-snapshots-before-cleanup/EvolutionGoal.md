# Evolution Goal Prompt

Stop the current PPO micro-engineering loop and use `EvolutionSearchPlan.md` as the active operating plan.

The original final target still matters: train a scratch/random SB3 PPO agent that completes a valid normal-start Monza lap near `<=80.0s`. But the immediate method changes. PPO has not made honest normal-start progress beyond the old `~970.775m` failure after hours of narrow curriculum work, so do not keep adding one-off micro-rungs and schedule presets.

Use Yosh-style elitist evolutionary search as the discovery engine. Spawn a large population of candidate driving controllers/action schedules, run them through the simulator, score them with strict progress/speed/line/yaw/steering/no-collision gates, keep elites, mutate/cross over action phases, add random immigrants, repeat for generations, and save elite telemetry/state libraries.

Use `src/f1rl/evolution_search.py` and the `f1-evolution-search` CLI. This is different from the old fixed `action_search.py`; do not fall back into manually enumerating tiny schedule presets unless it is a short diagnostic. The main loop should be population search, elite retention, mutation, and repeated generations.

First solve Rettifilo transfer by search: `945->1000`, then `945->1080`, then `650->1220`, then `520->1220`. After search finds recoverable elites, train PPO from those curriculum starts and test transfer back to honest normal-start eval. If search cannot find viable elites, change the action/controller/search representation or scoring contract instead of spending more time on tiny PPO guesses.

Do not count evolutionary/scripted/search trajectories as final success and do not initialize final scratch PPO policy weights from them. Use search to discover behavior, states, missing primitives, and curriculum targets. Final success still requires honest metadata-faithful PPO normal-start completion with scaffolds and assists disabled.

Be direct and progress-focused. Do not micro-engineer. Do not polish isolated completions. Do not treat runtime or artifacts as progress. The next proof must be search-discovered linked section transfer, then PPO reproduction, then honest normal-start improvement.
