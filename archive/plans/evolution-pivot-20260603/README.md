# Evolution Pivot Archive

This folder preserves the planning and goal files that were active before the 2026-06-03 pivot to reusable evolutionary search.

The archived files are kept for context only. The active next-step docs are:

- `EvolutionSearchPlan.md`
- `EvolutionGoal.md`

Reason for the pivot:

- More than six hours of curriculum/PPO micro-rungs produced useful local Rettifilo skills, but no honest normal-start full-lap progress beyond the old `~970.775m` failure.
- The agent was drifting into narrow hand-tuned gates, action sets, and one-off schedule presets.
- The project needs a Yosh-style population loop: spawn many candidate drivers, score them, keep elites, mutate, repeat, save elite trajectories/states, then use those discoveries for curriculum/PPO transfer.
